"""Q-switch audit (user's design): at bottleneck states s on the winning
route, for a_gate (search action) and a_safe (PPO's argmax):
  Q^pi(s,a)  = remaining arc under PPO continuation after forcing a
  Qhat*(s,a) = r + gamma*Vhat*(s'), Vhat* from a search probe at s'
Report/plot dQ^pi = gate-safe and dQ* = gate-safe. Units: remaining arc (m).
"""
import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2}), None, dev)
model = hl.StraightModel(env)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz'); rt = np.load(RUN / 'reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
aseq = rt['action_idx']
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]
verts = torch.tensor(np.stack(np.meshgrid(*[[-1.,1.]]*4, indexing='ij'),
                     -1).reshape(-1,4), dtype=torch.float32, device=dev)
ppo = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                  hidden_dim=y['ppo']['hidden_dim']).to(dev)
ppo.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev)); ppo.eval()

@torch.no_grad()
def arc_of(q):
    p, _, _, _ = env.kin.tcp_fk_jac(q[None]); return float(((p[0]-p0)*d).sum())

@torch.no_grad()
def q_pi(s, a):
    env.line_dist = SingleTaskDistribution(
        {'q0': s[None], 'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    env.step(verts[int(a)][None].expand(2,-1), auto_reset=False)
    for t in range(400):
        if bool(env.done_persistent.all()): break
        env.step(ppo.actor_mean(env.current_obs()), auto_reset=False)
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    return float(((p[0]-p0)*d).sum()) - arc_of(s)

@torch.no_grad()
def v_star(s_next, H=45, W=1024, grid=0.02):
    q = s_next[None]
    for depth in range(H):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1,16,-1).reshape(P*16,-1)
        ae = verts.unsqueeze(0).expand(P,-1,-1).reshape(P*16,-1)
        CH = 32768
        qn = torch.cat([model.step(qe[i:i+CH], d.expand(min(CH,P*16-i),3),
                                   n_t.expand(min(CH,P*16-i),3), ae[i:i+CH])
                        for i in range(0,P*16,CH)])
        m = torch.cat([model.margins(qn[i:i+CH], p0.expand(min(CH,P*16-i),3),
                                     d.expand(min(CH,P*16-i),3),
                                     n_t.expand(min(CH,P*16-i),3))
                       for i in range(0,P*16,CH)])
        alive = (m.amin(-1) > 0)
        if not bool(alive.any()): return depth
        qn = qn[alive]
        key = torch.round(qn/grid).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
        q = qn[keep]
    return H

@torch.no_grad()
def q_star(s, a):
    sn = model.step(s[None], d[None], n_t[None], verts[int(a)][None])
    m = model.margins(sn, p0[None], d[None], n_t[None])
    if not bool(m.amin() > 0): return 0.0
    return 0.01 * (1 + v_star(sn[0]))     # arc units: 1cm/step

rows = []
print(f"{'arc':>5} {'aG':>3} {'aS':>3} | {'Qpi_G':>7} {'Qpi_S':>7} "
      f"{'dQpi':>7} | {'Q*_G':>6} {'Q*_S':>6} {'dQ*':>7}")
for t in range(28, 72, 4):
    s = qr[t]
    aG = int(aseq[t])
    env.line_dist = SingleTaskDistribution(
        {'q0': s[None], 'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    logits = ppo._logits_head(ppo._actor_trunk(env.current_obs()))[0]
    aS = int(logits.argmax())
    qpg, qps = q_pi(s, aG), q_pi(s, aS)
    qsg, qss = q_star(s, aG), q_star(s, aS)
    rows.append((t*0.01, qpg, qps, qsg, qss))
    print(f"{t*0.01:5.2f} {aG:>3} {aS:>3} | {qpg:7.3f} {qps:7.3f} "
          f"{qpg-qps:+7.3f} | {qsg:6.3f} {qss:6.3f} {qsg-qss:+7.3f}",
          flush=True)
np.savez(RUN / 'q_switch_audit.npz', rows=np.array(rows))
print(f"wrote {RUN / 'q_switch_audit.npz'}")
