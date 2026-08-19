"""Generate Vhat* training labels: search-probe max-survival at ~1500 states
drawn from PPO rollouts, the reachtree bank, the GE archive and the winning
route (+ jitter). Label = min(D,40) probe steps, saved with the state's OBS
(computed by resetting the env to the state)."""
import numpy as np, torch, yaml, sys, time
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
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 256}), None, dev)
env1 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env1)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]
verts = torch.tensor(np.stack(np.meshgrid(*[[-1.,1.]]*4, indexing='ij'),
                     -1).reshape(-1,4), dtype=torch.float32, device=dev)
spec = {'q0': q0_t[None], 'line_dir': d[None], 'n_target': n_t[None]}

# ---- state pool ----
states = []
# (a) PPO stochastic rollout cloud
ppo = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                  hidden_dim=y['ppo']['hidden_dim']).to(dev)
ppo.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev)); ppo.eval()
env.line_dist = SingleTaskDistribution(spec)
env.reset()
with torch.no_grad():
    for t in range(90):
        logits = ppo._logits_head(ppo._actor_trunk(env.current_obs()))
        a = torch.distributions.Categorical(logits=logits).sample()
        env.step(ppo.vertices[a], auto_reset=False)
        if t % 15 == 7:
            alive = (~env.done_persistent).nonzero().squeeze(-1)
            if alive.numel():
                pick = alive[torch.randperm(alive.numel())[:80]]
                states.append(env.q[pick].clone())
# (b) reachtree bank
rb = np.load(RUN / 'reachtree_bank.npz')
ridx = np.random.default_rng(0).choice(len(rb['q']), 450, replace=False)
states.append(torch.tensor(rb['q'][ridx], device=dev, dtype=env.kin.dtype))
# (c) GE archive
ga = np.load(RUN / 'goexplore_archive.npz')
gidx = np.random.default_rng(1).choice(len(ga['q']), 350, replace=False)
states.append(torch.tensor(ga['q'][gidx], device=dev, dtype=env.kin.dtype))
# (d) winning route + jitter
rt = np.load(RUN / 'reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
states.append(qr)
states.append(qr + 0.01 * torch.randn(3 * 107, 7, device=dev,
              dtype=env.kin.dtype).reshape(3, 107, 7)[0])
S = torch.cat(states)[:1600]
print(f"state pool: {S.shape[0]}")

@torch.no_grad()
def v_star(s, H=40, W=256, grid=0.02):
    m0 = model.margins(s[None], p0[None], d[None], n_t[None])
    if not bool(m0.amin() > 0):
        return 0
    q = s[None]
    for depth in range(H):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1,16,-1).reshape(P*16,-1)
        ae = verts.unsqueeze(0).expand(P,-1,-1).reshape(P*16,-1)
        qn = model.step(qe, d.expand(P*16,3), n_t.expand(P*16,3), ae)
        m = model.margins(qn, p0.expand(P*16,3), d.expand(P*16,3),
                          n_t.expand(P*16,3))
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

# obs for each state (env reset trick) + label
obs_all, lab_all = [], []
t0 = time.time()
for i in range(S.shape[0]):
    env1.line_dist = SingleTaskDistribution(
        {'q0': S[i][None], 'line_dir': d[None], 'n_target': n_t[None]})
    env1.reset()
    obs_all.append(env1.current_obs()[0].clone())
    lab_all.append(v_star(S[i]))
    if i % 200 == 0:
        print(f"{i}/{S.shape[0]}  {time.time()-t0:.0f}s", flush=True)
obs = torch.stack(obs_all).cpu().numpy()
lab = np.array(lab_all, dtype=np.float32)
np.savez(RUN / 'vstar_labels.npz', obs=obs, label=lab)
print(f"labels: mean {lab.mean():.1f} max {lab.max()} zeros {(lab==0).mean():.2f}")
print(f"wrote {RUN / 'vstar_labels.npz'}  ({time.time()-t0:.0f}s)")
