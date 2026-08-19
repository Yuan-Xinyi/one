"""Value-estimation audit: does the RL critic UNDER-value the filament?

State level: at matched arc s, compare corridor-1 states (PPO trajectory)
vs filament states (winning trajectory) under three estimators:
  critic V (learned), MC survival under the policy (sanity), search max
  survival (V* proxy).
Action level: at fork-region states, rank all 16 actions by critic-V(next)
vs search-Dmax(next).
"""
import numpy as np
import torch
import yaml
from pathlib import Path
import sys, time

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
tr = np.load(RUN / 'traj_compare.npz')
rt = np.load(RUN / 'reachtree.npz')

envB = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 128}), None, dev)
env1 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env1)
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=env1.kin.dtype)
qr = torch.tensor(rt['q'], device=dev, dtype=env1.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env1.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env1.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env1.kin.dtype)
p0 = env1.kin.tcp_fk_jac(q0_t[None])[0][0]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)

agent = VertexAgent(obs_dim=envB.obs_dim, act_dim=envB.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev))
agent.eval()


@torch.no_grad()
def critic_of(q_state, env):
    env.line_dist = SingleTaskDistribution(
        {'q0': q_state[None], 'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    return float(agent.critic(env.current_obs()).mean())


@torch.no_grad()
def mc_survival(q_state, n_roll=128, max_steps=200):
    envB.line_dist = SingleTaskDistribution(
        {'q0': q_state[None], 'line_dir': d[None], 'n_target': n_t[None]})
    envB.reset()
    for t in range(max_steps):
        logits = agent._logits_head(agent._actor_trunk(envB.current_obs()))
        a_idx = torch.distributions.Categorical(logits=logits).sample()
        envB.step(agent.vertices[a_idx], auto_reset=False)
        if bool(envB.done_persistent.all()):
            break
    return float(envB.episode_steps.float().mean())


@torch.no_grad()
def search_survival(q_state, H=45, W=1024, grid=0.02):
    q = q_state[None]
    for depth in range(H):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
        CH = 32768
        qn = torch.cat([model.step(qe[i:i+CH], d.expand(min(CH, P*16-i), 3),
                                   n_t.expand(min(CH, P*16-i), 3), ae[i:i+CH])
                        for i in range(0, P*16, CH)])
        m = torch.cat([model.margins(qn[i:i+CH], p0.expand(min(CH, P*16-i), 3),
                                     d.expand(min(CH, P*16-i), 3),
                                     n_t.expand(min(CH, P*16-i), 3))
                       for i in range(0, P*16, CH)])
        alive = (m.amin(-1) > 0)
        if not bool(alive.any()):
            return depth
        qn = qn[alive]
        key = torch.round(qn / grid).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
        q = qn[keep]
    return H


print("=== state level: corridor-1 (PPO traj) vs FILAMENT (winning traj) ===")
print(f"{'s(m)':>5} | {'critic V':>17} | {'MC survival':>17} | "
      f"{'search survival':>17}")
print(f"{'':>5} | {'corr':>8}{'fil':>8} | {'corr':>8}{'fil':>8} | "
      f"{'corr':>8}{'fil':>8}")
rows = []
for t in (20, 30, 40, 50, 56, 62, 68):
    vc = critic_of(qp[t], env1)
    vf = critic_of(qr[t], env1)
    mc = mc_survival(qp[t])
    mf = mc_survival(qr[t])
    sc = search_survival(qp[t])
    sf = search_survival(qr[t])
    rows.append((t, vc, vf, mc, mf, sc, sf))
    print(f"{t*0.01:5.2f} | {vc:8.2f}{vf:8.2f} | {mc:8.1f}{mf:8.1f} | "
          f"{sc:8d}{sf:8d}", flush=True)

print("\n=== action level at fork states (all 16 vertices) ===")
for t in (40, 50):
    q_t = qp[t]
    vs, ss = [], []
    for a in range(16):
        qn = model.step(q_t[None], d[None], n_t[None], verts[a][None])
        m = model.margins(qn, p0[None], d[None], n_t[None])
        if not bool(m.amin() > 0):
            vs.append(float('nan')); ss.append(0); continue
        vs.append(critic_of(qn[0], env1))
        ss.append(1 + search_survival(qn[0], H=40))
    vs, ss = np.array(vs), np.array(ss)
    ok = ~np.isnan(vs)
    from scipy.stats import spearmanr
    rho = spearmanr(vs[ok], ss[ok]).statistic if ok.sum() > 2 else float('nan')
    b_search = int(np.argmax(ss))
    print(f"t={t} (s={t*0.01:.2f}): search-best action #{b_search} "
          f"(survives {ss[b_search]}) is critic-ranked "
          f"{int((vs[ok] > vs[b_search]).sum()) + 1}/{int(ok.sum())}; "
          f"spearman(critic, search) = {rho:.2f}")
    print("  search:", ss.tolist())
    print("  critic:", [f"{v:.2f}" for v in vs])
np.savez(RUN / 'value_audit.npz', rows=np.array(rows))
print(f"\nwrote {RUN / 'value_audit.npz'}")
