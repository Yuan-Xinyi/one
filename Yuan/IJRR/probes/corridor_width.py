"""How narrow is the corridor along the reachtree 1.06 m trajectory?

For every step t and every vertex action a: apply a at q_t, then measure the
maximum survivable depth D(t,a) with a small dispersion-kept reachtree probe
(width 1024, horizon 40). n_viable(t) = number of actions whose D matches the
executed route's survivability. The product of n_viable/16 over a window is
the probability that ANY open-loop explorer stays viable through it.
"""
import numpy as np
import torch
import yaml
from pathlib import Path
import sys, time

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env)          # terms=None -> all four margins

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
rt = np.load(RUN / 'reachtree.npz')
traj_q = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)   # (107, 7)
act = rt['action_idx']
T = len(act)                            # 106

d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
# original line origin: FK of q0
p0 = env.kin.tcp_fk_jac(traj_q[:1])[0][0]

verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
W, H, GRID = 1024, 40, 0.02


@torch.no_grad()
def survive_depth(q_start):
    """Max depth (<=H) any continuation from q_start survives."""
    q = q_start[None]
    for depth in range(H):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
        qn = model.step(qe, d.expand(P * 16, 3), n.expand(P * 16, 3), ae)
        m = model.margins(qn, p0.expand(P * 16, 3), d.expand(P * 16, 3),
                          n.expand(P * 16, 3))
        alive = (m.amin(-1) > 0)
        if not bool(alive.any()):
            return depth
        qn = qn[alive]
        key = torch.round(qn / GRID).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
        q = qn[keep]
    return H


@torch.no_grad()
def one_step(q, a_idx):
    qn = model.step(q[None], d[None], n[None], verts[a_idx][None])
    m = model.margins(qn, p0[None], d[None], n[None])
    return qn[0], bool(m.amin() > 0)


D = np.zeros((T, 16), dtype=np.int64)
t0 = time.time()
for t in range(T):
    for a in range(16):
        qn, alive = one_step(traj_q[t], a)
        D[t, a] = 1 + survive_depth(qn) if alive else 0
    if t % 20 == 0:
        print(f"t={t}/{T}  {time.time() - t0:.0f}s", flush=True)

need = np.minimum(H, T - np.arange(T))          # survive as long as the route
n_viable = (D >= need[:, None]).sum(1)
np.savez(RUN / 'corridor_width.npz', D=D, n_viable=n_viable, act=act,
         probe_width=W, horizon=H)

logp = np.log(np.maximum(n_viable, 1) / 16.0)
w0, w1 = 40, 90
print(f"\nn_viable: min {n_viable.min()}, "
      f"steps with <=2 viable: {(n_viable <= 2).sum()}/{T}, "
      f"<=1: {(n_viable <= 1).sum()}")
print(f"P(random explorer viable through steps {w0}-{w1}) = "
      f"exp({logp[w0:w1].sum():.1f}) = {np.exp(logp[w0:w1].sum()):.2e}")
print(f"P over full 0-{T} = {np.exp(logp.sum()):.2e}")
print(f"wrote {RUN / 'corridor_width.npz'}  ({time.time() - t0:.0f}s)")
