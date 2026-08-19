"""When does PPO's state become doomed, and in which joints does it differ
from the surviving reachtree state at the same arc length?

For each step t: D_max(state) = longest survivable continuation (width-1024
probe, horizon 45) from PPO's q(t) and from reachtree's q(t). Plus per-joint
|q_ppo - q_rt|(t).
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
model = hl.StraightModel(env)

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
tr = np.load(RUN / 'traj_compare.npz')
rt = np.load(RUN / 'reachtree.npz')
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=env.kin.dtype)     # (71,7)
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)         # (107,7)

d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(qr[:1])[0][0]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
H, W, GRID = 45, 1024, 0.02


@torch.no_grad()
def d_max(q0):
    q = q0[None]
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


ts = list(range(20, 70, 3))
deg = 180 / np.pi
print("t    s(m)  Dmax_PPO  Dmax_RT   |dq| per joint (deg, PPO-RT)")
res = []
t0 = time.time()
for t in ts:
    dp = d_max(qp[t])
    dr = d_max(qr[t])
    dq = (qp[t] - qr[t]).abs().cpu().numpy() * deg
    res.append((t, dp, dr, dq))
    print(f"{t:>3} {t*0.01:>6.2f} {dp:>8} {dr:>8}   "
          + " ".join(f"{v:5.1f}" for v in dq), flush=True)
np.savez(RUN / 'doom_onset.npz',
         t=np.array([r[0] for r in res]),
         dmax_ppo=np.array([r[1] for r in res]),
         dmax_rt=np.array([r[2] for r in res]),
         dq=np.stack([r[3] for r in res]))
print(f"({time.time()-t0:.0f}s)")
