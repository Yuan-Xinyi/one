"""From PPO's state at t0, can ANY feasible continuation merge into the
reachtree corridor? Forward tree (W=4096) for 35 steps; at each depth record
the closest approach (per-joint max |dq|) of the surviving pool to the
reachtree state at the same global step. The last t0 that can still merge is
the dynamic commitment point (the fork)."""
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
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=env.kin.dtype)
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)

d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(qr[:1])[0][0]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
W, GRID, H = 4096, 0.02, 35
deg = 180 / np.pi


@torch.no_grad()
def merge_probe(t0):
    q = qp[t0][None]
    closest = 1e9
    closest_depth = -1
    for depth in range(1, H + 1):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
        CH = 32768
        qn = torch.cat([model.step(qe[i:i + CH],
                                   d.expand(min(CH, P * 16 - i), 3),
                                   n.expand(min(CH, P * 16 - i), 3),
                                   ae[i:i + CH])
                        for i in range(0, P * 16, CH)])
        m = torch.cat([model.margins(qn[i:i + CH],
                                     p0.expand(min(CH, P * 16 - i), 3),
                                     d.expand(min(CH, P * 16 - i), 3),
                                     n.expand(min(CH, P * 16 - i), 3))
                       for i in range(0, P * 16, CH)])
        alive = (m.amin(-1) > 0)
        if not bool(alive.any()):
            return closest, closest_depth, depth   # died at this depth
        qn = qn[alive]
        key = torch.round(qn / GRID).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
        q = qn[keep]
        g = t0 + depth
        if g < qr.shape[0]:
            dist = (q - qr[g][None]).abs().amax(-1).amin() * deg
            if float(dist) < closest:
                closest = float(dist)
                closest_depth = depth
    return closest, closest_depth, -1


print("t0   s(m)   closest approach to RT corridor (deg, max-joint)  "
      "@depth  tree-died-at")
for t0 in (10, 20, 30, 38, 44, 50, 56):
    t0s = time.time()
    c, cd, died = merge_probe(t0)
    print(f"{t0:>3} {t0*0.01:>6.2f}   {c:>10.1f} {cd:>18} "
          f"{'alive@'+str(H) if died < 0 else 'died@'+str(died):>14}  "
          f"({time.time()-t0s:.0f}s)", flush=True)
