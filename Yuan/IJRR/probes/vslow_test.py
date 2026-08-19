"""If the task-space speed v were adjustable, could PPO's committed state
still swing over? From PPO's state at t0 (committed at v=0.2), run the
dispersion tree with v in {0.2, 0.1, 0.05} and record the farthest arc
progress any continuation reaches."""
import numpy as np
import torch
import yaml
import dataclasses
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

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
tr = np.load(RUN / 'traj_compare.npz')
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=env.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(qp[:1])[0][0]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
W, GRID, CH = 4096, 0.02, 32768


@torch.no_grad()
def max_arc(q0, v, horizon):
    model = hl.StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, v=v)
    q = q0[None]
    best = -1e9
    for depth in range(horizon):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
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
            return best, depth
        qn = qn[alive]
        p, _, _, _ = env.kin.tcp_fk_jac(qn)
        best = max(best, float(((p - p0) * d).sum(-1).max()))
        key = torch.round(qn / GRID).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
        q = qn[keep]
    return best, horizon


print("from PPO state at t0 (arc s0), farthest arc reachable at each v:")
for t0 in (50, 56, 62):
    row = [f"t0={t0} (s0={t0*0.01:.2f})"]
    for v, hor in ((0.2, 70), (0.1, 140), (0.05, 280)):
        t0s = time.time()
        best, died = max_arc(qp[t0], v, hor)
        row.append(f"v={v}: {best:.3f} m (tree died depth {died}, "
                   f"{time.time()-t0s:.0f}s)")
    print("  ".join(row), flush=True)
