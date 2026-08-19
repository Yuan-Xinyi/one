"""Was PPO's death near the joint-6 limit caused by the 50 ms action quantum?

Along the last steps of PPO's recorded trajectory:
  - per-step joint-6 motion vs distance-to-limit (quantum vs margin);
  - D50(t,a): survivable depth after each vertex at the 50 ms quantum;
  - D25 escape test: at states where NO 50 ms action survives deeply, does
    any 25 ms half-step command sequence survive? (256 combos per state,
    then a 25 ms-resolution survival probe)
"""
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
model50 = hl.StraightModel(env)
model25 = hl.StraightModel(env)
model25.cfg = dataclasses.replace(env.cfg, dt=env.cfg.dt / 2)

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
tr = np.load(RUN / 'traj_compare.npz')
q_traj = torch.tensor(tr['PPO_q'], device=dev, dtype=env.kin.dtype)  # (71,7)
T = q_traj.shape[0] - 1

d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q_traj[:1])[0][0]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)

# 1. quantum vs margin, last 12 steps
q_half = env.q_half
q_mid = env.q_mid
print("step |dq6|/step(deg)  dist_to_j6_limit(deg)  m_jl")
for t in range(T - 12, T):
    dq6 = float((q_traj[t + 1, 5] - q_traj[t, 5]).abs()) * 180 / np.pi
    dist = float((q_half[5] - (q_traj[t, 5] - q_mid[5]).abs())) * 180 / np.pi
    m = model50.margins(q_traj[t:t + 1], p0[None], d[None], n[None])[0]
    print(f"{t:>4} {dq6:>14.2f} {dist:>20.2f}  m_jl={float(m[0]):.4f} "
          f"m_cone={float(m[1]):.4f}")


@torch.no_grad()
def survive_depth(q_pool, model, substeps, horizon, W=1024, grid=0.02):
    q = q_pool
    for depth in range(horizon):
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
        qn = model.step(qe, d.expand(P * 16, 3), n.expand(P * 16, 3), ae,
                        substeps=substeps)
        m = model50.margins(qn, p0.expand(P * 16, 3), d.expand(P * 16, 3),
                            n.expand(P * 16, 3))
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
    return horizon


# 2. D50 per action on the last 8 states + 3. 25 ms escape test
H50 = 30
print(f"\nper-state: viable actions at 50 ms (D50>=depth_cap {H50}) "
      f"vs 25 ms escape")
for t in range(T - 8, T):
    q_t = q_traj[t]
    d50 = []
    for a in range(16):
        qn = model50.step(q_t[None], d[None], n[None], verts[a][None],
                          substeps=1)
        m = model50.margins(qn, p0[None], d[None], n[None])
        if not bool(m.amin() > 0):
            d50.append(0)
            continue
        d50.append(1 + survive_depth(qn, model50, 1, H50 - 1))
    d50 = np.array(d50)

    # 25 ms: all 256 half-step pairs covering one 50 ms period
    q1 = model25.step(q_t.expand(16, 7), d.expand(16, 3), n.expand(16, 3),
                      verts, substeps=1)
    m1 = model50.margins(q1, p0.expand(16, 3), d.expand(16, 3),
                         n.expand(16, 3))
    ok1 = (m1.amin(-1) > 0)
    q2 = model25.step(q1.repeat_interleave(16, 0), d.expand(256, 3),
                      n.expand(256, 3), verts.repeat(16, 1), substeps=1)
    m2 = model50.margins(q2, p0.expand(256, 3), d.expand(256, 3),
                         n.expand(256, 3))
    ok2 = (m2.amin(-1) > 0) & ok1.repeat_interleave(16)
    surv25 = 0
    if bool(ok2.any()):
        pool = q2[ok2]
        key = torch.round(pool / 0.02).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        pool = pool[torch.as_tensor(np.sort(first), device=dev)][:1024]
        surv25 = 1 + survive_depth(pool, model25, 1, 2 * H50 - 2) / 2
    print(f"t={t:>3}  D50 max {d50.max():>2}  "
          f"viable50(D>={H50}) {(d50 >= H50).sum():>2}/16  "
          f"alive-1step50 {(d50 > 0).sum():>2}/16  "
          f"alive-25ms-pairs {int(ok2.sum()):>3}/256  "
          f"surv25({H50} eq-steps) {surv25:.1f}")
