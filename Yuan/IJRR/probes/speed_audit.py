"""Audit: is 'no episode ever passed 0.73 despite the half-speed gear'
anomalous or expected?

1. unit check: the speed channel really halves the step in the env;
2. random rollouts AT HALF SPEED from q0 and from PPO's fork-region states:
   can undirected exploration use the wider corridor, or is it search-only?
3. viable-action count at v=0.1 vs v=0.2 at the fork states.
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
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
tr = np.load(RUN / 'traj_compare.npz')
spec = {k: torch.tensor(task[k], device=dev, dtype=torch.float32).unsqueeze(0)
        for k in ('q0', 'line_dir', 'n_target')}

# ---- 1. unit check of the speed channel ------------------------------------
env1 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2,
                                   'speed_levels': (1.0, 0.5)}), None, dev)
env1.line_dist = SingleTaskDistribution(
    {k: v.to(env1.kin.dtype) for k, v in spec.items()})
env1.reset()
p0u = env1.p_start.clone()
a = torch.zeros(2, 5, device=dev)
a[0, 4] = 1.0
a[1, 4] = 0.5
env1.step(a, auto_reset=False)
p1, _, _, _ = env1.kin.tcp_fk_jac(env1.q)
dp = ((p1 - p0u) * env1.line_dir).sum(-1)
print(f"unit check: full-speed step {float(dp[0])*1000:.2f} mm, "
      f"half-speed step {float(dp[1])*1000:.2f} mm "
      f"(expect ~10 / ~5)")

# ---- 2. random rollouts at half speed --------------------------------------
N = 4096
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=torch.float32)


def random_rollouts(v, q_start, rounds=5, max_steps=400):
    envr = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': N, 'v': v}),
                          None, dev)
    sp = {'q0': q_start[None].to(envr.kin.dtype),
          'line_dir': spec['line_dir'].to(envr.kin.dtype),
          'n_target': spec['n_target'].to(envr.kin.dtype)}
    envr.line_dist = SingleTaskDistribution(sp)
    # progress measured from the ORIGINAL task p0 for comparability
    env_probe = envr
    best = -1e9
    n = 0
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'),
                 -1).reshape(-1, 4), dtype=torch.float32, device=dev)
    p0_task = None
    for r in range(rounds):
        torch.manual_seed(4000 + r)
        envr.reset()
        if p0_task is None:
            # original line origin = FK of the task q0
            p0_task = envr.kin.tcp_fk_jac(
                torch.tensor(task['q0'], device=dev,
                             dtype=envr.kin.dtype)[None])[0][0]
        u = envr.line_dir[0]
        for t in range(max_steps):
            a_idx = torch.randint(0, 16, (N,), device=dev)
            envr.step(verts[a_idx], auto_reset=False)
            if bool(envr.done_persistent.all()):
                break
        p, _, _, _ = envr.kin.tcp_fk_jac(envr.q)
        prog = ((p - p0_task) * u).sum(-1)
        best = max(best, float(prog.max()))
        n += N
    return best, n


q0_t = torch.tensor(task['q0'], device=dev, dtype=torch.float32)
for name, qs in (('q0', q0_t), ('PPO@s=0.40', qp[40]), ('PPO@s=0.50', qp[50])):
    for v in (0.2, 0.1):
        best, n = random_rollouts(v, qs)
        print(f"random rollouts from {name:<11s} v={v}: {n} eps, "
              f"farthest arc {best:.4f} m", flush=True)

# ---- 3. viable-action count at half speed ----------------------------------
env2 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
d = torch.tensor(task['line_dir'], device=dev, dtype=env2.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env2.kin.dtype)
p0 = env2.kin.tcp_fk_jac(q0_t[None].to(env2.kin.dtype))[0][0]


def viable(q_state, v, H=40, W=1024, grid=0.02):
    model = hl.StraightModel(env2)
    model.cfg = dataclasses.replace(env2.cfg, v=v)
    out = []
    for ai in range(16):
        qn = model.step(q_state[None].to(env2.kin.dtype), d[None], n_t[None],
                        verts[ai][None])
        m = model.margins(qn, p0[None], d[None], n_t[None])
        if not bool(m.amin() > 0):
            out.append(0)
            continue
        q = qn
        depth = 1
        for _ in range(H - 1):
            P = q.shape[0]
            qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
            ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
            qq = torch.cat([model.step(qe[i:i+32768],
                                       d.expand(min(32768, P*16-i), 3),
                                       n_t.expand(min(32768, P*16-i), 3),
                                       ae[i:i+32768])
                            for i in range(0, P*16, 32768)])
            mm = torch.cat([model.margins(qq[i:i+32768],
                                          p0.expand(min(32768, P*16-i), 3),
                                          d.expand(min(32768, P*16-i), 3),
                                          n_t.expand(min(32768, P*16-i), 3))
                            for i in range(0, P*16, 32768)])
            alive = (mm.amin(-1) > 0)
            if not bool(alive.any()):
                break
            qq = qq[alive]
            key = torch.round(qq / grid).to(torch.int32)
            _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
            keep = torch.as_tensor(np.sort(first), device=dev)
            if keep.numel() > W:
                keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
            q = qq[keep]
            depth += 1
        out.append(depth)
    out = np.array(out)
    return out


for name, qs in (('PPO@s=0.40', qp[40]), ('PPO@s=0.50', qp[50])):
    for v in (0.2, 0.1):
        Dv = viable(qs, v)
        print(f"{name} v={v}: viable(D>=40) {(Dv >= 40).sum():>2}/16  "
              f"Dmax {Dv.max():>2}  D per action {Dv.tolist()}", flush=True)
