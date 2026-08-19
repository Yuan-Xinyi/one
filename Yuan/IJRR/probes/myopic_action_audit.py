"""Step-by-step audit of myopic on task 27: at every step, what could each of
the 16 vertices do to m_jl / m_cone one period ahead, and what did the argmax
of softmin(m_jl, m_cone) actually choose."""
import numpy as np
import torch
import yaml
from pathlib import Path
import sys

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env)
model.terms = [0, 1]
tau = 0.1

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
spec = {k: torch.tensor(task[k], device=dev, dtype=env.kin.dtype).unsqueeze(0)
        for k in ('q0', 'line_dir', 'n_target')}

from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
env.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
env.reset()
p0, d, n = env.p_start[0], env.line_dir[0], env.n_target[0]

verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
K = 16

print(f"{'t':>3} {'m_jl':>7} {'m_cone':>7} | next-step m_cone over 16 verts "
      f"{'best':>7} {'worst':>7} {'chosen':>7} | next m_jl {'chosen':>7} "
      f"{'@bestcone':>9} | limiting")
for t in range(200):
    q = env.q[0:1]
    m_now = model.margins(q, p0[None], d[None], n[None])[0]
    qe = q.expand(K, -1)
    qn = model.step(qe, d.expand(K, 3), n.expand(K, 3), verts)
    mn = model.margins(qn, p0.expand(K, 3), d.expand(K, 3), n.expand(K, 3))
    sm = -tau * torch.logsumexp(-mn[:, [0, 1]] / tau, dim=-1)
    pick = int(sm.argmax())
    best_cone = int(mn[:, 1].argmax())
    limiting = 'jl' if m_now[0] < m_now[1] else 'cone'
    print(f"{t:>3} {m_now[0]:>7.3f} {m_now[1]:>7.3f} | "
          f"{' ':31s}{mn[:, 1].max():>7.3f} {mn[:, 1].min():>7.3f} "
          f"{mn[pick, 1]:>7.3f} |           {mn[pick, 0]:>7.3f} "
          f"{mn[best_cone, 0]:>9.3f} | {limiting}")
    _, _, _, _, info = env.step(verts[pick:pick + 1], auto_reset=False)
    if bool(info['episode_done'][0]):
        print(f"dead at step {t + 1}, term "
              f"{ {0:'alive',2:'collision',3:'cone',4:'jl',5:'trunc',6:'lat'}[int(info['term_reason'][0])] }")
        break
