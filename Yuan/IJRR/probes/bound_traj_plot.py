"""Task 27: joint trajectories of myopic / PPO / the bound's IK witness chain,
all against arc length s along the task line."""
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

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
kin = env.kin
model = hl.StraightModel(env)

cs = np.load(RUN / 'task_cs.npz')
p0 = torch.tensor(cs['cs_p0'][0], device=dev, dtype=kin.dtype)
d = torch.tensor(cs['cs_line_dir'][0], device=dev, dtype=kin.dtype)
n = torch.tensor(cs['cs_n_target'][0], device=dev, dtype=kin.dtype)

tr = np.load(RUN / 'traj_compare.npz')
w = np.load(RUN / 'line_bound_task27_witness_q0.npz')

# replay-verify the reachtree open-loop sequence through the real env
rt = np.load(RUN / 'reachtree.npz')
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
task = np.load(RUN / 'task.npz')
env.line_dist = ScriptedLineDistribution(
    {k: torch.tensor(task[k], device=dev, dtype=kin.dtype).unsqueeze(0)
     for k in ('q0', 'line_dir', 'n_target')})
env.reset()
verts16 = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
for ai in rt['action_idx']:
    env.step(verts16[int(ai):int(ai) + 1], auto_reset=False)
p_end, _, _, _ = kin.tcp_fk_jac(env.q)
d_t = torch.tensor(task['line_dir'], device=dev, dtype=kin.dtype)
p0_t = torch.tensor(cs['cs_p0'][0], device=dev, dtype=kin.dtype)
replay_prog = float(((p_end[0] - p0_t) * d_t).sum())
print(f"reachtree replay through env: {replay_prog:.4f} m "
      f"(stored {float(rt['progress']):.4f}, "
      f"env alive at end: {not bool(env.done_persistent[0])})")
qw = w['q_witness'][0]                          # (n_steps+1, 7), NaN after death
valid = np.isfinite(qw).all(-1)
qw = qw[valid]
s_w = np.nonzero(valid)[0] * float(w['step'])
print(f'witness: {len(qw)} points, s in [0, {s_w.max():.2f}] m, '
      f'max |dq| between consecutive points = '
      f'{np.degrees(np.abs(np.diff(qw, axis=0)).max()):.1f} deg')

curves = {}
for name, q in (('myopic', tr['myopic_q']), ('PPO', tr['PPO_q']),
                ('search (reachtree)', rt['q'])):
    qt = torch.tensor(q, device=dev, dtype=kin.dtype)
    p, _, _, _ = kin.tcp_fk_jac(qt)
    s = ((p - p0) * d).sum(-1).cpu().numpy()
    m = model.margins(qt, p0.expand(len(qt), 3), d.expand(len(qt), 3),
                      n.expand(len(qt), 3)).cpu().numpy()
    curves[name] = (s, q, m)
qwt = torch.tensor(qw, device=dev, dtype=kin.dtype)
mw = model.margins(qwt, p0.expand(len(qwt), 3), d.expand(len(qwt), 3),
                   n.expand(len(qwt), 3)).cpu().numpy()
curves['bound witness'] = (s_w, qw, mw)
print('witness margins: m_jl min %.3f, m_cone min %.3f, m_lat min %.3f, '
      'm_coll min %.3f' % tuple(mw.min(0)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

style = {'myopic': dict(color='tab:blue', lw=1.6),
         'PPO': dict(color='tab:red', lw=1.6),
         'search (reachtree)': dict(color='tab:orange', lw=1.6),
         'bound witness': dict(color='tab:green', lw=1.2, ls='--')}
q_mid = env.q_mid.cpu().numpy()
q_half = env.q_half.cpu().numpy()
deg = 180.0 / np.pi

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
for j in range(7):
    ax = axes.flat[j]
    for name, (s, q, _) in curves.items():
        ax.plot(s, q[:, j] * deg, label=name, **style[name])
    for sgn in (-1, 1):
        ax.axhline((q_mid[j] + sgn * q_half[j]) * deg, color='k', lw=0.7,
                   ls=':', alpha=0.7)
    ax.set_title(f'joint {j + 1}', fontsize=10)
    ax.set_ylabel('deg', fontsize=8)
    ax.tick_params(labelsize=8)
for k, (mi, lab) in enumerate(((0, 'margin m_jl'), (1, 'margin m_cone'))):
    ax = axes.flat[7 + k]
    for name, (s, _, m) in curves.items():
        ax.plot(s, m[:, mi], label=name, **style[name])
    ax.axhline(0.0, color='k', lw=0.7, ls=':', alpha=0.7)
    ax.set_title(lab, fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes[-1]:
    ax.set_xlabel('arc length s along the line (m)', fontsize=9)
handles, labels = axes.flat[0].get_legend_handles_labels()
ends = {k: f'{v[0].max():.2f} m' for k, v in curves.items()}
fig.legend(handles, [f'{l} (reaches {ends[l]})' for l in labels],
           loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4,
           fontsize=9, frameon=False)
fig.suptitle('task 27, all starting from the same q0: rollouts vs the '
             'q0-seeded bound witness chain (per-point IK certificate)',
             y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = RUN / 'bound_traj_compare.png'
fig.savefig(out, dpi=160)
print(f'wrote {out}')
