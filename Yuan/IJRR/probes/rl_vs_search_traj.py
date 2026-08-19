"""Clean 7-joint trajectory figure: RL local optimum (0.70 m) vs the longer
search trajectory (1.06 m), against arc length, with joint limits."""
import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
tr = np.load(RUN / 'traj_compare.npz')     # PPO_q (71,7)
rt = np.load(RUN / 'reachtree.npz')        # q (107,7)
qp, qs = tr['PPO_q'], rt['q']
deg = 180 / np.pi
q_mid = env.q_mid.cpu().numpy(); q_half = env.q_half.cpu().numpy()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
sp = np.arange(len(qp)) * 0.01
ss = np.arange(len(qs)) * 0.01
for j in range(7):
    ax = axes.flat[j]
    ax.plot(sp, qp[:, j] * deg, c='tab:blue', lw=1.8,
            label='RL local optimum (PPO, 0.70 m)')
    ax.plot(ss, qs[:, j] * deg, c='tab:red', lw=1.8,
            label='search trajectory (1.06 m)')
    ax.axvline(0.70, c='tab:blue', lw=0.8, ls=':', alpha=0.8)
    for sgn in (-1, 1):
        lim = (q_mid[j] + sgn * q_half[j]) * deg
        if abs(lim) < 400:
            ax.axhline(lim, color='k', lw=0.7, ls='--', alpha=0.55)
    ax.set_title(f'joint {j + 1}', fontsize=11)
    ax.set_ylabel('deg', fontsize=9)
    ax.grid(alpha=0.2)
ax = axes.flat[7]
ax.plot(sp, np.linalg.norm((qp - qs[:len(qp)]) * deg, axis=1)
        if False else np.abs(qp - qs[:len(qp)]).max(1) * deg,
        c='tab:purple', lw=1.6)
ax.set_title('max joint difference |RL − search| (deg)', fontsize=11)
ax.set_ylabel('deg', fontsize=9)
ax.grid(alpha=0.2)
for a2 in axes[1]:
    a2.set_xlabel('arc length s (m)', fontsize=10)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2,
           fontsize=11, frameon=False)
fig.suptitle('task 27: RL local-optimum trajectory vs the longer search '
             'trajectory (dotted blue = where RL dies; dashed = joint '
             'limits)', y=1.06, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = RUN / 'rl_vs_search_traj.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'wrote {out}')
