"""Classical nullspace controller takes over task 27: how far does it get,
and which side of the j7 fan does it pick? Also runs sgnclassical."""
import matplotlib                       # must precede torch on this box
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import torch
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.eval import horizon_ladder as hl
from Yuan.IJRR.eval.single_task_ppo import (_env_and_yaml, _load_task,
                                            _record_traj, TERM_NAMES)
import Yuan.IJRR.eval.single_task_ppo as stp

dev = torch.device('cuda')
stp.OUT = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
y, env = _env_and_yaml(1, dev)
task, spec1 = _load_task(dev, env.kin.dtype)

classical = ClassicalNullspaceController(env.kin)
fcl = cn_action_fn(classical)
sgn = hl.make_sgnclassical(classical)

trajs = {}
for name, fn in (('classical', lambda e: fcl(e)),
                 ('sgnclassical', lambda e: sgn(e, e.done_persistent))):
    q, term = _record_traj(env, spec1, fn)
    p0 = env.p_start[0]
    d = env.line_dir[0]
    pf, _, _, _ = env.kin.tcp_fk_jac(q.to(dev))
    prog = float(((pf[-1] - p0) * d).sum())
    trajs[name] = (q.cpu().numpy(), prog, term)
    print(f"{name:<13s} {q.shape[0]-1:>3d} steps  progress {prog:.4f} m  "
          f"term {TERM_NAMES[term]}")

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
np.savez(RUN / 'classical_task27.npz',
         **{f'{k}_q': v[0] for k, v in trajs.items()},
         **{f'{k}_progress': v[1] for k, v in trajs.items()})

# overlay on the bundle figure
tr = np.load(RUN / 'traj_compare.npz')
bu = np.load(RUN / 'reachtree_bundle.npz')
qp = tr['PPO_q']; qm = tr['myopic_q']
flat, lens, progs = bu['q_flat'], bu['lens'], bu['progress']
offs = np.concatenate([[0], np.cumsum(lens)])
order = np.argsort(progs)
best = int(progs.argmax())
deg = 180 / np.pi
norm = Normalize(vmin=0.70, vmax=float(progs.max()))
cmap = plt.get_cmap('viridis')
q_mid = env.q_mid.cpu().numpy(); q_half = env.q_half.cpu().numpy()

fig, axes = plt.subplots(2, 4, figsize=(16.5, 8), sharex=True,
                         constrained_layout=True)
for j in range(7):
    ax = axes.flat[j]
    segs = [np.column_stack([np.arange(lens[i]) * 0.01,
                             flat[offs[i]:offs[i + 1], j] * deg])
            for i in order if i != best]
    cols = [cmap(norm(progs[i])) for i in order if i != best]
    ax.add_collection(LineCollection(segs, colors=cols, lw=0.4, alpha=0.08))
    qq = flat[offs[best]:offs[best + 1], j] * deg
    ax.plot(np.arange(lens[best]) * 0.01, qq, c='tab:red', lw=2.0, zorder=5,
            label='best search (1.06 m)')
    ax.plot(np.arange(len(qp)) * 0.01, qp[:, j] * deg, c='tab:blue', lw=1.8,
            zorder=6, label='PPO (0.73 m)')
    ax.plot(np.arange(len(qm)) * 0.01, qm[:, j] * deg, c='tab:purple',
            lw=1.8, zorder=6, label=f'myopic (0.17 m)')
    for name, c in (('classical', 'black'), ('sgnclassical', 'tab:orange')):
        q, prog, term = trajs[name]
        ax.plot(np.arange(len(q)) * 0.01, q[:, j] * deg, c=c, lw=1.8,
                zorder=7,
                label=f'{name} ({prog:.2f} m, {TERM_NAMES[term]})')
    for sgn_ in (-1, 1):
        lim = (q_mid[j] + sgn_ * q_half[j]) * deg
        if abs(lim) < 400:
            ax.axhline(lim, color='k', lw=0.7, ls='--', alpha=0.55)
    ax.set_title(f'joint {j + 1}', fontsize=11)
    if j % 4 == 0:
        ax.set_ylabel('joint angle (deg)', fontsize=9)
    ax.autoscale_view()
    ax.grid(alpha=0.2)

ax = axes.flat[7]
ax.hist(progs, bins=36, color='tab:green', alpha=0.75)
for name, c in (('classical', 'black'), ('sgnclassical', 'tab:orange')):
    ax.axvline(trajs[name][1], c=c, lw=1.5, label=name)
ax.axvline(0.73, c='tab:blue', ls=':', lw=1.5, label='PPO')
ax.axvline(float(progs.max()), c='tab:red', lw=1.5, label='best search')
ax.set_title('where each controller lands', fontsize=10)
ax.set_xlabel('final progress (m)', fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.2)

for a2 in axes[1][:3]:
    a2.set_xlabel('arc length s (m)', fontsize=10)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                    location='bottom', fraction=0.035, shrink=0.55, pad=0.02)
cbar.set_label('search trajectory final progress (m)', fontsize=10)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3,
           fontsize=10, frameon=False)
fig.suptitle('task 27: classical nullspace takeover vs the >0.70 m search '
             'bundle', fontsize=12)
out = RUN / 'classical_vs_bundle.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
