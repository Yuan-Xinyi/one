"""Figures for the EMG journal paper's Problem-Analysis section (Part A)
and, once the campaign lands, the system-level section.

Run after runs/emg_analysis/partA_report.json exists. Idempotent.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

A = Path('/home/lqin/one/Yuan/unified_rl/runs/emg_analysis')
FIG = Path('/home/lqin/one/Yuan/ikpool_paper/figures')
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42})
BLUE, ORANGE, GRAY, GREEN, RED = '#2b6cb0', '#dd6b20', '#718096', '#2f855a', '#c53030'

# ---- Fig: SMM progress-vs-arc curves (3 representative tasks) ----------
a1 = np.load(A / 'a1_smm.npz')
tasks_meta = json.loads((A / 'tasks.json').read_text())
# pick one task per bucket with >=3 branches and many points
cand_tasks = []
for t in sorted(set(a1['task'].tolist())):
    m = a1['task'] == t
    if len(set(a1['branch'][m].tolist())) >= 3 and m.sum() >= 25:
        cand_tasks.append((t, a1['progress'][m].max()))
cand_tasks.sort(key=lambda x: -x[1])
picks = [cand_tasks[0][0],
         cand_tasks[len(cand_tasks) // 2][0],
         cand_tasks[-1][0]] if len(cand_tasks) >= 3 else \
        [t for t, _ in cand_tasks]
fig, axes = plt.subplots(1, len(picks), figsize=(7.0, 2.0), sharey=False)
for ax, t in zip(np.atleast_1d(axes), picks):
    m = a1['task'] == t
    arc, br, pr = a1['arc'][m], a1['branch'][m], a1['progress'][m]
    for b in sorted(set(br.tolist())):
        mb = br == b
        o = np.argsort(arc[mb])
        ax.plot(arc[mb][o], pr[mb][o], 'o-', ms=2.5, lw=1.1,
                label=f'branch {b}')
    ax.set_xlabel('SMM arc length [rad]')
    ax.set_title(f'task {t}', fontsize=7)
axes_flat = np.atleast_1d(axes)
axes_flat[0].set_ylabel('achieved length [m]')
axes_flat[-1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(FIG / 'fig_smm_curves.pdf'); plt.close(fig)

# ---- Fig: variance attribution + gain scatter --------------------------
rep = json.loads((A / 'partA_report.json').read_text())
a0 = np.load(A / 'a0_grid.npz')
grid, tidx = a0['grid'], a0['task_indices']
gg = a0['gain_grid']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2),
                               gridspec_kw={'width_ratios': [1, 1.6]})
sh = rep['a0_variance_shares_pct']
ax1.bar(range(3), [sh['seed'], sh['controller_gains'], sh['interaction']],
        color=[BLUE, ORANGE, GRAY], width=.6)
ax1.set_xticks(range(3))
ax1.set_xticklabels(['initial\nconfig.', 'controller\ngains', 'inter-\naction'],
                    fontsize=7)
ax1.set_ylabel('variance share [%]')
# per-task optimal gain index histogram
best_gain = []
for t in sorted(set(tidx.tolist())):
    g = grid[tidx == t]
    r = g.max(1).argmax()
    best_gain.append(int(g[r].argmax()))
counts = np.bincount(best_gain, minlength=len(gg))
ax2.bar(range(len(gg)), counts, color=ORANGE, width=.8)
ax2.set_xlabel('gain setting index (27-point grid)')
ax2.set_ylabel('# tasks optimal')
fig.tight_layout(); fig.savefig(FIG / 'fig_attribution.pdf'); plt.close(fig)

print('EMG Part-A figures written:',
      sorted(p.name for p in FIG.glob('fig_smm*.pdf'))
      + sorted(p.name for p in FIG.glob('fig_attr*.pdf')))

# ---- Fig: reference-length distribution + per-bucket controller bars ----
import json as _json
mt = Path('/home/lqin/one/Yuan/unified_rl/runs/iksel_final_n48/main10k_tables.json')
if mt.exists():
    rep2 = _json.loads(mt.read_text())
    import numpy as _np
    import torch as _t
    import sys
    sys.path.insert(0, '/home/lqin/one')
    from Yuan.unified_rl.iksel_campaign import _load_pool_env as _lpe
    _G = Path('/home/lqin/one/Yuan/unified_rl/runs/iksel_final_n48')
    _, P, V = _lpe(_G / 'iksel_eval10k_candidates.npz',
                   _G / 'iksel_eval10k_returns_hybrid.npz', _t.device('cuda:0'))
    lref = _t.where(V, P, _t.tensor(-1e9, device=P.device)).max(1).values.cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.1),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    ax1.hist(lref, bins=60, color=BLUE, alpha=.85)
    for x, lab in ((0.45, '0.45'), (0.80, '0.80')):
        ax1.axvline(x, color=RED, lw=1, ls='--')
        ax1.text(x, ax1.get_ylim()[1] * .97, lab, color=RED, fontsize=6,
                 ha='center', va='top')
    ax1.set_xlabel(r'reference length $\ell^{\mathrm{ref}}$ [m]')
    ax1.set_ylabel('# tasks')
    ax1.text(.22, .9, 'Difficult', transform=ax1.transAxes, fontsize=6, color=GRAY)
    ax1.text(.52, .9, 'Medium', transform=ax1.transAxes, fontsize=6, color=GRAY)
    ax1.text(.8, .9, 'Easy', transform=ax1.transAxes, fontsize=6, color=GRAY)
    t1 = rep2['table1']
    buckets = ['Easy', 'Medium', 'Difficult']
    arms = [('proposed+classical', 'Classical', GRAY),
            ('proposed+rl', 'RL', ORANGE),
            ('proposed+hybrid', 'Hybrid', BLUE)]
    w = .26
    for k, (a, lab, col) in enumerate(arms):
        vals = [t1[a][b]['ratio'][0] for b in buckets]
        ax2.bar(np.arange(3) + (k - 1) * w, vals, w, color=col, label=lab)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(buckets, fontsize=7)
    ax2.set_ylabel('ratio to reference [%]'); ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, fontsize=6, loc='lower right')
    fig.tight_layout(); fig.savefig(FIG / 'fig_refdist.pdf'); plt.close(fig)
    print('fig_refdist.pdf written')
