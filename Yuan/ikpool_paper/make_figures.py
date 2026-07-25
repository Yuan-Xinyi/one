"""Generate all data figures for the IK-pool paper from cached artifacts."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

R = Path('/home/lqin/one/Yuan/unified_rl/runs')
D = R / 'ikpool_full_v1'
FIG = Path('/home/lqin/one/Yuan/ikpool_paper/figures')
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42})
BLUE, ORANGE, GRAY, GREEN, RED = '#2b6cb0', '#dd6b20', '#718096', '#2f855a', '#c53030'

ik = np.load(D / 'ikpool_returns.npz')
dif = np.load(R / 'r2_full_returns_v1/train_returns.npz', allow_pickle=True)
ti = ik['task_indices']
ikP, ikV = ik['progress_m'], ik['valid']
dP, dV = dif['progress_m'][ti], dif['valid'][ti]


def spread(P, V):
    out = []
    for i in range(len(P)):
        v = np.nan_to_num(P[i])[V[i]]
        if v.size >= 2:
            out.append(v.max() - v.min())
    return np.asarray(out) * 1e3


# ---- Fig: within-task spread histograms -------------------------------
s_ik, s_dif = spread(ikP, ikV), spread(dP, dV)
fig, ax = plt.subplots(figsize=(4.7, 1.9))
bins = np.linspace(0, 1200, 61)
ax.hist(s_dif, bins=bins, alpha=.65, color=GRAY, label='generator pool (K=8)', density=True)
ax.hist(s_ik, bins=bins, alpha=.65, color=BLUE, label='enumerated IK pool (K=32)', density=True)
ax.axvline(np.median(s_dif), color=GRAY, ls='--', lw=1)
ax.axvline(np.median(s_ik), color=BLUE, ls='--', lw=1)
ax.set_xlabel('within-task best-minus-worst path length [mm]')
ax.set_ylabel('density')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'fig_spread.pdf'); plt.close(fig)

# ---- Fig: oracle vs pool size -----------------------------------------
Kik = ikP.shape[1] - 1
Pm = np.where(ikV[:, :Kik], np.nan_to_num(ikP[:, :Kik], nan=-1e9), -1e9)
ks = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
curve = []
for k in ks:
    o = Pm[:, :k].max(1)
    curve.append(o[o > -1e8].mean())
dif_oracle = np.where(dV, np.nan_to_num(dP), -1e9).max(1).mean()
dif_first = dP[np.arange(len(ti)), dV.argmax(1)].mean()
fig, ax = plt.subplots(figsize=(4.7, 2.1))
ax.plot(ks, curve, 'o-', color=BLUE, ms=3.5, lw=1.4, label='IK-pool oracle (FPS prefix)')
ax.axhline(dif_oracle, color=GRAY, ls='--', lw=1.2, label='generator-pool oracle (K=8)')
ax.axhline(dif_first, color=GRAY, ls=':', lw=1.2, label='generator first-valid')
ax.set_xlabel('pool size $K$'); ax.set_ylabel('mean best path length [m]')
ax.set_xticks([1, 4, 8, 16, 24, 32]); ax.legend(frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(FIG / 'fig_oracle_vs_k.pdf'); plt.close(fig)

# ---- Fig: capture scaling ---------------------------------------------
e2 = json.loads((D / 'ikpool_e2_scaling.json').read_text())
sizes = sorted(int(s) for s in e2['sizes'])
mean = [e2['sizes'][str(s)]['test_capture_mean'] for s in sizes]
std = [e2['sizes'][str(s)]['test_capture_std'] for s in sizes]
ens = [e2['sizes'][str(s)]['ensemble3_capture'] for s in sizes]
fig, ax = plt.subplots(figsize=(4.7, 2.1))
ax.errorbar(sizes, mean, yerr=std, fmt='o-', color=BLUE, ms=3.5, lw=1.4,
            capsize=2, label='single selector (3 seeds)')
ax.plot(sizes, ens, 's--', color=ORANGE, ms=3.5, lw=1.2, label='3-seed ensemble')
ax.set_xscale('log'); ax.set_xticks(sizes)
ax.set_xticklabels([f'{s//1000}k' for s in sizes])
ax.set_xlabel('training tasks'); ax.set_ylabel('held-out oracle capture [%]')
ax.legend(frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(FIG / 'fig_scaling.pdf'); plt.close(fig)

# ---- Fig: novelty-stratified forward effect ---------------------------
res = {}
for which, oldp, difsrc in [
        ('validation', R / 'r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz', 'rank'),
        ('external', R / 'r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz', 'ext')]:
    g = np.load(D / f'gate_{which}_ikpool_c1_r2.npz')
    ikc = np.load(D / f'ikpool_{which}_candidates.npz')
    tids, pick = g['task_indices'], g['pick']
    if difsrc == 'rank':
        c = np.load('/home/lqin/one/Yuan/seed_selection/runs/rank_train/candidates_K8.npz',
                    allow_pickle=True)
        ds_, dok, dpil = c['seeds'][tids], c['ik_ok'][tids], c['q0_pilot'][tids]
    else:
        c = np.load(R / 'external_dev_v1/candidates_K8.npz', allow_pickle=True)
        ds_, dok, dpil = c['seeds'], c['ik_ok'], c['q0_pilot']
    n = len(tids)
    chosen = np.where((pick < 32)[:, None],
                      ikc['seeds'][np.arange(n), np.clip(pick, 0, 31)], dpil)
    pool = np.concatenate([np.where(dok[:, :, None], ds_, np.inf), dpil[:, None, :]], 1)
    dist = np.linalg.norm(pool - chosen[:, None, :], axis=-1)
    res[which] = (np.nanmin(np.where(np.isfinite(dist), dist, np.nan), 1),
                  (g['s0c1'] - g['s0c0']) * 1e3, g['s0c0'])
nov = np.concatenate([res[w][0] for w in res])
fwd = np.concatenate([res[w][1] for w in res])
base = np.concatenate([res[w][2] for w in res])
q = np.quantile(nov, [0, .25, .5, .75, 1.])
labels = [f'[{q[i]:.1f},{q[i+1]:.1f}]' for i in range(4)]
means, cis, bmeans = [], [], []
for i in range(4):
    m = (nov >= q[i]) & (nov < q[i + 1]) if i < 3 else (nov >= q[i])
    d = fwd[m]
    means.append(d.mean()); cis.append(1.96 * d.std(ddof=1) / np.sqrt(m.sum()))
    bmeans.append(base[m].mean())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.7, 1.9))
a1.bar(range(4), means, yerr=cis, color=ORANGE, alpha=.85, capsize=3, width=.6)
a1.axhline(0, color='k', lw=.8)
a1.set_xticks(range(4)); a1.set_xticklabels(labels, fontsize=6)
a1.set_xlabel('seed novelty [rad]'); a1.set_ylabel(r'forward effect $\Delta$ [mm]')
a2.bar(range(4), bmeans, color=BLUE, alpha=.85, width=.6)
a2.set_xticks(range(4)); a2.set_xticklabels(labels, fontsize=6)
a2.set_xlabel('seed novelty [rad]'); a2.set_ylabel('path length under $\\pi_0$ [m]')
a2.set_ylim(0.4, 0.65)
fig.tight_layout(); fig.savefig(FIG / 'fig_novelty.pdf'); plt.close(fig)

# ---- Fig: paired delta histogram --------------------------------------
deltas = []
for which, oldp in [
        ('validation', R / 'r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz'),
        ('external', R / 'r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz')]:
    g = np.load(D / f'gate_{which}_ikpool_c1_r2.npz')
    o = np.load(oldp, allow_pickle=True)
    order = {int(t): i for i, t in enumerate(o['task_indices'])}
    perm = np.array([order[int(t)] for t in g['task_indices']])
    deltas.append((g['s0c0'] - np.nan_to_num(o['policy_progress_m'])[perm]) * 1e3)
d = np.concatenate(deltas)
tr = np.sort(d); k = int(.05 * len(d)); trm = tr[k:-k].mean()
fig, ax = plt.subplots(figsize=(4.7, 1.9))
ax.hist(np.clip(d, -400, 400), bins=81, color=BLUE, alpha=.8)
ax.axvline(0, color='k', lw=.8)
ax.axvline(d.mean(), color=RED, lw=1.2, label=f'mean {d.mean():+.1f} mm')
ax.axvline(trm, color=GREEN, lw=1.2, ls='--', label=f'5% trimmed {trm:+.1f} mm')
ax.set_xlabel('per-task path-length difference vs. prior system [mm]')
ax.set_ylabel('tasks'); ax.set_yscale('log'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / 'fig_delta_hist.pdf'); plt.close(fig)

print('figures written:', sorted(p.name for p in FIG.glob('*.pdf')))
