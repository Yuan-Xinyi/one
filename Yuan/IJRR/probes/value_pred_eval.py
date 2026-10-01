"""Is the critic V(q, c) a good estimator of the remaining task extent?

Part A  start states: the K=32 candidate sets with their rolled strokes
        L (saved labels) and critic values V, per model and per family.
        Global Spearman, within-task Spearman and pairwise concordance
        (what argmax selection needs), top-1 accuracy, isotonic calibration
        (out-of-sample R^2 / MAE in metres), calibration curves.
Part B  along the stroke: 2000 straight tasks rolled with the flagship,
        V at every decision step against the remaining stroke
        L_final - s_t; same metrics, split by phase.

Outputs fam_unify/value_pred_rows.json and value_pred_calibration.png.
"""
import sys, dataclasses, json
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, yaml
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
dev = torch.device('cuda')
MODELS = {'flagship': 'fr3e8k_sel_{fam}.npz', 'curves': 'curves_sel_{fam}.npz', 'taskcond': 'taskcond_sel_{fam}.npz'}
FAMS = ('straight', 'serpentine', 'rot')
res = {}


def metrics(V, L, tag):
    """V, L: (N, K) with NaN for missing candidates."""
    valid = np.isfinite(V) & np.isfinite(L) & (L > 0)
    v, l = V[valid], L[valid]
    out = {'n_pairs': int(valid.sum())}
    out['spearman_global'] = float(spearmanr(v, l).correlation)
    # within-task rank agreement
    sp, conc, top1, n_t = [], [], [], 0
    for i in range(V.shape[0]):
        m = valid[i]
        if m.sum() < 3:
            continue
        vi, li = V[i, m], L[i, m]
        if li.max() - li.min() < 0.01:
            continue
        n_t += 1
        r = spearmanr(vi, li).correlation
        if np.isfinite(r):
            sp.append(r)
        dv = vi[:, None] - vi[None, :]; dl = li[:, None] - li[None, :]
        pair = np.abs(dl) > 0.01
        conc.append(float(((np.sign(dv) == np.sign(dl)) & pair).sum() / max(pair.sum(), 1)))
        top1.append(float(li[np.argmax(vi)] >= li.max() - 0.01))
    out['n_tasks'] = n_t
    out['spearman_within_task'] = float(np.mean(sp))
    out['pairwise_concordance'] = float(np.mean(conc))
    out['top1_within_1cm'] = float(np.mean(top1))
    # calibration: isotonic L ~ g(V), fit on even rows, test on odd rows
    rows = np.repeat(np.arange(V.shape[0])[:, None], V.shape[1], 1)[valid]
    tr, te = rows % 2 == 0, rows % 2 == 1
    iso = IsotonicRegression(out_of_bounds='clip').fit(v[tr], l[tr])
    pred = iso.predict(v[te])
    ss = ((l[te] - l[te].mean()) ** 2).sum()
    out['iso_r2'] = float(1 - ((l[te] - pred) ** 2).sum() / ss)
    out['iso_mae_m'] = float(np.abs(l[te] - pred).mean())
    lin = np.polyfit(v[tr], l[tr], 1); pl = np.polyval(lin, v[te])
    out['linear_r2'] = float(1 - ((l[te] - pl) ** 2).sum() / ss)
    # calibration curve: 20 quantile bins of V
    qs = np.quantile(v, np.linspace(0, 1, 21))
    cal = []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (v >= a) & (v <= b)
        cal.append((float(v[m].mean()), float(l[m].mean()), float(l[m].std())))
    out['calibration'] = cal
    print(f'  {tag:26s} pairs {out["n_pairs"]:6d}  Spearman global {out["spearman_global"]:.3f}  '
          f'within-task {out["spearman_within_task"]:.3f}  concordance {out["pairwise_concordance"]*100:.1f}%  '
          f'top-1 {out["top1_within_1cm"]*100:.1f}%  iso R2 {out["iso_r2"]:.3f}  MAE {out["iso_mae_m"]*100:.1f} cm  '
          f'(linear R2 {out["linear_r2"]:.3f})', flush=True)
    return out


# ------------------------------------------------------------- part A
print('=== part A: start states (K=32 candidate sets) ===', flush=True)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for j, fam in enumerate(FAMS):
    for model, pat in MODELS.items():
        f = FU / pat.format(fam=fam)
        if not f.exists():
            continue
        d = np.load(f); L, V, nf = d['L'], d['V'], d['n_found']
        M = np.arange(L.shape[1])[None, :] < nf[:, None]
        L = np.where(M, L, np.nan); V = np.where(M, V, np.nan)
        res[f'A_{model}_{fam}'] = metrics(V, L, f'{model} / {fam}')
        cal = np.array(res[f'A_{model}_{fam}']['calibration'])
        axes[0, j].errorbar(cal[:, 0], cal[:, 1], yerr=cal[:, 2], fmt='o-', ms=3, capsize=2, label=model)
    axes[0, j].set_title(f'start states: {fam}'); axes[0, j].set_xlabel('critic value V'); axes[0, j].set_ylabel('rolled stroke L [m]')
    axes[0, j].legend(fontsize=8); axes[0, j].grid(alpha=.3)

# ------------------------------------------------------------- part B
print('=== part B: along the stroke (flagship, 2000 straight tasks) ===', flush=True)
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
N = 2000
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev)); ag.eval()
tz = np.load(A / 'tasks_pool_fr3.npz')
sub = np.sort(np.random.default_rng(3).choice(len(tz['q0_seed']), N, replace=False))
rdt = env.kin.dtype
env.line_dist = ScriptedLineDistribution({'q0': torch.tensor(tz['q0_seed'][sub], dtype=rdt, device=dev),
                                         'line_dir': torch.tensor(tz['cs_line_dir'][sub], dtype=rdt, device=dev),
                                         'n_target': torch.tensor(tz['cs_n_target'][sub], dtype=rdt, device=dev)})
Vs, Ss, alive = [], [], []
with torch.no_grad():
    obs = env.reset()
    for _ in range(env.cfg.max_steps // 2):
        Vs.append(ag.get_value(obs).float().cpu().numpy()); Ss.append(env.arc_progress.float().cpu().numpy().copy())
        alive.append((~env.done_persistent).cpu().numpy().copy())
        a = ag.actor_mean(obs)
        for _ in range(2):
            obs, _, _, _, _ = env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
Vs, Ss, alive = np.stack(Vs, 1), np.stack(Ss, 1), np.stack(alive, 1)        # (N, T)
L_final = env.arc_progress.float().cpu().numpy()
L_rem = L_final[:, None] - Ss
phase = Ss / np.maximum(L_final[:, None], 1e-6)                             # fraction of the stroke done
Vm = np.where(alive, Vs, np.nan); Lm = np.where(alive, L_rem, np.nan)
res['B_flagship_all_steps'] = metrics(Vm, Lm, 'along stroke: all steps')
for lo, hi, nm in ((0.0, 0.33, 'early third'), (0.33, 0.66, 'middle third'), (0.66, 1.01, 'last third')):
    m = alive & (phase >= lo) & (phase < hi)
    res[f'B_flagship_{nm.replace(" ", "_")}'] = metrics(np.where(m, Vs, np.nan), np.where(m, L_rem, np.nan), f'along stroke: {nm}')
cal = np.array(res['B_flagship_all_steps']['calibration'])
ax = axes[1, 0]; ax.errorbar(cal[:, 0], cal[:, 1], yerr=cal[:, 2], fmt='o-', ms=3, capsize=2, color='k')
ax.set_title('along the stroke (flagship): V vs remaining stroke'); ax.set_xlabel('critic value V'); ax.set_ylabel('remaining stroke [m]'); ax.grid(alpha=.3)
ax = axes[1, 1]
idx = np.random.default_rng(0).choice(np.flatnonzero(alive.ravel()), 4000, replace=False)
ax.scatter(Vs.ravel()[idx], L_rem.ravel()[idx], s=3, alpha=.3, c=phase.ravel()[idx], cmap='viridis')
ax.set_title('scatter (colour = fraction of stroke done)'); ax.set_xlabel('V'); ax.set_ylabel('remaining stroke [m]'); ax.grid(alpha=.3)
ax = axes[1, 2]
for nm in ('early_third', 'middle_third', 'last_third'):
    c = np.array(res[f'B_flagship_{nm}']['calibration']); ax.plot(c[:, 0], c[:, 1], 'o-', ms=3, label=nm.replace('_', ' '))
ax.set_title('calibration by phase'); ax.set_xlabel('V'); ax.set_ylabel('remaining stroke [m]'); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FU / 'value_pred_calibration.png', dpi=130)
json.dump(res, open(FU / 'value_pred_rows.json', 'w'), indent=1)
print('saved', FU / 'value_pred_rows.json', FU / 'value_pred_calibration.png', flush=True)
