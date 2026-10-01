"""Where is the remaining gap to the pointwise bound: start selection or the
controller?  Flagship on the K=32 candidate sets (straight 2000-task subset,
serpentine 2500, rotating-axis 2500).

rows (start x controller), Ratio mean / p10 against ref = max(pointwise
bound, witnesses, every row):
  default start (pool seed)   x deterministic policy      [straight only]
  first / random / critic / oracle candidate x deterministic policy
  critic / oracle candidate   x best-of-16 stochastic rollouts (controller
                                headroom proxy: same start, sampled actions)
The decomposition then reads: selection gap = oracle - critic (policy);
controller headroom >= best-of-16(oracle) - oracle; remainder to the bound.
"""
import sys, dataclasses, json, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
dev = torch.device('cuda'); B = 2500; NS = 16
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2); kw['k_lateral'] = 5.0
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev)); ag.eval()
rdt = env.kin.dtype
T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)


def _sub(spec, lo, hi):
    pad = B - (hi - lo); out = {}
    for k, v in spec.items():
        t = v[lo:hi]
        if pad:
            t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
        out[k] = t.to(dev)
    return out


@torch.no_grad()
def roll(spec, N, stochastic=False):
    out = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = ScriptedLineDistribution(_sub(spec, lo, hi))
        obs = env.reset()
        for _ in range(env.cfg.max_steps // 2):
            if stochastic:
                z = ag.get_action_and_value(obs)[0]; a = torch.tanh(z)
            else:
                a = ag.actor_mean(obs)
            for _ in range(2):
                obs, _, _, _, _ = env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


def best_of(spec, N, q_start):
    """best of NS stochastic rollouts from q_start (N,7)."""
    rep = {k: v.repeat_interleave(NS, 0) for k, v in spec.items()}
    rep['q0'] = T(np.repeat(q_start, NS, 0))
    torch.manual_seed(0)
    out = roll(rep, N * NS, stochastic=True).reshape(N, NS)
    return out.max(1), out.mean(1)


cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt', weights_only=False)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)
tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
res = {}
t0 = time.time()
for fam, key in (('straight', 'benchmark'), ('serpentine', 'test_serpentine'), ('rot', 'test_nonplanar')):
    C = cands[key]['cands'].numpy(); nf = cands[key]['n_found'].numpy()
    if key == 'benchmark':
        rows = np.sort(np.random.default_rng(3).choice(C.shape[0], 2000, replace=False))
        spec = {'line_dir': T(tb['cs_line_dir'][rows]), 'n_target': T(tb['cs_n_target'][rows])}
        q_default = tb['q0_seed'][rows].astype(np.float32)
        b = np.load(MAIN / 'runs/paper_fill/bound_10000_final.npz'); w = np.load(MAIN / 'runs/paper_fill/witness_10k_v4.npz')
        saved = np.load(FU / 'fr3e8k_sel_straight.npz')
    else:
        rows = np.arange(C.shape[0]); sp = tasks[key]
        spec = {'p0': sp['p0'].float(), 'line_dir': sp['line_dir'].float(), 'n_target': sp['n_target'].float()}
        for kk in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
            if kk in sp:
                spec[kk] = sp[kk].float()
        q_default = None
        fam2 = key.replace('test_', '')
        b = np.load(A / f'bound_sel_{fam2}.npz'); w = np.load(A / f'witness_sel_{fam2}.npz')
        saved = np.load(FU / f'fr3e8k_sel_{fam}.npz')
    for kk in ('line_dir', 'n_target'):
        spec[kk] = spec[kk] / spec[kk].norm(dim=-1, keepdim=True)
    C, nf = C[rows], nf[rows]; N = len(rows)
    L, V = saved['L'][rows], saved['V'][rows]
    K = L.shape[1]; M = np.arange(K)[None, :] < nf[:, None]
    ref0 = np.maximum(b['L_hi'], w['prog'])[rows]
    orc_i = np.where(M, L, -1e9).argmax(1); crit_i = np.where(M & np.isfinite(V), V, -1e9).argmax(1)
    q_orc = np.nan_to_num(C[np.arange(N), orc_i]); q_crit = np.nan_to_num(C[np.arange(N), crit_i])
    rowsL = {'first candidate x policy': L[np.arange(N), M.argmax(1)],
             'random candidate x policy': np.where(M, L, 0).sum(1) / np.maximum(M.sum(1), 1),
             'critic start x policy': L[np.arange(N), crit_i],
             'oracle start x policy': L[np.arange(N), orc_i]}
    if q_default is not None:
        rowsL = {'default start x policy': roll({**spec, 'q0': T(q_default)}, N), **rowsL}
    print(f'[{fam}] deterministic rows ready [{time.time()-t0:.0f}s]; best-of-{NS} from critic starts...', flush=True)
    bc, mc = best_of(spec, N, q_crit)
    print(f'[{fam}] ... from oracle starts [{time.time()-t0:.0f}s]', flush=True)
    bo, mo = best_of(spec, N, q_orc)
    rowsL[f'critic start x best-of-{NS} stochastic'] = bc
    rowsL[f'critic start x mean-of-{NS} stochastic'] = mc
    rowsL[f'oracle start x best-of-{NS} stochastic'] = bo
    rowsL[f'oracle start x mean-of-{NS} stochastic'] = mo
    ok = np.where(M, L, -1e9).max(1) > 1e-6
    ref = np.maximum.reduce([ref0] + list(rowsL.values()))
    res[fam] = {}
    print(f'=== {fam} ({ok.sum()} tasks; ref = max(pointwise bound, witnesses, all rows)) ===', flush=True)
    for nm, v in rowsL.items():
        rt = v / np.maximum(ref, 1e-9)
        res[fam][nm] = [round(float(v[ok].mean()), 4), round(float(rt[ok].mean() * 100), 2), round(float(np.percentile(rt[ok], 10) * 100), 2)]
        print(f'  {nm:40s} stroke {v[ok].mean():.3f}  ratio {rt[ok].mean()*100:.1f} / {np.percentile(rt[ok],10)*100:.1f}', flush=True)
    rb = ref0 / np.maximum(ref, 1e-9)
    print(f'  {"pointwise bound + witnesses (ref0)":40s} stroke {ref0[ok].mean():.3f}  ratio {rb[ok].mean()*100:.1f} / {np.percentile(rb[ok],10)*100:.1f}', flush=True)
    r = lambda nm: res[fam][nm][1]
    dec = {'selection gap (oracle - critic, policy)': round(r('oracle start x policy') - r('critic start x policy'), 2),
           f'controller headroom proxy (best-of-{NS} - policy, oracle start)': round(r(f'oracle start x best-of-{NS} stochastic') - r('oracle start x policy'), 2),
           f'remainder to bound (100 - best-of-{NS} oracle start)': round(100 - r(f'oracle start x best-of-{NS} stochastic'), 2)}
    if q_default is not None:
        dec['selection gain over default start (critic - default)'] = round(r('critic start x policy') - r('default start x policy'), 2)
    res[fam]['decomposition'] = dec
    print('  decomposition:', json.dumps(dec), flush=True)
    np.savez(FU / f'gap_decomp_{fam}.npz', rows=rows, ref=ref, ref0=ref0, **{nm.replace(' ', '_'): v for nm, v in rowsL.items()})
json.dump(res, open(FU / 'gap_decomp_rows.json', 'w'), indent=1)
print('saved', FU / 'gap_decomp_rows.json', f'[{time.time()-t0:.0f}s]', flush=True)
