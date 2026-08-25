"""Selection-stage report under the DirFrac executor.

Rules (per set): oracle / first-feasible / random / manip / orient / jl /
phim / learned selector (conditioned ranker). Strokes come from the
DirFrac candidate labels; rankings are executor-independent and reuse the
selreport machinery. Ratios: test sets against max(bound_sel, witness_sel)
per task; benchmark against bound_10000_final + witness_10k_v4 (recipe
verified to reproduce the retired framework numbers exactly).
Near-oracle = within 1 cm of the within-pool oracle stroke."""
import sys, json, dataclasses, importlib.util
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv

_spec = importlib.util.spec_from_file_location(
    'selector_ood', MAIN / 'stage1_seed/selector_ood.py')
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
K = 32

y = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 128}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])

cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                   weights_only=False)
rk = torch.load(MAIN / 'runs/selector_ood/v3_k32_fixedgate/rankers.pt',
                weights_only=False)
tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
bench = {'p0': torch.tensor(tb['cs_p0'], dtype=torch.float32),
         'line_dir': torch.tensor(tb['cs_line_dir'], dtype=torch.float32),
         'n_target': torch.tensor(tb['cs_n_target'], dtype=torch.float32),
         'q0': torch.tensor(tb['q0_seed'], dtype=torch.float32)}
bench['line_dir'] /= bench['line_dir'].norm(dim=-1, keepdim=True)
bench['n_target'] /= bench['n_target'].norm(dim=-1, keepdim=True)
ALL = {**tasks, 'benchmark': bench}


def selector_scores(key):
    spec = ALL[key]
    Xc = so.cand_features(env, cands[key]['cands'].numpy(), spec)
    rt = so.road_table(spec)
    Xr = torch.tensor(rt.reshape(rt.shape[0], -1))
    net = so.Ranker(Xc.shape[-1], Xr.shape[-1], conditioned=True).to(dev)
    net.load_state_dict(rk['cond'])
    net.eval()
    with torch.no_grad():
        return net(Xc.to(dev).float(), Xr.to(dev).float()).cpu()


def heuristic_scores(key):
    spec = ALL[key]
    C = cands[key]['cands'].to(dev)
    N = C.shape[0]
    nj = C.shape[-1]
    fq = C.reshape(-1, nj)
    fp = spec['p0'].to(dev).repeat_interleave(K, 0)
    fd = spec['line_dir'].to(dev).repeat_interleave(K, 0)
    fn = spec['n_target'].to(dev).repeat_interleave(K, 0)
    CH = 4096
    mjl, mc, ph, wd = [], [], [], []
    with torch.no_grad():
        for i in range(0, fq.shape[0], CH):
            m = model.margins(fq[i:i + CH], fp[i:i + CH], fd[i:i + CH],
                              fn[i:i + CH])
            mjl.append(m[:, 0]); mc.append(m[:, 1])
            ph.append(-0.1 * torch.logsumexp(-m / 0.1, dim=-1))
            _, _, J, _ = env.kin.tcp_fk_jac(fq[i:i + CH])
            Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                                env.cfg.sigma_thr)
            wd.append(1.0 / ((Jp @ fd[i:i + CH].unsqueeze(-1)).squeeze(-1)
                             .norm(dim=-1) + 1e-9))
    r = lambda t: torch.cat(t).reshape(N, K).cpu()
    return {'manip': r(wd), 'orient': r(mc), 'jl': r(mjl), 'phim': r(ph)}


def refs_for(key, n):
    if key == 'benchmark':
        b = np.load(MAIN / 'runs/paper_fill/bound_10000_final.npz')
        w = np.load(MAIN / 'runs/paper_fill/witness_10k_v4.npz')
    else:
        fam = key.replace('test_', '')
        b = np.load(A / f'bound_sel_{fam}.npz')
        w = np.load(A / f'witness_sel_{fam}.npz')
    return np.maximum(b['L_hi'], w['prog'])[:n]


out, rep = {}, {}
for key in ('test_straight', 'test_arc', 'test_serpentine',
            'test_nonplanar', 'benchmark'):
    Y = torch.tensor(np.load(FU / f'dirfrac_labels_{key}.npz')['L']).float()
    nf = cands[key]['n_found'][:Y.shape[0]]
    M = (torch.arange(K)[None, :] < nf[:, None])
    Ym = torch.where(M, Y, torch.full_like(Y, -1e9))
    orc = Ym.max(1).values
    per = {'oracle': orc.numpy(),
           'first': Y.gather(1, M.float().argmax(1)[:, None])
                     .squeeze(1).numpy(),
           'random': ((Y * M).sum(1) / M.sum(1).clamp(min=1)).numpy()}
    scr = heuristic_scores(key)
    scr['selector'] = selector_scores(key)
    for name, S in scr.items():
        pick = torch.where(M, S, torch.full_like(S, -1e9)).argmax(1)
        per[name] = Y.gather(1, pick[:, None]).squeeze(1).numpy()
    ok = (orc > 1e-6).numpy()
    ref = refs_for(key, Y.shape[0])
    rep[key] = {}
    for name, v in per.items():
        rt = v / np.maximum(np.maximum(ref, v), 1e-9)
        near = (v >= per['oracle'] - 0.01)
        rep[key][name] = {
            'stroke': round(float(v[ok].mean()), 3),
            'ratio_mean': round(float(rt[ok].mean() * 100), 1),
            'ratio_p10': round(float(np.percentile(rt[ok], 10) * 100), 1),
            'near_oracle': round(float(near[ok].mean() * 100), 1)}
    out.update({f'{key}__{n}': v for n, v in per.items()})
    out[f'{key}__valid'] = ok
    print(key, json.dumps(rep[key]), flush=True)

np.savez_compressed(FU / 'sel_report_dirfrac.npz', **out)
json.dump(rep, open(FU / 'sel_report_dirfrac.json', 'w'), indent=1)
print('done', flush=True)
