"""Unified selection table rows, FR3 + xArm7, three families
(straight / serpentine / rot-axis), rules: first-feasible / random /
manip / orient / jl / phim / shared-critic (ours) / oracle.
FR3 uses the existing candidate pools + dirfrac labels + saved critic
scores; xArm7 uses the xarm7_sel_* files (heuristic scores computed
here). Ratios against the established per-set references."""
import sys, json, math, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
K = 32


def build_model(robot):
    y = yaml.safe_load(open(REPO / hl.ROBOTS[robot][0]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2
    kw['max_steps'] = int(y['env']['max_steps'] * 2)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 128}), None, dev)
    model = hl.StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    return env, model


def heuristics(env, model, C, nf, p0, ld, nt):
    N = C.shape[0]
    nj = C.shape[-1]
    fq = torch.tensor(np.nan_to_num(C.reshape(-1, nj)),
                      dtype=env.kin.dtype, device=dev)
    rep = lambda x: torch.tensor(np.repeat(x, K, 0), dtype=env.kin.dtype,
                                 device=dev)
    fp, fd, fn = rep(p0), rep(ld), rep(nt)
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
    r = lambda t: torch.cat(t).reshape(N, K).float().cpu().numpy()
    return {'manip': r(wd), 'orient': r(mc), 'jl': r(mjl), 'phim': r(ph)}


def rows(L, V, nf, scr, ref):
    N = L.shape[0]
    M = np.arange(K)[None, :] < nf[:, None]
    Lm = np.where(M, L, -1e9)
    orc = Lm.max(1)
    ok = orc > 1e-6
    out = {}
    per = {'first': L[np.arange(N), M.argmax(1)],
           'random': (np.where(M, L, 0).sum(1)
                      / np.maximum(M.sum(1), 1)),
           'oracle': orc}
    for nme, S in {**scr, 'critic': V}.items():
        Sm = np.where(M & np.isfinite(S), S, -1e9)
        per[nme] = L[np.arange(N), Sm.argmax(1)]
    # margin-value consensus: sum of within-task ranks of phim and critic
    def _ranks(S):
        Sm = np.where(M & np.isfinite(S), S, -np.inf)
        return Sm.argsort(1).argsort(1)
    RS = _ranks(scr['phim']) + _ranks(V)
    per['consensus'] = L[np.arange(N), np.where(M, RS, -1).argmax(1)]
    for nme, v in per.items():
        rt = v / np.maximum(np.maximum(ref, v), 1e-9)
        near = (v >= per['oracle'] - 0.01)
        out[nme] = dict(stroke=round(float(v[ok].mean()), 3),
                        r=round(float(rt[ok].mean() * 100), 1),
                        p10=round(float(np.percentile(rt[ok], 10) * 100), 1),
                        near=round(float(near[ok].mean() * 100), 1))
    return out


REP = {}
# ------------------------- FR3 ------------------------------------------
env3, model3 = build_model('fr3')
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                   weights_only=False)
tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
for fam, key in (('straight', 'benchmark'),
                 ('serpentine', 'test_serpentine'),
                 ('rot', 'test_nonplanar')):
    L = np.load(FU / f'dirfrac_labels_{key}.npz')['L']
    V = np.load(FU / f'critic_scores_{key}.npz')['V']
    nf = cands[key]['n_found'].numpy()[:L.shape[0]]
    C = cands[key]['cands'][:L.shape[0]].numpy()
    if key == 'benchmark':
        p0 = tb['cs_p0'][:L.shape[0]].astype(np.float32)
        ld = tb['cs_line_dir'][:L.shape[0]].astype(np.float32)
        nt = tb['cs_n_target'][:L.shape[0]].astype(np.float32)
        b = np.load(MAIN / 'runs/paper_fill/bound_10000_final.npz')
        w = np.load(MAIN / 'runs/paper_fill/witness_10k_v4.npz')
    else:
        sp = tasks[key]
        p0 = sp['p0'].numpy()[:L.shape[0]]
        ld = sp['line_dir'].numpy()[:L.shape[0]]
        nt = sp['n_target'].numpy()[:L.shape[0]]
        fam2 = key.replace('test_', '')
        b = np.load(A / f'bound_sel_{fam2}.npz')
        w = np.load(A / f'witness_sel_{fam2}.npz')
    ld = ld / np.linalg.norm(ld, axis=1, keepdims=True)
    nt = nt / np.linalg.norm(nt, axis=1, keepdims=True)
    ref = np.maximum(b['L_hi'], w['prog'])[:L.shape[0]]
    scr = heuristics(env3, model3, C, nf, p0, ld, nt)
    REP[f'fr3_{fam}'] = rows(L, V, nf, scr, ref)
    print('fr3', fam, json.dumps(REP[f'fr3_{fam}']), flush=True)
del env3, model3
torch.cuda.empty_cache()

# ------------------------- xArm7 ----------------------------------------
env7, model7 = build_model('xarm7')
for fam, (fname, tz_f, bset, wset) in {
    'straight': ('xarm7_sel_straight.npz', A / 'tasks_pool_xarm7.npz',
                 A / 'bound_pool_xarm7_v2.npz',
                 A / 'witness_pool_xarm7.npz'),
    'serpentine': ('xarm7_sel_serpentine.npz',
                   A / 'tasks_selx_serpentine_xarm7.npz',
                   A / 'bound_selx_serpentine_xarm7.npz',
                   A / 'witness_selx_serpentine_xarm7.npz'),
    'rot': ('xarm7_sel_nonplanar.npz',
            A / 'tasks_selx_nonplanar_xarm7.npz',
            A / 'bound_selx_nonplanar_xarm7.npz',
            A / 'witness_selx_nonplanar_xarm7.npz'),
}.items():
    d = np.load(FU / fname)
    L, V, nf, C = d['L'], d['V'], d['n_found'], d['cands']
    tz = np.load(tz_f)
    N = L.shape[0]
    if 'cs_p0' in tz.files:
        p0 = tz['cs_p0'][:N].astype(np.float32)
    else:
        q0s = torch.tensor(tz['q0_seed'][:N], dtype=env7.kin.dtype,
                           device=dev)
        p0 = env7.kin.tcp_fk_jac(q0s)[0].cpu().numpy().astype(np.float32)
    ld = tz['cs_line_dir'][:N].astype(np.float32)
    nt = tz['cs_n_target'][:N].astype(np.float32)
    ld = ld / np.linalg.norm(ld, axis=1, keepdims=True)
    nt = nt / np.linalg.norm(nt, axis=1, keepdims=True)
    b = np.load(bset); w = np.load(wset)
    ref = np.maximum(b['L_hi'], w['prog'])[:N]
    scr = heuristics(env7, model7, C, nf, p0, ld, nt)
    REP[f'xarm7_{fam}'] = rows(L, V, nf, scr, ref)
    print('xarm7', fam, json.dumps(REP[f'xarm7_{fam}']), flush=True)

json.dump(REP, open(FU / 'unified_sel_rows.json', 'w'), indent=1)
print('done', flush=True)
