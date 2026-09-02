"""Definitive zero-step selection eval with the true-reset-obs retrains.

Per (robot, family): reuse the CACHED candidate pools (policy-independent),
re-roll every candidate with the retrained controller (labels L), and score
every candidate with the same run's critic at the RESET observation --
zero steps executed, margins and projection scales real by true_reset_obs.
Rebuilds all selection-table rows. argv: robot ('xarm7' | 'fr3')."""
import sys, json, math, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

robot = sys.argv[1]
dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
K = 32
B = 2500
RM = {'fr3': ('config_line_cont_dirfrac_e8kXXL_rm.yaml',
              'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'),
      'xarm7': ('config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml',
                'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt')}

keys = {f.name for f in dataclasses.fields(EnvConfig)}

# margin-field model (heuristics; policy-independent)
y = yaml.safe_load(open(REPO / hl.ROBOTS[robot][0]))
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
menv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 128}), None, dev)
model = hl.StraightModel(menv)
model.cfg = dataclasses.replace(menv.cfg, dt=y['env']['dt'])

# retrained policy env (true_reset_obs on via the _rm yaml)
y2 = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / RM[robot][0]))
kw2 = {k: v for k, v in y2['env'].items() if k in keys}
kw2['dt'] /= 2
kw2['max_steps'] = int(y2['env']['max_steps'] * 2)
renv = NSRLBatchedEnv(EnvConfig(**{**kw2, 'n_envs': B}), None, dev)
assert getattr(renv.cfg, 'true_reset_obs', False)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y2['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO / RM[robot][1], map_location=dev))
ag.eval()
rdt = renv.kin.dtype


def heuristics(C, p0, ld, nt):
    N = C.shape[0]
    nj = C.shape[-1]
    fq = torch.tensor(np.nan_to_num(C.reshape(-1, nj)),
                      dtype=menv.kin.dtype, device=dev)
    rep = lambda x: torch.tensor(np.repeat(x, K, 0), dtype=menv.kin.dtype,
                                 device=dev)
    fp, fd, fn = rep(p0), rep(ld), rep(nt)
    mjl, mc, ph, wd = [], [], [], []
    with torch.no_grad():
        for i in range(0, fq.shape[0], 4096):
            m = model.margins(fq[i:i + 4096], fp[i:i + 4096], fd[i:i + 4096],
                              fn[i:i + 4096])
            mjl.append(m[:, 0]); mc.append(m[:, 1])
            ph.append(-0.1 * torch.logsumexp(-m / 0.1, dim=-1))
            _, _, J, _ = menv.kin.tcp_fk_jac(fq[i:i + 4096])
            Jp, _ = damped_pinv(J[:, :3, :], menv.cfg.lambda_0,
                                menv.cfg.sigma_thr)
            wd.append(1.0 / ((Jp @ fd[i:i + 4096].unsqueeze(-1)).squeeze(-1)
                             .norm(dim=-1) + 1e-9))
    r = lambda t: torch.cat(t).reshape(N, K).float().cpu().numpy()
    return {'manip': r(wd), 'orient': r(mc), 'jl': r(mjl), 'phim': r(ph)}


def label_and_score(C, nf, spec, N):
    """L = stroke of the retrained controller; V = critic at reset obs."""
    L = np.zeros((N, K), np.float32)
    V = np.full((N, K), np.nan, np.float32)
    t0 = time.time()
    for k in range(K):
        valid = nf > k
        for lo in range(0, N, B):
            hi = min(lo + B, N)
            pad = B - (hi - lo)
            q0b = np.nan_to_num(C[lo:hi, k], nan=0.0)
            sub = {'q0': torch.tensor(q0b, dtype=rdt)}
            for kk, v in spec.items():
                sub[kk] = v[lo:hi]
            if pad:
                sub = {kk: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                       for kk, v in sub.items()}
            for kk in ('q0', 'line_dir', 'n_target'):
                sub[kk] = sub[kk].to(device=dev, dtype=rdt)
            renv.line_dist = ScriptedLineDistribution(sub)
            renv.reset()
            with torch.no_grad():
                V[lo:hi, k] = ag.get_value(
                    renv.current_obs()).float().cpu().numpy()[:hi - lo]
                for _ in range(renv.cfg.max_steps // 2):
                    a = ag.actor_mean(renv.current_obs())
                    for _ in range(2):
                        renv.step(a, auto_reset=False)
                    if bool(renv.done_persistent.all()):
                        break
            L[lo:hi, k] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
        L[~valid, k] = 0.0
        V[~valid, k] = np.nan
        if (k + 1) % 8 == 0:
            print(f'  cand {k + 1}/{K} ({(time.time() - t0) / 60:.1f} min)',
                  flush=True)
    return L, V


def rows(L, V, nf, scr, ref):
    N = L.shape[0]
    M = np.arange(K)[None, :] < nf[:, None]
    orc = np.where(M, L, -1e9).max(1)
    ok = orc > 1e-6
    def _ranks(S):
        Sm = np.where(M & np.isfinite(S), S, -np.inf)
        return Sm.argsort(1).argsort(1)
    per = {'first': L[np.arange(N), M.argmax(1)],
           'random': np.where(M, L, 0).sum(1) / np.maximum(M.sum(1), 1),
           'oracle': orc}
    for nme, S in {**scr, 'critic': V}.items():
        Sm = np.where(M & np.isfinite(S), S, -1e9)
        per[nme] = L[np.arange(N), Sm.argmax(1)]
    RS = _ranks(scr['phim']) + _ranks(V)
    per['consensus'] = L[np.arange(N), np.where(M, RS, -1).argmax(1)]
    out = {}
    for nme, v in per.items():
        rt = v / np.maximum(np.maximum(ref, v), 1e-9)
        near = (v >= per['oracle'] - 0.01)
        out[nme] = dict(stroke=round(float(v[ok].mean()), 3),
                        r=round(float(rt[ok].mean() * 100), 1),
                        p10=round(float(np.percentile(rt[ok], 10) * 100), 1),
                        near=round(float(near[ok].mean() * 100), 1))
    return out


REP = {}
if robot == 'fr3':
    cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                       weights_only=False)
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
    SETS = []
    for fam, key in (('straight', 'benchmark'),
                     ('serpentine', 'test_serpentine'),
                     ('rot', 'test_nonplanar')):
        nf = cands[key]['n_found'].numpy()
        C = cands[key]['cands'].numpy()
        N = C.shape[0]
        if key == 'benchmark':
            p0 = tb['cs_p0'][:N].astype(np.float32)
            ld = tb['cs_line_dir'][:N].astype(np.float32)
            nt = tb['cs_n_target'][:N].astype(np.float32)
            b = np.load(MAIN / 'runs/paper_fill/bound_10000_final.npz')
            w = np.load(MAIN / 'runs/paper_fill/witness_10k_v4.npz')
            spec_np = {}
        else:
            sp = tasks[key]
            p0 = sp['p0'].numpy()[:N]
            ld = sp['line_dir'].numpy()[:N]
            nt = sp['n_target'].numpy()[:N]
            fam2 = key.replace('test_', '')
            b = np.load(A / f'bound_sel_{fam2}.npz')
            w = np.load(A / f'witness_sel_{fam2}.npz')
            spec_np = {kk: sp[kk] for kk in ('kappa', 'amp', 'wavelen',
                                             'n_rot_axis', 'n_rot_rate')
                       if kk in sp}
        SETS.append((fam, C, nf, p0, ld, nt, b, w, spec_np))
else:
    SETS = []
    for fam, fname, tzn, bn, wn in (
        ('straight', 'xarm7_sel_straight.npz', 'tasks_pool_xarm7.npz',
         'bound_pool_xarm7_v2.npz', 'witness_pool_xarm7.npz'),
        ('serpentine', 'xarm7_sel_serpentine.npz',
         'tasks_selx_serpentine_xarm7.npz',
         'bound_selx_serpentine_xarm7.npz',
         'witness_selx_serpentine_xarm7.npz'),
        ('rot', 'xarm7_sel_nonplanar.npz', 'tasks_selx_nonplanar_xarm7.npz',
         'bound_selx_nonplanar_xarm7.npz',
         'witness_selx_nonplanar_xarm7.npz')):
        d = np.load(FU / fname)
        C, nf = d['cands'], d['n_found']
        N = C.shape[0]
        tz = np.load(A / tzn)
        if 'cs_p0' in tz.files:
            p0 = tz['cs_p0'][:N].astype(np.float32)
        else:
            q0s = torch.tensor(tz['q0_seed'][:N], dtype=menv.kin.dtype,
                               device=dev)
            p0 = menv.kin.tcp_fk_jac(q0s)[0].cpu().numpy().astype(np.float32)
        ld = tz['cs_line_dir'][:N].astype(np.float32)
        nt = tz['cs_n_target'][:N].astype(np.float32)
        spec_np = {kk: tz[kk][:N] for kk in ('kappa', 'amp', 'wavelen',
                                             'n_rot_axis', 'n_rot_rate')
                   if kk in tz.files}
        SETS.append((fam, C, nf, p0, ld, nt, np.load(A / bn),
                     np.load(A / wn), spec_np))

for fam, C, nf, p0, ld, nt, b, w, spec_np in SETS:
    N = C.shape[0]
    ld = ld / np.linalg.norm(ld, axis=1, keepdims=True)
    nt = nt / np.linalg.norm(nt, axis=1, keepdims=True)
    ref = np.maximum(b['L_hi'], w['prog'])[:N]
    spec = {'line_dir': torch.tensor(ld), 'n_target': torch.tensor(nt)}
    for kk, v in spec_np.items():
        spec[kk] = v if torch.is_tensor(v) else torch.tensor(v)
    scr = heuristics(C, p0, ld, nt)
    print(f'[{robot} {fam}] labeling {N}x{K} with retrained policy...',
          flush=True)
    L, V = label_and_score(C, nf, spec, N)
    np.savez_compressed(FU / f'{robot}e8k_sel_{fam}.npz', L=L, V=V,
                        n_found=nf)
    REP[f'{robot}_{fam}'] = rows(L, V, nf, scr, ref)
    print(robot, fam, json.dumps(REP[f'{robot}_{fam}']), flush=True)

json.dump(REP, open(FU / f'sel_e8k_rows_{robot}.json', 'w'), indent=1)
print('done', flush=True)
