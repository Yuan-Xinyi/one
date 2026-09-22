"""Implicit-force straight-line analysis, eval side (mirrors cone5_eval.py):
2000-task subsample of the FR3 straight pool, fresh starts admissible under
the stiffness cap k_n <= KN_MAX (cone 30), pointwise march with the same
cap, classical + flagship-zero-shot rollouts in the implicit-force env.
Saves candidates for the later selection stage.

Usage: force_eval.py [ckpt_tag ...]   extra tags roll additional policies:
    force   -> runs/rl_dirfrac_e8kXXL_force/agent.pt with its own config
"""
import sys, math, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent

CONE = 30.0
_kn = [a for a in sys.argv[1:] if a.startswith('--kn=')]
KN_MAX = float(_kn[0][5:]) if _kn else 2000.0
F_SET, F_TOL, K_LAT = 5.0, 2.0, 5.0
_mu = [a for a in sys.argv[1:] if a.startswith('--mu=')]
MU = float(_mu[0][5:]) if _mu else 0.0
FORCE_KW = dict(force_kn_max=KN_MAX, force_set=F_SET, force_tol=F_TOL,
                k_lateral=K_LAT, force_mu=MU)
dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'
FU = MAIN / 'runs/paper_fill/fam_unify'
FULL = '--all' in sys.argv[1:]
extra = [a for a in sys.argv[1:] if not a.startswith('--')]
_tag = [a for a in sys.argv[1:] if a.startswith('--tag=')]
_sfx = ('' if KN_MAX == 2000.0 else f'_k{int(KN_MAX)}') + ('' if MU == 0.0 else f'_mu{MU}') + (('_' + _tag[0][6:]) if _tag else '')
OUTF = FU / (f'force_eval_10k{_sfx}.npz' if FULL else (f'force_eval{_sfx}.npz' if _sfx else 'force_eval_v1.npz'))

env0 = lb.build_env(dev, 'stock', 512)
dt0 = env0.kin.dtype
kq0 = torch.as_tensor(env0.kq_joint, device=dev, dtype=dt0)
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1).astype(np.float32))
cosc = math.cos(math.radians(CONE)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)
tz = np.load(A / 'tasks_pool_fr3.npz')
b30 = np.load(A / 'bound_pool_fr3.npz'); w30 = np.load(A / 'witness_pool_fr3.npz')
ref30 = np.maximum(b30['L_hi'], w30['prog'])
rng = np.random.default_rng(3)
sub = (np.arange(len(tz['q0_seed'])) if FULL else
       np.sort(rng.choice(len(tz['q0_seed']), 2000, replace=False)))
p0 = tz['cs_p0'][sub].astype(np.float32)
dd = tz['cs_line_dir'][sub].astype(np.float32)
dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32)
nt /= np.linalg.norm(nt, axis=1, keepdims=True)
N = len(sub)


def stiffness(q_t, z_t, t_t=None):
    return lb.stiffness_along(env0, q_t, z_t, kq0, t_t, MU)


if not OUTF.exists():
    # --- fresh starts: 4 cone dirs x 96 kNN, keep all admissible incl. k_n cap
    CQ, CT, CK = [], [], []
    for m in range(4):
        if m == 0:
            zs = nt.copy()
        else:
            zs = np.stack([_sample_in_cone(torch.as_tensor(nt[i]), CONE, 1,
                           np.random.default_rng(m * 777 + i)).numpy()[0]
                           for i in range(N)])
        feat = np.concatenate([p0 * POS_SCALE, zs], 1).astype(np.float32)
        _, ids = tree.query(feat, k=96, workers=-1)
        CH = 128
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, 7), device=dev, dtype=dt0)
            fp = torch.as_tensor(np.repeat(p0[lo:hi], 96, 0), device=dev, dtype=dt0)
            fz = torch.as_tensor(np.repeat(zs[lo:hi], 96, 0), device=dev, dtype=dt0)
            fn = torch.as_tensor(np.repeat(nt[lo:hi], 96, 0), device=dev, dtype=dt0)
            ft = torch.as_tensor(np.repeat(dd[lo:hi], 96, 0), device=dev, dtype=dt0)
            q_o, _, _ = _batched_ik_project(env0.kin, fq, fp, _build_R_with_z(fz, hint),
                                            branch_action=None)
            coll = env0.collision.is_collided(env0.kin.link_transforms(q_o))
            p_fk, R_fk, J_fk, _ = env0.kin.tcp_fk_jac(q_o)
            kn = stiffness(q_o, R_fk[:, :, 2], ft)
            in_lmt = ((q_o >= env0.kin.lmt_lo - 1e-5) & (q_o <= env0.kin.lmt_up + 1e-5)).all(-1)
            fine = ((~coll) & in_lmt & ((p_fk - fp).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * fn).sum(-1) >= cosc) & (kn <= KN_MAX))
            f = fine.cpu().numpy()
            rows = (np.arange(len(f)) // 96) + lo
            CQ.append(q_o[fine].cpu().numpy()); CT.append(rows[f])
            CK.append(kn[fine].float().cpu().numpy())
        print(f'starts dir {m + 1}/4', flush=True)
    CQ = np.concatenate(CQ).astype(np.float32); CT = np.concatenate(CT)
    CK = np.concatenate(CK).astype(np.float32)
    order = np.argsort(CT, kind='stable'); CQ, CT, CK = CQ[order], CT[order], CK[order]
    key = np.concatenate([CT[:, None], np.round(CQ, 2)], 1)
    _, ui = np.unique(key, axis=0, return_index=True)
    ui = np.sort(ui); CQ, CT, CK = CQ[ui], CT[ui], CK[ui]
    cnt = np.bincount(CT, minlength=N)
    q0_first = np.zeros((N, 7), np.float32); has = cnt > 0
    lo = 0
    while lo < len(CT):
        hi = lo
        while hi < len(CT) and CT[hi] == CT[lo]:
            hi += 1
        q0_first[CT[lo]] = CQ[lo]
        lo = hi
    print(f'force starts (k_n<={KN_MAX:.0f}): {has.mean() * 100:.1f}% tasks, '
          f'median cands {int(np.median(cnt[has]))}, cand k_n median {np.median(CK):.0f}',
          flush=True)

    # --- pointwise march under the stiffness cap (cone 30, 8 dirs/point) ---
    kn_tab = []
    for lo in range(0, len(T['q']), 16384):
        qt = torch.as_tensor(T['q'][lo:lo + 16384], device=dev, dtype=dt0)
        zt = torch.as_tensor(T['zax'][lo:lo + 16384], device=dev, dtype=dt0)
        kn_tab.append(stiffness(qt, zt).float().cpu().numpy())
    Td = {k: T[k] for k in T.files}; Td['kn'] = np.concatenate(kn_tab)
    M = 8
    STEPM = 0.02
    # chained march (iterative deepening, as line_bound.main): every alive task
    # is probed at the same arc length; the witness found at the previous point
    # seeds the next one; a stiffness-descent pass rescues postures that fail
    # only the cap; executed rollouts (--witness=) certify points directly and
    # the march then runs along the executed ray of that witness file
    _wit = [a for a in sys.argv[1:] if a.startswith('--witness=')]
    Wit = None; p_march = p0.copy()
    if _wit:
        wz = np.load(_wit[0][10:])
        Wit = wz['W']; assert abs(float(wz['step']) - STEPM) < 1e-9
        ps = wz['p_start'].astype(np.float32)
        p_march = np.where(np.isfinite(ps).all(1)[:, None], ps, p0)
        print(f'[march] witnesses: {int(np.isfinite(Wit[:, :, 0]).sum())} points from {_wit[0][10:]}', flush=True)
    dirs = np.empty((N, M, 3), np.float32)
    for i in range(N):
        dirs[i] = np.concatenate([nt[i][None], _sample_in_cone(
            torch.as_tensor(nt[i]), CONE, 16, np.random.default_rng(50 + i)).numpy()[:M - 1]], 0)
    # no horizon inherited from the position-only bound: that bound is itself a
    # search estimate and rollouts exceed it (task 4657: 1.56 m vs 0.88 m)
    smax = np.full(N, 1.8, np.float32)
    _sub = [a for a in sys.argv[1:] if a.startswith('--subset=')]
    if _sub:   # validation on a few tasks: everything else drops out at once
        keep = np.zeros(N, bool); keep[[int(v) for v in _sub[0][9:].split(',')]] = True
        smax = np.where(keep, smax, -1.0).astype(np.float32)
    n_steps = int(round(1.8 / STEPM))
    first_bad = np.full(N, -1, np.int64); alive = np.arange(N)
    q_prev = np.full((N, 7), np.nan, np.float32)
    q_prev[has] = q0_first[has]
    n_wc = 0
    for r in range(n_steps + 1):
        alive = alive[(r * STEPM) <= smax[alive] + 1e-9]
        if not len(alive):
            break
        s_ = r * STEPM
        certified = np.zeros(N, bool)
        if Wit is not None and r < Wit.shape[1]:
            w = Wit[alive, r]; have = np.isfinite(w).all(1)
            if have.any():
                rows = alive[have]
                fine = lb.witness_rows(env0, w[have].astype(np.float32), p_march[rows] + s_ * dd[rows],
                                       nt[rows], cosc, tube, kn_lim=(None, KN_MAX), t_rows=dd[rows], mu=MU)
                certified[rows[fine]] = True; q_prev[rows[fine]] = w[have][fine]; n_wc += int(fine.sum())
        search = alive[~certified[alive]]
        if len(search):
            pts = np.repeat(p_march[search] + s_ * dd[search], M, 0)
            zs = dirs[search].reshape(-1, 3); nrf = np.repeat(nt[search], M, 0); trw = np.repeat(dd[search], M, 0)
            hint = np.repeat(q_prev[search], M, 0)
            ok, q = lb.feasible_rows(env0, tree, Td, pts, zs, nrf, cosc, tube, k_nn=100, n_try=12,
                                     q_hint=None if r == 0 else hint, kn_lim=(None, KN_MAX),
                                     kn_descend=True, t_rows=trw, mu=MU)
            ok = ok.reshape(len(search), M); q = q.reshape(len(search), M, 7)
            any_ok = ok.any(1); pick = ok.argmax(1)
            q_prev[search[any_ok]] = q[np.arange(len(search)), pick][any_ok]
            # second chance before declaring a task dead: a heavier search at
            # this point only (more cone directions, more warm starts)
            dead = search[~any_ok]
            if len(dead):
                M2 = 24
                d2 = np.stack([np.concatenate([nt[i][None], _sample_in_cone(
                    torch.as_tensor(nt[i]), CONE, 64, np.random.default_rng(900 + i)).numpy()[:M2 - 1]], 0)
                    for i in dead]).astype(np.float32)
                pts2 = np.repeat(p_march[dead] + s_ * dd[dead], M2, 0)
                ok2, q2 = lb.feasible_rows(env0, tree, Td, pts2, d2.reshape(-1, 3), np.repeat(nt[dead], M2, 0),
                                           cosc, tube, k_nn=200, n_try=24,
                                           q_hint=None if r == 0 else np.repeat(q_prev[dead], M2, 0),
                                           kn_lim=(None, KN_MAX), kn_descend=True,
                                           t_rows=np.repeat(dd[dead], M2, 0), mu=MU)
                ok2 = ok2.reshape(len(dead), M2); q2 = q2.reshape(len(dead), M2, 7)
                any2 = ok2.any(1); pick2 = ok2.argmax(1)
                q_prev[dead[any2]] = q2[np.arange(len(dead)), pick2][any2]
                first_bad[dead[~any2]] = r
        alive = alive[first_bad[alive] < 0]
        if r % 10 == 0:
            print(f'[march] s={s_:.2f} alive {len(alive)}/{N} witness-certified {n_wc}', flush=True)
    lpwf = np.where(first_bad < 0, smax + STEPM, first_bad * STEPM).astype(np.float32)
    p0 = p_march
    print(f'lpw_force: mean {lpwf[has].mean():.3f} '
          f'(30-deg ref mean {ref30[sub][has].mean():.3f})', flush=True)
    np.savez(OUTF, sub=sub, q0_first=q0_first, has=has, lpwf=lpwf,
             cands_q=CQ, cands_t=CT, cands_kn=CK)
else:
    d = dict(np.load(OUTF))
    sub, q0_first, has, lpwf = d['sub'], d['q0_first'], d['has'], d['lpwf']
    print('loaded starts + bound from', OUTF, flush=True)
del env0; torch.cuda.empty_cache()


def roll(cfgfile, ckpt, classical=False):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw.update(FORCE_KW)
    kw['cone_deg'] = CONE
    B = 2000
    renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    rdt = renv.kin.dtype
    if classical:
        fn = cn_action_fn(ClassicalNullspaceController(renv.kin))
    else:
        ag = Agent(renv.obs_dim, renv.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO / ckpt, map_location=dev))
        ag.eval()
    out = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        def _t(a):
            t = torch.tensor(a[lo:hi], dtype=rdt, device=dev)
            return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])]) if pad else t
        renv.line_dist = ScriptedLineDistribution({'q0': _t(q0_first), 'line_dir': _t(dd), 'n_target': _t(nt)})
        renv.reset()
        with torch.no_grad():
            for _ in range(renv.cfg.max_steps // 2):
                a = fn(renv) if classical else ag.actor_mean(renv.current_obs())
                for _ in range(2):
                    renv.step(a, auto_reset=False)
                if bool(renv.done_persistent.all()):
                    break
        out[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
    del renv; torch.cuda.empty_cache()
    return out


d = dict(np.load(OUTF))
if 'p_cls' not in d:
    d['p_cls'] = roll(str(Path(hl.ROBOTS['fr3'][0]).name), None, classical=True)
    print('classical rolled', flush=True)
    np.savez(OUTF, **d)
if 'p_rl' not in d:
    # the flagship needs its 35-dim obs: observe_force stays off for it
    d['p_rl'] = roll('config_line_cont_dirfrac_e8kXXL_rm.yaml',
                     'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt')
    print('flagship rolled', flush=True)
    np.savez(OUTF, **d)
for tag in extra:
    d[f'p_{tag}'] = roll(f'config_line_cont_dirfrac_e8kXXL_{tag}.yaml',
                         f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{tag}/agent.pt')
    print(f'{tag} rolled', flush=True)
    np.savez(OUTF, **d)
rows = [('classical', d['p_cls']), ('flagship-0shot', d['p_rl'])] + \
       [(tag, d[f'p_{tag}']) for tag in extra]
ref = np.maximum.reduce([lpwf] + [v for _, v in rows])
for tag, v in rows:
    rt = v[has] / np.maximum(ref[has], 1e-9)
    print(f'{tag:15s} stroke {v[has].mean():.3f}  ratio {rt.mean() * 100:.1f} / '
          f'{np.percentile(rt, 10) * 100:.1f}', flush=True)
