"""Native OMPL on straight lines under the implicit-force constraint
(mu = 0.3, k_eff <= 2000 N/m, log-stiffness slope <= rate_max), 10-task
trial. Same fairness ladder as the one-stroke campaign: tolerance tube as
the constraint tolerance, the whole admissible candidate pool as start
states of one query, an informed bank sampler along the line, 300 s per
query; every solution is re-checked with the torch model. The policy row
is the friction-aware full-budget model with critic-picked starts, read
from force_eval_10k_mu0.3.npz."""
import sys, math, time, subprocess
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.env.env import LATERAL_SAFETY_NET
from onestroke_ompl import dump_spheres

MU, KN_MAX, F_SET, F_TOL, K_LAT, V = 0.3, 2000.0, 5.0, 2.0, 5.0, 0.2
RATE_MAX = F_TOL * K_LAT / (F_SET * V)          # |d ln k / ds| bound [1/m] = 10
CONE, STEP, TUBE, BUDGET = 30.0, 0.002, LATERAL_SAFETY_NET, 300.0
QD = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BIN = SCR / 'ompl_force_line'; WORK = SCR / 'ompl_force_work'; WORK.mkdir(exist_ok=True)
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
dev = torch.device('cuda')

d = dict(np.load(FU / 'force_eval_10k_mu0.3.npz'))
sub, lpwf, CQ, CT = d['sub'], d['lpwf'], d['cands_q'], d['cands_t']
has = np.bincount(CT, minlength=len(sub)) > 0
p_pol = d['p_sel_force_mu03full_mu0.3']
tz = np.load(A / 'tasks_pool_fr3.npz')
p0 = tz['cs_p0'][sub].astype(np.float64)
dd = tz['cs_line_dir'][sub].astype(np.float64); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float64); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
rng = np.random.default_rng(11)
pool = np.nonzero(has & (lpwf >= 0.5) & (lpwf <= 1.0))[0]
tasks = np.sort(rng.choice(pool, N_TASKS, replace=False))
print(f'tasks {tasks.tolist()}  bound {np.round(lpwf[tasks], 2).tolist()}', flush=True)

env0 = lb.build_env(dev, 'stock', 512); dt0 = env0.kin.dtype
kq0 = torch.as_tensor(env0.kq_joint, device=dev, dtype=dt0)
T0 = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1).astype(np.float32))
sph = WORK / 'spheres.txt'
if not sph.exists():
    dump_spheres(sph)


def bank_for(i, L, node=0.02, n_dirs=4):
    s_nodes = np.arange(0.0, L + 1e-9, node)
    P = (p0[i][None] + s_nodes[:, None] * dd[i][None]).astype(np.float32)
    N = len(P); pts, zs, ss = [], [], []
    for m in range(n_dirs):
        z = (np.tile(nt[i], (N, 1)) if m == 0 else np.stack([
            _sample_in_cone(torch.as_tensor(nt[i].astype(np.float32)), CONE, 1,
                            np.random.default_rng(m * 131 + k)).numpy()[0] for k in range(N)]))
        pts.append(P); zs.append(z.astype(np.float32)); ss.append(s_nodes)
    pts, zs, ss = np.concatenate(pts), np.concatenate(zs), np.concatenate(ss)
    nrf = np.tile(nt[i].astype(np.float32), (len(pts), 1))
    trow = np.tile(dd[i].astype(np.float32), (len(pts), 1))
    ok, q = lb.feasible_rows(env0, tree, T0, pts, zs, nrf, math.cos(math.radians(CONE)), TUBE,
                             k_nn=64, n_try=8, kn_lim=(None, KN_MAX), kn_descend=False,
                             t_rows=trow, mu=MU)
    return np.concatenate([q[ok], ss[ok, None]], 1).astype(np.float64)


@torch.no_grad()
def verify(i, Q, L):
    """First violation along the planner's path decides the verified s_end."""
    qt = torch.as_tensor(Q[:, :7], device=dev, dtype=dt0)
    s = Q[:, 7]
    P = torch.as_tensor(p0[i][None] + s[:, None] * dd[i][None], device=dev, dtype=dt0)
    p, R, _, _ = env0.kin.tcp_fk_jac(qt)
    trk = (p - P).norm(dim=-1)
    cn = (R[:, :, 2] * torch.as_tensor(nt[i], device=dev, dtype=dt0)).sum(-1)
    lim = ((qt >= env0.kin.lmt_lo) & (qt <= env0.kin.lmt_up)).all(-1)
    cm = env0.collision.min_margin(env0.kin.link_transforms(qt))
    tt = torch.as_tensor(np.tile(dd[i], (len(Q), 1)), device=dev, dtype=dt0)
    kn = lb.stiffness_along(env0, qt, R[:, :, 2], kq0, tt, MU)
    ok_state = ((trk <= TUBE) & (cn >= math.cos(math.radians(CONE))) & lim & (cm >= 0)
                & (kn <= KN_MAX)).cpu().numpy()
    ds = np.diff(s); dq = np.abs(np.diff(Q[:, :7], axis=0))
    still = ds <= 1e-12
    vel_ok = np.where(still, (dq <= 1e-9).all(1), (dq <= QD[None] * np.maximum(ds, 1e-12)[:, None] / V + 1e-9).all(1))
    lk = np.log(kn.cpu().numpy()); rate_ok = np.abs(np.diff(lk)) <= RATE_MAX * np.maximum(ds, 1e-12) + 1e-6
    mono = ds >= -1e-12
    edge_ok = vel_ok & rate_ok & mono
    good = ok_state[0]
    s_end = 0.0
    for k in range(len(Q) - 1):
        if not (good and edge_ok[k] and ok_state[k + 1]):
            break
        s_end = float(s[k + 1])
    return dict(s_end=s_end, L=L, track=float(trk.max()), kn_max=float(kn.max()),
                rate_max=float(np.abs(np.diff(lk)).max() / max(np.median(ds), 1e-9)) if len(ds) else 0.0,
                full=bool(s_end >= L - 1e-6))


res = []
for i in tasks:
    L = float(lpwf[i]); ss = np.arange(0.0, L + 1e-9, STEP)
    W = p0[i][None] + ss[:, None] * dd[i][None]
    fig = WORK / f'line_{i}.txt'
    with open(fig, 'w') as f:
        f.write(f'{len(W)} {STEP:.9g}\n{nt[i][0]:.9g} {nt[i][1]:.9g} {nt[i][2]:.9g}\n')
        for w in W:
            f.write(' '.join(f'{v:.9g}' for v in (*w, *dd[i])) + '\n')
    q0 = CQ[CT == i]
    q0f = WORK / f'q0_{i}.txt'; q0f.write_text('\n'.join(' '.join(f'{v:.12g}' for v in r) for r in q0))
    t0 = time.time(); bank = bank_for(i, L); tb = time.time() - t0
    bf = WORK / f'bank_{i}.txt'; bf.write_text('\n'.join(' '.join(f'{v:.12g}' for v in r) for r in bank))
    sol = WORK / f'sol_{i}.txt'
    t0 = time.time()
    r = subprocess.run([str(BIN), str(sph), str(fig), str(q0f), str(CONE), str(BUDGET), str(sol), 'rrt',
                        'projected', '0.05', '0.01', str(bf), str(MU), str(KN_MAX), str(RATE_MAX)],
                       capture_output=True, text=True, timeout=BUDGET + 300)
    el = time.time() - t0
    head = ' | '.join(l for l in r.stdout.splitlines() if l.startswith(('STARTS', 'BANK', 'STATUS', 'SOLVED')))
    if sol.exists() and 'SOLVED' in r.stdout:
        Q = np.loadtxt(sol, skiprows=1).reshape(-1, 8)
        chk = verify(i, Q, L)
    else:
        chk = dict(s_end=0.0, L=L, track=0.0, kn_max=0.0, rate_max=0.0, full=False)
    ref = max(L, float(p_pol[i]), chk['s_end'])
    res.append((int(i), L, chk['s_end'], float(p_pol[i]), el, len(q0), len(bank), chk['full']))
    print(f'task {i}: bound {L:.2f}  starts {len(q0)}  bank {len(bank)} ({tb:.0f}s)  {head}  '
          f'| verified s_end {chk["s_end"]:.2f} (kn max {chk["kn_max"]:.0f}, track {chk["track"]*1e3:.1f} mm)  '
          f'ompl {chk["s_end"]/ref*100:.0f}%  policy {p_pol[i]/ref*100:.0f}%  ({el:.0f}s)', flush=True)

R = np.array([(a, b, c, e, f) for a, b, c, dpol, e, *_ in [(x[0], x[1], x[2], x[3], x[4]) for x in res] for f in [dpol]])
L_ = np.array([x[1] for x in res]); so = np.array([x[2] for x in res]); pp = np.array([x[3] for x in res])
ref = np.maximum.reduce([L_, so, pp])
print(f'\nSUMMARY {len(res)} tasks: OMPL ratio {np.mean(so/ref)*100:.1f} (full {sum(x[7] for x in res)}/{len(res)}, '
      f'mean plan time {np.mean([x[4] for x in res]):.0f}s)   policy ratio {np.mean(pp/ref)*100:.1f}', flush=True)
np.savez(FU / 'ompl_force_10.npz', tasks=np.array([x[0] for x in res]), bound=L_, ompl=so, policy=pp,
         t_plan=np.array([x[4] for x in res]), full=np.array([x[7] for x in res]))
