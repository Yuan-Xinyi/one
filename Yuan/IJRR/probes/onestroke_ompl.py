"""Driver for the native OMPL one-stroke planner.

Writes the problem out, runs the C++ binary, and then checks whatever comes
back with the SAME torch model the rest of the study uses -- the planner's own
verdict is not taken on trust. A silent disagreement between the C++ and torch
kinematics would show up immediately here as a tracking error of centimetres.
"""
import subprocess, sys, math, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch

BIN = SCR / 'ompl_onestroke'
WORK = SCR / 'ompl_work'
STEP = 0.002                    # figure sampling [m]
V = 0.2
QD = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
TUBE = 0.02
NDOWN = np.float32([0, 0, -1])


def dump_spheres(path):
    from one.robots.manipulators.franka.fr3.sphere_collision import (
        FR3SphereCollision)
    c = FR3SphereCollision(device='cpu')
    ce, ra = c.centers.numpy(), c.radii.numpy()
    li = c.link_indices.numpy()
    ij = np.argwhere(c.mask.numpy())
    with open(path, 'w') as f:
        f.write(f'{len(ce)}\n')
        for k in range(len(ce)):
            f.write(f'{ce[k,0]:.9g} {ce[k,1]:.9g} {ce[k,2]:.9g} '
                    f'{ra[k]:.9g} {li[k]}\n')
        f.write(f'{len(ij)}\n')
        for a, b in ij:
            f.write(f'{a} {b}\n')
    return len(ce), len(ij)


def _sharpen(W, min_turn_deg=10.0):
    """Collapse smeared corners back to exact vertices.

    Arc-length resampling splits a polygon corner across two neighbouring
    joints (a 120-degree turn becomes two 60-degree turns 2 mm apart), and a
    per-joint fillet then rounds each with a near-zero radius and leaves the
    corner effectively sharp. Consecutive turning joints are therefore merged
    first: the entry and exit lines of each cluster are intersected in the
    figure plane and the cluster is replaced by that single vertex, so the
    fillet sees one corner with the full turn."""
    W = np.asarray(W, np.float64)
    sg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    u = np.diff(W, axis=0) / sg[:, None]
    turn = np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0))
    thr = math.radians(min_turn_deg)
    hot = turn > thr                      # joint j is point index j+1
    out, i = [W[0]], 1
    n = len(W)
    while i < n - 1:
        if not hot[i - 1]:
            out.append(W[i]); i += 1
            continue
        j = i
        while j < n - 2 and hot[j]:
            j += 1
        # cluster spans point indices i..j; entry line through W[i-1] along
        # u[i-1], exit line through W[j+1] along u[j] (backwards)
        P0, d0 = W[i - 1, :2], u[i - 1, :2]
        P1, d1 = W[j + 1, :2], u[j, :2]
        den = d0[0] * (-d1[1]) - d0[1] * (-d1[0])
        if abs(den) > 1e-9:
            rhs = P1 - P0
            a = (rhs[0] * (-d1[1]) - rhs[1] * (-d1[0])) / den
            V = np.array([*(P0 + a * d0), W[i, 2]])
            out.append(V)
        else:
            out.extend(W[i:j + 1])
        i = j + 1
    out.append(W[-1])
    return np.asarray(out)


def fillet(W, dev=0.008, min_turn_deg=10.0):
    """Round the corners of a dense polyline within a deviation budget.

    The other controllers are graded inside a 20 mm tolerance tube; the
    planner's equality constraint pins it to the polyline itself, which at a
    corner means passing exactly through a tangent discontinuity -- a task
    nobody else is being asked to solve. Each corner is replaced by the
    circular fillet whose maximum deviation from the original vertex is
    ``dev``; the radius follows from the turn angle, so sharp corners get
    small fillets and the rounded curve stays well inside everyone's tube.
    Smooth figures have no corners above the threshold and pass unchanged.

    Bookkeeping is done in arc length: straight stretches are copied by
    interpolation between trim points, so a mistake cannot silently leave a
    corner unrounded -- every joint above the threshold is either rounded or
    was clamped against its neighbour, and both show up in the deviation
    check the caller runs."""
    W = _sharpen(np.asarray(W, np.float64), min_turn_deg)
    sg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    kp = sg > 1e-12
    W = np.concatenate([W[:1], W[1:][kp]])
    sg = sg[kp]
    arc = np.concatenate([[0.0], np.cumsum(sg)])
    u = np.diff(W, axis=0) / sg[:, None]
    turn = np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1.0, 1.0))
    cid = np.nonzero(turn > math.radians(min_turn_deg))[0] + 1
    if not len(cid):
        return W
    h = float(np.median(sg))

    def at(a):
        """point on the original polyline at arc position a"""
        a = min(max(a, 0.0), arc[-1])
        k = min(np.searchsorted(arc, a, side='right') - 1, len(sg) - 1)
        w = (a - arc[k]) / sg[k]
        return W[k] * (1 - w) + W[k + 1] * w

    # per-corner fillet radius and trim, clamped pairwise so trims never
    # overlap a neighbouring corner or run off the ends
    R, T = [], []
    for ci in cid:
        th = turn[ci - 1]
        alpha = (math.pi - th) / 2.0
        sa, ca = math.sin(alpha), math.cos(alpha)
        r = dev * sa / max(1.0 - sa, 1e-9)
        R.append(r); T.append(r * ca / max(sa, 1e-9))
    A = arc[cid]
    for k in range(len(cid)):
        lo = A[k] - (A[k - 1] + T[k - 1] if k else 0.0)
        hi = (A[k + 1] - T[k + 1] if k + 1 < len(cid) else arc[-1]) - A[k]
        c = min(T[k], 0.45 * max(lo, 0.0), 0.45 * max(hi, 0.0) if k + 1 < len(cid)
                else max(hi, 0.0))
        if c < T[k]:
            T[k] = max(c, 0.0)
            R[k] = T[k] * math.tan((math.pi - turn[cid[k] - 1]) / 2.0)

    pieces, cur = [], 0.0
    for k, ci in enumerate(cid):
        t, r = T[k], R[k]
        th = turn[ci - 1]
        if t < 2e-4 or r < 2e-4:
            continue
        a0, a1 = arc[ci] - t, arc[ci] + t
        n_st = max(2, int((a0 - cur) / h))
        pieces.append(np.stack([at(a) for a in np.linspace(cur, a0, n_st)]))
        ui, uo = u[ci - 1], u[ci]
        alpha = (math.pi - th) / 2.0
        bis = uo - ui
        C = W[ci] + bis / np.linalg.norm(bis) * (r / math.sin(alpha))
        va, vb = at(a0) - C, at(a1) - C
        axis = np.cross(va, vb)
        na = np.linalg.norm(axis)
        if na < 1e-12:
            cur = a1
            continue
        axis /= na
        ang = math.atan2(na, float(va @ vb))
        n_arc = max(3, int(math.ceil(r * ang / h)))
        for q_ in range(1, n_arc + 1):
            a = ang * q_ / n_arc
            pieces.append((C + va * math.cos(a)
                           + np.cross(axis, va) * math.sin(a)
                           + axis * (axis @ va) * (1 - math.cos(a)))[None])
        cur = a1
    n_st = max(2, int((arc[-1] - cur) / h))
    pieces.append(np.stack([at(a) for a in np.linspace(cur, arc[-1], n_st)]))
    Rl = np.concatenate(pieces)
    sg2 = np.linalg.norm(np.diff(Rl, axis=0), axis=1)
    kp2 = sg2 > 1e-12
    Rl = np.concatenate([Rl[:1], Rl[1:][kp2]]); sg2 = sg2[kp2]
    a2 = np.concatenate([[0.0], np.cumsum(sg2)])
    g = np.arange(0.0, a2[-1] + 1e-12, h)
    return np.stack([np.interp(g, a2, Rl[:, k]) for k in range(3)], 1)


def dump_figure(path, W, T):
    with open(path, 'w') as f:
        f.write(f'{len(W)} {STEP:.9g}\n')
        for k in range(len(W)):
            f.write(' '.join(f'{v:.9g}' for v in (*W[k], *T[k])) + '\n')


def verify(kin, coll, Q, W, cos_lim, W_orig=None):
    """Same admissibility test the environment terminates on, plus the
    velocity box between consecutive planner states."""
    dev, dt = kin.lmt_lo.device, kin.lmt_lo.dtype
    qt = torch.as_tensor(Q[:, :7], device=dev, dtype=dt)
    s = Q[:, 7]
    idx = np.clip(np.rint(s / STEP).astype(int), 0, len(W) - 1)
    p, R, _, _ = kin.tcp_fk_jac(qt)
    trk = (p - torch.as_tensor(W[idx], device=dev, dtype=dt)).norm(dim=-1)
    cn = (R[:, :, 2] * torch.as_tensor(NDOWN, device=dev, dtype=dt)).sum(-1)
    lim = ((qt >= kin.lmt_lo) & (qt <= kin.lmt_up)).all(-1)
    cm = coll.min_margin(kin.link_transforms(qt))
    # the fillet is an implementation device, not a change of task: the tip
    # must still be inside the ORIGINAL figure's tolerance tube
    d_orig = 0.0
    if W_orig is not None:
        from scipy.spatial import cKDTree as _KD
        d_orig = float(_KD(W_orig).query(p.cpu().numpy(), k=1,
                                          workers=-1)[0].max())
    ds = np.diff(s)
    dq = np.abs(np.diff(Q[:, :7], axis=0))
    # ds == 0 is a duplicated state, which is harmless ONLY if the arm does not
    # move there; a standstill with joint motion is reconfiguring in place,
    # which at a constant feed is lifting the pen and must be rejected.
    still = ds <= 1e-12
    vel = bool(np.all(dq[~still] <= (QD[None, :] * ds[~still, None] / V) + 1e-9)
               and np.all(dq[still] <= 1e-9))
    return dict(track=float(trk.max()), cone=float(cn.min()),
                coll=float(cm.min()), lim=bool(lim.all()),
                mono=bool((ds >= -1e-12).all()), vel=bool(vel),
                d_orig=d_orig,
                s_end=float(s[-1]), L=float(STEP * (len(W) - 1)),
                ok=bool(trk.max() <= TUBE and cn.min() >= cos_lim
                        and lim.all() and cm.min() >= 0
                        and (ds >= -1e-12).all()
                        and d_orig <= TUBE
                        and vel and s[-1] >= STEP * (len(W) - 1) - 1e-6))


CTOL = 0.010                    # constraint tolerance = the shared tube share
DELTA = 0.05                    # OMPL's own default geodesic step
SPACE = 'projected'


def build_bank(env, tree, T0, W, cone_deg, n_dirs=4, k_nn=64, node=0.02):
    """Admissible (q, s) states along the whole figure, from the same
    cone-constrained inverse kinematics that builds every method's start pool
    and the pointwise bound. One state per (node, cone direction) that
    converges; the sampler diversifies from there."""
    import math as _m
    from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone
    cos_lim = _m.cos(_m.radians(cone_deg))
    sg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(sg)])
    s_nodes = np.arange(0.0, arc[-1] + 1e-9, node)
    idx = np.searchsorted(arc, s_nodes).clip(0, len(W) - 1)
    P = W[idx].astype(np.float32)
    N = len(P)
    rows_p, rows_z, rows_n, rows_s = [], [], [], []
    rng = np.random.default_rng(7)
    for m in range(n_dirs):
        if m == 0:
            zs = np.tile(NDOWN, (N, 1))
        else:
            zs = np.stack([_sample_in_cone(
                torch.as_tensor(NDOWN), cone_deg, 1,
                np.random.default_rng(m * 131 + i)).numpy()[0]
                for i in range(N)]).astype(np.float32)
        rows_p.append(P); rows_z.append(zs)
        rows_n.append(np.tile(NDOWN, (N, 1))); rows_s.append(s_nodes)
    pts = np.concatenate(rows_p); zs = np.concatenate(rows_z)
    nrf = np.concatenate(rows_n); ss = np.concatenate(rows_s)
    from Yuan.IJRR.eval import line_bound as lb
    ok, q = lb.feasible_rows(env, tree, T0, pts, zs, nrf, cos_lim, TUBE,
                             k_nn=k_nn, n_try=8, q_hint=None, chunk=8192)
    bank = np.concatenate([q[ok], ss[ok, None]], 1)
    return bank.astype(np.float64)


def plan(key, S, place, q0, cone_deg, budget=300.0, planner='rrt',
         kin=None, coll=None, verbose=True, bank=None, delta=None):
    """q0 may be one configuration or the whole candidate pool (N, 7): every
    row becomes a start state of the same query, which is the planner-side
    equivalent of the framework's selection stage."""
    import onestroke_lib as osl
    WORK.mkdir(exist_ok=True)
    sph = WORK / 'spheres.txt'
    if not sph.exists():
        n, m = dump_spheres(sph)
        if verbose:
            print(f'[ompl] collision model {n} spheres, {m} checked pairs')
    _, p2, Lu, _ = osl.build(key)
    pts = osl.resample(p2, STEP / S)
    W_orig = (np.concatenate([S * pts, np.zeros((len(pts), 1), np.float32)], 1)
              + place).astype(np.float64)
    W = fillet(W_orig)
    T = np.gradient(W, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
    fig = WORK / f'{key}_{int(S*100)}.txt'
    dump_figure(fig, W, T)
    q0 = np.atleast_2d(np.asarray(q0, np.float64))
    q0f = WORK / 'q0.txt'
    q0f.write_text('\n'.join(' '.join(f'{v:.12g}' for v in row)
                              for row in q0))
    bankf = '-'
    if bank is not None and len(bank):
        bf = WORK / f'bank_{key}_{int(S*100)}.txt'
        bf.write_text('\n'.join(' '.join(f'{v:.12g}' for v in row)
                                 for row in bank))
        bankf = str(bf)
    sol = WORK / f'sol_{key}_{int(S*100)}.txt'

    t0 = time.time()
    r = subprocess.run([str(BIN), str(sph), str(fig), str(q0f),
                        str(cone_deg), str(budget), str(sol), planner,
                        SPACE, str(DELTA if delta is None else delta),
                        str(CTOL), bankf],
                       capture_output=True, text=True, timeout=budget + 300)
    el = time.time() - t0
    line = [l for l in r.stdout.splitlines() if l.startswith('SOLVED')]
    if not line or not sol.exists():
        if verbose:
            print(f'[ompl] {key} S={S:.2f}: no path  ({el:.0f}s)  '
                  f'{r.stdout.strip()[:80]} {r.stderr.strip()[:80]}')
        return None, dict(ok=False, t_plan=el)
    Q = np.loadtxt(sol, skiprows=1).reshape(-1, 8)
    chk = verify(kin, coll, Q, W, math.cos(math.radians(cone_deg)),
                 W_orig=W_orig)
    chk['t_plan'] = el
    if verbose:
        print(f'[ompl] {key} S={S*100:.0f}cm  {r.stdout.splitlines()[-1]}  '
              f'| torch check: track {chk["track"]*1000:.1f} mm  '
              f'cone {math.degrees(math.acos(min(1,chk["cone"]))):.1f} deg  '
              f'coll {chk["coll"]*1000:.1f} mm  dorig {chk["d_orig"]*1000:.1f} mm  '
              f'mono {chk["mono"]}  '
              f'vel {chk["vel"]}  s {chk["s_end"]:.2f}/{chk["L"]:.2f}  '
              f'-> {chk["ok"]}  ({el:.0f}s)', flush=True)
    return Q, chk
