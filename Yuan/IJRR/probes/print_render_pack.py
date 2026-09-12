"""Joint trajectories for the object demos.

The arm is marched along each object's own toolpath, solving the cone IK at
every waypoint warm-started from the previous one. That is the witness chain
the paper already uses to certify a pointwise bound -- it demonstrates that the
whole toolpath is coverable, and because each solution starts from its
predecessor it stays on one branch and reads as a continuous motion. It is NOT
a controller's execution: the continuity question is measured separately in
print_raster.py, and the classical law does not survive these paths.

A coarser layer height and infill than a real print is used so the animation is
tens of seconds rather than tens of minutes; the geometry is unchanged.
"""
import sys, math, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
PACK = OUT / 'render'
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from scipy.spatial import cKDTree

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
import print_objects as po

DEV = torch.device('cuda')
CONE = 5.0
TUBE = 0.002
MAXW = 820            # waypoints per animation
# coarser than a real print so the clip is watchable; geometry unchanged
RENDER_KW = {
    'cube':      dict(dz=0.22, pitch=0.30),
    'penholder': dict(dz=0.15),
    'bowl':      dict(dz=0.12),
    'vase':      dict(dz=0.10),
    'plate':     dict(dz=0.055),
    'dome':      dict(dz=0.115),
    'mug':       dict(dz=0.16),
}


JUMP = 0.15           # rad, per-joint step counted as a hard step in the stats
INS_JUMP = 0.50       # rad, step large enough that it is a branch change and
                      # has to be played as an explicit travel move


def march(env, tree, T, pts, cone_deg, n_dirs=7):
    """Cone IK at every waypoint, continued from the one before.

    The cone leaves the roll about the tool axis free, so an unconstrained
    solver drifts across that null direction and the chain snaps by radians
    between neighbouring points. Each step is therefore solved with
    ``preserve_seed`` from the previous posture -- no pull toward mid-range --
    and the admissible candidate closest to it is kept. When nothing near the
    previous posture works, the full search is allowed to change branch and the
    step is counted: those counts are re-seats the toolpath actually needs."""
    cos_lim = math.cos(math.radians(cone_deg))
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, math.radians(cone_deg * 0.85), n_dirs)
    az = rng.uniform(0, 2 * math.pi, n_dirs)
    dirs = np.stack([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az),
                     -np.cos(ang)], 1).astype(np.float32)
    dirs[0] = (0, 0, -1)
    nref = np.tile(np.float32([0, 0, -1]), (n_dirs, 1))
    dt_, NJ = env.kin.dtype, int(env.kin.lmt_lo.shape[0])
    nref_t = torch.as_tensor(nref[0], device=DEV, dtype=dt_)
    hintx = torch.tensor([1.0, 0.0, 0.0], dtype=dt_, device=DEV)
    dirs_t = torch.as_tensor(dirs, device=DEV, dtype=dt_)

    Q = np.full((len(pts), NJ), np.nan, np.float32)
    q_prev, miss, switch = None, 0, 0
    for i, p in enumerate(pts):
        p_t = torch.as_tensor(np.tile(p, (n_dirs, 1)), device=DEV, dtype=dt_)
        cand = None
        if q_prev is not None:
            seed = torch.as_tensor(np.tile(q_prev, (n_dirs, 1)),
                                   device=DEV, dtype=dt_)
            q_o, _, _ = _bip(env.kin, seed, p_t,
                             _brz(dirs_t, hintx), None, preserve_seed=True)
            fine = _admits(env, q_o, p_t, nref_t, cos_lim)
            if bool(fine.any()):
                qn = q_o[fine].cpu().numpy()
                cand = qn[np.abs(qn - q_prev).max(1).argmin()]
        if cand is None:
            ok, q = lb.feasible_rows(
                env, tree, T, np.tile(p, (n_dirs, 1)).astype(np.float32),
                dirs, nref, cos_lim, TUBE, k_nn=64, n_try=6,
                q_hint=None if q_prev is None
                else np.tile(q_prev, (n_dirs, 1)).astype(np.float32), chunk=64)
            if ok.any():
                cand = q[int(np.argmax(ok))]
                if q_prev is not None:
                    switch += 1
            else:
                miss += 1
        if cand is not None:
            q_prev = cand
        if q_prev is not None:
            Q[i] = q_prev
    return Q, miss, switch


def expand_reseats(Q, path, n_ins=26):
    """Turn each branch jump into an explicit travel move.

    A jump in the witness chain is not a rendering artefact, it is a re-seat:
    the arm has to leave the surface, change posture and come back. Playing it
    as a joint-space transition with deposition switched off shows what it
    actually costs instead of teleporting the arm between frames."""
    out_q, out_p, dep = [Q[0]], [path[0]], [True]
    for i in range(1, len(Q)):
        if np.isfinite(Q[i - 1, 0]) and np.isfinite(Q[i, 0]) \
                and np.abs(Q[i] - Q[i - 1]).max() > INS_JUMP:
            for a in np.linspace(0, 1, n_ins)[1:-1]:
                out_q.append((1 - a) * Q[i - 1] + a * Q[i])
                out_p.append(path[i])
                dep.append(False)
        out_q.append(Q[i]); out_p.append(path[i]); dep.append(True)
    return (np.stack(out_q).astype(np.float32),
            np.stack(out_p).astype(np.float32), np.array(dep, bool))


def _brz(z, hint):
    from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z
    return _build_R_with_z(z, hint)


def _bip(*a, **k):
    from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
    return _batched_ik_project(*a, **k)


def _admits(env, q_o, p_t, nref_t, cos_lim):
    coll = env.collision.is_collided(env.kin.link_transforms(q_o))
    p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
    in_l = ((q_o >= env.kin.lmt_lo - 1e-5)
            & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
    return ((~coll) & in_l & ((p_fk - p_t).norm(dim=-1) <= TUBE)
            & ((R_fk[:, :, 2] * nref_t).sum(-1) >= cos_lim))


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    oq = np.load(OUT / 'objects_query.npz')
    env = lb.build_env(DEV, 'stock', 64)
    T = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
                   .astype(np.float32))
    want = sys.argv[1:] if len(sys.argv) > 1 else [o[1] for o in po.OBJECTS]

    for cn, key, fn, dim in po.OBJECTS:
        if key not in want:
            continue
        t0 = time.time()
        s, cx, cy, zb = [float(v) for v in oq[f'{key}_c5']]
        raw = fn(**RENDER_KW.get(key, {}))
        p = po.densify(raw, 0.012) * s + np.float32([cx, cy, zb])
        if len(p) > MAXW:
            p = p[np.linspace(0, len(p) - 1, MAXW).astype(int)]
        Q, miss, switch = march(env, tree, T, p.astype(np.float32), CONE)
        good = np.isfinite(Q[:, 0])
        dq = (np.abs(np.diff(Q[good], axis=0)).max(1) if good.sum() > 1
              else np.zeros(1))
        n_big = int((dq > JUMP).sum())
        Q, p, dep = expand_reseats(Q, p.astype(np.float32))
        p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(
            torch.as_tensor(np.nan_to_num(Q), device=DEV, dtype=env.kin.dtype))
        tip = p_fk.cpu().numpy().astype(np.float32)
        zax = R_fk[:, :, 2].cpu().numpy().astype(np.float32)
        np.savez_compressed(PACK / f'{key}_pack.npz', q=Q.astype(np.float32),
                            tip=tip, zax=zax, path=p.astype(np.float32),
                            dep=dep, size=np.float32([s, cx, cy, zb]),
                            stats=np.int32([len(p), miss, switch, n_big]),
                            name=cn, dim=dim)
        print(f'{cn:<6s} {key:<9s} size {s*100:.0f} cm  {len(p):4d} waypoints  '
              f'IK miss {miss:3d}  branch switches {switch:3d}  '
              f'steps >{JUMP} rad {n_big:3d}  max |dq| {np.max(dq):.3f}  '
              f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
