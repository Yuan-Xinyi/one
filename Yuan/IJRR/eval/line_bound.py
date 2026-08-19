"""Per-task pointwise-feasibility bound on the achievable path length.

A continuous stroke has to pass through every point of the prescribed path up
to the point where it stops, so the first point along that path at which *no*
admissible configuration exists is an upper bound on the achievable length.
That bound depends on the kinematics alone, which is what makes it usable as
the denominator of "how much of the pointwise reachability was realised".

The voxel map in ``reach_map.py`` answers a coarser question and cannot serve
as that denominator: it snaps the query to a 5 cm voxel centre and it asks
whether the point is reachable in *any* of 50 globally sampled tool
directions, ignoring the task's own orientation cone. Measured on the 10,000
task set, a chord read off that map is exceeded by the achieved length on
5.3% of tasks (worst case 68x), so it is not a bound at all.

This module instead marches along each task's own line and runs the same
cone-constrained IK used to build the seed candidates, with the task's own
n_target and cone half-angle. There is no voxel snapping and no direction
mismatch, so the only remaining source of error is IK incompleteness, which is
pushed down by warm-starting from the CVT table, retrying, and carrying the
previous point's solution forward.

The march is iterative-deepening: every alive task is probed at the same arc
length in one batch, and a task drops out as soon as it fails. Two bounds are
reported per task,

    L_lo   the last arc length still certified feasible
    L_hi   the first arc length at which the IK found nothing

with the true bound lying in between; ``L_hi`` is the conservative choice and
the one to use as a denominator.

Usage:
    python -m Yuan.IJRR.eval.line_bound --n-tasks 500 --out runs/line_bound_500.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from Yuan.IJRR.env.env import (
    NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET,
)
from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z, _sample_in_cone
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import _minimal_rotvec, POS_SCALE
from Yuan.IJRR.kinematics.pen_collision import PenSphereCollision

REPO = Path(__file__).resolve().parents[3]   # Yuan/IJRR/<sub>/<file>.py -> repo root
TABLE = "Yuan/IJRR/runs/iksel_clean_v1/cvt_table_201600.npz"
ENV_YAML = "Yuan/IJRR/stage2_traj/config.yaml"
TASKS = "Yuan/IJRR/runs/eval_10k_systematic/eval_set_10k.npz"
CONE_DEG = 30.0


@torch.no_grad()
def feasible_rows(env, tree, T, pts, zs, n_refs, cos_lim, tube,
                  k_nn=200, n_try=12, q_hint=None, chunk=8192):
    """Per row: is there a collision-free q at ``pts`` with the tool along
    ``zs`` and within the cone around ``n_refs``?

    Every row carries its own tool direction and its own cone axis, which is
    what distinguishes this from the shared-direction sweep in
    ``fig_slice_capacity.ik_feasible``. ``q_hint`` supplies one extra warm
    start per row -- the solution found at the previous point of the same
    march, which is the single most effective seed because consecutive points
    are 2 cm apart.

    ``tube`` is the position tolerance of Eq. (3): a stroke is admissible while
    the tip stays within that distance of the prescribed path, so the feasible
    set at arc length s is the ball of that radius around p(s), not the single
    point. Testing the point alone under-reports the bound and lets a rollout
    that drifts inside the tolerance overtake it.

    Returns (ok, q) of shapes (P,) and (P, 7).
    """
    dev, dt = env.device, env.kin.dtype
    P = pts.shape[0]
    ok = np.zeros(P, bool)
    q_out = np.full((P, int(env.kin.lmt_lo.shape[0])), np.nan, np.float32)
    hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
    Tpos, Tzax, Tq, Tji = T["pos"], T["zax"], T["q"], T["jinv6"]

    cand = np.empty((P, n_try, int(env.kin.lmt_lo.shape[0])), np.float32)
    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        feat = np.concatenate([pts[lo:hi] * POS_SCALE, zs[lo:hi]], 1).astype(np.float32)
        _, ids = tree.query(feat, k=k_nn, workers=-1)
        C = hi - lo
        dp = pts[lo:hi, None, :] - Tpos[ids]
        rv = _minimal_rotvec(Tzax[ids].reshape(-1, 3),
                             np.repeat(zs[lo:hi], k_nn, 0)).reshape(C, k_nn, 3)
        d6 = np.concatenate([dp, rv], -1).astype(np.float32)
        dq = np.einsum('ckje,cke->ckj', Tji[ids], d6)
        order = (dq * dq).sum(-1).argsort(1)[:, :n_try]
        cand[lo:hi] = Tq[np.take_along_axis(ids, order, 1)]

    slots = [cand[:, t] for t in range(n_try)]
    if q_hint is not None:
        for h in (q_hint if isinstance(q_hint, list) else [q_hint]):
            slots.insert(0, np.nan_to_num(h, nan=0.0).astype(np.float32))

    pend = np.arange(P)
    for q_slot in slots:
        if not len(pend):
            break
        # The projection and the collision check are chunked: at the first
        # march step every task is alive, so pend can hold hundreds of
        # thousands of rows and the pairwise sphere margins alone would not
        # fit on the device.
        for c_lo in range(0, len(pend), chunk):
            rows = pend[c_lo:c_lo + chunk]
            q0 = torch.as_tensor(q_slot[rows], device=dev, dtype=dt)
            p_t = torch.as_tensor(pts[rows], device=dev, dtype=dt)
            R_t = _build_R_with_z(
                torch.as_tensor(zs[rows], device=dev, dtype=dt), hint)
            q_o, _, _ = _batched_ik_project(env.kin, q0, p_t, R_t,
                                            branch_action=None)
            # The projector's own convergence flag demands the exact point to
            # within 5 mm; the admissible set here is the tolerance tube, so
            # every projected configuration is scored rather than only the
            # converged ones.
            coll = env.collision.is_collided(env.kin.link_transforms(q_o))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
            nt = torch.as_tensor(n_refs[rows], device=dev, dtype=dt)
            in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                      & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
            fine = ((~coll) & in_lmt
                    & ((p_fk - p_t).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * nt).sum(-1) >= cos_lim))
            f = fine.cpu().numpy()
            ok[rows[f]] = True
            q_out[rows[f]] = q_o[fine].cpu().numpy()
        # A row that projected but failed the collision, position or cone check
        # is still pending: the next warm start may land on another branch.
        pend = pend[~ok[pend]]
    return ok, q_out


@torch.no_grad()
def witness_rows(env, qw, pts, n_refs, cos_lim, tube):
    """Constraint check of externally supplied witness configurations,
    without any IK projection: a witness comes from an executed rollout, so
    it either satisfies the point's constraints as it stands or it does not
    count."""
    dev, dt = env.device, env.kin.dtype
    q_t = torch.as_tensor(qw, device=dev, dtype=dt)
    p_t = torch.as_tensor(pts, device=dev, dtype=dt)
    nt = torch.as_tensor(n_refs, device=dev, dtype=dt)
    coll = env.collision.is_collided(env.kin.link_transforms(q_t))
    p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_t)
    in_lmt = ((q_t >= env.kin.lmt_lo - 1e-5)
              & (q_t <= env.kin.lmt_up + 1e-5)).all(dim=-1)
    fine = ((~coll) & in_lmt
            & ((p_fk - p_t).norm(dim=-1) <= tube)
            & ((R_fk[:, :, 2] * nt).sum(-1) >= cos_lim))
    return fine.cpu().numpy()


def build_env(device, collision, chunk, robot="fr3"):
    from Yuan.IJRR.eval.horizon_ladder import ROBOTS as _R
    cfg_path = ENV_YAML if robot == "fr3" else _R[robot][0]
    with open(REPO / cfg_path) as f:
        y = yaml.safe_load(f)
    keys = {fl.name for fl in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y["env"].items() if k in keys}
    env = NSRLBatchedEnv(EnvConfig(**{**kw, "n_envs": chunk}), None, device)
    if collision == "pen":
        env.collision = PenSphereCollision(env.kin.tcp_offset, device=device)
        print(f"[bound] collision model includes hand and pen "
              f"({len(env.collision.radii)} spheres)")
    else:
        print("[bound] collision model is the stock link0..link7 set, "
              "matching env.py used by every cached rollout")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--robot", default="fr3", choices=["fr3","xarm7","cobotta"],
                    help="env config per eval.horizon_ladder.ROBOTS")
    ap.add_argument("--table", default=None,
                    help="warm-start table npz (pos, zax, q, jinv6); "
                         "defaults to the FR3 CVT table")
    ap.add_argument("--n-tasks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--max-len", type=float, default=1.8)
    ap.add_argument("--n-dirs", type=int, default=24,
                    help="tool directions tried inside the cone at each point")
    ap.add_argument("--dir-pool", type=int, default=64,
                    help="pool the directions are drawn from before truncation "
                         "to --n-dirs; keep fixed across a convergence sweep so "
                         "the direction sets nest")
    ap.add_argument("--n-try", type=int, default=12)
    ap.add_argument("--k-nn", type=int, default=200)
    ap.add_argument("--cone-deg", type=float, default=CONE_DEG)
    ap.add_argument("--tube", type=float, default=None,
                    help="position tolerance of Eq. (3); defaults to the "
                         "env's LATERAL_SAFETY_NET so the bound admits every "
                         "stroke the rollout itself would admit")
    ap.add_argument("--collision", choices=["stock", "pen"], default="stock")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-q0", action="store_true",
                    help="seed the march at s=0 with the task's own start "
                         "configuration (requires cs_q0 in the tasks npz); "
                         "the witness chain then grows out of the same "
                         "posture the rollouts start from")
    ap.add_argument("--save-witness", action="store_true",
                    help="also save the feasibility-witness configuration "
                         "at every certified march point (q_witness, NaN "
                         "where infeasible); a chain of IK solutions, not "
                         "a dynamically consistent trajectory")
    ap.add_argument("--witness", default=None,
                    help="npz with W (N_all, n_grid, 7) and step: "
                         "configurations recorded along executed rollouts at "
                         "the march arc grid; a witness that passes the "
                         "constraint check certifies the point without "
                         "search, so the estimate is never below any "
                         "evaluated rollout")
    a = ap.parse_args()

    dev = torch.device(a.device)
    t = np.load(REPO / a.tasks, allow_pickle=False)
    N_all = len(t["cs_p0"])
    rng = np.random.default_rng(a.seed)
    sel = (np.arange(N_all) if a.n_tasks >= N_all
           else np.sort(rng.choice(N_all, a.n_tasks, replace=False)))
    p0 = t["cs_p0"][sel].astype(np.float32)
    d = t["cs_line_dir"][sel].astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n_t = t["cs_n_target"][sel].astype(np.float32)
    n_t /= np.linalg.norm(n_t, axis=1, keepdims=True)
    N = len(sel)

    env = build_env(dev, a.collision, a.chunk, robot=a.robot)
    NJ = int(env.kin.lmt_lo.shape[0])
    T = np.load(a.table) if a.table else np.load(REPO / TABLE)
    tree = cKDTree(np.concatenate(
        [T["pos"] * POS_SCALE, T["zax"]], 1).astype(np.float32))
    cos_lim = math.cos(math.radians(a.cone_deg))
    tube = LATERAL_SAFETY_NET if a.tube is None else a.tube
    print(f"[bound] position tolerance tube {tube * 1000:.0f} mm, "
          f"cone {a.cone_deg} deg, {a.n_dirs} directions per point")

    # Directions are sampled across the full cone and the axis is always
    # included; a converged solution may sit up to THETA_MAX off the requested
    # direction, so the cone membership is re-checked exactly per row rather
    # than assumed from the sampling.
    #
    # The pool is always drawn at --dir-pool and then truncated, so the set
    # used at n_dirs=24 is a subset of the one used at 48. Drawing n_dirs
    # samples directly would not nest -- _sample_in_cone consumes the rng
    # twice, so changing the count shifts the second draw and yields a
    # different set -- and the bound would then not be monotone in the search
    # budget, which is what the convergence check relies on.
    M = a.n_dirs
    dirs = np.empty((N, M, 3), np.float32)
    for i in range(N):
        pool = _sample_in_cone(torch.as_tensor(n_t[i]), a.cone_deg,
                               a.dir_pool, np.random.default_rng(a.seed + i)
                               ).numpy()
        pool[0] = n_t[i]
        dirs[i] = pool[:M]

    PATH_PTS = PATH_AXES = None
    if "path_pts" in t.files:
        PATH_PTS = t["path_pts"][sel].astype(np.float32)
        PATH_AXES = t["path_axes"][sel].astype(np.float32)
        print(f"[bound] per-task path tables: {PATH_PTS.shape}")
        # canonical cone directions around +z, rotated to the local axis
        rngc = np.random.default_rng(a.seed)
        zc = np.zeros((a.dir_pool, 3), np.float32); zc[:, 2] = 1.0
        angc = rngc.uniform(0, math.radians(a.cone_deg * 0.8), a.dir_pool)
        azc = rngc.uniform(0, 2 * math.pi, a.dir_pool)
        zc[:, 0] = np.sin(angc) * np.cos(azc)
        zc[:, 1] = np.sin(angc) * np.sin(azc)
        zc[:, 2] = np.cos(angc)
        zc[0] = (0.0, 0.0, 1.0)
        CDIRS = zc[:M]

        def _rot_to(axis_rows):
            z = np.array([0.0, 0.0, 1.0], np.float32)
            v = np.cross(np.broadcast_to(z, axis_rows.shape), axis_rows)
            c = axis_rows[:, 2:3]
            vn = np.linalg.norm(v, axis=1, keepdims=True)
            out = np.empty((axis_rows.shape[0], 3, 3), np.float32)
            for i in range(axis_rows.shape[0]):
                if vn[i, 0] < 1e-8:
                    out[i] = np.eye(3, dtype=np.float32) * (1 if c[i, 0] > 0 else -1)
                    if c[i, 0] < 0:
                        out[i][2, 2] = 1.0  # degenerate; rare
                    continue
                vv = v[i] / vn[i, 0]
                K = np.array([[0, -vv[2], vv[1]],
                              [vv[2], 0, -vv[0]],
                              [-vv[1], vv[0], 0]], np.float32)
                th = math.acos(max(-1.0, min(1.0, float(c[i, 0]))))
                out[i] = np.eye(3, dtype=np.float32) + math.sin(th) * K \
                    + (1 - math.cos(th)) * (K @ K)
            return out

    n_steps = int(round(a.max_len / a.step))
    if PATH_PTS is not None:
        n_steps = min(n_steps, PATH_PTS.shape[1] - 1)
    first_bad = np.full(N, -1, np.int64)
    alive = np.arange(N)
    q_prev = np.full((N, NJ), np.nan, np.float32)
    seeded = False
    if a.start_q0:
        if "cs_q0" not in t.files:
            raise SystemExit("--start-q0 needs cs_q0 in the tasks npz")
        q_prev[:] = t["cs_q0"][sel].astype(np.float32)
        seeded = True
    witness = (np.full((N, n_steps + 1, NJ), np.nan, np.float32)
               if a.save_witness else None)
    Wit = None
    if a.witness:
        wz = np.load(a.witness)
        assert abs(float(wz["step"]) - a.step) < 1e-9, \
            "witness grid step must equal the march step"
        Wit = wz["W"][sel]
        print(f"[bound] rollout witnesses: {a.witness}  "
              f"({int(np.isfinite(Wit[:, :, 0]).sum())} points)")
        if "p_start" in wz.files:
            # march along the EXECUTED ray: the deployed start configuration
            # sits within the position tolerance of the nominal p0, so the
            # rollout's own ray is offset by up to that tolerance; both sides
            # of the bracket must share one ray for the arc lengths to be
            # comparable
            p0 = wz["p_start"][sel].astype(np.float32)
            print("[bound] marching along the executed ray "
                  "(p_start from the witness file)")
    n_wcert = 0
    t0 = time.time()

    for r in range(n_steps + 1):
        if not len(alive):
            break
        s = r * a.step
        if PATH_PTS is not None:
            cur_pts = PATH_PTS[:, r]
            cur_axes = PATH_AXES[:, r]
        certified = np.zeros(N, bool)
        if Wit is not None and r < Wit.shape[1]:
            w = Wit[alive, r]
            have = np.isfinite(w).all(axis=1)
            if have.any():
                rows = alive[have]
                wq = w[have].astype(np.float32)
                for c_lo in range(0, len(rows), a.chunk):
                    rr = rows[c_lo:c_lo + a.chunk]
                    ww = wq[c_lo:c_lo + a.chunk]
                    ptsw = (cur_pts[rr] if PATH_PTS is not None
                            else p0[rr] + d[rr] * s)
                    nrfw = (cur_axes[rr] if PATH_PTS is not None
                            else n_t[rr])
                    fine = witness_rows(env, ww, ptsw, nrfw, cos_lim, tube)
                    certified[rr[fine]] = True
                    q_prev[rr[fine]] = ww[fine]
                    n_wcert += int(fine.sum())
        search = alive[~certified[alive]]
        if len(search):
            if PATH_PTS is not None:
                pts = np.repeat(cur_pts[search], M, axis=0)
                Rrot = _rot_to(cur_axes[search])
                zs = np.einsum('nij,mj->nmi', Rrot, CDIRS).reshape(-1, 3)
                nrf = np.repeat(cur_axes[search], M, axis=0)
            else:
                pts = np.repeat(p0[search] + d[search] * s, M, axis=0)
                zs = dirs[search].reshape(-1, 3)
                nrf = np.repeat(n_t[search], M, axis=0)
            hint = np.repeat(q_prev[search], M, axis=0)
            ok, q = feasible_rows(env, tree, T, pts, zs, nrf, cos_lim, tube,
                                  k_nn=a.k_nn, n_try=a.n_try,
                                  q_hint=None if (r == 0 and not seeded)
                                  else hint)
            ok = ok.reshape(len(search), M)
            q = q.reshape(len(search), M, NJ)
            any_ok = ok.any(axis=1)
            pick = ok.argmax(axis=1)
            q_prev[search[any_ok]] = q[np.arange(len(search)), pick][any_ok]
            first_bad[search[~any_ok]] = r
        if witness is not None:
            still = alive[first_bad[alive] < 0]
            witness[still, r] = q_prev[still]
        alive = alive[first_bad[alive] < 0]
        if r % 10 == 0 or not len(alive):
            print(f"[bound] s={s:.2f} m  alive {len(alive):5d}/{N}  "
                  f"witness-certified {n_wcert}  "
                  f"{time.time() - t0:6.1f}s", flush=True)

    censored = first_bad < 0
    first_bad[censored] = n_steps + 1
    L_hi = first_bad * a.step                      # conservative: use as denominator
    L_lo = (first_bad - 1).clip(0) * a.step        # last certified-feasible sample

    print(f"\ncensored (never failed within {a.max_len} m): {int(censored.sum())}")
    print(f"zero-length bound (start itself infeasible): {int((first_bad == 0).sum())}")
    print(f"L_hi  mean {L_hi.mean():.4f}  median {np.median(L_hi):.4f}")
    print(f"L_lo  mean {L_lo.mean():.4f}  median {np.median(L_lo):.4f}")

    print(f'\n{"achieved":<20s} {"bound":<6s} {"ratio med":>10s} {"ratio mean":>11s} '
          f'{">1 frac":>9s} {"max":>8s}')
    ach = {k: t[f].astype(np.float64)[sel]
           for k, f in (("L_seed", "L_seed"),
                        ("L_oracle(labels)", "max_label_L")) if f in t}
    hyb = REPO / "Yuan/IJRR/runs/eval_10k_systematic/cell_oracle_hyb_results.npz"
    if hyb.exists():
        z = np.load(hyb, allow_pickle=False)
        if len(z["L_best"]) == N_all:
            ach["oracle hybrid"] = np.asarray(z["L_best"], np.float64)[sel]
    for name, v in ach.items():
        for bn, b in (("L_lo", L_lo), ("L_hi", L_hi)):
            m = np.isfinite(v) & (b > 0)
            r = v[m] / b[m]
            print(f"{name:<20s} {bn:<6s} {np.median(r):>10.4f} {r.mean():>11.4f} "
                  f"{(r > 1).mean():>9.2%} {r.max():>8.3f}")

    out = Path(a.out) if a.out else (Path(__file__).parent / "runs" /
                                     f"line_bound_{N}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, idx=sel, L_lo=L_lo, L_hi=L_hi, first_bad=first_bad,
                        censored=censored, step=np.float32(a.step),
                        cone_deg=np.float32(a.cone_deg),
                        collision=a.collision, n_dirs=np.int32(M),
                        **({"q_witness": witness} if witness is not None
                           else {}))
    print(f"\n[bound] wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
