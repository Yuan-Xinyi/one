"""Pointwise vs. continuous reachability for the ISRR FR3 + pen.

The slice IS the workpiece: a horizontal plane at z = z0. Its normal n = +/- z
is the tool-orientation reference, and every path direction d lies in the
plane, so n is orthogonal to d by construction and the whole task stays inside
the picture.

Two fields over the same (x, y) grid and shared mask, both in metres, for a
fixed path direction d = +x:

  1. ell_reach   longest straight segment from (x, y) whose every point the TCP
                 can occupy with the tool axis inside the 30 deg cone around
                 n = -z, respecting joint limits and collision constraints.
                 Every point is tested independently at 1 cm spacing.

  2. ell_max     longest arc the arm can actually traverse without stopping,
                 maximised over start configurations, under the ISRR hybrid
                 redundancy-resolution law and the same 30 deg tool cone.

1 >= 2 pointwise, so the pair reads as bound vs. actual, and ell_max/ell_reach
is the fraction of the geometrically permitted travel that is attainable.

    python -m Yuan.system_eval.fig_slice_capacity --draft \
        --out Yuan/system_eval/runs/curvature_scan/fig_slice_z000_posx.png
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _e = dict(os.environ)
    _e["LD_LIBRARY_PATH"] = _conda_lib + ":" + _e.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _e)

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z, _sample_in_cone, _dedup_q
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.unified_rl.iksel_clean_pilot import _minimal_rotvec, POS_SCALE
from Yuan.system_eval.pen_collision import PenSphereCollision

TABLE = "Yuan/unified_rl/runs/iksel_clean_v1/cvt_table_201600.npz"
RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
ENV_YAML = "Yuan/RL_controller/config.yaml"
CONE_DEG = 30.0


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi),
                     np.sin(th) * np.sin(phi), np.cos(phi)], 1).astype(np.float32)


@torch.no_grad()
def ik_feasible(env, tree, T, pts, tool_dirs, rng, cone_deg=180.0,
                n_ref=None, k_nn=200, n_try=12, chunk=4096, verbose=False):
    """Is there a collision-free configuration at `pts` with the tool along
    each of `tool_dirs`?

    The 201,600-entry CVT table cannot answer this by lookup: the pose space
    here is 5-D (position + tool axis) and the table is far too sparse once it
    is restricted to a thin slab and a cone, so a lookup can certify
    reachability but never refute it. It is used the way it was built to be
    used instead -- as a warm-start bank for Newton projection, ranked by the
    precomputed 6-D Jacobian pseudo-inverse `jinv6`:

        dq = jinv6 . (delta_position, minimal_rotation_vector)

    is a first-order estimate of the joint motion needed to carry that sample
    onto the query pose, which orders the neighbours far better than raw
    feature distance does. Same recipe as unified_rl/iksel_campaign.stage_gen.

    Returns (q, ok) of shapes (P, D, 7) and (P, D).
    """
    dev, dt = env.device, env.kin.dtype
    P, D = pts.shape[0], len(tool_dirs)
    q_out_all = np.full((P, D, 7), np.nan, np.float32)
    ok = np.zeros((P, D), bool)
    hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
    cos_lim = math.cos(math.radians(cone_deg)) if n_ref is not None else -2.0
    n_t = (torch.as_tensor(n_ref, device=dev, dtype=dt)
           if n_ref is not None else None)
    Tpos, Tzax, Tq, Tji = T["pos"], T["zax"], T["q"], T["jinv6"]

    for k in range(D):
        z = np.tile(tool_dirs[k], (P, 1)).astype(np.float32)
        cand = np.empty((P, n_try, 7), np.float32)
        for lo in range(0, P, chunk):
            hi = min(lo + chunk, P)
            feat = np.concatenate([pts[lo:hi] * POS_SCALE, z[lo:hi]],
                                  1).astype(np.float32)
            _, ids = tree.query(feat, k=k_nn, workers=-1)
            C = hi - lo
            dp = pts[lo:hi, None, :] - Tpos[ids]
            rv = _minimal_rotvec(Tzax[ids].reshape(-1, 3),
                                 np.repeat(z[lo:hi], k_nn, 0)).reshape(C, k_nn, 3)
            d6 = np.concatenate([dp, rv], -1).astype(np.float32)
            dq = np.einsum('ckje,cke->ckj', Tji[ids], d6)
            order = (dq * dq).sum(-1).argsort(1)[:, :n_try]
            cand[lo:hi] = Tq[np.take_along_axis(ids, order, 1)]

        pend = np.arange(P)
        for t in range(n_try):
            if not len(pend):
                break
            q0 = torch.as_tensor(cand[pend, t], device=dev, dtype=dt)
            p_t = torch.as_tensor(pts[pend], device=dev, dtype=dt)
            R_t = _build_R_with_z(
                torch.as_tensor(z[pend], device=dev, dtype=dt), hint)
            q_o, conv, _ = _batched_ik_project(env.kin, q0, p_t, R_t,
                                               branch_action=None)
            c = conv.cpu().numpy()
            good = pend[c]
            if len(good):
                qg = q_o[conv]
                coll = env.collision.is_collided(env.kin.link_transforms(qg))
                p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(qg)
                fine = (~coll) & ((p_fk - torch.as_tensor(
                    pts[good], device=dev, dtype=dt)).norm(-1) < 5e-3)
                if n_t is not None:
                    fine &= ((R_fk[:, :, 2] * n_t).sum(-1) >= cos_lim)
                f = fine.cpu().numpy()
                q_out_all[good[f], k] = qg[fine].cpu().numpy()
                ok[good[f], k] = True
            pend = pend[~c]
        if verbose:
            print(f"[ik] dir {k+1}/{D}: {ok[:, k].sum()}/{P} feasible",
                  flush=True)
    return q_out_all, ok


def ray_lengths(mask, xs, ys, dirs, step, max_len):
    """For every in-mask grid point, the longest ray staying inside `mask`."""
    nx, ny = mask.shape
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    n_steps = int(max_len / step)
    out = np.zeros((nx, ny), np.float32)
    best_dir = np.full((nx, ny), np.nan, np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for th in dirs:
        ux, uy = math.cos(th), math.sin(th)
        alive = mask.copy()
        reach = np.zeros((nx, ny), np.float32)
        for s in range(1, n_steps + 1):
            px = gx + ux * s * step
            py = gy + uy * s * step
            ix = np.rint((px - xs[0]) / dx).astype(int)
            iy = np.rint((py - ys[0]) / dy).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            inside = np.zeros((nx, ny), bool)
            inside[ok] = mask[ix[ok], iy[ok]]
            alive &= inside
            reach[alive] = s * step
            if not alive.any():
                break
        upd = reach > out
        out[upd] = reach[upd]
        best_dir[upd] = th
    out[~mask] = np.nan
    best_dir[~mask] = np.nan
    return out, best_dir


@torch.no_grad()
def cone_ik_seeds(env, tree, T, pts, n_vec, n_seeds, rng, max_try=3):
    """Distinct start configurations at each point, tool inside the cone."""
    axis = torch.as_tensor(n_vec, dtype=torch.float32)
    dirs = _sample_in_cone(axis, CONE_DEG - 0.5, n_seeds, rng).numpy()
    dirs[0] = n_vec                                  # always include the axis
    q, ok = ik_feasible(env, tree, T, pts, dirs, rng,
                        cone_deg=CONE_DEG, n_ref=n_vec)
    return np.nan_to_num(q, nan=0.0).astype(np.float32), ok


@torch.no_grad()
def rollout_batch(env, agent, classical, q0, p0, d0, nt, tau_e, tau_x):
    env.line_dist = ScriptedLineDistribution(
        {"q0": q0, "line_dir": d0, "n_target": nt, "p0": p0,
         "kappa": torch.zeros(q0.shape[0], device=env.device,
                              dtype=env.kin.dtype)})
    env.reset()
    qm, qh = env.q_mid, env.q_half
    mx = lambda q: ((q - qm).abs() / qh).max(-1).values
    using = mx(env.q) < tau_e
    # The environment advances q and then tests the hard constraints.  Its
    # arc_progress therefore includes the fatal step.  The requested length is
    # the feasible prefix, so freeze the pre-step arc when an episode first
    # terminates.
    feasible_arc = torch.zeros_like(env.arc_progress)
    for _ in range(env.max_steps + 1):
        cq = mx(env.q)
        using = torch.where(using, cq < tau_e, cq < tau_x)
        obs = env.current_obs()
        rl = agent.actor_mean(obs).clamp(-1.0, 1.0)
        B, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        qd = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cl = ((B.transpose(-1, -2) @ qd.unsqueeze(-1)).squeeze(-1)
              / env.a_max).clamp(-1.0, 1.0)
        arc_before = env.arc_progress.clone()
        _, _, _, _, info = env.step(
            torch.where(using.unsqueeze(-1), rl, cl), auto_reset=False)
        newly_done = info['episode_done']
        feasible_arc = torch.where(newly_done, arc_before, feasible_arc)
        if bool(env.done_persistent.all().item()):
            break
    return torch.where(env.done_persistent, feasible_arc,
                       env.arc_progress).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--half", type=float, default=1.15)
    ap.add_argument("--mask-step", type=float, default=0.025)
    ap.add_argument("--grid-step", type=float, default=0.08)
    ap.add_argument("--n-dirs", type=int, default=16)
    ap.add_argument("--fixed-dir-deg", type=float, default=0.0,
                    help="use this single in-plane direction instead "
                         "of maximising over directions")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-dir-pos", type=int, default=16)
    ap.add_argument("--n-dir-cone", type=int, default=32)
    ap.add_argument("--eps-reach", type=float, default=0.035)
    ap.add_argument("--march-step", type=float, default=0.01)
    ap.add_argument("--max-len", type=float, default=2.4)
    ap.add_argument("--k-lateral", type=float, default=5.0)
    ap.add_argument("--tau-enter", type=float, default=0.98)
    ap.add_argument("--tau-exit", type=float, default=0.94)
    ap.add_argument("--chunk", type=int, default=1300)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--no-tool-collision", action="store_true",
                    help="use the bare link0-7 sphere model, which "
                         "does not see the hand or the pen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()
    if args.draft:
        args.grid_step, args.n_dirs, args.n_seeds = 0.10, 8, 5
        args.mask_step, args.n_dir_pos, args.n_dir_cone = 0.04, 10, 16
    dev = torch.device(args.device)
    t0 = time.time()

    T = np.load(REPO / TABLE)
    zax = T["zax"]

    # ---- masks by actual inverse kinematics, not by table density --------
    # The pose table is far too sparse once it is restricted to a thin z-slab
    # AND a 30 deg cone (720 entries), which makes a nearest-neighbour mask a
    # sampling artefact rather than a reachability statement. Both masks are
    # therefore built by running the same cone-constrained IK used for the
    # seeds, over the fine grid.
    mxs = np.arange(-args.half, args.half + 1e-9, args.mask_step)
    mys = mxs.copy()
    gx, gy = np.meshgrid(mxs, mys, indexing="ij")
    fine = np.stack([gx.ravel(), gy.ravel(),
                     np.full(gx.size, args.z0)], 1).astype(np.float32)

    with open(REPO / ENV_YAML) as f:
        y = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y["env"].items() if k in keys}
    env = NSRLBatchedEnv(EnvConfig(**{**kw, "n_envs": args.chunk,
                                      "k_lateral": args.k_lateral}), None, dev)
    # The stock sphere model covers link0..link7 only; the hand and the 0.2034 m
    # pen are invisible to it, and near the base axis the pen shaft passes 4.6 cm
    # inside link1/link2 without being flagged. Swapping the checker here makes
    # BOTH fields use the same model: the IK feasibility test behind level 1 and
    # 2, and the termination check inside every rollout behind level 3.
    if not args.no_tool_collision:
        env.collision = PenSphereCollision(env.kin.tcp_offset, device=dev)
        print(f"[slice] collision model includes the hand and pen "
              f"({len(env.collision.radii)} spheres, "
              f"{env.collision.n_tool_spheres} on the tool)")
    feat_tree = cKDTree(np.concatenate(
        [T["pos"] * (1.0 / 0.05), T["zax"]], 1).astype(np.float32))
    rng = np.random.default_rng(0)

    # Level 1: any tool orientation. Level 2: only orientations inside the cone.
    sphere = fib_sphere(args.n_dir_pos)
    _, ok_any = ik_feasible(env, feat_tree, T, fine, sphere, rng, cone_deg=180.0)
    from scipy.ndimage import binary_closing
    st = np.ones((3, 3), bool)
    mask_pos_raw = ok_any.any(1).reshape(gx.shape)
    mask_pos = binary_closing(mask_pos_raw, structure=st)
    print(f"[slice] level-1 mask: {mask_pos_raw.sum()} cells, "
          f"{mask_pos.sum() - mask_pos_raw.sum()} pinholes closed")
    print(f"[slice] level-1 mask done ({time.time()-t0:.0f}s)")

    n_vec = np.array([0.0, 0.0, -1.0], np.float32)
    cd = _sample_in_cone(torch.as_tensor(n_vec), CONE_DEG - 0.5,
                         args.n_dir_cone, np.random.default_rng(1)).numpy()
    cd[0] = n_vec
    _, okc = ik_feasible(env, feat_tree, T, fine, cd, rng, cone_deg=CONE_DEG,
                         n_ref=n_vec)
    in_cone = T["zax"] @ n_vec >= math.cos(math.radians(CONE_DEG))
    if in_cone.sum():
        tt = cKDTree(T["pos"][in_cone])
        dd, _ = tt.query(fine[:, :3], workers=-1)
        cert = (dd < args.mask_step).reshape(gx.shape)
    else:
        cert = np.zeros(gx.shape, bool)
    m_raw = okc.any(1).reshape(gx.shape) | cert
    print(f"[slice] table certificates added "
          f"{int((cert & ~okc.any(1).reshape(gx.shape)).sum())} cells")
    mask_cone = binary_closing(m_raw, structure=st) & mask_pos
    print(f"[slice] raw {m_raw.sum()}, after closing {mask_cone.sum()}")
    print(f"[slice] fixed surface normal n = -z (tool points down): "
          f"{mask_cone.sum()} cone-reachable cells ({time.time()-t0:.0f}s)")
    print(f"[slice] mask cells: position-reachable {mask_pos.sum()}, "
          f"cone-reachable {mask_cone.sum()}  ({time.time()-t0:.0f}s)")

    if args.fixed_dir_deg is None:
        dirs = np.linspace(0, 2 * np.pi, args.n_dirs, endpoint=False)
    else:
        dirs = np.array([math.radians(args.fixed_dir_deg)])
        args.n_dirs = 1
    print(f"[slice] {len(dirs)} path direction(s)"
          + ("" if args.fixed_dir_deg is None else
             f" fixed at {args.fixed_dir_deg:.0f} deg"))
    L_pos, _ = ray_lengths(mask_pos, mxs, mys, dirs, args.march_step, args.max_len)
    L_cone, D_cone = ray_lengths(mask_cone, mxs, mys, dirs, args.march_step,
                                 args.max_len)
    print(f"[slice] ray marching done ({time.time()-t0:.0f}s)")

    # ---- level 3: rollouts on a coarser grid ------------------------------
    xs = np.arange(-args.half, args.half + 1e-9, args.grid_step)
    ys = xs.copy()
    cx, cy = np.meshgrid(xs, ys, indexing="ij")
    cand = np.stack([cx.ravel(), cy.ravel()], 1)
    ixm = np.clip(np.rint((cand[:, 0] - mxs[0]) / args.mask_step).astype(int),
                  0, len(mxs) - 1)
    iym = np.clip(np.rint((cand[:, 1] - mys[0]) / args.mask_step).astype(int),
                  0, len(mys) - 1)
    keep = mask_cone[ixm, iym]
    pts2 = cand[keep]
    pts = np.concatenate([pts2, np.full((len(pts2), 1), args.z0)], 1).astype(np.float32)
    print(f"[slice] rollout grid: {len(pts)} cone-reachable points of {len(cand)}")

    classical = ClassicalNullspaceController(env.kin)
    with open(REPO / RL_CKPT / "config.yaml") as f:
        rc = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim, hidden_dim=rc["ppo"]["hidden_dim"],
                  init_log_std=rc["ppo"]["init_log_std"]).to(dev)
    agent.load_state_dict(torch.load(REPO / RL_CKPT / "agent.pt", map_location=dev))
    agent.eval()

    seeds, sok = cone_ik_seeds(env, feat_tree, T, pts, n_vec, args.n_seeds, rng)
    print(f"[slice] cone-IK: {sok.sum()} seeds over {len(pts)} points "
          f"(mean {sok.sum(1).mean():.1f}/point, "
          f"{100*(sok.sum(1) > 0).mean():.0f}% of points have >=1)  "
          f"({time.time()-t0:.0f}s)")

    P, S = len(pts), args.n_seeds
    D = args.n_dirs
    pi, di, si = np.meshgrid(np.arange(P), np.arange(D), np.arange(S),
                             indexing="ij")
    job = sok[pi.ravel(), si.ravel()]
    pi, di, si = pi.ravel()[job], di.ravel()[job], si.ravel()[job]
    dvec = np.stack([np.cos(dirs), np.sin(dirs), np.zeros(D)], 1).astype(np.float32)
    N = len(pi)
    print(f"[slice] {N} rollouts ({P} pts x {D} dirs x valid seeds)")
    arc = np.zeros(N, np.float32)
    dt_t = env.kin.dtype
    q_all = torch.as_tensor(seeds[pi, si], device=dev, dtype=dt_t)
    p_all = torch.as_tensor(pts[pi], device=dev, dtype=dt_t)
    d_all = torch.as_tensor(dvec[di], device=dev, dtype=dt_t)
    n_all = torch.as_tensor(np.tile(n_vec, (N, 1)), device=dev, dtype=dt_t)
    nch = math.ceil(N / args.chunk)
    for c in range(nch):
        lo, hi = c * args.chunk, min((c + 1) * args.chunk, N)
        pad = args.chunk - (hi - lo)
        sl = lambda x: (torch.cat([x[lo:hi], x[hi - 1:hi].expand(pad, *x.shape[1:])])
                        if pad else x[lo:hi])
        r = rollout_batch(env, agent, classical, sl(q_all), sl(p_all), sl(d_all),
                          sl(n_all), args.tau_enter, args.tau_exit)
        arc[lo:hi] = r[:hi - lo].float().cpu().numpy()
        if c % 10 == 0 or c == nch - 1:
            print(f"[slice]   chunk {c+1}/{nch}  ({time.time()-t0:.0f}s)", flush=True)

    best_arc = np.zeros((P, D), np.float32)
    np.maximum.at(best_arc, (pi, di), arc)
    L_max_pts = best_arc.max(1)
    best_dir_pts = dirs[best_arc.argmax(1)]
    # Best-of-the-pool is an oracle over start configurations. What a single
    # start configuration achieves is a different quantity, and the gap between
    # them is exactly what a seed-selection module is worth. Take the median
    # over the valid seeds as the typical single choice.
    per_seed = np.full((P, D, S), np.nan, np.float32)
    per_seed[pi, di, si] = arc
    with np.errstate(invalid="ignore"):
        med_arc = np.nanmedian(per_seed, axis=2)
    L_med_pts = np.nanmax(np.where(np.isfinite(med_arc), med_arc, -np.inf), axis=1)
    L_med_pts = np.where(np.isfinite(L_med_pts) & (L_med_pts > -1e30),
                         L_med_pts, np.nan)
    # A point where the cone-IK found no start configuration at all has an
    # UNKNOWN capacity, not a zero one: the enumerator failing is not evidence
    # that the arm cannot move. Leaving these as 0 would charge the enumerator's
    # misses to the continuity constraint, and they concentrate exactly where
    # the constraint looks most dramatic.
    no_seed = sok.sum(1) == 0
    L_max_pts = np.where(no_seed, np.nan, L_max_pts)
    L_med_pts = np.where(no_seed, np.nan, L_med_pts)
    print(f"[slice] {int(no_seed.sum())}/{P} rollout points had no cone-IK "
          f"solution; their capacity is masked out, not set to zero")

    L_max = np.full(cx.shape, np.nan, np.float32)
    L_med = np.full(cx.shape, np.nan, np.float32)
    D_max = np.full(cx.shape, np.nan, np.float32)
    idx = np.nonzero(keep)[0]
    L_max.ravel()[idx] = L_max_pts
    L_med.ravel()[idx] = L_med_pts
    D_max.ravel()[idx] = best_dir_pts

    # A pointwise feasibility mask built from sampled IK carries a small
    # per-cell false-negative rate, and a ray crossing ~60 cells amplifies it
    # into a badly truncated level-2 field. Every point a rollout actually
    # traversed is a proof that the point is cone-reachable (the episode would
    # have ended otherwise), so folding those witnesses in can only remove
    # false negatives -- it never certifies a point that is not reachable.
    # Cells no rollout visited stay as they were, so the level-2 field remains
    # an under-estimate there and the continuity cost stays conservative.
    wit = np.zeros_like(mask_cone)
    n_w = 0
    for i in range(len(pts)):
        ell = float(L_max_pts[i]) if np.isfinite(L_max_pts[i]) else 0.0
        if not np.isfinite(ell) or ell <= 0:
            continue
        th = float(best_dir_pts[i])
        ss = np.arange(0.0, ell + 1e-9, args.mask_step * 0.5)
        wx = pts[i, 0] + math.cos(th) * ss
        wy = pts[i, 1] + math.sin(th) * ss
        ix = np.clip(np.rint((wx - mxs[0]) / args.mask_step).astype(int),
                     0, len(mxs) - 1)
        iy = np.clip(np.rint((wy - mys[0]) / args.mask_step).astype(int),
                     0, len(mys) - 1)
        wit[ix, iy] = True
        n_w += len(ss)
    added = int((wit & ~mask_cone).sum())
    mask_cone = mask_cone | wit
    print(f"[slice] rollout witnesses: {added} cells certified that the "
          f"sampled IK had missed ({100*added/max(mask_cone.sum(),1):.1f}% of "
          f"the cone mask); recomputing level 2")
    L_cone, D_cone = ray_lengths(mask_cone, mxs, mys, dirs, args.march_step,
                                 args.max_len)

    # ---- coarse-grid versions of levels 1 and 2 for a like-for-like ratio --
    def sample_fine(F):
        ix = np.clip(np.rint((cx - mxs[0]) / args.mask_step).astype(int),
                     0, len(mxs) - 1)
        iy = np.clip(np.rint((cy - mys[0]) / args.mask_step).astype(int),
                     0, len(mys) - 1)
        return F[ix, iy]

    Lp_c, Lc_c = sample_fine(L_pos), sample_fine(L_cone)
    good = np.isfinite(L_max) & np.isfinite(Lc_c) & (Lc_c > 0.05)
    ratio = np.full(cx.shape, np.nan, np.float32)
    ratio[good] = L_max[good] / Lc_c[good]
    print(f"\n[slice] ==== headline numbers ({good.sum()} grid points) ====")
    print(f"  ell_reach_pos   median {np.nanmedian(Lp_c[good]):.3f} m")
    print(f"  ell_reach_cone  median {np.nanmedian(Lc_c[good]):.3f} m")
    print(f"  ell_max         median {np.nanmedian(L_max[good]):.3f} m")
    print(f"  ell_med (typical seed)   median "
          f"{np.nanmedian(L_med[good]):.3f} m   "
          f"typical/best median "
          f"{np.nanmedian(L_med[good]/np.maximum(L_max[good],1e-9)):.3f}")
    print(f"  ell_max / ell_reach_cone : median {np.nanmedian(ratio[good]):.3f}"
          f"  IQR [{np.nanpercentile(ratio[good],25):.3f}, "
          f"{np.nanpercentile(ratio[good],75):.3f}]")
    if args.fixed_dir_deg is None:
        bd = np.abs(((D_max - sample_fine(D_cone) + np.pi) % (2*np.pi)) - np.pi)
        print(f"  best direction agrees with the geometric best within 45 deg "
              f"on {100*np.nanmean(bd < np.pi/4):.0f}% of points")
    v21 = np.nanmean(Lc_c[good] > Lp_c[good] + 1e-6) * 100
    v32 = np.nanmean(L_max[good] > Lc_c[good] + 1e-6) * 100
    v31 = np.nanmean(L_max[good] > Lp_c[good] + 1e-6) * 100
    print(f"  bound violations: level2>level1 {v21:.1f}%, "
          f"level3>level2 {v32:.1f}%, level3>level1 {v31:.1f}%")
    print(f"  level 2 / level 1 : median "
          f"{np.nanmedian(Lc_c[good]/np.maximum(Lp_c[good],1e-9)):.3f}"
          f"   level 3 / level 2 : median "
          f"{np.nanmedian(L_max[good]/np.maximum(Lc_c[good],1e-9)):.3f}")

    if args.npz:
        n_seed_grid = np.full(cx.shape, -1, np.int32)
        n_seed_grid.ravel()[idx] = sok.sum(1)
        np.savez_compressed(REPO / args.npz, L_pos=L_pos, L_cone=L_cone,
                            L_max=L_max, L_med=L_med, ratio=ratio,
                            mxs=mxs, mys=mys,
                            xs=xs, ys=ys, n_vec=n_vec, z0=args.z0,
                            n_seeds_per_point=n_seed_grid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.65), constrained_layout=True)
    ec = [xs[0] - args.grid_step/2, xs[-1] + args.grid_step/2,
          ys[0] - args.grid_step/2, ys[-1] + args.grid_step/2]
    msk = lambda F: np.where(good, F, np.nan)
    F2, F3 = msk(Lc_c), msk(L_max)
    vmax = float(np.nanmax([np.nanmax(F2), np.nanmax(F3)]))
    titles = [
        "(a) Pointwise reachability",
        "(b) Continuous reachability"]
    im = None
    for a, F, ti in zip(ax, (F2, F3), titles):
        im = a.imshow(F.T, origin="lower", extent=ec, cmap=args.cmap,
                      vmin=0, vmax=vmax, interpolation="nearest")
        a.set_title(ti, fontsize=11)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")
        a.plot(0, 0, "w^", ms=10, mec="k"); a.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="reachable length [m]", shrink=0.88, pad=0.03)
    out = (REPO / args.out if not Path(args.out).is_absolute()
           else Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[slice] saved -> {out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
