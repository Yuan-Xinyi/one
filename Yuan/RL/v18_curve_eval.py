"""v18 zero-shot curve-path generalization eval.

Tests whether the v18 backward CFM (trained ONLY on linear paths) generalizes
to curved paths by feeding per-segment local tangents as the `direction`
conditioning. No retraining, no data regeneration.

Path types:
  - line     : same family as training (sanity baseline)
  - arc      : circular arc on a plane
  - s_curve  : sinusoidal one-period S on a plane

Per-task pipeline (same as v18_eval_random):
  1. Sample task (plane_normal, p0, in-plane axis, length, curve params),
     reject if path leaves reachable workspace or goal IK fails.
  2. Discretize path into N+1 checkpoints.
  3. Enumerate K IK branches at goal x_T.
  4. For each q_T: run backward_sample with direction_per_step =
     normalize(path_pts[i+1] - path_pts[i]). Capture FK residual at every
     checkpoint after manifold_snap.

Reported metrics per curve type:
  - Per-checkpoint FK error (mean / p90 / max), in mm.
  - Per-task success rate (max FK err < 1cm at every checkpoint, for at
    least one of K samples).
  - Branch coverage (unique q_0 signatures across K).

The line baseline lets you read "ratio vs line" — values close to 1.0
mean the curve is no harder than the training distribution → zero-shot
generalization holds.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


# ---------- in-plane geometry ----------

def _orthonormal_in_plane(plane_normal: np.ndarray,
                          hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (e_axis, e_perp) orthonormal in the plane perpendicular to
    plane_normal. e_axis is `hint` projected to the plane; e_perp = n × e_axis."""
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    e = hint - n * (hint @ n)
    en = np.linalg.norm(e)
    if en < 1e-6:
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e = ref - n * (ref @ n)
        en = np.linalg.norm(e)
    e = (e / en).astype(np.float32)
    f = np.cross(n, e).astype(np.float32)
    return e, f


def make_line_path(p0, e_axis, L, n_pts):
    s = np.linspace(0.0, L, n_pts)
    return (p0[None, :] + s[:, None] * e_axis[None, :]).astype(np.float32)


def make_arc_path(p0, e_axis, e_perp, L, R, n_pts):
    """Arc of length L on the plane spanned by (e_axis, e_perp), starting
    at p0 with initial tangent e_axis, curving toward +e_perp."""
    theta_max = L / R
    theta = np.linspace(0.0, theta_max, n_pts)
    C = p0 + R * e_perp                                 # arc center
    pts = (C[None, :]
           + R * (-np.cos(theta)[:, None] * e_perp[None, :]
                  + np.sin(theta)[:, None] * e_axis[None, :]))
    return pts.astype(np.float32)


def make_s_curve_path(p0, e_axis, e_perp, L, A, n_pts):
    """One-period sinusoid: p(s) = p0 + s*e_axis + A*sin(2π s/L)*e_perp."""
    s = np.linspace(0.0, L, n_pts)
    pts = (p0[None, :]
           + s[:, None] * e_axis[None, :]
           + A * np.sin(2 * np.pi * s / L)[:, None] * e_perp[None, :])
    return pts.astype(np.float32)


def make_circle_path(p0, e_axis, e_perp, R, n_pts):
    """Closed circle on plane (e_axis, e_perp). p0 = first point, initial
    tangent = e_axis, curving toward +e_perp side. Returns n_pts points
    where pts[-1] ≈ pts[0]."""
    theta = np.linspace(0.0, 2 * np.pi, n_pts)
    C = p0 + R * e_perp
    pts = (C[None, :]
           + R * (-np.cos(theta)[:, None] * e_perp[None, :]
                  + np.sin(theta)[:, None] * e_axis[None, :]))
    return pts.astype(np.float32)


def make_ellipse_path(p0, e_axis, e_perp, a, b, n_pts):
    """Closed ellipse with semi-axes (a along e_axis, b along e_perp).
    p0 = first point. Returns n_pts points, pts[-1] ≈ pts[0]."""
    theta = np.linspace(0.0, 2 * np.pi, n_pts)
    C = p0 + b * e_perp
    pts = (C[None, :]
           + a * np.sin(theta)[:, None] * e_axis[None, :]
           - b * np.cos(theta)[:, None] * e_perp[None, :])
    return pts.astype(np.float32)


def make_figure8_path(p0, e_axis, e_perp, a, b, n_pts):
    """Lissajous figure-8: x = a·sin(t), y = (b/2)·sin(2t).
    Crosses itself once at p0 (also pts[-1] ≈ pts[0])."""
    t = np.linspace(0.0, 2 * np.pi, n_pts)
    pts = (p0[None, :]
           + a * np.sin(t)[:, None] * e_axis[None, :]
           + (b / 2.0) * np.sin(2 * t)[:, None] * e_perp[None, :])
    return pts.astype(np.float32)


def per_step_tangents(path_pts: np.ndarray) -> np.ndarray:
    """Forward differences: t[i] = normalize(p[i+1] - p[i]) for i in [0, N-1]."""
    diffs = path_pts[1:] - path_pts[:-1]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-12)
    return (diffs / norms).astype(np.float32)


def make_sphere_path(C: np.ndarray, R: float,
                     p0: np.ndarray, tangent0: np.ndarray,
                     L: float, n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle arc on a sphere of radius R centered at C.
    p0 must satisfy ||p0 - C|| ≈ R; tangent0 will be projected onto
    tangent plane at p0. Returns (path_pts(n_pts,3), n_outward(n_pts,3))
    where n_outward is the outward surface normal at each point."""
    e_rad0 = (p0 - C) / np.linalg.norm(p0 - C)
    e_tan0 = tangent0 - e_rad0 * (tangent0 @ e_rad0)
    e_tan0 = e_tan0 / max(np.linalg.norm(e_tan0), 1e-12)
    theta = np.linspace(0.0, L / R, n_pts)
    pts = (C[None, :]
           + R * (np.cos(theta)[:, None] * e_rad0[None, :]
                  + np.sin(theta)[:, None] * e_tan0[None, :]))
    n_outward = (pts - C[None, :]) / R
    return pts.astype(np.float32), n_outward.astype(np.float32)


def branch_signature(q):
    return (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))


def sample_surface_task(rng, kin, surface_type: str, n_checkpoints: int,
                        L_range=(0.20, 0.40),
                        sphere_R_range=(0.6, 1.2),
                        max_tilt_deg: float = 30.0):
    """Sample a path on a curved surface. Returns the same dict shape as
    sample_curve_task, plus 'n_per_step' (per-segment outward surface
    normal). For surface_type='sphere' the path is a great-circle arc.
    'sphere' is supported now; extend with cylinder / saddle later."""
    if surface_type != "sphere":
        raise ValueError(f"unsupported surface_type: {surface_type}")
    cos_max = float(np.cos(np.deg2rad(max_tilt_deg)))
    for _ in range(120):
        R_sph = float(rng.uniform(*sphere_R_range))
        # contact point in reachable workspace
        u = rng.normal(size=3)
        u = u / (np.linalg.norm(u) + 1e-12)
        if u[2] < cos_max:
            continue
        n_outward = u.astype(np.float32)
        ok = False
        for _ in range(15):
            p0 = rng.uniform(np.array([-0.30, -0.40, 0.10]),
                             np.array([0.55, 0.40, 0.55])).astype(np.float32)
            if 0.32 < float(np.linalg.norm(p0)) < 0.72:
                ok = True
                break
        if not ok:
            continue
        # sphere center placed behind the contact point along -n_outward
        C = p0 - R_sph * n_outward
        # initial tangent: random in tangent plane
        v = rng.normal(size=3)
        v = v - n_outward * (v @ n_outward)
        nv = float(np.linalg.norm(v))
        if nv < 0.1:
            continue
        tangent0 = (v / nv).astype(np.float32)
        L = float(rng.uniform(*L_range))
        path_pts, n_at_pts = make_sphere_path(C, R_sph, p0, tangent0, L,
                                              n_checkpoints + 1)
        fine_pts, _ = make_sphere_path(C, R_sph, p0, tangent0, L, 120)
        # reachability sanity
        norms = np.linalg.norm(path_pts, axis=1)
        if (norms > 0.85).any() or (norms < 0.20).any():
            continue
        if (path_pts[:, 2] < 0.02).any():
            continue
        # per-segment normal (segment midpoint, re-normalized) and tangent
        n_mid = 0.5 * (n_at_pts[:-1] + n_at_pts[1:])
        n_per_step = (n_mid / np.linalg.norm(n_mid, axis=1, keepdims=True)).astype(np.float32)
        d_per_step = per_step_tangents(path_pts)
        # goal R built from final segment's local frame
        R_goal = _build_R_from_normal_direction(n_at_pts[-1], d_per_step[-1])

        # IK feasibility at goal
        device = kin.device
        x_T = torch.as_tensor(path_pts[-1], device=device, dtype=torch.float32)
        R_T = torch.as_tensor(R_goal,       device=device, dtype=torch.float32)
        seeds_np = rng.uniform(kin.lmt_lo.cpu().numpy()[None, :],
                               kin.lmt_up.cpu().numpy()[None, :],
                               size=(16, 7)).astype(np.float32)
        q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
        p_rep = x_T.unsqueeze(0).expand(16, 3)
        R_rep = R_T.unsqueeze(0).expand(16, 3, 3)
        _, ok_ik, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=None)
        if not bool(ok_ik.any().item()):
            continue

        return dict(
            path_pts=path_pts,
            fine_path_pts=fine_pts,
            plane_normal=n_at_pts[0].astype(np.float32),    # back-compat (start normal)
            direction_axis=d_per_step[0].astype(np.float32),# back-compat (start tangent)
            d_per_step=d_per_step,
            n_per_step=n_per_step,
            R_target_at_goal=R_goal,
            L=L,
            curve_type=surface_type,                        # reuse field
            sphere_C=C,
            sphere_R=R_sph,
        )
    return None


# ---------- task sampler ----------

def sample_curve_task(rng, kin, curve_type: str, n_checkpoints: int,
                      L_range=(0.30, 0.55),
                      arc_R_range=(0.40, 1.0),
                      s_amp_frac_range=(0.05, 0.15),
                      max_tilt_deg: float = 30.0):
    """Returns dict (path_pts, plane_normal, direction_axis, d_per_step,
    R_target_at_goal, L) or None if rejection-sampled out."""
    cos_max = float(np.cos(np.deg2rad(max_tilt_deg)))
    for _ in range(80):
        u = rng.normal(size=3)
        u = u / (np.linalg.norm(u) + 1e-12)
        if u[2] < cos_max:
            continue
        plane_normal = u.astype(np.float32)

        p0_ok = False
        for _ in range(15):
            p0 = rng.uniform(np.array([-0.30, -0.40, 0.10]),
                             np.array([0.55, 0.40, 0.55])).astype(np.float32)
            if 0.30 < float(np.linalg.norm(p0)) < 0.75:
                p0_ok = True
                break
        if not p0_ok:
            continue

        hint = rng.normal(size=3)
        e_axis, e_perp = _orthonormal_in_plane(plane_normal, hint)
        L = float(rng.uniform(*L_range))

        if curve_type == "line":
            path_pts = make_line_path(p0, e_axis, L, n_checkpoints + 1)
            fine_pts = make_line_path(p0, e_axis, L, 120)
        elif curve_type == "arc":
            R = float(rng.uniform(*arc_R_range))
            if rng.random() < 0.5:
                e_perp = -e_perp
            path_pts = make_arc_path(p0, e_axis, e_perp, L, R, n_checkpoints + 1)
            fine_pts = make_arc_path(p0, e_axis, e_perp, L, R, 120)
        elif curve_type == "s_curve":
            A = L * float(rng.uniform(*s_amp_frac_range))
            path_pts = make_s_curve_path(p0, e_axis, e_perp, L, A, n_checkpoints + 1)
            fine_pts = make_s_curve_path(p0, e_axis, e_perp, L, A, 120)
        elif curve_type == "circle":
            R = float(rng.uniform(0.08, 0.13))
            if rng.random() < 0.5:
                e_perp = -e_perp
            L = float(2 * np.pi * R)                       # perimeter
            path_pts = make_circle_path(p0, e_axis, e_perp, R, n_checkpoints + 1)
            fine_pts = make_circle_path(p0, e_axis, e_perp, R, 120)
        elif curve_type == "ellipse":
            a = float(rng.uniform(0.10, 0.14))
            b = a * float(rng.uniform(0.55, 0.85))
            if rng.random() < 0.5:
                e_perp = -e_perp
            L = float(np.pi * (a + b))                     # Ramanujan-1 approx
            path_pts = make_ellipse_path(p0, e_axis, e_perp, a, b, n_checkpoints + 1)
            fine_pts = make_ellipse_path(p0, e_axis, e_perp, a, b, 120)
        elif curve_type == "figure8":
            a = float(rng.uniform(0.10, 0.13))
            b = float(rng.uniform(0.06, 0.09))
            if rng.random() < 0.5:
                e_perp = -e_perp
            L = float(2 * np.pi * np.sqrt(a * a + b * b / 4.0))   # rough
            path_pts = make_figure8_path(p0, e_axis, e_perp, a, b, n_checkpoints + 1)
            fine_pts = make_figure8_path(p0, e_axis, e_perp, a, b, 120)
        else:
            raise ValueError(curve_type)

        norms = np.linalg.norm(path_pts, axis=1)
        if (norms > 0.85).any() or (norms < 0.20).any():
            continue
        if (path_pts[:, 2] < 0.02).any():
            continue

        d_per_step = per_step_tangents(path_pts)            # (n_checkpoints, 3)
        d_T = d_per_step[-1]
        R_goal = _build_R_from_normal_direction(plane_normal, d_T)

        # goal IK feasibility check
        device = kin.device
        x_T = torch.as_tensor(path_pts[-1], device=device, dtype=torch.float32)
        R_T = torch.as_tensor(R_goal, device=device, dtype=torch.float32)
        seeds_np = rng.uniform(kin.lmt_lo.cpu().numpy()[None, :],
                               kin.lmt_up.cpu().numpy()[None, :],
                               size=(16, 7)).astype(np.float32)
        q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
        p_rep = x_T.unsqueeze(0).expand(16, 3)
        R_rep = R_T.unsqueeze(0).expand(16, 3, 3)
        _, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=None)
        if not bool(ok.any().item()):
            continue

        return dict(
            path_pts=path_pts,
            fine_path_pts=fine_pts,                         # (120, 3) hi-res curve for viz
            plane_normal=plane_normal,
            direction_axis=e_axis,                          # global axis (line-style cond)
            d_per_step=d_per_step,                          # (n_checkpoints, 3) per-segment tangents
            R_target_at_goal=R_goal,
            L=L,
            curve_type=curve_type,
        )
    return None


# ---------- single-task eval ----------

def evaluate_task(model, kin, task, K_branches, n_ode_steps, snap_iters,
                  rng, use_per_step_dir: bool = True):
    device = kin.device
    path_pts = torch.as_tensor(task['path_pts'], device=device, dtype=torch.float32)
    plane_normal = torch.as_tensor(task['plane_normal'], device=device, dtype=torch.float32)
    direction_axis = torch.as_tensor(task['direction_axis'], device=device, dtype=torch.float32)
    d_per_step = torch.as_tensor(task['d_per_step'], device=device, dtype=torch.float32)
    R_T = torch.as_tensor(task['R_target_at_goal'], device=device, dtype=torch.float32)

    x_T = path_pts[-1]
    q_Ts, _ = _dense_ik_at(kin, x_T, R_T, K_branches * 4, rng)
    if q_Ts.shape[0] == 0:
        return None
    if q_Ts.shape[0] > K_branches:
        idx = rng.permutation(q_Ts.shape[0])[:K_branches]
        q_Ts = q_Ts[idx]

    fk_errs = []                        # list of (n_ckpt+1,) arrays
    sigs = []
    for k in range(q_Ts.shape[0]):
        q_traj = backward_sample(
            model, kin, q_Ts[k], path_pts, plane_normal, direction_axis,
            n_ode_steps=n_ode_steps, snap_iters=snap_iters,
            direction_per_step=(d_per_step if use_per_step_dir else None))
        p_pred, _ = kin.fk_batch(q_traj)
        err = (p_pred - path_pts).norm(dim=-1).cpu().numpy()
        fk_errs.append(err)
        sigs.append(branch_signature(q_traj[0].cpu().numpy()))

    return dict(
        n_samples=q_Ts.shape[0],
        fk_err=np.stack(fk_errs, axis=0),                   # (K, n_ckpt+1)
        unique_branches=len(set(sigs)),
    )


# ---------- main ----------

def _print_curve_block(name: str, all_errs: np.ndarray, per_task_succ: np.ndarray,
                       all_uniq: np.ndarray, wall: float, n_tasks: int):
    per_seg_mean = 1000.0 * all_errs.mean(axis=0)
    per_seg_p90 = 1000.0 * np.percentile(all_errs, 90, axis=0)
    max_mm = 1000.0 * all_errs.max()
    print(f"  n_tasks={n_tasks}  K-samples per task aggregated  wall={wall:.1f}s")
    print(f"  FK err mm  (mean/ckpt): " + "  ".join(f"{v:5.2f}" for v in per_seg_mean))
    print(f"  FK err mm  (p90 /ckpt): " + "  ".join(f"{v:5.2f}" for v in per_seg_p90))
    print(f"  FK err mm  (max ovrll): {max_mm:.2f}")
    print(f"  per-task succ rate (max-err<1cm any-of-K): "
          f"{per_task_succ.mean()*100:.1f}%   ({int(per_task_succ.sum())}/{len(per_task_succ)})")
    print(f"  unique q_0 branches per task: mean={all_uniq.mean():.2f}  max={all_uniq.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_multi/best.pt")
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-checkpoints", type=int, default=5)
    ap.add_argument("--K-branches", type=int, default=8)
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--snap-iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--curve-types", nargs="+",
                    default=["line", "arc", "s_curve"],
                    help="subset of {line, arc, s_curve}")
    ap.add_argument("--succ-tol", type=float, default=0.01,
                    help="max FK err threshold for per-task success (m)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = CFMFlowModel(q_dim=7, cond_dim=COND_DIM,
                         hidden=margs.get("hidden", 512),
                         depth=margs.get("depth", 6)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt}, epoch {state.get('epoch', '?')}")
    print(f"n-tasks={args.n_tasks}  n-ckpt={args.n_checkpoints}  "
          f"K={args.K_branches}  ode={args.n_ode_steps}  snap={args.snap_iters}")

    rng = np.random.default_rng(args.seed)
    summary = {}

    for ctype in args.curve_types:
        print(f"\n===== {ctype} =====")
        all_errs = []
        all_uniq = []
        per_task_succ = []
        n_completed = 0
        n_attempts = 0
        t0 = time.perf_counter()
        max_attempts = args.n_tasks * 12
        while n_completed < args.n_tasks and n_attempts < max_attempts:
            n_attempts += 1
            task = sample_curve_task(rng, kin, ctype, args.n_checkpoints)
            if task is None:
                continue
            out = evaluate_task(
                model, kin, task,
                K_branches=args.K_branches,
                n_ode_steps=args.n_ode_steps,
                snap_iters=args.snap_iters,
                rng=rng)
            if out is None:
                continue
            errs = out['fk_err']                            # (K, n_ckpt+1)
            all_errs.append(errs)
            all_uniq.append(out['unique_branches'])
            # any-of-K success: at least one sample has max err < tol
            any_ok = bool((errs.max(axis=1) < args.succ_tol).any())
            per_task_succ.append(any_ok)
            n_completed += 1

        wall = time.perf_counter() - t0
        if n_completed == 0:
            print(f"  no tasks completed in {n_attempts} attempts")
            continue
        errs_cat = np.concatenate(all_errs, axis=0)
        per_task_succ = np.array(per_task_succ, dtype=bool)
        all_uniq = np.array(all_uniq)
        _print_curve_block(ctype, errs_cat, per_task_succ, all_uniq, wall, n_completed)
        summary[ctype] = {
            'mean_per_ckpt': errs_cat.mean(axis=0),
            'p90_per_ckpt': np.percentile(errs_cat, 90, axis=0),
            'task_succ': float(per_task_succ.mean()),
            'mean_unique': float(all_uniq.mean()),
        }

    if "line" in summary and len(summary) > 1:
        print(f"\n===== ratio (curve / line, per-checkpoint mean FK err) =====")
        line_mean = summary["line"]['mean_per_ckpt']
        for ctype, d in summary.items():
            if ctype == "line":
                continue
            ratio = d['mean_per_ckpt'] / np.maximum(line_mean, 1e-6)
            print(f"  {ctype:8s}: " + "  ".join(f"{r:.2f}" for r in ratio)
                  + f"   succ ratio: {d['task_succ']/max(summary['line']['task_succ'], 1e-6):.2f}")


if __name__ == "__main__":
    main()
