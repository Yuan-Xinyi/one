"""v18 curve-path visualization in the one-world interactive viewer.

Samples ONE curve task (line / arc / s_curve), runs K backward CFM samples
with per-segment local tangents, then spawns K FR3+pen arms in the world
viewer. Each arm plays its own dense q-trajectory in lockstep, ping-pong
through the path. Pen tip leaves no automatic trail (use trace markers
in the scene below).

Run:
    python -m Yuan.flow_connectivity.v18_curve_world --curve-type arc --K-arms 4
"""
from __future__ import annotations
import argparse
import builtins
import numpy as np
import torch

import Yuan.flow_connectivity.config as cfg
from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import batched_rollout_segment
from Yuan.flow_connectivity.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.flow_connectivity.v18_inference import backward_sample
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at, _build_R_from_normal_direction
from Yuan.flow_connectivity.v18_curve_eval import (
    sample_curve_task, sample_surface_task, branch_signature,
)


BRANCH_COLORS = {
    (+1, -1, +1): (0.20, 0.50, 0.95),                      # blue
    (+1, -1, -1): (1.00, 0.55, 0.10),                      # orange
    (-1, -1, +1): (0.20, 0.75, 0.30),                      # green
    (-1, -1, -1): (0.95, 0.20, 0.30),                      # red
    (+1, +1, +1): (0.65, 0.30, 0.85),                      # purple
    (+1, +1, -1): (0.55, 0.40, 0.30),                      # brown
    (-1, +1, +1): (0.95, 0.50, 0.75),                      # pink
    (-1, +1, -1): (0.75, 0.75, 0.20),                      # olive
}


def branch_color(sig):
    return BRANCH_COLORS.get(tuple(int(s) for s in sig), (0.5, 0.5, 0.5))


def dls_forward_rollout(kin: BatchedFR3Kinematics,
                        q_traj_cfm: torch.Tensor,
                        path_pts: torch.Tensor,
                        plane_normal: torch.Tensor,
                        v_path: float = 0.10,
                        eps_p: float = 0.05,
                        verbose: bool = False,
                        plane_normal_per_step: torch.Tensor | None = None,
                        fine_path_pts: torch.Tensor | None = None,
                        fine_path_normals: torch.Tensor | None = None,
                        track_fine: bool = True,
                        ) -> tuple[torch.Tensor, bool]:
    """Run DLS Cartesian-tracking controller from q_traj_cfm[0].

    q_traj_cfm accepts (T, 7) [single arm] or (T, B, 7) [B arms in parallel].
    Returns (q_dense, all_ok) with shapes (M, 7) / bool, or (M, B, 7) /
    (B,) bool. All arms share the same path / cond geometry; only q's are
    per-arm. Internal calls to batched_rollout_segment use B in batch dim.

    Two tracking modes:

      track_fine=False (legacy): controller chord-tracks coarse path_pts
        segments. Within each chord, p_ref moves linearly. Works fine for
        line; on curves the chord-vs-curve gap (~mm-cm) is the dominant
        TCP error source.

      track_fine=True + fine_path_pts given (default): controller tracks
        the FINE curve discretization point-by-point as micro-chords
        (typically ~3 mm each). Chord-vs-curve gap → sub-mm. q_ref is
        linearly interpolated between consecutive coarse CFM predictions
        (q_traj_cfm) by arc-length fraction.
    """
    device = kin.device
    dt = float(cfg.DT)
    # auto-batch: accept (T, 7) and treat as B=1
    single = (q_traj_cfm.ndim == 2)
    if single:
        q_traj_cfm = q_traj_cfm.unsqueeze(1)                # (T, 1, 7)
    B = q_traj_cfm.shape[1]
    plane_n_np = plane_normal.detach().cpu().numpy()
    n_per_step_np = (plane_normal_per_step.detach().cpu().numpy()
                      if plane_normal_per_step is not None else None)
    branch_action = torch.tensor([1.0, 0.0, 1.0, 0.0],
                                  device=device, dtype=torch.float32
                                  ).unsqueeze(0).expand(B, 4)

    # ----- choose tracking discretization -----
    use_fine = track_fine and fine_path_pts is not None
    if use_fine:
        track_pts = fine_path_pts                           # (n_fine, 3)
        track_n_np = (fine_path_normals.detach().cpu().numpy()
                      if fine_path_normals is not None else None)
        n_track = track_pts.shape[0]
        n_coarse = q_traj_cfm.shape[0]
        fracs = torch.linspace(0.0, 1.0, n_track, device=device) * (n_coarse - 1)
        i_co = fracs.long().clamp(0, n_coarse - 2)
        local_co = (fracs - i_co.float()).view(-1, 1, 1)    # (n_track, 1, 1)
        # q_anchor: (n_track, B, 7)
        q_anchor = ((1.0 - local_co) * q_traj_cfm[i_co]
                    + local_co * q_traj_cfm[i_co + 1])
    else:
        track_pts = path_pts
        track_n_np = None
        q_anchor = q_traj_cfm                                # (n_coarse, B, 7)

    q_curr = q_anchor[0].clone()                            # (B, 7)
    chunks = [q_curr.unsqueeze(0).clone()]                  # (1, B, 7) initial
    alive = torch.ones((B,), device=device, dtype=torch.bool)
    all_ok_per_arm = torch.ones((B,), device=device, dtype=torch.bool)

    for i in range(track_pts.shape[0] - 1):
        seg_vec = track_pts[i + 1] - track_pts[i]
        L_seg = float(seg_vec.norm().item())
        if L_seg < 1e-8:
            continue
        d_dir_1 = (seg_vec / L_seg)                         # (3,)
        if track_n_np is not None:
            n_local_np = track_n_np[i]
        elif n_per_step_np is not None:
            ci = min(int(i * (len(n_per_step_np)) / max(track_pts.shape[0] - 1, 1)),
                     len(n_per_step_np) - 1)
            n_local_np = n_per_step_np[ci]
        else:
            n_local_np = plane_n_np
        R_np = _build_R_from_normal_direction(n_local_np,
                                              d_dir_1.cpu().numpy())
        R_seg_1 = torch.as_tensor(R_np, device=device, dtype=torch.float32)
        T_seg = max(1, int(round(L_seg / (v_path * dt))))
        # broadcast per-segment shared tensors to (B, ·)
        p_a = track_pts[i].unsqueeze(0).expand(B, 3)
        d_dir = d_dir_1.unsqueeze(0).expand(B, 3)
        R_seg = R_seg_1.unsqueeze(0).expand(B, 3, 3)
        # per-arm moving q_ref: (T+1, B, 7) interp anchor[i] → anchor[i+1]
        alphas = torch.linspace(0.0, 1.0, T_seg + 1,
                                 device=device).view(-1, 1, 1)
        q_ref_traj = ((1.0 - alphas) * q_anchor[i].unsqueeze(0)
                      + alphas * q_anchor[i + 1].unsqueeze(0))   # (T+1, B, 7)
        out = batched_rollout_segment(
            q_init=q_curr,
            R_tgt=R_seg,
            branch_action=branch_action,
            p0=p_a,
            d_dir=d_dir,
            v_path=torch.full((B,), v_path, device=device, dtype=torch.float32),
            eps_p=torch.full((B,), eps_p, device=device, dtype=torch.float32),
            T_total=torch.full((B,), T_seg, device=device, dtype=torch.long),
            start_step=0,
            end_step=T_seg,
            kin=kin,
            q_ref=q_ref_traj,
            alive_mask=alive,
            record_traj=True,
        )
        q_curr = out['q_final']                             # (B, 7)
        chunks.append(out['q_record'][1:])                  # (T_seg, B, 7)
        seg_alive = out['alive_out']                        # (B,) bool
        all_ok_per_arm = all_ok_per_arm & seg_alive
        alive = seg_alive

        if verbose and (not seg_alive.all()):
            n_died = int((~seg_alive).sum().item())
            mode = "fine" if use_fine else "coarse"
            print(f"      [{mode}] seg {i}->{i+1}: {n_died}/{B} arms died this segment")

    q_traj_full = torch.cat(chunks, dim=0)                  # (M, B, 7)
    if single:
        return q_traj_full.squeeze(1), bool(all_ok_per_arm.item())
    return q_traj_full, all_ok_per_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/flow_connectivity/checkpoints_v18_multi/best.pt")
    ap.add_argument("--surface-type", default="flat",
                    choices=["flat", "sphere"],
                    help="flat: planar surface (use --curve-type for path "
                         "shape); sphere: great-circle arc on a sphere "
                         "(--curve-type is ignored)")
    ap.add_argument("--curve-type", default="line",
                    choices=["line", "arc", "s_curve",
                             "circle", "ellipse", "figure8"],
                    help="open: line / arc / s_curve   closed: circle / "
                         "ellipse / figure8 (path returns to start)")
    ap.add_argument("--sphere-R-min", type=float, default=0.6,
                    help="sphere radius lower bound (m); smaller = more curved")
    ap.add_argument("--sphere-R-max", type=float, default=1.2)
    ap.add_argument("--n-checkpoints", type=int, default=5)
    ap.add_argument("--ctrl-stride", type=int, default=4,
                    help="subsample DLS rollout this many control steps per "
                         "animation frame (1 = every step ≈ 50 fps target)")
    ap.add_argument("--v-path", type=float, default=0.10,
                    help="DLS controller path velocity [m/s]")
    ap.add_argument("--K-arms", type=int, default=4,
                    help="how many arms to visualize after diversity-pick")
    ap.add_argument("--K-pool", type=int, default=24,
                    help="how many CFM samples to attempt; from the successes "
                         "we pick the K-arms most q_0-diverse via greedy max-min")
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--snap-iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for fresh randomness each run; set int to "
                         "reproduce CFM samples + task draw")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="parallel-mode arm transparency")
    ap.add_argument("--mode", default="serial",
                    choices=["serial", "parallel"],
                    help="serial: one arm cycling through K samples; "
                         "parallel: K colored arms moving in lockstep")
    ap.add_argument("--cycle", default="forward",
                    choices=["forward", "pingpong"],
                    help="forward: jump back to start when reaching end "
                         "(short inter-sample gap); pingpong: reverse the "
                         "sweep before switching")
    ap.add_argument("--debug-orient", action="store_true",
                    help="print z-axis angle (deg) before/after manifold_snap "
                         "per checkpoint; flags samples where snap pushes "
                         "orient past THETA_MAX=5°")
    ap.add_argument("--show-failed", action="store_true",
                    help="include arms whose DLS rollout failed mid-trajectory "
                         "in the viewer (default: only show fully-completed)")
    ap.add_argument("--no-track-fine", action="store_true",
                    help="legacy: chord-track coarse path_pts (cm-scale TCP "
                         "deviation on curves). Default ON: track fine "
                         "discretization for sub-mm TCP precision")
    ap.add_argument("--strict-track", action="store_true",
                    help="enforce <5mm TCP deviation: snap_iters→30 (squeeze "
                         "CFM-snap residual) + eps_p→5mm (strict failure "
                         "threshold). Surviving arms are guaranteed in-band; "
                         "may need larger --K-pool to keep enough OK arms")
    args = ap.parse_args()

    # strict-track: tighter snap + stricter pos tolerance
    if args.strict_track:
        snap_iters_eff = max(args.snap_iters, 30)
        eps_p_eff = 0.005                                  # 5mm
        print(f"  [strict-track] snap_iters={snap_iters_eff}  eps_p={eps_p_eff*1000:.0f}mm")
    else:
        snap_iters_eff = args.snap_iters
        eps_p_eff = 0.05                                   # 5cm (legacy)

    # determinism only if --seed given; otherwise let everything draw fresh
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        print(f"seed={args.seed} (deterministic)")
    else:
        print("seed=None (fresh randomness each run)")

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

    rng = np.random.default_rng(args.seed)
    task = None
    for _ in range(50):
        if args.surface_type == "flat":
            t = sample_curve_task(rng, kin, args.curve_type, args.n_checkpoints)
        else:
            t = sample_surface_task(
                rng, kin, args.surface_type, args.n_checkpoints,
                sphere_R_range=(args.sphere_R_min, args.sphere_R_max))
        if t is not None:
            task = t
            break
    if task is None:
        kind = args.surface_type if args.surface_type != "flat" else args.curve_type
        print(f"could not sample {kind} task — retry with another --seed")
        return

    desc = (f"surface=sphere R={task['sphere_R']:.2f}m" if args.surface_type == "sphere"
            else f"flat curve={args.curve_type}")
    print(f"\ntask: {desc}  L={task['L']:.2f}m  "
          f"start_normal_z={task['plane_normal'][2]:.2f}")

    path_pts = torch.as_tensor(task['path_pts'], device=device, dtype=torch.float32)
    fine_pts = task['fine_path_pts']                       # (120, 3) numpy
    plane_normal_t = torch.as_tensor(task['plane_normal'], device=device, dtype=torch.float32)
    direction_axis = torch.as_tensor(task['direction_axis'], device=device, dtype=torch.float32)
    d_per_step = torch.as_tensor(task['d_per_step'], device=device, dtype=torch.float32)
    n_per_step_t = (torch.as_tensor(task['n_per_step'], device=device, dtype=torch.float32)
                     if 'n_per_step' in task else None)
    fine_pts_t = torch.as_tensor(task['fine_path_pts'], device=device, dtype=torch.float32)
    fine_normals_t = (torch.as_tensor(task['fine_path_normals'], device=device, dtype=torch.float32)
                       if 'fine_path_normals' in task else None)
    R_T = torch.as_tensor(task['R_target_at_goal'], device=device, dtype=torch.float32)
    x_T = path_pts[-1]

    K_pool = max(args.K_arms, args.K_pool)
    q_Ts, _ = _dense_ik_at(kin, x_T, R_T, K_pool * 4, rng)
    if q_Ts.shape[0] == 0:
        print("no IK at goal — retry with another --seed")
        return
    if q_Ts.shape[0] > K_pool:
        idx = rng.permutation(q_Ts.shape[0])[:K_pool]
        q_Ts = q_Ts[idx]
    print(f"  pool: {q_Ts.shape[0]} candidate q_T's at goal\n")

    # ----- batched CFM reverse chain across K_pool arms -----
    K_actual = q_Ts.shape[0]
    import time
    t0 = time.perf_counter()
    q_traj_all = backward_sample(                            # (T, K, 7)
        model, kin, q_Ts, path_pts, plane_normal_t, direction_axis,
        n_ode_steps=args.n_ode_steps, snap_iters=snap_iters_eff,
        direction_per_step=d_per_step,
        plane_normal_per_step=n_per_step_t,
        debug_orient=False)
    t_cfm = time.perf_counter() - t0
    # snap-fit per-arm: max FK err across ckpts
    p_pred_all, _ = kin.fk_batch(q_traj_all.reshape(-1, 7))
    p_pred_all = p_pred_all.view(q_traj_all.shape[0], K_actual, 3)
    snap_err_mm_all = (1000.0 * (p_pred_all - path_pts.unsqueeze(1)).norm(dim=-1)
                        .max(dim=0).values).detach().cpu().numpy()
    # branch sigs per arm
    sigs = [branch_signature(q_traj_all[0, k].detach().cpu().numpy())
            for k in range(K_actual)]

    # ----- batched DLS forward rollout across K arms -----
    t0 = time.perf_counter()
    q_dense_full_all, ok_all = dls_forward_rollout(           # (M, K, 7), (K,)
        kin, q_traj_cfm=q_traj_all, path_pts=path_pts,
        plane_normal=plane_normal_t, v_path=args.v_path,
        eps_p=eps_p_eff,
        verbose=args.debug_orient,
        plane_normal_per_step=n_per_step_t,
        fine_path_pts=fine_pts_t,
        fine_path_normals=fine_normals_t,
        track_fine=not args.no_track_fine)
    t_dls = time.perf_counter() - t0
    print(f"  batched: CFM {t_cfm:.2f}s + DLS {t_dls:.2f}s for K={K_actual} arms")

    # subsample + per-arm reporting
    stride = max(1, int(args.ctrl_stride))
    q_dense_subs = q_dense_full_all[::stride].detach().cpu().numpy()  # (M', K, 7)
    with torch.no_grad():
        p_dense_all, _ = kin.fk_batch(q_dense_full_all.reshape(-1, 7))
        p_dense_all = p_dense_all.view(q_dense_full_all.shape[0], K_actual, 3)
        d_to_ckpt = (p_dense_all.unsqueeze(2) - path_pts.unsqueeze(0).unsqueeze(0)
                      ).norm(dim=-1)                          # (M, K, T_ckpt)
        ctrl_err_mm_all = (1000.0 * d_to_ckpt.min(dim=-1).values
                            .max(dim=0).values).detach().cpu().numpy()
    success_flags = [bool(ok_all[k].item()) for k in range(K_actual)]
    q_trajs_dense = [q_dense_subs[:, k, :] for k in range(K_actual)]

    for k in range(K_actual):
        tag = "OK" if success_flags[k] else "FAILED"
        print(f"  arm {k}: branch={sigs[k]}  CFM-snap max err={snap_err_mm_all[k]:.2f}mm  "
              f"DLS frames={q_trajs_dense[k].shape[0]}  "
              f"ctrl-trace max-to-ckpt={ctrl_err_mm_all[k]:.1f}mm  [{tag}]")

    # filter to successes (unless --show-failed)
    n_total = len(q_trajs_dense)
    n_ok = sum(success_flags)
    if args.show_failed:
        keep_pool = list(range(n_total))
    elif n_ok > 0:
        keep_pool = [i for i, s in enumerate(success_flags) if s]
    else:
        print(f"\n  ⚠ no arm completed; falling back to all {n_total} (failed) arms")
        keep_pool = list(range(n_total))

    # diversity pick: greedy max-min over q_0 in joint-space L2
    target = min(args.K_arms, len(keep_pool))
    if len(keep_pool) <= target:
        keep = keep_pool
    else:
        q0_pool = np.stack([q_trajs_dense[i][0] for i in keep_pool], axis=0)  # (P, 7)
        # start from the q_0 farthest from the pool centroid (most extreme)
        centroid = q0_pool.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(q0_pool - centroid, axis=1)))
        picked_local = [first]
        while len(picked_local) < target:
            # for every candidate, distance to nearest already-picked
            d_to_picked = np.full(q0_pool.shape[0], np.inf)
            for j in range(q0_pool.shape[0]):
                if j in picked_local:
                    d_to_picked[j] = -np.inf                # exclude
                    continue
                for p in picked_local:
                    d_to_picked[j] = min(d_to_picked[j],
                                          float(np.linalg.norm(q0_pool[j] - q0_pool[p])))
            picked_local.append(int(np.argmax(d_to_picked)))
        keep = [keep_pool[j] for j in picked_local]

    q_trajs_dense = [q_trajs_dense[i] for i in keep]
    sigs = [sigs[i] for i in keep]
    print(f"\n  → pool: {n_total} attempted, {n_ok} OK; "
          f"diversity-picked {len(keep)} for viewer")

    K = len(q_trajs_dense)
    M_max = max(q.shape[0] for q in q_trajs_dense)

    # ----- build the world scene -----
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from Yuan.flow_connectivity.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    cam_lookat = list(np.mean(fine_pts, axis=0))
    base = ovw.World(cam_pos=(1.4, 1.0, 0.9),
                     cam_lookat_pos=cam_lookat,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # arm spawn — serial = one neutral arm; parallel = K colored arms
    arms = []
    if args.mode == "serial":
        arm, _ = make_fr3_with_pen()
        arm.attach_to(base.scene)
        arm.rgb = (0.55, 0.55, 0.60)                       # neutral gray
        arm.alpha = 0.95
        attach_pen_visual(arm, rgb=(0.55, 0.55, 0.60), alpha=0.95)
        arm.toggle_tcp(length_scale=0.10, radius_scale=0.4)
        arms.append(arm)
    else:
        for k in range(K):
            arm, _ = make_fr3_with_pen()
            rgb = branch_color(sigs[k])
            arm.attach_to(base.scene)
            arm.rgb = rgb
            arm.alpha = float(args.alpha)
            attach_pen_visual(arm, rgb=rgb, alpha=0.85)
            arm.toggle_tcp(length_scale=0.10, radius_scale=0.4)
            arms.append(arm)

    # base coord frame
    ossop.frame(length_scale=0.20, radius_scale=0.8).attach_to(base.scene)

    # desired curve as a faint reference line (continuous polyline)
    fine_segs = np.stack([fine_pts[:-1], fine_pts[1:]], axis=1)   # (n_fine-1, 2, 3)
    ossop.linsegs(segs=fine_segs, radius=0.0009,
                  srgbs=np.array([0.55, 0.55, 0.58]),
                  alpha=0.55).attach_to(base.scene)

    # in serial mode, render K colored TCP traces as small dots
    if args.mode == "serial":
        for k in range(K):
            rgb = branch_color(sigs[k])
            tcps_k = kin.fk_batch(
                torch.as_tensor(q_trajs_dense[k], device=device,
                                dtype=torch.float32))[0].detach().cpu().numpy()
            stride = max(1, len(tcps_k) // 40)
            for j in range(0, len(tcps_k), stride):
                ossop.sphere(pos=tuple(tcps_k[j]), radius=0.0028,
                             rgb=rgb, alpha=0.85).attach_to(base.scene)

    # checkpoints as small markers (where CFM operates)
    for ckpt_pt in task['path_pts']:
        ossop.sphere(pos=tuple(ckpt_pt), radius=0.006,
                     rgb=(0.10, 0.10, 0.10), alpha=0.95).attach_to(base.scene)

    # start / goal markers (still distinct but smaller)
    ossop.sphere(pos=tuple(task['path_pts'][0]), radius=0.011,
                 rgb=(0.10, 0.65, 0.25), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task['path_pts'][-1]), radius=0.013,
                 rgb=(0.85, 0.20, 0.20), alpha=0.95).attach_to(base.scene)

    # surface visualization
    if args.surface_type == "flat":
        plane_center = np.mean(task['path_pts'], axis=0)
        ossop.plane(pos=tuple(plane_center),
                    normal=tuple(task['plane_normal']),
                    size=(0.6, 0.6),
                    rgb=(0.55, 0.55, 0.6), alpha=0.15).attach_to(base.scene)
        ossop.arrow(spos=tuple(plane_center),
                    epos=tuple(plane_center + 0.12 * task['plane_normal']),
                    shaft_radius=0.0028, head_radius=0.0075, head_length=0.015,
                    rgb=(0.95, 0.20, 0.85), alpha=0.80).attach_to(base.scene)
    else:                                                  # sphere
        # translucent sphere shell
        ossop.sphere(pos=tuple(task['sphere_C']),
                     radius=task['sphere_R'],
                     rgb=(0.55, 0.55, 0.62), alpha=0.07).attach_to(base.scene)
        # per-checkpoint outward normal as small arrows (porcupine)
        for j, ckpt_pt in enumerate(task['path_pts']):
            n_local = task['n_per_step'][min(j, len(task['n_per_step']) - 1)]
            ossop.arrow(spos=tuple(ckpt_pt),
                        epos=tuple(ckpt_pt + 0.045 * n_local),
                        shaft_radius=0.0018, head_radius=0.0048,
                        head_length=0.009,
                        rgb=(0.95, 0.20, 0.85), alpha=0.75).attach_to(base.scene)

    # initial pose
    if args.mode == "serial":
        arms[0].fk(q_trajs_dense[0][0])
    else:
        for k, arm in enumerate(arms):
            arm.fk(q_trajs_dense[k][0])

    sample_idx = [0]
    indices = [0] * K
    step_dirs = [+1] * K

    if args.mode == "serial":
        def tick(_dt):
            k = sample_idx[0]
            traj = q_trajs_dense[k]
            M_k = traj.shape[0]
            arms[0].fk(traj[indices[k]])
            if args.cycle == "forward":
                indices[k] += 1
                if indices[k] >= M_k:
                    indices[k] = 0
                    sample_idx[0] = (sample_idx[0] + 1) % K
                    q0 = q_trajs_dense[sample_idx[0]][0]
                    q0_str = "[" + ", ".join(f"{v:+.2f}" for v in q0) + "]"
                    print(f"  → sample {sample_idx[0]}/{K}  "
                          f"branch={sigs[sample_idx[0]]}  "
                          f"q_0={q0_str}")
            else:
                indices[k] += step_dirs[k]
                if indices[k] >= M_k - 1:
                    indices[k] = M_k - 1
                    step_dirs[k] = -1
                elif indices[k] <= 0:
                    indices[k] = 0
                    step_dirs[k] = +1
                    sample_idx[0] = (sample_idx[0] + 1) % K
                    q0 = q_trajs_dense[sample_idx[0]][0]
                    q0_str = "[" + ", ".join(f"{v:+.2f}" for v in q0) + "]"
                    print(f"  → sample {sample_idx[0]}/{K}  "
                          f"branch={sigs[sample_idx[0]]}  "
                          f"q_0={q0_str}")
        print(f"\n[viewer-serial]  1 gray arm cycling through K={K} samples  "
              f"M_max={M_max} frames  cycle={args.cycle}  fps={args.fps:.1f}")
    else:
        def tick(_dt):
            for k, arm in enumerate(arms):
                traj = q_trajs_dense[k]
                M_k = traj.shape[0]
                arm.fk(traj[indices[k]])
                if args.cycle == "forward":
                    indices[k] += 1
                    if indices[k] >= M_k:
                        indices[k] = 0
                else:
                    indices[k] += step_dirs[k]
                    if indices[k] >= M_k - 1:
                        indices[k] = M_k - 1
                        step_dirs[k] = -1
                    elif indices[k] <= 0:
                        indices[k] = 0
                        step_dirs[k] = +1
        print(f"\n[viewer-parallel]  K={K} colored arms (each its own length)  "
              f"M_max={M_max}  cycle={args.cycle}  fps={args.fps:.1f}")

    base.schedule_interval(tick, interval=1.0 / args.fps)
    print("          drag-mouse: rotate cam   |   close window: exit")
    base.run()


if __name__ == "__main__":
    main()
