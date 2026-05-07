"""v18 curve-path visualization in the one-world interactive viewer.

Samples ONE curve task (line / arc / s_curve), runs K backward CFM samples
with per-segment local tangents, then spawns K FR3+pen arms in the world
viewer. Each arm plays its own dense q-trajectory in lockstep, ping-pong
through the path. Pen tip leaves no automatic trail (use trace markers
in the scene below).

Run:
    python -m Yuan.RL.v18_curve_world --curve-type arc --K-arms 4
"""
from __future__ import annotations
import argparse
import builtins
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import batched_rollout_segment
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample
from Yuan.RL.v18_data_prep import _dense_ik_at, _build_R_from_normal_direction
from Yuan.RL.v18_curve_eval import sample_curve_task, branch_signature


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
                        verbose: bool = False
                        ) -> tuple[torch.Tensor, bool]:
    """Run DLS Cartesian-tracking controller from q_traj_cfm[0], using
    q_traj_cfm[i+1] as a moving null-space target within segment i.

    Per segment i (between path_pts[i] and path_pts[i+1]):
      - q_init  := q at segment start (continuous from prior segment)
      - q_ref(t) := q_traj_cfm[i] + (t/T_seg) * (q_traj_cfm[i+1] - q_traj_cfm[i])
        i.e., a moving null-space target that linearly walks from q_i to q_{i+1}.

    This gives the controller intermediate joint-space waypoints (CFM's
    reverse-chain output) instead of relying on default null-space attractors
    alone, while keeping the per-step pull `g_knull · (q_ref(t) - q(t))` small
    enough to avoid joint-limit pushes. q_0 still determines the starting
    branch.

    Returns (M, 7) joint trajectory at every control step (dt = cfg.DT).
    """
    device = kin.device
    dt = float(cfg.DT)
    plane_n_np = plane_normal.detach().cpu().numpy()
    branch_action = torch.tensor([[1.0, 0.0, 1.0, 0.0]],
                                  device=device, dtype=torch.float32)

    q_curr = q_traj_cfm[0].unsqueeze(0).clone()             # (1, 7)
    chunks = [q_curr.clone().unsqueeze(0)]                  # (1, 1, 7) initial
    all_segments_ok = True

    for i in range(path_pts.shape[0] - 1):
        p_a = path_pts[i].unsqueeze(0)
        seg_vec = path_pts[i + 1] - path_pts[i]
        L_seg = float(seg_vec.norm().item())
        if L_seg < 1e-6:
            continue
        d_dir = (seg_vec / L_seg).unsqueeze(0)
        R_np = _build_R_from_normal_direction(plane_n_np,
                                              d_dir.squeeze(0).cpu().numpy())
        R_seg = torch.as_tensor(R_np, device=device,
                                 dtype=torch.float32).unsqueeze(0)
        T_seg = max(1, int(round(L_seg / (v_path * dt))))
        # moving q_ref: linear interp from CFM-predicted q_i to q_{i+1}
        # over T_seg + 1 control steps (0..T_seg). At t=0 it equals
        # q_traj_cfm[i] which equals q_curr at segment start (continuous).
        alphas = torch.linspace(0.0, 1.0, T_seg + 1,
                                 device=device).view(-1, 1, 1)
        q_ref_traj = ((1.0 - alphas) * q_traj_cfm[i].view(1, 1, 7)
                      + alphas * q_traj_cfm[i + 1].view(1, 1, 7))   # (T+1, 1, 7)
        out = batched_rollout_segment(
            q_init=q_curr,
            R_tgt=R_seg,
            branch_action=branch_action,
            p0=p_a,
            d_dir=d_dir,
            v_path=torch.full((1,), v_path, device=device, dtype=torch.float32),
            eps_p=torch.full((1,), eps_p, device=device, dtype=torch.float32),
            T_total=torch.full((1,), T_seg, device=device, dtype=torch.long),
            start_step=0,
            end_step=T_seg,
            kin=kin,
            q_ref=q_ref_traj,
            record_traj=True,
        )
        q_curr = out['q_final']
        chunks.append(out['q_record'][1:])                  # skip dup of seg-start q
        seg_alive = bool(out['alive_out'].item())
        if not seg_alive:
            all_segments_ok = False

        if verbose:
            T_reached = int(out['lengths'].item())
            pos_e_mm = float(out['last_pos_err'].item()) * 1000.0
            ori_e_deg = float(out['last_orient_err'].item()) * 180.0 / 3.14159265
            if seg_alive:
                tag = "ok"
            elif pos_e_mm > eps_p * 1000.0:
                tag = "fail_pos"
            elif ori_e_deg > 5.0:
                tag = "fail_ori"
            else:
                tag = "fail_lmt|other"
            print(f"      seg {i}->{i+1}:  T={T_reached:>3d}/{T_seg:<3d}  "
                  f"pos_err={pos_e_mm:6.2f}mm  ori_err={ori_e_deg:6.2f}°  "
                  f"[{tag}]")

    return torch.cat(chunks, dim=0).squeeze(1), all_segments_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_multi/best.pt")
    ap.add_argument("--curve-type", default="arc",
                    choices=["line", "arc", "s_curve"])
    ap.add_argument("--n-checkpoints", type=int, default=5)
    ap.add_argument("--ctrl-stride", type=int, default=4,
                    help="subsample DLS rollout this many control steps per "
                         "animation frame (1 = every step ≈ 50 fps target)")
    ap.add_argument("--v-path", type=float, default=0.10,
                    help="DLS controller path velocity [m/s]")
    ap.add_argument("--K-arms", type=int, default=6)
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
    args = ap.parse_args()

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
        t = sample_curve_task(rng, kin, args.curve_type, args.n_checkpoints)
        if t is not None:
            task = t
            break
    if task is None:
        print(f"could not sample {args.curve_type} task — retry with another --seed")
        return

    print(f"\ntask: curve={args.curve_type}  L={task['L']:.2f}m  "
          f"plane_normal_z={task['plane_normal'][2]:.2f}")

    path_pts = torch.as_tensor(task['path_pts'], device=device, dtype=torch.float32)
    fine_pts = task['fine_path_pts']                       # (120, 3) numpy
    plane_normal_t = torch.as_tensor(task['plane_normal'], device=device, dtype=torch.float32)
    direction_axis = torch.as_tensor(task['direction_axis'], device=device, dtype=torch.float32)
    d_per_step = torch.as_tensor(task['d_per_step'], device=device, dtype=torch.float32)
    R_T = torch.as_tensor(task['R_target_at_goal'], device=device, dtype=torch.float32)
    x_T = path_pts[-1]

    q_Ts, _ = _dense_ik_at(kin, x_T, R_T, args.K_arms * 4, rng)
    if q_Ts.shape[0] == 0:
        print("no IK at goal — retry with another --seed")
        return
    if q_Ts.shape[0] > args.K_arms:
        idx = rng.permutation(q_Ts.shape[0])[:args.K_arms]
        q_Ts = q_Ts[idx]

    q_trajs_dense = []
    sigs = []
    success_flags = []
    for k in range(q_Ts.shape[0]):
        if args.debug_orient:
            print(f"  --- arm {k} CFM reverse chain ---")
        q_traj = backward_sample(
            model, kin, q_Ts[k], path_pts, plane_normal_t, direction_axis,
            n_ode_steps=args.n_ode_steps, snap_iters=args.snap_iters,
            direction_per_step=d_per_step,
            debug_orient=args.debug_orient)
        p_pred, _ = kin.fk_batch(q_traj)
        snap_err_mm = float(1000.0 * (p_pred - path_pts).norm(dim=-1).max().item())
        sig = branch_signature(q_traj[0].detach().cpu().numpy())
        if args.debug_orient:
            print(f"  --- arm {k} DLS forward rollout ---")
        q_dense_full, ok = dls_forward_rollout(
            kin, q_traj_cfm=q_traj, path_pts=path_pts,
            plane_normal=plane_normal_t, v_path=args.v_path,
            verbose=args.debug_orient)
        stride = max(1, int(args.ctrl_stride))
        q_dense = q_dense_full[::stride].detach().cpu().numpy()
        with torch.no_grad():
            p_dense, _ = kin.fk_batch(q_dense_full)
            d_to_ckpt = (p_dense.unsqueeze(1) - path_pts.unsqueeze(0)).norm(dim=-1)
            ctrl_err_mm = float(1000.0 * d_to_ckpt.min(dim=-1).values.max().item())
        q_trajs_dense.append(q_dense)
        sigs.append(sig)
        success_flags.append(ok)
        tag = "OK" if ok else "FAILED"
        print(f"  arm {k}: branch={sig}  CFM-snap max err={snap_err_mm:.2f}mm  "
              f"DLS frames={q_dense.shape[0]}  ctrl-trace max-to-ckpt={ctrl_err_mm:.1f}mm  "
              f"[{tag}]")

    # filter to completed-only by default
    n_total = len(q_trajs_dense)
    n_ok = sum(success_flags)
    if not args.show_failed and n_ok > 0:
        keep = [i for i, s in enumerate(success_flags) if s]
        q_trajs_dense = [q_trajs_dense[i] for i in keep]
        sigs = [sigs[i] for i in keep]
        print(f"\n  → showing {len(keep)}/{n_total} arms that completed all "
              f"segments (use --show-failed to include partial)")
    elif n_ok == 0 and not args.show_failed:
        print(f"\n  ⚠ no arm completed; falling back to showing all "
              f"{n_total} (failed) arms")

    K = len(q_trajs_dense)
    M_max = max(q.shape[0] for q in q_trajs_dense)

    # ----- build the world scene -----
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

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

    # in serial mode, render K colored TCP traces statically so the user
    # always sees all branch outcomes while the gray arm sweeps one at a time
    if args.mode == "serial":
        for k in range(K):
            rgb = branch_color(sigs[k])
            tcps_k = kin.fk_batch(
                torch.as_tensor(q_trajs_dense[k], device=device,
                                dtype=torch.float32))[0].detach().cpu().numpy()
            stride = max(1, len(tcps_k) // 35)
            for j in range(0, len(tcps_k), stride):
                ossop.sphere(pos=tuple(tcps_k[j]), radius=0.0048,
                             rgb=rgb, alpha=0.75).attach_to(base.scene)

    # desired curve as a chain of small black spheres
    step = max(1, len(fine_pts) // 60)                     # ~60 markers along curve
    for ci in range(0, len(fine_pts), step):
        ossop.sphere(pos=tuple(fine_pts[ci]), radius=0.004,
                     rgb=(0.05, 0.05, 0.05), alpha=0.95).attach_to(base.scene)

    # checkpoints as larger spheres (where CFM operates)
    for ckpt_pt in task['path_pts']:
        ossop.sphere(pos=tuple(ckpt_pt), radius=0.012,
                     rgb=(0.10, 0.10, 0.10), alpha=0.95).attach_to(base.scene)

    # start / goal markers
    ossop.sphere(pos=tuple(task['path_pts'][0]), radius=0.018,
                 rgb=(0.10, 0.65, 0.25), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task['path_pts'][-1]), radius=0.020,
                 rgb=(0.85, 0.20, 0.20), alpha=0.95).attach_to(base.scene)

    # plane (translucent disc)
    plane_center = np.mean(task['path_pts'], axis=0)
    ossop.plane(pos=tuple(plane_center),
                normal=tuple(task['plane_normal']),
                size=(0.6, 0.6),
                rgb=(0.55, 0.55, 0.6), alpha=0.15).attach_to(base.scene)

    # plane-normal arrow at curve midpoint
    ossop.arrow(spos=tuple(plane_center),
                epos=tuple(plane_center + 0.15 * task['plane_normal']),
                shaft_radius=0.0045, head_radius=0.011, head_length=0.022,
                rgb=(0.95, 0.20, 0.85), alpha=0.85).attach_to(base.scene)

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
