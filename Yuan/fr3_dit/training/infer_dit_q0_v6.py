"""v6 inference: predict per-keypoint q sequence, IK-refine each keypoint to its
exact target vertex, and Cartesian-IK interpolate dense per-frame q's between
consecutive keypoints. Replaces tracker rollout entirely.

Pipeline:
    tokens → DDIM sample (B, T, 7) keypoint sequence → mask out non-keypoint positions
           → q7 snap (each keypoint q7 = 0)
           → per-keypoint IK refine (TCP exact = vertex_world, orient = seed's own)
           → segment-by-segment Cartesian IK interp at 1mm/step (seed = previous q)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3, PenFrankaResearch3GPU
from Yuan.fr3_dit.data_generation.generate_fr3_plane_dataset import (
    nullspace_projector_batch,
    position_jacobian_batch,
)
from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    FR3_JOINT_LIMITS,
    Q_CENTER,
    Q_HALF,
    denormalize_q,
)
from Yuan.fr3_dit.training.task_cond_dit_q0_v6 import (
    DiTq0Config_v6,
    TaskCondDiTq0_v6,
    ddim_sample_keypoints,
)
from Yuan.fr3_dit.training.ik_refine import refine_q0_seed


def project_q_to_cone_nullspace(
    q_raw: np.ndarray,                     # (B, 7) raw joint angles
    fr3_gpu,                                # PenFrankaResearch3GPU
    desk_normal: np.ndarray,                # (3,) world
    cos_threshold: float = 0.866,           # cos(30°)
    max_iters: int = 30,
    step_size: float = 0.05,                # rad per step in null-space direction
    damping: float = 1e-3,
) -> np.ndarray:
    """Project a batch of q's onto the cone-respecting subset by null-space gradient
    ascent on cos(TCP_z, -desk_normal). Each step moves q along the position-Jacobian's
    null space (TCP position invariant, IK branch preserved), increasing cos_theta until
    it crosses the threshold.

    This is the inference-time analog of the tracker's interior attractor (v5):
    same physics, applied at sampling instead of rollout.
    """
    device = fr3_gpu.robot.jnt_ranges.device
    n = q_raw.shape[0]
    q = torch.from_numpy(q_raw.astype(np.float32)).to(device)
    pen_axis = torch.tensor(-desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12),
                             device=device, dtype=torch.float32)

    for _ in range(max_iters):
        q_eval = q.detach().clone().requires_grad_(True)
        tcp_pos, tcp_rot = fr3_gpu.robot.fk_batch(q_eval)
        tcp_z = tcp_rot[:, :, 2]                                          # (B, 3)
        cos_theta = (tcp_z * pen_axis).sum(dim=-1)                         # (B,)
        if (cos_theta >= cos_threshold).all():
            break
        # Gradient of cos_theta wrt q.
        grad_cos = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=False)[0]   # (B, 7)
        # Position Jacobian + null-space projector.
        j_pos, _ = position_jacobian_batch(fr3_gpu.robot, q_eval.detach(), create_graph=False)
        proj = nullspace_projector_batch(j_pos, damping)                                  # (B, 7, 7)
        d = (proj @ grad_cos.unsqueeze(-1)).squeeze(-1)                                   # (B, 7)
        # Step only the violators.
        violators = (cos_theta < cos_threshold).float().unsqueeze(-1)                     # (B, 1)
        # Normalize step direction to magnitude `step_size` (avoid huge updates).
        d_norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        q = q + step_size * (d / d_norm) * violators

        # Soft clamp into joint limits.
        lo = fr3_gpu.robot.jnt_ranges[:, 0]; hi = fr3_gpu.robot.jnt_ranges[:, 1]
        q = torch.maximum(q, lo + 1e-3)
        q = torch.minimum(q, hi - 1e-3)
        q = q.detach()

    return q.cpu().numpy().astype(np.float32)


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_q0_v6_ckpts" / "final.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"

DIR_LOCAL_OFFSET = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--sampler-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg-w", type=float, default=3.0)
    p.add_argument("--clip-x0", type=float, default=1.2)
    p.add_argument("--no-snap-q7", action="store_true", default=False)
    p.add_argument("--out-prefix", type=str, default="infer_q0_v6")
    p.add_argument("--step-mm", type=float, default=1.0,
                   help="Cartesian IK interpolation step (mm) along each segment.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_task(h5_path, idx, max_tokens):
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        tok_off = ts["token_offset"][()]
        if idx < 0 or idx >= len(tok_off) - 1:
            raise IndexError(f"task_idx={idx} out of range")
        t_lo, t_hi = int(tok_off[idx]), int(tok_off[idx + 1])
        n_tok = t_hi - t_lo
        token_dim = int(f["meta"].attrs["token_dim"])
        length_ref = float(f["meta"].attrs["length_ref"])
        tokens = np.zeros((max_tokens, token_dim), dtype=np.float32)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_kind = np.zeros((max_tokens,), dtype=np.uint8)
        token_kind[:n_tok] = ts["token_kind"][t_lo:t_hi]
        token_mask = np.zeros((max_tokens,), dtype=np.float32)
        token_mask[:n_tok] = 1.0
        local_origin = np.asarray(ts["local_origin"][idx], dtype=np.float32)
        local_frame = np.asarray(ts["local_frame"][idx], dtype=np.float32)
        seg_count = int(ts["seg_count"][idx])
        total_length = float(ts["total_length"][idx])
        start_q = np.asarray(ts["start_q"][idx], dtype=np.float32)
        desk_normal = np.asarray(f["meta"].attrs["source_desk_normal"], dtype=np.float32)
    desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)

    # Vertex world positions: vertex 0 = local_origin, walk segments.
    seg_token_positions = np.where(token_kind == 1)[0]
    vertices_world = np.zeros((seg_count + 1, 3), dtype=np.float32)
    vertices_world[0] = local_origin
    for k in range(seg_count):
        tok_pos = int(seg_token_positions[k])
        dir_local = tokens[tok_pos, DIR_LOCAL_OFFSET:DIR_LOCAL_OFFSET + 3].astype(np.float32)
        length_m = float(tokens[tok_pos, 6]) * length_ref
        dir_world = local_frame @ dir_local
        n = float(np.linalg.norm(dir_world))
        if n > 1e-9:
            vertices_world[k + 1] = vertices_world[k] + (dir_world / n) * length_m
        else:
            vertices_world[k + 1] = vertices_world[k]

    return {
        "tokens": tokens, "token_kind": token_kind, "token_mask": token_mask,
        "n_tokens": n_tok, "seg_count": seg_count, "total_length": total_length,
        "start_q": start_q, "local_origin": local_origin, "local_frame": local_frame,
        "vertices_world": vertices_world, "desk_normal": desk_normal,
    }


def load_ckpt(ckpt_path, device, use_ema):
    d = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DiTq0Config_v6(**d["cfg"])
    model = TaskCondDiTq0_v6(cfg).to(device)
    if use_ema and "ema" in d and d["ema"] is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in d["ema"]:
                    p.copy_(d["ema"][n].to(device))
    else:
        model.load_state_dict(d["model"])
    schedule = DDPMCosineSchedule(T=int(d["T"])).to(device)
    return model.eval(), cfg, schedule, int(d.get("step", -1))


def ik_interpolate_segment(pen_robot, q_start, q_end, p_start, p_end, rot, step_m):
    """Cartesian-linear interp from p_start to p_end, IK each waypoint with seed
    chained from the previous solution. q_start fixes the first frame; q_end
    is the target end (used as IK seed for the last frame to guide branch)."""
    seg_len = float(np.linalg.norm(np.asarray(p_end) - np.asarray(p_start)))
    n_steps = max(2, int(np.ceil(seg_len / step_m)))
    out = [np.asarray(q_start, dtype=np.float64).reshape(7).copy()]
    seed = out[-1]
    for k in range(1, n_steps):
        a = k / n_steps
        p = (1.0 - a) * np.asarray(p_start) + a * np.asarray(p_end)
        sol = pen_robot.ik(tgt_pos=p, tgt_rotmat=rot, seed_jnt_values=seed)
        if sol is None:
            sol = seed
        sol = np.asarray(sol, dtype=np.float64)
        out.append(sol); seed = sol
    out.append(np.asarray(q_end, dtype=np.float64).reshape(7).copy())
    return np.stack(out, axis=0).astype(np.float32)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[ckpt] loading {args.ckpt}")
    model, cfg, schedule, step = load_ckpt(args.ckpt, device, args.use_ema)
    print(f"[ckpt] step={step} d_model={cfg.d_model} max_tokens={cfg.max_tokens}")

    task = load_task(args.data, args.task_idx, cfg.max_tokens)
    print(f"[task] idx={args.task_idx} seg_count={task['seg_count']} "
          f"total_len={task['total_length']*100:.1f}cm n_tokens={task['n_tokens']}")

    tokens_t = torch.from_numpy(task["tokens"]).unsqueeze(0).expand(args.n_samples, -1, -1).contiguous().to(device)
    token_mask_t = torch.from_numpy(task["token_mask"]).unsqueeze(0).expand(args.n_samples, -1).contiguous().to(device)

    print(f"[sample] DDIM steps={args.sampler_steps} eta={args.eta} cfg_w={args.cfg_w} n={args.n_samples}")
    kp_norm = ddim_sample_keypoints(
        model, schedule, tokens_t, token_mask_t,
        device=device, num_steps=args.sampler_steps, eta=args.eta,
        cfg_w=args.cfg_w, clip_x0=args.clip_x0,
    )
    # (N, T, 7) — denormalize and extract keypoints at START + SEGMENT positions only.
    kp_raw = (kp_norm.cpu().numpy() * Q_HALF + Q_CENTER).astype(np.float32)   # (N, T, 7)
    if not args.no_snap_q7:
        kp_raw[..., 6] = 0.0

    # Position list: START at 0, SEGMENT at all kind==1 positions (in order).
    kind = task["token_kind"]
    kp_token_positions = [0]                                              # vertex 0 (START)
    seg_positions = list(np.where(kind == 1)[0])
    kp_token_positions += seg_positions                                   # vertex 1..K
    kp_token_positions = sorted(set(kp_token_positions))                  # dedupe (just in case)
    # Map each keypoint index to vertex index.
    # First entry is vertex 0; each subsequent SEGMENT in order is vertex i (i = 1..K).

    pen_robot = PenFrankaResearch3(name="pen", enable_cc=False)
    fr3_gpu = PenFrankaResearch3GPU(device)                                # for null-space projection
    desk_normal = task["desk_normal"]
    vertices_world = task["vertices_world"]                                # (K+1, 3)
    n_v = vertices_world.shape[0]

    # Null-space cone projection: nudge any out-of-cone keypoint q back into the cone
    # along the position-Jacobian's null space. This preserves TCP position and IK branch
    # while pulling TCP_z toward -desk_normal. Replaces the previous "forced rotmat"
    # approach which broke IK whenever seed orientation was far from cone.
    flat_kp_q = kp_raw.reshape(-1, 7)                                       # (N*T, 7)
    flat_kp_q = project_q_to_cone_nullspace(
        flat_kp_q, fr3_gpu, desk_normal,
        cos_threshold=float(np.cos(np.deg2rad(30.0))),
        max_iters=30, step_size=0.05, damping=1e-3,
    )
    kp_raw = flat_kp_q.reshape(args.n_samples, -1, 7)
    if not args.no_snap_q7:
        kp_raw[..., 6] = 0.0    # re-snap q7 (projection may have nudged it)

    # Pre-compute per-vertex pen-into-desk rotation matrices. Each vertex's rotation is
    # built from desk_normal + the OUTGOING segment direction (last vertex uses incoming
    # direction). This forces every keypoint's IK refine to land at an in-cone TCP_z,
    # eliminating the angle_violation cascades caused by the model occasionally
    # predicting tilted poses.
    vertex_rotmats = []
    for vi in range(n_v):
        if vi < n_v - 1:
            d_w = vertices_world[vi + 1] - vertices_world[vi]
        else:
            d_w = vertices_world[vi] - vertices_world[vi - 1]
        d_n = float(np.linalg.norm(d_w))
        if d_n < 1e-9:
            d_w = np.array([1.0, 0.0, 0.0])
        else:
            d_w = d_w / d_n
        z_ax = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
        x_ax = d_w - z_ax * float(np.dot(d_w, z_ax))
        xn = float(np.linalg.norm(x_ax))
        if xn < 1e-9:
            helper = np.array([1.0, 0.0, 0.0]) if abs(z_ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_ax = helper - z_ax * float(np.dot(helper, z_ax))
            x_ax /= max(float(np.linalg.norm(x_ax)), 1e-12)
        else:
            x_ax /= xn
        y_ax = np.cross(z_ax, x_ax); y_ax /= max(float(np.linalg.norm(y_ax)), 1e-12)
        vertex_rotmats.append(np.column_stack((x_ax, y_ax, z_ax)).astype(np.float64))

    # For each candidate, refine each keypoint to FORCED in-cone TCP rotation,
    # then IK-interpolate between them.
    candidate_summaries = []
    for ci in range(args.n_samples):
        kp_for_cand = kp_raw[ci]                                            # (T, 7)
        vertex_qs = []
        ik_oks = []
        tcp_errs = []
        for vi in range(n_v):
            tok_pos = kp_token_positions[vi] if vi < len(kp_token_positions) else kp_token_positions[-1]
            q_seed = kp_for_cand[tok_pos]
            # IK refine: target TCP = vertex_world[vi]; target_rotmat = None (use seed's
            # own rotation, which has already been projected into cone via null-space ascent).
            q_ref, ok, info = refine_q0_seed(
                pen_robot, q_seed, vertices_world[vi],
                target_rotmat=None,
                desk_normal=desk_normal, theta_max_deg=30.0,
            )
            vertex_qs.append(q_ref.astype(np.float64))
            ik_oks.append(bool(ok))
            tcp_errs.append(float(info["tcp_err_refined_m"]))

        # Now Cartesian IK interp between consecutive vertex_qs along the world polyline.
        full_q = [np.asarray(vertex_qs[0]).reshape(1, 7).copy()]
        # Build pen-into-desk rotation per segment using local first-seg dir.
        for k in range(n_v - 1):
            d_w = vertices_world[k + 1] - vertices_world[k]
            d_n = float(np.linalg.norm(d_w))
            if d_n < 1e-9:
                d_w = np.array([1.0, 0.0, 0.0])
            else:
                d_w = d_w / d_n
            # Pen-into-desk: z=-desk_normal, x=d_w (projected), y=z×x.
            z_axis = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
            x_axis = d_w - z_axis * float(np.dot(d_w, z_axis))
            xn = float(np.linalg.norm(x_axis))
            if xn < 1e-9:
                helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                x_axis = helper - z_axis * float(np.dot(helper, z_axis))
                x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
            else:
                x_axis /= xn
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
            rot = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float64)

            seg_q = ik_interpolate_segment(
                pen_robot, vertex_qs[k], vertex_qs[k + 1],
                vertices_world[k], vertices_world[k + 1], rot, args.step_mm / 1000.0,
            )
            full_q.append(seg_q[1:])     # skip first (== vertex_qs[k])
        full_q = np.concatenate(full_q, axis=0).astype(np.float32)
        candidate_summaries.append({
            "candidate": ci,
            "n_vertices": n_v,
            "n_frames_total": int(full_q.shape[0]),
            "ik_ok_per_vertex": ik_oks,
            "tcp_err_per_vertex_cm": [round(e * 100, 4) for e in tcp_errs],
            "max_tcp_err_cm": float(max(tcp_errs) * 100),
            "all_ik_ok": bool(all(ik_oks)),
        })
        # Save this candidate's q-trajectory.
        np.save(args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_cand{ci}_qtraj.npy",
                full_q)
    # Save also the GT vertex info for evaluation.
    np.save(args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_vertices.npy",
            vertices_world)
    meta_path = args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task_idx": int(args.task_idx),
            "seg_count": int(task["seg_count"]),
            "n_vertices": int(n_v),
            "total_length_m": float(task["total_length"]),
            "ckpt_step": int(step),
            "sampler": {"steps": args.sampler_steps, "eta": args.eta,
                        "cfg_w": args.cfg_w, "clip_x0": args.clip_x0,
                        "step_mm": args.step_mm, "seed": args.seed},
            "n_candidates": args.n_samples,
            "candidates": candidate_summaries,
        }, f, indent=2)
    print(f"[saved] {meta_path}")
    n_all_ok = sum(1 for c in candidate_summaries if c["all_ik_ok"])
    max_err_cm = max(c["max_tcp_err_cm"] for c in candidate_summaries)
    print(f"[summary] {n_all_ok}/{args.n_samples} candidates have all keypoints IK-OK; "
          f"max TCP err across all candidates = {max_err_cm:.3f} cm")


if __name__ == "__main__":
    main()
