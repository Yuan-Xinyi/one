#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import one.utils.math as oum
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers

from Yuan.fr3_dit.data_generation.generate_fr3_plane_dataset import (
    PlaneConstrainedTracker,
    TrackerConfig,
    directional_manipulability_batch,
    joint_margin_mask,
    position_jacobian_batch,
)
from Yuan.fr3_dit.core.pen_fr3_robot import PEN_LENGTH, PenFrankaResearch3, PenFrankaResearch3GPU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test how much rollout capability varies across different valid start configurations under one fixed task.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-starts", type=int, default=64, help="How many valid start configurations to collect for the same task.")
    parser.add_argument("--oversample", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-max-deg", type=float, default=30.0)
    parser.add_argument("--angle-margin-deg", type=float, default=8.0)
    parser.add_argument("--angle-null-gain", type=float, default=0.4)
    parser.add_argument("--joint-margin-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show-topk", type=int, default=3)
    parser.add_argument("--show-bottomk", type=int, default=3)
    parser.add_argument("--ik-max-trials", type=int, default=512)
    parser.add_argument("--ik-seeds-per-pose", type=int, default=6)
    parser.add_argument("--curve-out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "same_task_start_conf_gap_curves.svg")
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def build_tracker(device: torch.device, args: argparse.Namespace) -> PlaneConstrainedTracker:
    fr3 = PenFrankaResearch3GPU(device)
    helpers = make_pen_collision_helpers(fr3.robot)
    return PlaneConstrainedTracker(
        robot=fr3.robot,
        self_collision_fn=helpers.self_collision_fn,
        sphere_positions_fn=helpers.sphere_positions_fn,
        sphere_radii=helpers.sphere_radii,
        sphere_link_indices=helpers.sphere_link_indices,
        config=TrackerConfig(
            theta_max_deg=float(args.theta_max_deg),
            angle_margin_deg=float(args.angle_margin_deg),
            angle_null_gain=float(args.angle_null_gain),
            joint_margin_ratio=float(args.joint_margin_ratio),
            max_steps=int(args.max_steps),
        ),
    )


def summarize(lengths: np.ndarray) -> None:
    print(
        f"[lengths] mean={float(lengths.mean()):.4f}m "
        f"median={float(np.median(lengths)):.4f}m "
        f"std={float(lengths.std()):.4f}m "
        f"min={float(lengths.min()):.4f}m "
        f"max={float(lengths.max()):.4f}m"
    )
    p10, p25, p75, p90 = np.percentile(lengths, [10, 25, 75, 90])
    print(f"[lengths] p10={p10:.4f}m p25={p25:.4f}m p75={p75:.4f}m p90={p90:.4f}m gap={float(lengths.max() - lengths.min()):.4f}m")


def print_ranked_cases(trajectories: list[dict], topk: int, bottomk: int) -> None:
    lengths = np.asarray([float(t["total_projected_length"]) for t in trajectories], dtype=np.float32)
    order = np.argsort(-lengths)
    print("")
    print("=== Top Starts ===")
    for rank, idx in enumerate(order[:topk], start=1):
        traj = trajectories[int(idx)]
        print(
            f"#{rank} idx={int(idx):03d} len={float(traj['total_projected_length']):.4f}m "
            f"reason={traj['termination_reason']} q={np.array2string(traj['start_q'], precision=4, suppress_small=True)}"
        )
    print("")
    print("=== Bottom Starts ===")
    for rank, idx in enumerate(order[-bottomk:][::-1], start=1):
        traj = trajectories[int(idx)]
        print(
            f"#{rank} idx={int(idx):03d} len={float(traj['total_projected_length']):.4f}m "
            f"reason={traj['termination_reason']} q={np.array2string(traj['start_q'], precision=4, suppress_small=True)}"
        )


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def sample_same_task_starts(
    tracker: PlaneConstrainedTracker,
    device: torch.device,
    num_starts: int,
    seed: int,
    ik_max_trials: int,
    ik_seeds_per_pose: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)

    # First get one valid anchor task. Only the task geometry is reused.
    q_anchor, plane_point, direction, plane_normal, plane_side = tracker.sample_valid_batch(1, device)
    plane_point_1 = plane_point[0].detach().cpu().numpy().astype(np.float32)
    direction_1 = direction[0].detach().cpu().numpy().astype(np.float32)
    plane_normal_1 = plane_normal[0].detach().cpu().numpy().astype(np.float32)
    plane_side_1 = float(plane_side[0].item())

    target_z = plane_normal_1 / max(float(np.linalg.norm(plane_normal_1)), 1e-12)
    ref_rot = rotation_matrix_from_normal(target_z)

    wrs_robot = PenFrankaResearch3(name="pen", enable_cc=True)

    q_solutions: list[np.ndarray] = []
    seen: list[np.ndarray] = []

    for trial in range(ik_max_trials):
        tilt_axis = rng.normal(size=3).astype(np.float32)
        tilt_axis = tilt_axis - np.dot(tilt_axis, target_z) * target_z
        tilt_axis_norm = float(np.linalg.norm(tilt_axis))
        if tilt_axis_norm < 1e-8:
            continue
        tilt_axis = tilt_axis / tilt_axis_norm
        tilt_angle = float(rng.uniform(0.0, np.deg2rad(tracker.config.theta_max_deg)))
        roll = float(rng.uniform(-np.pi, np.pi))
        target_dir = oum.rotmat_from_axangle(tilt_axis, tilt_angle) @ target_z
        target_rot = rotation_matrix_from_normal(target_dir) @ oum.rotmat_from_axangle(np.array([0.0, 0.0, 1.0], dtype=np.float32), roll)

        found_this_pose = 0
        for _ in range(ik_seeds_per_pose):
            seed_q = wrs_robot.rand_conf()
            q_sol = wrs_robot.ik(
                tgt_pos=plane_point_1,
                tgt_rotmat=target_rot,
                seed_jnt_values=seed_q,
                option="single",
            )
            if q_sol is None:
                continue
            q_sol = np.asarray(q_sol, dtype=np.float32)
            if not wrs_robot.are_jnts_in_ranges(q_sol):
                continue
            wrs_robot.goto_given_conf(q_sol)
            if wrs_robot.is_collided():
                continue
            if any(np.linalg.norm(q_sol - prev) < 1e-3 for prev in seen):
                continue

            q_t = torch.from_numpy(q_sol).to(device=device, dtype=torch.float32).unsqueeze(0)
            if not bool(joint_margin_mask(tracker.robot, q_t, tracker.config.joint_margin_ratio)[0].item()):
                continue
            tcp_pos_t, tcp_rot_t = tracker.robot.fk_batch(q_t)
            tcp_z_t = tcp_rot_t[:, :, 2]
            cos_theta = torch.sum(tcp_z_t * torch.from_numpy(plane_normal_1).to(device=device).view(1, 3), dim=-1)
            if float(cos_theta.item()) < tracker.theta_cos:
                continue
            if float(tracker.self_collision_fn(q_t)[0].item()) > 0.0:
                continue
            plane_ok = tracker.plane_clearance_mask(
                q_t,
                torch.from_numpy(plane_point_1).to(device=device).view(1, 3),
                torch.from_numpy(plane_normal_1).to(device=device).view(1, 3),
                torch.tensor([plane_side_1], device=device, dtype=torch.float32),
            )
            if not bool(plane_ok[0].item()):
                continue
            j_pos, _ = position_jacobian_batch(tracker.robot, q_t, create_graph=False)
            mu = directional_manipulability_batch(
                j_pos,
                torch.from_numpy(direction_1).to(device=device).view(1, 3),
                tracker.config.damping,
            )
            if float(mu.item()) <= tracker.config.mu_threshold:
                continue

            pos_err = float(np.linalg.norm(tcp_pos_t[0].detach().cpu().numpy() - plane_point_1))
            if pos_err > 1e-3:
                continue

            q_solutions.append(q_sol)
            seen.append(q_sol.copy())
            found_this_pose += 1
            print(
                f"[ik] collected={len(q_solutions)}/{num_starts} trial={trial + 1}/{ik_max_trials} "
                f"tilt={np.rad2deg(tilt_angle):.2f}deg roll={roll:.3f} rad"
            )
            if len(q_solutions) >= num_starts:
                break
        if len(q_solutions) >= num_starts:
            break

    if len(q_solutions) < num_starts:
        raise RuntimeError(f"Only found {len(q_solutions)} valid IK starts for the fixed task, need {num_starts}.")

    q_batch = torch.from_numpy(np.stack(q_solutions, axis=0)).to(device=device, dtype=torch.float32)
    plane_point_batch = torch.from_numpy(np.repeat(plane_point_1[None, :], num_starts, axis=0)).to(device=device, dtype=torch.float32)
    direction_batch = torch.from_numpy(np.repeat(direction_1[None, :], num_starts, axis=0)).to(device=device, dtype=torch.float32)
    plane_normal_batch = torch.from_numpy(np.repeat(plane_normal_1[None, :], num_starts, axis=0)).to(device=device, dtype=torch.float32)
    plane_side_batch = torch.full((num_starts,), plane_side_1, device=device, dtype=torch.float32)
    return q_batch, plane_point_batch, direction_batch, plane_normal_batch, plane_side_batch


def visualize(task: dict, trajectories: list[dict], topk: int, bottomk: int) -> None:
    lengths = np.asarray([float(t["total_projected_length"]) for t in trajectories], dtype=np.float32)
    order = np.argsort(-lengths)

    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    plane_point = np.asarray(task["plane_point"], dtype=np.float32)
    plane_normal = np.asarray(task["plane_normal"], dtype=np.float32)
    direction = np.asarray(task["direction"], dtype=np.float32)

    plane_rotmat = rotation_matrix_from_normal(plane_normal)
    plane_center = plane_point + 0.35 * direction
    ossop.box(
        half_extents=(0.35, 0.35, 0.0005),
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=np.array([0.80, 0.85, 0.90], dtype=np.float32),
        alpha=0.25,
    ).attach_to(world.scene)
    ossop.sphere(pos=plane_point, radius=0.01,
                 rgb=np.array([0.0, 0.7, 1.0], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)
    ossop.arrow(spos=plane_point, epos=plane_point + direction * 0.20,
                rgb=np.array([1.0, 0.15, 0.15], dtype=np.float32),
                alpha=0.95).attach_to(world.scene)
    ossop.arrow(spos=plane_point, epos=plane_point + plane_normal * 0.20,
                rgb=np.array([0.10, 0.45, 1.0], dtype=np.float32),
                alpha=0.95).attach_to(world.scene)

    robot = PenFrankaResearch3(name="pen", enable_cc=True)

    top_indices = order[:topk]
    bottom_indices = order[-bottomk:]

    for idx in top_indices:
        traj = trajectories[int(idx)]
        q = np.asarray(traj["start_q"], dtype=np.float32)
        robot.goto_given_conf(q)
        robot.gen_meshmodel(rgb=np.array([0.10, 0.80, 0.20], dtype=np.float32),
                            alpha=0.45, toggle_tcp_frame=True).attach_to(world)
        end = plane_point + direction * float(traj["total_projected_length"])
        ossop.cylinder(spos=plane_point, epos=end, radius=0.0045,
                       rgb=np.array([0.10, 0.80, 0.20], dtype=np.float32),
                       alpha=0.95).attach_to(world.scene)

    for idx in bottom_indices:
        traj = trajectories[int(idx)]
        q = np.asarray(traj["start_q"], dtype=np.float32)
        robot.goto_given_conf(q)
        robot.gen_meshmodel(rgb=np.array([0.90, 0.18, 0.18], dtype=np.float32),
                            alpha=0.25, toggle_tcp_frame=False).attach_to(world)
        end = plane_point + direction * float(traj["total_projected_length"])
        ossop.cylinder(spos=plane_point, epos=end, radius=0.0035,
                       rgb=np.array([0.90, 0.18, 0.18], dtype=np.float32),
                       alpha=0.80).attach_to(world.scene)

    world.run()


def _svg_polyline(points: list[tuple[float, float]], color: str, dashed: bool, width: float = 2.0) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = ' stroke-dasharray="7 5"' if dashed else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width:.2f}"{dash_attr} points="{pts}" />'


def _traj_q_path(traj: dict) -> np.ndarray:
    if "q" in traj:
        return np.asarray(traj["q"], dtype=np.float32)
    if "q_path" in traj:
        return np.asarray(traj["q_path"], dtype=np.float32)
    raise KeyError("Trajectory does not contain 'q' or 'q_path'.")


def export_joint_curve_svg(trajectories: list[dict], topk: int, bottomk: int, out_path: Path) -> None:
    lengths = np.asarray([float(t["total_projected_length"]) for t in trajectories], dtype=np.float32)
    order = np.argsort(-lengths)
    top_indices = [int(i) for i in order[:topk]]
    bottom_indices = [int(i) for i in order[-bottomk:][::-1]]

    selected = top_indices + bottom_indices
    q_paths = [_traj_q_path(trajectories[i]) for i in selected]
    n_joints = q_paths[0].shape[1]
    max_steps = max(q.shape[0] for q in q_paths) - 1

    colors_top = ["#1b9e77", "#2ca25f", "#66c2a4"]
    colors_bottom = ["#d95f02", "#e6550d", "#fdae6b"]

    width = 1600
    row_h = 170
    left = 90
    right = 30
    top = 40
    bottom = 35
    plot_w = width - left - right
    total_h = top + n_joints * row_h + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" viewBox="0 0 {width} {total_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .small{font-size:12px} .mid{font-size:14px} .title{font-size:18px;font-weight:bold}</style>',
        f'<text x="{left}" y="24" class="title">Same-Task Start Configuration Joint Trajectories</text>',
        f'<text x="{left}" y="44" class="small">Top {topk}: solid green lines | Bottom {bottomk}: dashed orange lines | x-axis: rollout step</text>',
    ]

    global_min = min(float(q[:, j].min()) for q in q_paths for j in range(n_joints))
    global_max = max(float(q[:, j].max()) for q in q_paths for j in range(n_joints))
    if abs(global_max - global_min) < 1e-6:
        global_max = global_min + 1.0

    for j in range(n_joints):
        y0 = top + j * row_h
        plot_h = row_h - 45
        y1 = y0 + plot_h

        joint_vals = [q[:, j] for q in q_paths]
        vmin = min(float(v.min()) for v in joint_vals)
        vmax = max(float(v.max()) for v in joint_vals)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        pad = 0.08 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        parts.append(f'<text x="18" y="{y0 + 18:.2f}" class="mid">Joint {j + 1}</text>')
        parts.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#cccccc" stroke-width="1"/>')

        for frac in (0.0, 0.5, 1.0):
            yy = y1 - frac * plot_h
            val = vmin + frac * (vmax - vmin)
            parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#eeeeee" stroke-width="1"/>')
            parts.append(f'<text x="24" y="{yy + 4:.2f}" class="small">{val:.2f}</text>')

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            xx = left + frac * plot_w
            step_val = int(round(frac * max_steps))
            parts.append(f'<line x1="{xx:.2f}" y1="{y0}" x2="{xx:.2f}" y2="{y1}" stroke="#f2f2f2" stroke-width="1"/>')
            parts.append(f'<text x="{xx - 8:.2f}" y="{y1 + 16:.2f}" class="small">{step_val}</text>')

        for rank, idx in enumerate(top_indices):
            q = _traj_q_path(trajectories[idx])
            pts = []
            denom = max(q.shape[0] - 1, 1)
            for step_i, val in enumerate(q[:, j]):
                xx = left + plot_w * (step_i / max(max_steps, 1))
                yy = y1 - plot_h * ((float(val) - vmin) / (vmax - vmin))
                pts.append((xx, yy))
            parts.append(_svg_polyline(pts, colors_top[rank % len(colors_top)], dashed=False))

        for rank, idx in enumerate(bottom_indices):
            q = _traj_q_path(trajectories[idx])
            pts = []
            for step_i, val in enumerate(q[:, j]):
                xx = left + plot_w * (step_i / max(max_steps, 1))
                yy = y1 - plot_h * ((float(val) - vmin) / (vmax - vmin))
                pts.append((xx, yy))
            parts.append(_svg_polyline(pts, colors_bottom[rank % len(colors_bottom)], dashed=True))

    legend_y = total_h - 12
    parts.append(f'<line x1="{left}" y1="{legend_y - 5}" x2="{left + 34}" y2="{legend_y - 5}" stroke="#1b9e77" stroke-width="2"/>')
    parts.append(f'<text x="{left + 42}" y="{legend_y - 1}" class="small">Good starts (top-{topk})</text>')
    parts.append(f'<line x1="{left + 210}" y1="{legend_y - 5}" x2="{left + 244}" y2="{legend_y - 5}" stroke="#d95f02" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{left + 252}" y="{legend_y - 1}" class="small">Bad starts (bottom-{bottomk})</text>')
    parts.append('</svg>')

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[curve] wrote joint trajectory plot to {out_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tracker = build_tracker(torch.device(args.device), args)

    q0, plane_point, direction, plane_normal, plane_side = sample_same_task_starts(
        tracker=tracker,
        device=torch.device(args.device),
        num_starts=int(args.num_starts),
        seed=int(args.seed),
        ik_max_trials=int(args.ik_max_trials),
        ik_seeds_per_pose=int(args.ik_seeds_per_pose),
    )
    trajectories = tracker.collect_batch_trajectories(q0, plane_point, direction, plane_normal, plane_side)

    lengths = np.asarray([float(t["total_projected_length"]) for t in trajectories], dtype=np.float32)
    task = {
        "plane_point": trajectories[0]["plane_point"],
        "plane_normal": trajectories[0]["plane_normal"],
        "direction": trajectories[0]["direction"],
    }

    print("=== Fixed Task ===")
    print(f"plane_point  = {np.array2string(task['plane_point'], precision=4, suppress_small=True)}")
    print(f"plane_normal = {np.array2string(task['plane_normal'], precision=4, suppress_small=True)}")
    print(f"direction    = {np.array2string(task['direction'], precision=4, suppress_small=True)}")
    print("")
    summarize(lengths)
    print_ranked_cases(trajectories, int(args.show_topk), int(args.show_bottomk))
    export_joint_curve_svg(
        trajectories=trajectories,
        topk=int(args.show_topk),
        bottomk=int(args.show_bottomk),
        out_path=args.curve_out,
    )

    if not args.no_vis:
        visualize(task, trajectories, int(args.show_topk), int(args.show_bottomk))


if __name__ == "__main__":
    main()
