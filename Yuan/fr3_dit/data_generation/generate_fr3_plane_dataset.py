#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers
from Yuan.fr3_dit.core.pen_fr3_robot import PEN_LENGTH, PenFrankaResearch3GPU


# DEFAULT_URDF is kept for backward compatibility with downstream importers
# (e.g. eval_tracker.py). The new collision pipeline does not use a URDF: it
# pulls FR3 sphere data straight from ``one.robots.manipulators.franka.fr3``.
DEFAULT_URDF = None
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_plane_trajectories.hdf5"


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_unit_vectors_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    return normalize_batch(torch.randn(batch_size, 3, device=device))


def project_to_plane_batch(v: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    normal = normalize_batch(normal)
    v_plane = v - torch.sum(v * normal, dim=-1, keepdim=True) * normal
    tiny = v_plane.norm(dim=-1, keepdim=True) < 1e-8
    if tiny.any():
        fallback = torch.zeros_like(v_plane)
        fallback[..., 0] = 1.0
        fallback = fallback - torch.sum(fallback * normal, dim=-1, keepdim=True) * normal
        v_plane = torch.where(tiny, fallback, v_plane)
    return normalize_batch(v_plane)


def position_jacobian_batch(robot, q_batch: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
    q_eval = q_batch.detach().clone().requires_grad_(True)
    tcp_pos, _ = robot.fk_batch(q_eval)
    grads = []
    for dim in range(3):
        grads.append(torch.autograd.grad(tcp_pos[:, dim].sum(), q_eval, retain_graph=True, create_graph=create_graph)[0])
    return torch.stack(grads, dim=1), q_eval


def damped_pseudoinverse_batch(j: torch.Tensor, damping: float) -> torch.Tensor:
    batch, task_dim, _ = j.shape
    eye = torch.eye(task_dim, device=j.device, dtype=j.dtype).unsqueeze(0).expand(batch, -1, -1)
    return j.transpose(1, 2) @ torch.linalg.inv(j @ j.transpose(1, 2) + (damping ** 2) * eye)


def nullspace_projector_batch(j: torch.Tensor, damping: float) -> torch.Tensor:
    batch, _, dof = j.shape
    eye = torch.eye(dof, device=j.device, dtype=j.dtype).unsqueeze(0).expand(batch, -1, -1)
    return eye - damped_pseudoinverse_batch(j, damping) @ j


def directional_manipulability_batch(j_pos: torch.Tensor, direction: torch.Tensor, damping: float) -> torch.Tensor:
    batch = j_pos.shape[0]
    eye = torch.eye(3, device=j_pos.device, dtype=j_pos.dtype).unsqueeze(0).expand(batch, -1, -1)
    metric = j_pos @ j_pos.transpose(1, 2) + (damping ** 2) * eye
    d = direction.unsqueeze(-1)
    return (d.transpose(1, 2) @ torch.linalg.inv(metric) @ d).squeeze(-1).squeeze(-1).clamp_min(1e-12).pow(-0.5)


def joint_margin_mask(robot, q_batch: torch.Tensor, margin_ratio: float) -> torch.Tensor:
    lower = robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = robot.jnt_ranges[:, 1].unsqueeze(0)
    span = upper - lower
    inner_lower = lower + margin_ratio * span
    inner_upper = upper - margin_ratio * span
    return ((q_batch >= inner_lower) & (q_batch <= inner_upper)).all(dim=1)


def sample_normals_near_tcp_z(tcp_z: torch.Tensor, max_angle_deg: float) -> torch.Tensor:
    batch = tcp_z.shape[0]
    noise = random_unit_vectors_batch(batch, tcp_z.device)
    tangent = project_to_plane_batch(noise, tcp_z)
    angles = torch.rand(batch, device=tcp_z.device) * np.deg2rad(max_angle_deg)
    return normalize_batch(torch.cos(angles).unsqueeze(1) * tcp_z + torch.sin(angles).unsqueeze(1) * tangent)


def termination_label(code: int) -> str:
    labels = {
        0: "low_mu",
        1: "joint_margin",
        2: "self_collision",
        3: "max_steps",
        4: "pos_tracking_error",
        5: "plane_clearance",
        6: "angle_violation",
    }
    return labels.get(int(code), "unknown")


@dataclass
class TrackerConfig:
    dt: float = 0.01
    task_speed: float = 0.1
    damping: float = 1e-3
    null_gain: float = 0.6
    joint_limit_gain: float = 0.2
    theta_max_deg: float = 30.0
    angle_margin_deg: float = 8.0
    angle_null_gain: float = 0.4
    angle_attract_gain: float = 0.0
    joint_margin_ratio: float = 0.05
    boundary_gain: float = 10.0
    max_steps: int = 2000
    mu_threshold: float = 0.01
    pos_error_threshold: float = 0.01


@dataclass
class DeskConfig:
    """Fixed desk plane shared by every sampled trajectory.

    Convention: ``normal`` is the physical OUTWARD normal of the tabletop (the side the
    robot reaches from). The pen is driven to point opposite ``normal``. ``center`` MUST
    sit below every non-pen robot link so that plane-clearance passes; for a base-at-origin
    FR3 that means ``center[2] <= 0``.
    """
    center: tuple[float, float, float] = (0.5, 0.0, -0.05)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    x_half: float = 0.20
    y_half: float = 0.20
    pos_tol: float = 0.02         # max TCP-to-desk offset accepted at start (m)


class PlaneConstrainedTracker:
    def __init__(self, robot, self_collision_fn, sphere_positions_fn, sphere_radii: np.ndarray, sphere_link_indices: np.ndarray, config: TrackerConfig):
        self.robot = robot
        self.self_collision_fn = self_collision_fn
        self.sphere_positions_fn = sphere_positions_fn
        tensor_device = robot.device
        self.sphere_radii = torch.tensor(sphere_radii, dtype=torch.float32, device=tensor_device)
        self.sphere_link_indices = torch.tensor(sphere_link_indices, dtype=torch.long, device=tensor_device)
        self.config = config
        self.theta_cos = float(np.cos(np.deg2rad(config.theta_max_deg)))
        self.theta_margin_cos = float(np.cos(np.deg2rad(config.theta_max_deg + config.angle_margin_deg)))
        max_link_index = int(np.max(sphere_link_indices))
        protected_from_plane = max(2, max_link_index - 1)
        self.keep_mask = self.sphere_link_indices < protected_from_plane

    def plane_clearance_mask(self, q_batch: torch.Tensor, plane_point: torch.Tensor, plane_normal: torch.Tensor, plane_side: torch.Tensor) -> torch.Tensor:
        sphere_pos = self.sphere_positions_fn(q_batch)
        signed = torch.sum((sphere_pos - plane_point[:, None, :]) * plane_normal[:, None, :], dim=-1)
        signed = signed * plane_side[:, None]
        required = self.sphere_radii.unsqueeze(0)
        return (signed[:, self.keep_mask] >= required[:, self.keep_mask]).all(dim=1)

    def sample_valid_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_list, p_list, d_list, n_list, s_list = [], [], [], [], []
        remaining = batch_size
        oversample = max(256, batch_size * 8)
        trial = 0
        while remaining > 0:
            trial += 1
            q = self.robot.rand_conf_batch(oversample).to(device)
            q = q[joint_margin_mask(self.robot, q, self.config.joint_margin_ratio)]
            if q.shape[0] == 0:
                print(f"[sample] trial={trial} after-margin=0/{oversample}")
                continue
            tcp_pos, tcp_rot = self.robot.fk_batch(q)
            tcp_z = tcp_rot[:, :, 2]
            normals = sample_normals_near_tcp_z(tcp_z, self.config.theta_max_deg)
            directions = project_to_plane_batch(torch.randn_like(normals), normals)
            j_pos, _ = position_jacobian_batch(self.robot, q, create_graph=False)
            mu = directional_manipulability_batch(j_pos, directions, self.config.damping)
            coll = self.self_collision_fn(q)
            plane_point = tcp_pos
            plane_side = -torch.ones(q.shape[0], device=device, dtype=torch.float32)
            plane_ok = self.plane_clearance_mask(q, plane_point, normals, plane_side)
            valid = (mu > self.config.mu_threshold) & (coll <= 0.0) & plane_ok
            print(
                f"[sample] trial={trial} after-margin={q.shape[0]}/{oversample} "
                f"plane-ok={int(plane_ok.sum().item())} valid={int(valid.sum().item())} "
                f"collected={batch_size - remaining}/{batch_size}"
            )
            if valid.any():
                take = min(int(valid.sum().item()), remaining)
                idx = torch.where(valid)[0][:take]
                q_list.append(q[idx])
                p_list.append(plane_point[idx])
                d_list.append(directions[idx])
                n_list.append(normals[idx])
                s_list.append(plane_side[idx])
                remaining -= take
                print(f"[sample] collected={batch_size - remaining}/{batch_size} valid starts")
        return (
            torch.cat(q_list, dim=0),
            torch.cat(p_list, dim=0),
            torch.cat(d_list, dim=0),
            torch.cat(n_list, dim=0),
            torch.cat(s_list, dim=0),
        )

    def sample_desk_valid_batch(
        self, batch_size: int, desk: DeskConfig, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rejection-sample valid starts on a single shared desk plane.

        Convention: ``desk.normal`` is the physical OUTWARD normal (points away from the
        desk surface, e.g. +Z for a horizontal tabletop). The pen points INTO the desk,
        so TCP_z is driven toward ``-desk.normal``. Internally we store
        ``plane_normal = -desk.normal`` (the pen-axis convention used by the tracker),
        which together with the existing ``plane_side = -1`` keeps the arm on the
        +desk.normal side (above the tabletop).
        """
        desk_center_t = torch.tensor(desk.center, dtype=torch.float32, device=device)
        desk_normal_t = torch.tensor(desk.normal, dtype=torch.float32, device=device)
        desk_normal_t = desk_normal_t / desk_normal_t.norm().clamp_min(1e-12)
        pen_axis = -desk_normal_t  # pen points INTO the desk
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
        if abs(float(desk_normal_t[0])) >= 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        dx = torch.linalg.cross(helper, desk_normal_t); dx = dx / dx.norm().clamp_min(1e-12)
        dy = torch.linalg.cross(desk_normal_t, dx)
        cos_theta_max = float(np.cos(np.deg2rad(self.config.theta_max_deg)))

        q_list, p_list, d_list = [], [], []
        remaining = batch_size
        oversample = max(4096, batch_size * 128)
        trial = 0
        while remaining > 0:
            trial += 1
            q = self.robot.rand_conf_batch(oversample).to(device)
            q = q[joint_margin_mask(self.robot, q, self.config.joint_margin_ratio)]
            if q.shape[0] == 0:
                print(f"[desk-sample] trial={trial} after-margin=0/{oversample}")
                continue
            tcp_pos, tcp_rot = self.robot.fk_batch(q)
            tcp_z = tcp_rot[:, :, 2]
            cos_theta = torch.sum(tcp_z * pen_axis.unsqueeze(0), dim=-1)
            aligned = cos_theta > cos_theta_max

            offset = tcp_pos - desk_center_t.unsqueeze(0)
            plane_dist = torch.sum(offset * desk_normal_t.unsqueeze(0), dim=-1)
            lx = torch.sum(offset * dx.unsqueeze(0), dim=-1)
            ly = torch.sum(offset * dy.unsqueeze(0), dim=-1)
            on_desk = (
                (plane_dist.abs() < desk.pos_tol)
                & (lx.abs() < desk.x_half)
                & (ly.abs() < desk.y_half)
            )

            # Early filter: keep only configs that pass the cheap gates before running
            # expensive collision + plane-clearance checks (100× speed-up at typical yields).
            pre_mask = aligned & on_desk
            if not pre_mask.any():
                print(f"[desk-sample] trial={trial} after-margin={q.shape[0]}/{oversample} aligned/on_desk=0")
                continue
            q_f = q[pre_mask]
            lx_f, ly_f = lx[pre_mask], ly[pre_mask]
            plane_point = (
                desk_center_t.unsqueeze(0)
                + lx_f.unsqueeze(-1) * dx.unsqueeze(0)
                + ly_f.unsqueeze(-1) * dy.unsqueeze(0)
            )
            pen_axis_batch = pen_axis.unsqueeze(0).expand(q_f.shape[0], 3)
            directions = project_to_plane_batch(torch.randn_like(pen_axis_batch), pen_axis_batch)
            coll = self.self_collision_fn(q_f)
            plane_side = -torch.ones(q_f.shape[0], device=device, dtype=torch.float32)
            plane_ok = self.plane_clearance_mask(q_f, plane_point, pen_axis_batch, plane_side)

            # Only collision-level gating (on-desk / alignment already applied above).
            valid = (coll <= 0.0) & plane_ok
            print(
                f"[desk-sample] trial={trial} margin={q.shape[0]}/{oversample} "
                f"pre={q_f.shape[0]} coll_ok={int((coll<=0).sum().item())} "
                f"plane_ok={int(plane_ok.sum().item())} valid={int(valid.sum().item())} "
                f"collected={batch_size - remaining}/{batch_size}"
            )
            if valid.any():
                take = min(int(valid.sum().item()), remaining)
                idx = torch.where(valid)[0][:take]
                q_list.append(q_f[idx])
                p_list.append(plane_point[idx])
                d_list.append(directions[idx])
                remaining -= take
        n_total = sum(t.shape[0] for t in q_list)
        return (
            torch.cat(q_list, dim=0),
            torch.cat(p_list, dim=0),
            torch.cat(d_list, dim=0),
            pen_axis.unsqueeze(0).expand(n_total, 3).contiguous(),
            -torch.ones(n_total, dtype=torch.float32, device=device),
        )

    def collect_batch_trajectories(
        self,
        q0_batch: torch.Tensor,
        plane_point_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        plane_normal_batch: torch.Tensor,
        plane_side_batch: torch.Tensor,
    ) -> list[dict]:
        q = q0_batch.clone()
        direction = project_to_plane_batch(direction_batch, plane_normal_batch)
        start_pos, _ = self.robot.fk_batch(q)
        batch_size = q.shape[0]
        print(f"[rollout] batch_size={batch_size} max_steps={self.config.max_steps}")
        active = torch.ones(batch_size, dtype=torch.bool, device=q.device)
        termination = torch.full((batch_size,), 3, dtype=torch.long, device=q.device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=q.device)

        q_hist = torch.empty((self.config.max_steps + 1, batch_size, q.shape[1]), device=q.device)
        tcp_hist = torch.empty((self.config.max_steps + 1, batch_size, 3), device=q.device)
        q_hist[0] = q
        tcp_hist[0] = start_pos

        for step_idx in range(self.config.max_steps):
            if not active.any():
                break

            q_eval = q.detach().clone().requires_grad_(True)
            tcp_pos, tcp_rot = self.robot.fk_batch(q_eval)
            tcp_z = tcp_rot[:, :, 2]
            cos_theta = torch.sum(tcp_z * plane_normal_batch, dim=-1)

            j_pos, _ = position_jacobian_batch(self.robot, q_eval, create_graph=True)
            j_g = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=True, create_graph=False)[0].unsqueeze(1)
            mu = directional_manipulability_batch(j_pos, direction, self.config.damping)
            low_mu = active & (mu < self.config.mu_threshold)
            termination[low_mu] = 0
            active = active & (~low_mu)
            if not active.any():
                break

            grad_mu = torch.autograd.grad(
                mu.sum(),
                q_eval,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_mu is None:
                grad_mu = torch.zeros_like(q_eval)
            grad_cos = torch.autograd.grad(
                cos_theta.sum(),
                q_eval,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_cos is None:
                grad_cos = torch.zeros_like(q_eval)
            on_boundary = (cos_theta <= self.theta_cos).view(-1, 1, 1)
            j_task = torch.cat([j_pos, j_g * on_boundary], dim=1)
            j_pinv = damped_pseudoinverse_batch(j_task, self.config.damping)
            projector = nullspace_projector_batch(j_task, self.config.damping)

            v_pos = (self.config.task_speed * direction).unsqueeze(-1)
            v_ang = (self.config.boundary_gain * torch.clamp(self.theta_cos - cos_theta, min=0.0)).view(-1, 1, 1)
            v_task = torch.cat([v_pos, v_ang * on_boundary.squeeze(-1).float().unsqueeze(-1)], dim=1)
            q_dot_task = (j_pinv @ v_task).squeeze(-1)
            lower = self.robot.jnt_ranges[:, 0].unsqueeze(0)
            upper = self.robot.jnt_ranges[:, 1].unsqueeze(0)
            center = 0.5 * (lower + upper)
            span = (upper - lower).clamp_min(1e-6)
            q_dot_joint = -self.config.joint_limit_gain * (q_eval.detach() - center) / span
            # Boundary gate: ramps 0→1 between margin and theta_cos. Brakes when near/past cone edge.
            boundary_gate = torch.clamp(
                (self.theta_margin_cos - cos_theta) / max(self.theta_margin_cos - self.theta_cos, 1e-6),
                min=0.0,
                max=1.0,
            ).unsqueeze(-1)
            # Interior attractor: always-on pull toward perfect alignment, scales with deviation.
            # Off by default (gain=0) to preserve original data-gen behavior; enable at eval/viz time.
            # Use the angle in radians (≈ θ) instead of (1-cos θ ≈ θ²/2). Linear gate gives
            # a meaningful pull even at small deviations (mid-cone), where the quadratic form
            # is essentially zero and lets drift accumulate.
            theta_rad = torch.acos(torch.clamp(cos_theta, min=-1.0 + 1e-6, max=1.0 - 1e-6))
            interior_gate = theta_rad.unsqueeze(-1)
            q_dot_angle = (
                self.config.angle_null_gain * boundary_gate
                + self.config.angle_attract_gain * interior_gate
            ) * grad_cos
            q_dot_null = self.config.null_gain * grad_mu + q_dot_joint + q_dot_angle
            q_dot = q_dot_task + (projector @ q_dot_null.unsqueeze(-1)).squeeze(-1)
            q_next = q + self.config.dt * q_dot

            margin_ok = joint_margin_mask(self.robot, q_next, self.config.joint_margin_ratio)
            fail_margin = active & (~margin_ok)
            termination[fail_margin] = 1

            coll = self.self_collision_fn(q_next)
            fail_coll = active & margin_ok & (coll > 0.0)
            termination[fail_coll] = 2

            plane_ok = self.plane_clearance_mask(q_next, plane_point_batch, plane_normal_batch, plane_side_batch)
            fail_plane = active & margin_ok & (~fail_coll) & (~plane_ok)
            termination[fail_plane] = 5

            tcp_pos_next, tcp_rot_next = self.robot.fk_batch(q_next)
            cos_theta_next = torch.sum(tcp_rot_next[:, :, 2] * plane_normal_batch, dim=-1)
            fail_angle = active & margin_ok & (~fail_coll) & (~fail_plane) & (cos_theta_next < self.theta_cos)
            termination[fail_angle] = 6

            expected_pos = start_pos + direction * ((step_idx + 1) * self.config.dt * self.config.task_speed)
            pos_err = torch.linalg.norm(tcp_pos_next - expected_pos, dim=1)
            fail_pos = active & margin_ok & (~fail_coll) & (~fail_plane) & (~fail_angle) & (pos_err > self.config.pos_error_threshold)
            termination[fail_pos] = 4

            advance = active & margin_ok & (~fail_coll) & (~fail_plane) & (~fail_angle) & (~fail_pos)
            q[advance] = q_next[advance]
            steps[advance] += 1
            active = advance

            if (step_idx + 1) % 100 == 0 or not active.any():
                print(
                    f"[rollout] step={step_idx + 1} active={int(active.sum().item())}/{batch_size} "
                    f"mean_step={float(steps.float().mean().item()):.1f}"
                )

            tcp_now, _ = self.robot.fk_batch(q)
            q_hist[step_idx + 1] = q
            tcp_hist[step_idx + 1] = tcp_now

        trajectories = []
        for i in range(batch_size):
            num_points = int(steps[i].item()) + 1
            q_path = q_hist[:num_points, i].detach().cpu().numpy().astype(np.float32)
            tcp_path = tcp_hist[:num_points, i].detach().cpu().numpy().astype(np.float32)
            progress = np.sum((tcp_path - tcp_path[0]) * direction[i].detach().cpu().numpy()[None, :], axis=1).astype(np.float32)
            trajectories.append(
                {
                    "start_q": q0_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_point": plane_point_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_normal": plane_normal_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_side": float(plane_side_batch[i].item()),
                    "direction": direction[i].detach().cpu().numpy().astype(np.float32),
                    "termination_code": int(termination[i].item()),
                    "termination_reason": termination_label(int(termination[i].item())),
                    "num_points": num_points,
                    "total_projected_length": float(progress[-1]),
                    "q": q_path,
                    "tcp_pos": tcp_path,
                    "progress_length": progress,
                }
            )
        return trajectories


def init_hdf5(path: Path, config: TrackerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["dt"] = config.dt
        f.attrs["task_speed"] = config.task_speed
        f.attrs["theta_max_deg"] = config.theta_max_deg
        f.attrs["angle_margin_deg"] = config.angle_margin_deg
        f.attrs["angle_null_gain"] = config.angle_null_gain
        f.attrs["joint_margin_ratio"] = config.joint_margin_ratio
        f.attrs["plane_clearance_m"] = 0.0
        f.attrs["robot_name"] = "pen_fr3"
        f.attrs["pen_length_m"] = PEN_LENGTH
        f.attrs["plane_collision_policy"] = "plane_through_pen_tcp_keep_arm_on_negative_normal_side_ignore_distal_two_links"
        f.attrs["num_trajectories"] = 0


def append_trajectories_hdf5(path: Path, trajectories: list[dict]) -> int:
    with h5py.File(path, "a") as f:
        start_idx = int(f.attrs.get("num_trajectories", 0))
        for offset, traj in enumerate(trajectories):
            g = f.create_group(f"traj_{start_idx + offset:06d}")
            for key, value in traj.items():
                if isinstance(value, str):
                    g.attrs[key] = value
                elif np.isscalar(value):
                    g.attrs[key] = value
                else:
                    g.create_dataset(key, data=value, compression="gzip")
        f.attrs["num_trajectories"] = start_idx + len(trajectories)
        return int(f.attrs["num_trajectories"])


def robot_visualization_test() -> None:
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3

    base = ovw.World(cam_pos=[2.0, -1.8, 1.2], cam_lookat_pos=[0.2, 0.0, 0.4])
    ossop.frame().attach_to(base.scene)

    robot = PenFrankaResearch3(name="pen", enable_cc=True)
    robot.gen_meshmodel(alpha=0.6, toggle_tcp_frame=True).attach_to(base)

    flange_pos = robot.manipulator.gl_flange_pos
    flange_rotmat = robot.manipulator.gl_flange_rotmat
    pen_tip = flange_pos + flange_rotmat[:, 2] * PEN_LENGTH
    ossop.frame(pos=pen_tip, rotmat=flange_rotmat, length_scale=0.5).attach_to(base.scene)

    print(f"[robot-vis] pen length = {PEN_LENGTH:.3f} m")
    print(f"[robot-vis] flange_pos = {np.array2string(flange_pos, precision=4, suppress_small=True)}")
    print(f"[robot-vis] pen_tip    = {np.array2string(pen_tip, precision=4, suppress_small=True)}")
    base.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FR3 plane-constrained straight-line trajectories.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-max-deg", type=float, default=45.0)
    parser.add_argument("--angle-margin-deg", type=float, default=15.0)
    parser.add_argument("--angle-null-gain", type=float, default=0.4)
    parser.add_argument("--joint-margin-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    # Fixed-desk (IK-targeted) sampling is the default.
    parser.add_argument("--random-plane", action="store_true",
                        help="Legacy: per-trajectory random plane (rejection sampler).")
    parser.add_argument("--desk-center", type=float, nargs=3, default=[0.5, 0.0, -0.05],
                        metavar=("CX", "CY", "CZ"),
                        help="World-frame center of the shared desk plane (meters). CZ must sit "
                             "below every non-pen robot link for plane-clearance to pass.")
    parser.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        metavar=("NX", "NY", "NZ"),
                        help="World-frame normal of the shared desk plane.")
    parser.add_argument("--desk-x-half", type=float, default=0.20,
                        help="Half-width along the desk's local x-axis (meters).")
    parser.add_argument("--desk-y-half", type=float, default=0.20,
                        help="Half-width along the desk's local y-axis (meters).")
    parser.add_argument("--desk-pos-tol", type=float, default=0.02,
                        help="Max TCP-to-desk offset accepted at start (meters).")
    parser.add_argument("--min-length-m", type=float, default=0.3,
                        help="Drop trajectories whose total_projected_length is below this (meters). "
                             "The save counter only advances on trajectories that pass.")
    parser.add_argument("--robot-vis-test", action="store_true", help="Only visualize the pen robot and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.robot_vis_test:
        robot_visualization_test()
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    fr3 = PenFrankaResearch3GPU(device)
    helpers = make_pen_collision_helpers(fr3.robot)

    tracker = PlaneConstrainedTracker(
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

    desk = DeskConfig(
        center=tuple(args.desk_center),
        normal=tuple(args.desk_normal),
        x_half=float(args.desk_x_half),
        y_half=float(args.desk_y_half),
        pos_tol=float(args.desk_pos_tol),
    )

    init_hdf5(args.output, tracker.config)
    with h5py.File(args.output, "a") as f:
        if args.random_plane:
            f.attrs["sampling_mode"] = "random_plane"
        else:
            f.attrs["sampling_mode"] = "fixed_desk"
            f.attrs["desk_center"] = np.asarray(desk.center, dtype=np.float32)
            f.attrs["desk_normal"] = np.asarray(desk.normal, dtype=np.float32)
            f.attrs["desk_x_half"] = desk.x_half
            f.attrs["desk_y_half"] = desk.y_half
            f.attrs["desk_pos_tol"] = desk.pos_tol

    total_written = 0
    batch_idx = 0
    while total_written < int(args.num_trajectories):
        batch_idx += 1
        remaining = int(args.num_trajectories) - total_written
        take = min(int(args.batch_size), remaining)
        print(f"[batch] {batch_idx} target_batch={take} collected_total={total_written}/{int(args.num_trajectories)}")
        if args.random_plane:
            q0, plane_point, direction, plane_normal, plane_side = tracker.sample_valid_batch(take, device)
        else:
            q0, plane_point, direction, plane_normal, plane_side = tracker.sample_desk_valid_batch(take, desk, device)
        batch_trajs = tracker.collect_batch_trajectories(q0, plane_point, direction, plane_normal, plane_side)
        batch_lengths_all = np.asarray([float(t["total_projected_length"]) for t in batch_trajs], dtype=np.float32)
        # Length filter: only keep trajectories at least --min-length-m meters long.
        kept = [t for t in batch_trajs if float(t["total_projected_length"]) >= args.min_length_m]
        dropped = len(batch_trajs) - len(kept)
        if kept:
            batch_lengths = np.asarray([float(t["total_projected_length"]) for t in kept], dtype=np.float32)
            print(
                f"[length] batch={batch_idx} kept={len(kept)}/{len(batch_trajs)} "
                f"(dropped {dropped} < {args.min_length_m}m) "
                f"mean={float(batch_lengths.mean()):.4f}m median={float(np.median(batch_lengths)):.4f}m "
                f"min={float(batch_lengths.min()):.4f}m max={float(batch_lengths.max()):.4f}m"
            )
        else:
            print(
                f"[length] batch={batch_idx} kept=0/{len(batch_trajs)} "
                f"(all dropped < {args.min_length_m}m; raw max={float(batch_lengths_all.max()):.4f}m)"
            )
        if kept:
            total_written = append_trajectories_hdf5(args.output, kept)
        print(f"[save] collected={total_written}/{int(args.num_trajectories)}")

    print(f"[done] wrote {total_written} trajectories to {args.output}")


if __name__ == "__main__":
    main()
