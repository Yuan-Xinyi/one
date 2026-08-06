"""Fail-closed deployment for one continuously generated seed.

Each task follows one fixed route:

``DIRECT``
    The generated joint vector already passes the complete hard validity gate.
``REFINED``
    The generated vector is used as the warm start of exactly one batched,
    z-axis IK projection with ``preserve_seed=True``.
``FALLBACK``
    Direct and refined routes failed; return the supplied fallback bit-exactly.
``INVALID``
    Even the supplied fallback is unsafe.  The returned tensor is still the
    bit-exact fallback, but ``valid=False`` tells the caller not to execute it.

There is no candidate enumeration, model rollout, alpha search, or retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math
import torch

from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.unified_rl.direct_seed_model import direct_seed_task


ROUTE_DIRECT = 0
ROUTE_REFINED = 1
ROUTE_FALLBACK = 2
ROUTE_INVALID = 3
ROUTE_NAMES = ('direct', 'refined', 'fallback', 'invalid')
_UNIT_TOL = 1e-4


@dataclass(frozen=True)
class DirectSeedProjectionConfig:
    """Hard-gate and one-shot projection settings."""

    position_tol_m: float = 5e-3
    cone_deg: float = 30.0
    projection_cone_deg: float = 24.5
    joint_margin_rad: float = 0.02
    collision_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if (not math.isfinite(self.position_tol_m)
                or self.position_tol_m <= 0.0):
            raise ValueError('position_tol_m must be finite and positive')
        if (not math.isfinite(self.cone_deg)
                or not 0.0 < self.cone_deg <= 30.0):
            raise ValueError('cone_deg must be finite and in (0, 30]')
        if (not math.isfinite(self.projection_cone_deg)
                or self.projection_cone_deg <= 0.0):
            raise ValueError(
                'projection_cone_deg must be finite and positive')
        # The shared IK solver declares z-axis convergence at 5 degrees.
        # Keep another 0.5-degree numerical buffer so solver success implies a
        # target no farther than 29.5 degrees from the 30-degree hard cone.
        if self.projection_cone_deg + 5.0 > self.cone_deg - 0.5 + 1e-12:
            raise ValueError(
                'projection_cone_deg + 5deg IK tolerance must leave a '
                '0.5deg buffer inside cone_deg')
        if (not math.isfinite(self.joint_margin_rad)
                or self.joint_margin_rad < 0.0):
            raise ValueError(
                'joint_margin_rad must be finite and non-negative')
        if (not math.isfinite(self.collision_margin_m)
                or self.collision_margin_m < 0.0):
            raise ValueError(
                'collision_margin_m must be finite and non-negative')


@dataclass(frozen=True)
class StrictSeedValidity:
    """Per-row hard validity diagnostics."""

    valid: torch.Tensor
    input_valid: torch.Tensor
    finite: torch.Tensor
    joint_limits: torch.Tensor
    position: torch.Tensor
    cone: torch.Tensor
    collision_free: torch.Tensor
    position_error_m: torch.Tensor
    cone_cosine: torch.Tensor
    collision_margin_m: torch.Tensor
    joint_margin_rad: torch.Tensor
    z_tool: torch.Tensor


@dataclass(frozen=True)
class GeneratedSeedResult:
    """One-shot routing result and all audit fields."""

    q_raw: torch.Tensor
    q_projected: torch.Tensor
    q: torch.Tensor
    route: torch.Tensor
    valid: torch.Tensor
    ik_attempted: torch.Tensor
    ik_ok: torch.Tensor
    raw: StrictSeedValidity
    projected: StrictSeedValidity
    fallback: StrictSeedValidity

    @property
    def used_direct(self) -> torch.Tensor:
        return self.route == ROUTE_DIRECT

    @property
    def used_refined(self) -> torch.Tensor:
        return self.route == ROUTE_REFINED

    @property
    def used_fallback(self) -> torch.Tensor:
        return self.route == ROUTE_FALLBACK


Projector = Callable[..., tuple[torch.Tensor, torch.Tensor, object]]


def _require_float_batch(
    value: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a tensor')
    if value.shape != shape:
        raise ValueError(
            f'{name} must have shape {shape}, got {tuple(value.shape)}')
    if value.device != device or value.dtype != dtype:
        raise ValueError(f'{name} must match q device and dtype')
    return value


def _kin_vector(kin, name: str, q: torch.Tensor) -> torch.Tensor:
    if not hasattr(kin, name):
        raise ValueError(f'kinematics object is missing {name}')
    value = torch.as_tensor(
        getattr(kin, name), device=q.device, dtype=q.dtype)
    if value.shape != (7,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'kinematics {name} must be finite with shape (7,)')
    return value


@torch.no_grad()
def strict_seed_validity(
    kin,
    collision,
    q: torch.Tensor,
    p0: torch.Tensor,
    line_dir: torch.Tensor,
    n_target: torch.Tensor,
    config: DirectSeedProjectionConfig | None = None,
) -> StrictSeedValidity:
    """Validate a single seed per task under the deployment constraints.

    Missing or non-finite collision evidence fails closed.
    """
    config = DirectSeedProjectionConfig() if config is None else config
    if not isinstance(config, DirectSeedProjectionConfig):
        raise TypeError('config must be a DirectSeedProjectionConfig')
    if not isinstance(q, torch.Tensor):
        raise TypeError('q must be a tensor')
    if q.ndim != 2 or q.shape[-1] != 7:
        raise ValueError(f'q must have shape (B, 7), got {tuple(q.shape)}')
    if q.dtype not in (torch.float32, torch.float64):
        raise TypeError('q must use float32 or float64')
    batch_size = q.shape[0]
    if batch_size < 1:
        raise ValueError('q batch must be non-empty')
    device, dtype = q.device, q.dtype
    p0 = _require_float_batch(
        p0, 'p0', (batch_size, 3), device=device, dtype=dtype)
    line_dir = _require_float_batch(
        line_dir, 'line_dir', (batch_size, 3), device=device, dtype=dtype)
    n_target = _require_float_batch(
        n_target, 'n_target', (batch_size, 3), device=device, dtype=dtype)

    q_lower = _kin_vector(kin, 'lmt_lo', q)
    q_upper = _kin_vector(kin, 'lmt_up', q)
    q_mid = _kin_vector(kin, 'q_mid', q)
    if not bool((q_lower < q_upper).all()):
        raise ValueError('kinematics joint limits must satisfy lower < upper')

    q_finite = torch.isfinite(q).all(dim=-1)
    p0_finite = torch.isfinite(p0).all(dim=-1)
    line_finite = torch.isfinite(line_dir).all(dim=-1)
    target_finite = torch.isfinite(n_target).all(dim=-1)
    line_norm = torch.where(
        line_finite, line_dir.norm(dim=-1),
        torch.full((batch_size,), float('inf'), device=device, dtype=dtype))
    target_norm = torch.where(
        target_finite, n_target.norm(dim=-1),
        torch.full((batch_size,), float('inf'), device=device, dtype=dtype))
    task_valid = (
        p0_finite
        & line_finite
        & target_finite
        & ((line_norm - 1.0).abs() <= _UNIT_TOL)
        & ((target_norm - 1.0).abs() <= _UNIT_TOL)
    )
    joint_margin = torch.minimum(q - q_lower, q_upper - q).amin(dim=-1)
    joint_margin = torch.nan_to_num(
        joint_margin, nan=-torch.finfo(dtype).max,
        posinf=torch.finfo(dtype).max, neginf=-torch.finfo(dtype).max)
    joint_limits = q_finite & (
        joint_margin >= config.joint_margin_rad)
    safe_fk = task_valid & joint_limits
    q_eval = torch.where(
        safe_fk.unsqueeze(-1), q, q_mid.expand_as(q))
    p_tcp, rotation, _, _ = kin.tcp_fk_jac(q_eval)
    fk_finite = (
        torch.isfinite(p_tcp).all(dim=-1)
        & torch.isfinite(rotation).all(dim=(-1, -2))
    )
    finite = q_finite & fk_finite
    z_tool = rotation[:, :, 2]
    position_error = (p_tcp - p0).norm(dim=-1)
    cone_cosine = (z_tool * n_target).sum(dim=-1)
    max_value = torch.finfo(dtype).max
    evidence_valid = safe_fk & fk_finite
    position_error = torch.where(
        evidence_valid, position_error,
        torch.full_like(position_error, max_value))
    cone_cosine = torch.where(
        evidence_valid, cone_cosine,
        torch.full_like(cone_cosine, -1.0))
    position = evidence_valid & (position_error <= config.position_tol_m)
    cone = evidence_valid & (
        cone_cosine >= math.cos(math.radians(config.cone_deg)))

    collision_margin = torch.full(
        (batch_size,), -max_value, device=device, dtype=dtype)
    collision_free = torch.zeros(
        (batch_size,), device=device, dtype=torch.bool)
    if collision is not None:
        checker_margin = float(getattr(collision, 'margin', 0.0))
        if not math.isfinite(checker_margin):
            raise ValueError('collision checker margin must be finite')
        required_collision_margin = max(
            config.collision_margin_m, checker_margin)
        link_transforms = kin.link_transforms(q_eval)
        links_finite = torch.isfinite(link_transforms).all(dim=(-1, -2, -3))
        collision_input = evidence_valid & links_finite
        if hasattr(collision, 'min_margin'):
            raw_margin = torch.as_tensor(
                collision.min_margin(link_transforms),
                device=device, dtype=dtype)
            if raw_margin.shape != (batch_size,):
                raise ValueError(
                    'collision.min_margin must return shape (B,)')
            margin_finite = torch.isfinite(raw_margin)
            collision_margin = torch.where(
                collision_input & margin_finite, raw_margin,
                torch.full_like(raw_margin, -max_value))
            collision_free = (
                collision_input & margin_finite
                & (raw_margin >= required_collision_margin)
            )
        elif (hasattr(collision, 'is_collided')
              and required_collision_margin == checker_margin):
            collided = torch.as_tensor(
                collision.is_collided(link_transforms),
                device=device, dtype=torch.bool)
            if collided.shape != (batch_size,):
                raise ValueError(
                    'collision.is_collided must return shape (B,)')
            collision_free = collision_input & ~collided
            collision_margin = torch.where(
                collision_free,
                torch.full_like(collision_margin, max_value),
                collision_margin)

    valid = (
        task_valid & finite & joint_limits & position & cone & collision_free)
    return StrictSeedValidity(
        valid=valid,
        input_valid=task_valid,
        finite=finite,
        joint_limits=joint_limits,
        position=position,
        cone=cone,
        collision_free=collision_free,
        position_error_m=position_error,
        cone_cosine=cone_cosine,
        collision_margin_m=collision_margin,
        joint_margin_rad=joint_margin,
        z_tool=z_tool,
    )


def _normalise(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(value.dtype).eps)


@torch.no_grad()
def _project_z_to_cone(
    z_tool: torch.Tensor,
    n_target: torch.Tensor,
    cone_deg: float,
) -> torch.Tensor:
    """Map a finite tool-z direction to the nearest spherical-cap boundary."""
    z_tool = _normalise(z_tool)
    n_target = _normalise(n_target)
    cosine = (z_tool * n_target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = z_tool - cosine * n_target
    tangent_norm = tangent.norm(dim=-1, keepdim=True)

    ref_x = z_tool.new_tensor([1.0, 0.0, 0.0]).expand_as(z_tool)
    ref_y = z_tool.new_tensor([0.0, 1.0, 0.0]).expand_as(z_tool)
    reference = torch.where(
        ((ref_x * n_target).sum(dim=-1, keepdim=True).abs() < 0.9),
        ref_x, ref_y)
    fallback = _normalise(
        reference
        - (reference * n_target).sum(dim=-1, keepdim=True) * n_target)
    tangent_unit = torch.where(
        tangent_norm > 1e-7,
        tangent / tangent_norm.clamp_min(1e-7),
        fallback)
    angle = math.radians(float(cone_deg))
    boundary = math.cos(angle) * n_target + math.sin(angle) * tangent_unit
    inside = cosine >= math.cos(angle)
    return _normalise(torch.where(inside, z_tool, boundary))


@torch.no_grad()
def _rotation_with_z(
    z_axis: torch.Tensor,
    line_dir: torch.Tensor,
) -> torch.Tensor:
    """Construct stable rotations; only their z column is used by z-axis IK."""
    z_axis = _normalise(z_axis)
    x_raw = (
        line_dir
        - (line_dir * z_axis).sum(dim=-1, keepdim=True) * z_axis)
    ref_x = z_axis.new_tensor([1.0, 0.0, 0.0]).expand_as(z_axis)
    ref_y = z_axis.new_tensor([0.0, 1.0, 0.0]).expand_as(z_axis)
    reference = torch.where(
        ((ref_x * z_axis).sum(dim=-1, keepdim=True).abs() < 0.9),
        ref_x, ref_y)
    fallback = (
        reference
        - (reference * z_axis).sum(dim=-1, keepdim=True) * z_axis)
    x_raw = torch.where(
        (x_raw.norm(dim=-1, keepdim=True) > 1e-7), x_raw, fallback)
    x_axis = _normalise(x_raw)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


@torch.no_grad()
def route_generated_seed(
    kin,
    collision,
    q_raw: torch.Tensor,
    p0: torch.Tensor,
    line_dir: torch.Tensor,
    n_target: torch.Tensor,
    fallback_q: torch.Tensor,
    config: DirectSeedProjectionConfig | None = None,
    *,
    projector: Projector | None = None,
) -> GeneratedSeedResult:
    """Apply the fixed DIRECT → one-shot REFINE → FALLBACK routing rule."""
    config = DirectSeedProjectionConfig() if config is None else config
    if not isinstance(config, DirectSeedProjectionConfig):
        raise TypeError('config must be a DirectSeedProjectionConfig')
    if not isinstance(q_raw, torch.Tensor):
        raise TypeError('q_raw must be a tensor')
    if q_raw.ndim != 2 or q_raw.shape[-1] != 7:
        raise ValueError(
            f'q_raw must have shape (B, 7), got {tuple(q_raw.shape)}')
    batch_size = q_raw.shape[0]
    fallback_q = _require_float_batch(
        fallback_q, 'fallback_q', tuple(q_raw.shape),
        device=q_raw.device, dtype=q_raw.dtype)
    p0 = _require_float_batch(
        p0, 'p0', (batch_size, 3),
        device=q_raw.device, dtype=q_raw.dtype)
    line_dir = _require_float_batch(
        line_dir, 'line_dir', (batch_size, 3),
        device=q_raw.device, dtype=q_raw.dtype)
    n_target = _require_float_batch(
        n_target, 'n_target', (batch_size, 3),
        device=q_raw.device, dtype=q_raw.dtype)

    raw = strict_seed_validity(
        kin, collision, q_raw, p0, line_dir, n_target, config)
    fallback = strict_seed_validity(
        kin, collision, fallback_q, p0, line_dir, n_target, config)
    q_projected = q_raw.clone()
    ik_attempted = torch.zeros(
        batch_size, device=q_raw.device, dtype=torch.bool)
    ik_ok = torch.zeros_like(ik_attempted)

    projectable = (
        ~raw.valid & raw.input_valid & raw.finite & raw.joint_limits)
    project_index = torch.nonzero(
        projectable, as_tuple=False).flatten()
    if project_index.numel() > 0:
        ik_attempted[project_index] = True
        q_seed = q_raw.index_select(0, project_index)
        p_target = p0.index_select(0, project_index)
        line_target = line_dir.index_select(0, project_index)
        normal_target = n_target.index_select(0, project_index)
        generated_z = raw.z_tool.index_select(0, project_index)
        projected_z = _project_z_to_cone(
            generated_z, normal_target, config.projection_cone_deg)
        rotation_target = _rotation_with_z(projected_z, line_target)
        project = _batched_ik_project if projector is None else projector
        projected_q, projected_ok, _ = project(
            kin, q_seed, p_target, rotation_target,
            branch_action=None, preserve_seed=True)
        projected_q = torch.as_tensor(
            projected_q, device=q_raw.device, dtype=q_raw.dtype)
        projected_ok = torch.as_tensor(
            projected_ok, device=q_raw.device, dtype=torch.bool)
        if projected_q.shape != q_seed.shape:
            raise ValueError(
                'projector q output must match projected input shape')
        if projected_ok.shape != (project_index.numel(),):
            raise ValueError(
                'projector success output must have shape (N_projected,)')
        q_projected.index_copy_(0, project_index, projected_q)
        ik_ok.index_copy_(0, project_index, projected_ok)

    projected = strict_seed_validity(
        kin, collision, q_projected, p0, line_dir, n_target, config)
    use_direct = raw.valid
    use_refined = ~use_direct & ik_attempted & ik_ok & projected.valid
    use_fallback = ~use_direct & ~use_refined & fallback.valid
    valid = use_direct | use_refined | use_fallback

    route = torch.full(
        (batch_size,), ROUTE_INVALID, device=q_raw.device, dtype=torch.int8)
    route = torch.where(
        use_fallback, torch.full_like(route, ROUTE_FALLBACK), route)
    route = torch.where(
        use_refined, torch.full_like(route, ROUTE_REFINED), route)
    route = torch.where(
        use_direct, torch.full_like(route, ROUTE_DIRECT), route)

    # Start with a clone of the user-supplied fallback.  Rows taking FALLBACK
    # or INVALID are never touched, preserving signed zero and all other bits.
    q = fallback_q.clone()
    if bool(use_refined.any()):
        q[use_refined] = q_projected[use_refined]
    if bool(use_direct.any()):
        q[use_direct] = q_raw[use_direct]
    return GeneratedSeedResult(
        q_raw=q_raw,
        q_projected=q_projected,
        q=q,
        route=route,
        valid=valid,
        ik_attempted=ik_attempted,
        ik_ok=ik_ok,
        raw=raw,
        projected=projected,
        fallback=fallback,
    )


@torch.no_grad()
def generate_or_refine_seed(
    generator,
    kin,
    collision,
    p0: torch.Tensor,
    line_dir: torch.Tensor,
    n_target: torch.Tensor,
    fallback_q: torch.Tensor,
    config: DirectSeedProjectionConfig | None = None,
    *,
    projector: Projector | None = None,
) -> GeneratedSeedResult:
    """Generate exactly one q, then apply :func:`route_generated_seed`."""
    task = direct_seed_task(p0, line_dir, n_target)
    q_raw = generator(task)
    if not isinstance(q_raw, torch.Tensor):
        raise TypeError('generator must return a tensor')
    return route_generated_seed(
        kin, collision, q_raw, p0, line_dir, n_target, fallback_q,
        config, projector=projector)


__all__ = [
    'DirectSeedProjectionConfig',
    'GeneratedSeedResult',
    'ROUTE_DIRECT',
    'ROUTE_FALLBACK',
    'ROUTE_INVALID',
    'ROUTE_NAMES',
    'ROUTE_REFINED',
    'StrictSeedValidity',
    'generate_or_refine_seed',
    'route_generated_seed',
    'strict_seed_validity',
]
