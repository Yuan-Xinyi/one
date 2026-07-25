"""Safety shield for a continuous residual around a discrete IK seed.

The residual is expressed in the four-dimensional, task-aligned nullspace
basis used by the continuous controller.  A short position-only DLS solve
projects the perturbed configuration back to the task origin.  The shield then
tries a fixed sequence of residual scales and returns the largest one that
satisfies every hard constraint.

This module is deliberately independent of the seed policy and trainer.  In
particular, it is deterministic, has no learned state, and accepts lightweight
CPU kinematics/collision stubs implementing the same small interface as the
FR3 classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math
import torch

from Yuan.RL_controller.env.env import build_task_aligned_basis


_DEFAULT_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)
_TASK_UNIT_TOL = 1e-5


@dataclass(frozen=True)
class ResidualSeedConfig:
    """Numerical and safety settings for residual-seed projection.

    ``rho`` is the maximum Euclidean joint-space distance in radians from the
    discrete seed.  Projection uses a fixed iteration count to keep execution
    deterministic across a batch.
    """

    rho: float = 0.08
    dls_steps: int = 6
    dls_damping: float = 1e-3
    max_dls_step: float = 0.05
    position_tol: float = 5e-3
    cone_deg: float = 30.0
    collision_margin: float = 0.0
    basis_damping: float = 1e-3
    task_norm_eps: float = 1e-8
    solve_position_tol: float = 1e-7
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS

    def __post_init__(self) -> None:
        if not math.isfinite(self.rho) or not 0.0 < self.rho <= 0.08:
            raise ValueError('rho must be in (0, 0.08]')
        if not 5 <= self.dls_steps <= 8:
            raise ValueError('dls_steps must be in [5, 8]')
        if not math.isfinite(self.dls_damping) or self.dls_damping <= 0.0:
            raise ValueError('dls_damping must be finite and positive')
        if not math.isfinite(self.max_dls_step) or self.max_dls_step <= 0.0:
            raise ValueError('max_dls_step must be finite and positive')
        if (not math.isfinite(self.position_tol)
                or not 0.0 < self.position_tol <= 5e-3):
            raise ValueError('position_tol must be in (0, 5e-3]')
        if not math.isfinite(self.cone_deg) or not 0.0 < self.cone_deg <= 30.0:
            raise ValueError('cone_deg must be in (0, 30]')
        if (not math.isfinite(self.collision_margin)
                or self.collision_margin < 0.0):
            raise ValueError('collision_margin must be finite and non-negative')
        if not math.isfinite(self.basis_damping) or self.basis_damping <= 0.0:
            raise ValueError('basis_damping must be finite and positive')
        if not math.isfinite(self.task_norm_eps) or self.task_norm_eps <= 0.0:
            raise ValueError('task_norm_eps must be finite and positive')
        if (not math.isfinite(self.solve_position_tol)
                or self.solve_position_tol <= 0.0):
            raise ValueError('solve_position_tol must be finite and positive')
        if tuple(float(value) for value in self.alphas) != _DEFAULT_ALPHAS:
            raise ValueError(
                f'alphas must be the fixed safety schedule {_DEFAULT_ALPHAS}')


@dataclass(frozen=True)
class ResidualSeedDiagnostics:
    """Diagnostics for the returned (selected) configuration.

    Scalar tensors are shaped ``(B,)``.  ``basis_fallback`` is ``(B, 3)`` and
    identifies fallback use for the first three task-aligned basis vectors.
    ``attempted_valid`` and ``projection_ok_by_alpha`` are ``(B, 5)`` in the
    order specified by :attr:`ResidualSeedConfig.alphas`.
    """

    position_error: torch.Tensor
    cone_cosine: torch.Tensor
    collision_margin: torch.Tensor
    joint_margin: torch.Tensor
    branch_distance: torch.Tensor
    basis_fallback: torch.Tensor
    finite: torch.Tensor
    joint_limits: torch.Tensor
    position: torch.Tensor
    cone: torch.Tensor
    collision_free: torch.Tensor
    branch: torch.Tensor
    projection_ok: torch.Tensor
    input_finite: torch.Tensor
    basis_finite: torch.Tensor
    attempted_valid: torch.Tensor
    projection_ok_by_alpha: torch.Tensor
    selected_index: torch.Tensor


@dataclass(frozen=True)
class ResidualSeedResult:
    """Shielded seed and its acceptance metadata."""

    q: torch.Tensor
    accepted_alpha: torch.Tensor
    valid: torch.Tensor
    diagnostics: ResidualSeedDiagnostics

    @property
    def alpha(self) -> torch.Tensor:
        """Short alias for callers that treat the shield as an action map."""
        return self.accepted_alpha


BasisBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _same_effective_device(left: torch.device | str,
                           right: torch.device | str) -> bool:
    """Treat an unspecified CUDA index as the process's current device."""
    left = torch.device(left)
    right = torch.device(right)
    if left.type != right.type:
        return False
    if left.type != 'cuda':
        return left == right
    left_index = torch.cuda.current_device() if left.index is None else left.index
    right_index = torch.cuda.current_device() if right.index is None else right.index
    return left_index == right_index


def _require_batch_tensor(
    value: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(value.shape)}')
    return value


def _kin_vector(
    kin,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(kin, name):
        raise ValueError(f'kinematics object is missing {name}')
    value = torch.as_tensor(getattr(kin, name), device=device, dtype=dtype)
    if value.shape != (7,):
        raise ValueError(f'kinematics {name} must have shape (7,)')
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f'kinematics {name} must be finite')
    return value


def _normalise_or_default(
    value: torch.Tensor,
    finite_nonzero: torch.Tensor,
    default: tuple[float, float, float],
    eps: float,
) -> torch.Tensor:
    safe_default = value.new_tensor(default).expand_as(value)
    safe = torch.where(finite_nonzero.unsqueeze(-1), value, safe_default)
    return safe / safe.norm(dim=-1, keepdim=True).clamp_min(eps)


def _limit_norm(value: torch.Tensor, limit: torch.Tensor | float) -> torch.Tensor:
    norm = value.norm(dim=-1, keepdim=True)
    limit_tensor = torch.as_tensor(limit, device=value.device, dtype=value.dtype)
    if limit_tensor.ndim > 0:
        limit_tensor = limit_tensor.unsqueeze(-1)
    scale = (limit_tensor / norm.clamp_min(torch.finfo(value.dtype).eps)).clamp(max=1.0)
    return value * scale


@torch.no_grad()
def _project_position_dls(
    kin,
    q_initial: torch.Tensor,
    q_base: torch.Tensor,
    p_target: torch.Tensor,
    radius: torch.Tensor,
    active: torch.Tensor,
    config: ResidualSeedConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-step batched DLS, retaining failed rows without NaNs."""
    q = q_initial.clone()
    projection_ok = active.clone()
    batch_size = q.shape[0]
    eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(batch_size, 3, 3)

    for _ in range(config.dls_steps):
        p_tcp, _, jacobian, _ = kin.tcp_fk_jac(q)
        jacobian_p = jacobian[:, :3, :]
        error = p_target - p_tcp
        system_finite = (
            torch.isfinite(p_tcp).all(dim=-1)
            & torch.isfinite(jacobian_p).all(dim=(-1, -2))
            & torch.isfinite(error).all(dim=-1)
        )
        error_norm = torch.where(
            system_finite,
            error.norm(dim=-1),
            torch.full((batch_size,), float('inf'), device=q.device, dtype=q.dtype),
        )
        converged = system_finite & (error_norm <= config.solve_position_tol)
        needs_step = projection_ok & ~converged

        safe_jacobian = torch.where(
            needs_step.view(-1, 1, 1), jacobian_p,
            torch.zeros_like(jacobian_p))
        safe_error = torch.where(
            needs_step.unsqueeze(-1), error, torch.zeros_like(error))
        system = safe_jacobian @ safe_jacobian.transpose(-1, -2)
        system = system + (config.dls_damping ** 2) * eye3
        solution, info = torch.linalg.solve_ex(
            system, safe_error.unsqueeze(-1), check_errors=False)
        solution = solution.squeeze(-1)
        solve_ok = (
            needs_step
            & (info == 0)
            & torch.isfinite(solution).all(dim=-1)
        )
        projection_ok = projection_ok & (converged | solve_ok)

        correction = (
            safe_jacobian.transpose(-1, -2) @ solution.unsqueeze(-1)
        ).squeeze(-1)
        correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
        correction = _limit_norm(correction, config.max_dls_step)
        proposal = q + correction
        branch_delta = _limit_norm(proposal - q_base, radius)
        proposal = q_base + branch_delta
        proposal_finite = torch.isfinite(proposal).all(dim=-1)
        update = solve_ok & proposal_finite
        q = torch.where(update.unsqueeze(-1), proposal, q)
        projection_ok = projection_ok & (converged | proposal_finite)

    return q, projection_ok


@torch.no_grad()
def _evaluate_candidates(
    kin,
    collision,
    q_candidates: torch.Tensor,
    q_base: torch.Tensor,
    p0: torch.Tensor,
    n_target: torch.Tensor,
    task_input_valid: torch.Tensor,
    config: ResidualSeedConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate every hard constraint without exposing unsafe rows to FK."""
    batch_size, n_alpha, _ = q_candidates.shape
    flat_q = q_candidates.reshape(-1, 7)
    flat_base = q_base[:, None, :].expand(-1, n_alpha, -1).reshape(-1, 7)
    flat_task_valid = task_input_valid[:, None].expand(-1, n_alpha).reshape(-1)
    lmt_lo = _kin_vector(kin, 'lmt_lo', device=q_base.device, dtype=q_base.dtype)
    lmt_up = _kin_vector(kin, 'lmt_up', device=q_base.device, dtype=q_base.dtype)
    q_mid = _kin_vector(kin, 'q_mid', device=q_base.device, dtype=q_base.dtype)

    q_finite = torch.isfinite(flat_q).all(dim=-1)
    joint_margin = torch.minimum(flat_q - lmt_lo, lmt_up - flat_q).amin(dim=-1)
    joint_limits = q_finite & (joint_margin >= 0.0)
    safe_fk = flat_task_valid & joint_limits
    q_eval = torch.where(safe_fk.unsqueeze(-1), flat_q, q_mid.expand_as(flat_q))
    p_tcp, rotation, _, _ = kin.tcp_fk_jac(q_eval)
    fk_finite = (
        torch.isfinite(p_tcp).all(dim=-1)
        & torch.isfinite(rotation).all(dim=(-1, -2))
    )
    finite = q_finite & fk_finite & flat_task_valid

    flat_p0 = p0[:, None, :].expand(-1, n_alpha, -1).reshape(-1, 3)
    flat_normal = n_target[:, None, :].expand(-1, n_alpha, -1).reshape(-1, 3)
    position_error = (p_tcp - flat_p0).norm(dim=-1)
    cone_cosine = (rotation[:, :, 2] * flat_normal).sum(dim=-1)

    max_value = torch.finfo(q_base.dtype).max
    position_error = torch.where(
        finite, position_error,
        torch.full_like(position_error, max_value))
    cone_cosine = torch.where(
        finite, cone_cosine,
        torch.full_like(cone_cosine, -1.0))
    joint_margin = torch.nan_to_num(
        joint_margin, nan=-max_value, posinf=max_value, neginf=-max_value)
    position = finite & (position_error <= config.position_tol)
    cone = finite & (cone_cosine >= math.cos(math.radians(config.cone_deg)))

    if collision is None:
        # Collision state is a hard constraint, not optional evidence.  A
        # missing checker must therefore reject every candidate rather than
        # silently treating it as collision-free.
        collision_margin = torch.full_like(position_error, -max_value)
        collision_free = torch.zeros_like(finite)
    else:
        link_transforms = kin.link_transforms(q_eval)
        link_finite = torch.isfinite(link_transforms).all(dim=(-1, -2, -3))
        checker_margin = float(getattr(collision, 'margin', 0.0))
        if not math.isfinite(checker_margin):
            raise ValueError('collision checker margin must be finite')
        required_collision_margin = max(
            config.collision_margin, checker_margin)
        if hasattr(collision, 'min_margin'):
            collision_margin = collision.min_margin(link_transforms)
            collision_margin = torch.as_tensor(
                collision_margin, device=q_base.device, dtype=q_base.dtype)
            if collision_margin.shape != (flat_q.shape[0],):
                raise ValueError('collision.min_margin must return shape (B,)')
            margin_finite = torch.isfinite(collision_margin)
            collision_margin = torch.nan_to_num(
                collision_margin, nan=-max_value,
                posinf=-max_value, neginf=-max_value)
            collision_free = (
                finite & link_finite & margin_finite
                & (collision_margin >= required_collision_margin)
            )
        elif hasattr(collision, 'is_collided'):
            if required_collision_margin > 0.0:
                raise ValueError(
                    'positive collision margin requires collision.min_margin')
            collided = torch.as_tensor(
                collision.is_collided(link_transforms),
                device=q_base.device, dtype=torch.bool)
            if collided.shape != (flat_q.shape[0],):
                raise ValueError('collision.is_collided must return shape (B,)')
            collision_free = finite & link_finite & ~collided
            collision_margin = torch.where(
                collision_free,
                torch.full_like(position_error, max_value),
                torch.full_like(position_error, -max_value),
            )
        else:
            raise ValueError(
                'collision object must define min_margin or is_collided')
        collision_margin = torch.where(
            finite & link_finite, collision_margin,
            torch.full_like(collision_margin, -max_value))

    branch_distance = (flat_q - flat_base).norm(dim=-1)
    branch_distance = torch.nan_to_num(
        branch_distance, nan=max_value, posinf=max_value, neginf=max_value)
    # Projection itself enforces this ball.  A small dtype-scaled comparison
    # allowance avoids rejecting a boundary point solely due to sqrt rounding.
    branch_allowance = 16.0 * torch.finfo(q_base.dtype).eps * max(config.rho, 1.0)
    branch = finite & (branch_distance <= config.rho + branch_allowance)

    def shaped(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch_size, n_alpha)

    return {
        'finite': shaped(finite),
        'joint_limits': shaped(joint_limits),
        'position': shaped(position),
        'cone': shaped(cone),
        'collision_free': shaped(collision_free),
        'branch': shaped(branch),
        'position_error': shaped(position_error),
        'cone_cosine': shaped(cone_cosine),
        'collision_margin': shaped(collision_margin),
        'joint_margin': shaped(joint_margin),
        'branch_distance': shaped(branch_distance),
    }


@torch.no_grad()
def apply_residual_seed(
    kin,
    collision,
    base_q: torch.Tensor,
    p0: torch.Tensor,
    line_dir: torch.Tensor,
    n_target: torch.Tensor,
    latent: torch.Tensor,
    enabled: torch.Tensor | bool = True,
    config: ResidualSeedConfig | None = None,
    *,
    basis_builder: BasisBuilder | None = None,
) -> ResidualSeedResult:
    """Apply and shield a four-dimensional residual seed action.

    Args:
        kin: Batched kinematics with ``tcp_fk_jac`` and ``link_transforms``.
        collision: Collision checker. ``None`` fails closed for every row.
        base_q: Discrete IK seeds, shape ``(B, 7)``.
        p0: Task-space origins to preserve, shape ``(B, 3)``.
        line_dir: Task directions, shape ``(B, 3)``.
        n_target: Target tool-z directions, shape ``(B, 3)``.
        latent: Residual coordinates in the task-aligned basis, shape ``(B, 4)``.
        enabled: Per-row residual gate.  Disabled rows return ``base_q`` exactly.
        config: Shield settings.
        basis_builder: Test seam; defaults to ``build_task_aligned_basis``.

    Returns:
        A :class:`ResidualSeedResult`.  Rows accepting alpha zero, including
        every disabled row, are copied directly from ``base_q`` without any
        floating-point arithmetic, preserving their bit pattern.
    """
    config = ResidualSeedConfig() if config is None else config
    if not isinstance(config, ResidualSeedConfig):
        raise TypeError('config must be a ResidualSeedConfig')
    if not isinstance(base_q, torch.Tensor):
        raise TypeError('base_q must be a torch.Tensor')
    if base_q.ndim != 2 or base_q.shape[1] != 7:
        raise ValueError(f'base_q must have shape (B, 7), got {tuple(base_q.shape)}')
    if not torch.is_floating_point(base_q):
        raise TypeError('base_q must have a floating dtype')
    if base_q.dtype not in (torch.float32, torch.float64):
        raise TypeError('base_q dtype must be float32 or float64')
    batch_size = base_q.shape[0]
    device, dtype = base_q.device, base_q.dtype
    if batch_size < 1:
        raise ValueError('base_q batch must be non-empty')
    if (hasattr(kin, 'device')
            and not _same_effective_device(kin.device, device)):
        raise ValueError('base_q and kinematics must be on the same device')
    if hasattr(kin, 'dtype') and kin.dtype != dtype:
        raise ValueError('base_q and kinematics must use the same dtype')

    p0 = _require_batch_tensor(
        p0, 'p0', (batch_size, 3), device=device, dtype=dtype)
    line_dir = _require_batch_tensor(
        line_dir, 'line_dir', (batch_size, 3), device=device, dtype=dtype)
    n_target = _require_batch_tensor(
        n_target, 'n_target', (batch_size, 3), device=device, dtype=dtype)
    latent = _require_batch_tensor(
        latent, 'latent', (batch_size, 4), device=device, dtype=dtype)

    if isinstance(enabled, bool):
        enabled_tensor = torch.full(
            (batch_size,), enabled, device=device, dtype=torch.bool)
    else:
        enabled_tensor = torch.as_tensor(enabled, device=device)
        if enabled_tensor.dtype != torch.bool:
            raise TypeError('enabled tensor must have dtype bool')
        if enabled_tensor.shape != (batch_size,):
            raise ValueError(
                f'enabled must have shape ({batch_size},), '
                f'got {tuple(enabled_tensor.shape)}')

    lmt_lo = _kin_vector(kin, 'lmt_lo', device=device, dtype=dtype)
    lmt_up = _kin_vector(kin, 'lmt_up', device=device, dtype=dtype)
    if not bool((lmt_lo < lmt_up).all().item()):
        raise ValueError('kinematics joint limits must have positive width')
    q_mid = _kin_vector(kin, 'q_mid', device=device, dtype=dtype)
    q_half = 0.5 * (lmt_up - lmt_lo)

    base_finite = torch.isfinite(base_q).all(dim=-1)
    base_in_limits = (
        base_finite
        & ((base_q >= lmt_lo) & (base_q <= lmt_up)).all(dim=-1)
    )
    p0_finite = torch.isfinite(p0).all(dim=-1)
    line_finite = torch.isfinite(line_dir).all(dim=-1)
    normal_finite = torch.isfinite(n_target).all(dim=-1)
    line_norm = line_dir.norm(dim=-1)
    normal_norm = n_target.norm(dim=-1)
    line_nonzero = line_finite & (line_norm > config.task_norm_eps)
    normal_nonzero = normal_finite & (normal_norm > config.task_norm_eps)
    # The environment consumes the original task vectors, while the shield
    # uses unit vectors for its basis and cone check.  Requiring canonical
    # unit inputs prevents the two paths from disagreeing near a boundary.
    line_unit_length = line_nonzero & ((line_norm - 1.0).abs() <= _TASK_UNIT_TOL)
    normal_unit_length = (
        normal_nonzero & ((normal_norm - 1.0).abs() <= _TASK_UNIT_TOL))
    task_input_valid = (
        base_finite & p0_finite & line_unit_length & normal_unit_length)
    latent_finite = torch.isfinite(latent).all(dim=-1)
    latent_nonzero = (
        latent_finite & (latent.norm(dim=-1) > config.task_norm_eps))
    residual_input_valid = (
        task_input_valid & base_in_limits & latent_nonzero & enabled_tensor)

    safe_base = torch.where(
        (task_input_valid & base_in_limits).unsqueeze(-1),
        base_q,
        q_mid.expand_as(base_q),
    )
    line_unit = _normalise_or_default(
        line_dir, line_nonzero, (1.0, 0.0, 0.0), config.task_norm_eps)
    normal_unit = _normalise_or_default(
        n_target, normal_nonzero, (0.0, 0.0, 1.0), config.task_norm_eps)
    safe_p0 = torch.where(p0_finite.unsqueeze(-1), p0, torch.zeros_like(p0))
    safe_latent = torch.where(latent_finite.unsqueeze(-1), latent, torch.zeros_like(latent))

    basis = torch.zeros((batch_size, 7, 4), device=device, dtype=dtype)
    basis_fallback = torch.zeros((batch_size, 3), device=device, dtype=torch.bool)
    basis_finite = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    active_index = torch.nonzero(residual_input_valid, as_tuple=False).flatten()
    if active_index.numel() > 0:
        builder = build_task_aligned_basis if basis_builder is None else basis_builder
        active_basis, active_fallback = builder(
            kin,
            safe_base.index_select(0, active_index),
            line_unit.index_select(0, active_index),
            normal_unit.index_select(0, active_index),
            q_mid,
            q_half,
            config.basis_damping,
        )
        active_basis = torch.as_tensor(active_basis, device=device, dtype=dtype)
        active_fallback = torch.as_tensor(
            active_fallback, device=device, dtype=torch.bool)
        expected_basis_shape = (active_index.numel(), 7, 4)
        expected_fallback_shape = (active_index.numel(), 3)
        if active_basis.shape != expected_basis_shape:
            raise ValueError(
                'basis builder must return basis with shape '
                f'{expected_basis_shape}, got {tuple(active_basis.shape)}')
        if active_fallback.shape != expected_fallback_shape:
            raise ValueError(
                'basis builder must return fallback mask with shape '
                f'{expected_fallback_shape}, got {tuple(active_fallback.shape)}')
        active_finite = torch.isfinite(active_basis).all(dim=(-1, -2))
        active_basis = torch.where(
            active_finite.view(-1, 1, 1), active_basis,
            torch.zeros_like(active_basis))
        basis.index_copy_(0, active_index, active_basis)
        basis_fallback.index_copy_(0, active_index, active_fallback)
        basis_finite.index_copy_(0, active_index, active_finite)

    raw_delta = (basis @ safe_latent.unsqueeze(-1)).squeeze(-1)
    bounded_delta = _limit_norm(raw_delta, config.rho)
    residual_active = (
        residual_input_valid
        & basis_finite
        & (bounded_delta.norm(dim=-1) > config.task_norm_eps)
    )
    alphas = base_q.new_tensor(config.alphas)
    n_alpha = alphas.numel()
    n_nonzero = n_alpha - 1

    nonzero_alphas = alphas[:-1]
    q_initial = (
        safe_base[:, None, :]
        + nonzero_alphas.view(1, -1, 1) * bounded_delta[:, None, :]
    )
    flat_q_initial = q_initial.reshape(-1, 7)
    flat_base = safe_base[:, None, :].expand(-1, n_nonzero, -1).reshape(-1, 7)
    flat_p0 = safe_p0[:, None, :].expand(-1, n_nonzero, -1).reshape(-1, 3)
    # Projection may repair numerical task-space drift, but it must never
    # amplify a small residual into a rho-sized joint move.  Bound each
    # branch by the norm actually requested at that alpha.
    requested_radius = bounded_delta.norm(dim=-1, keepdim=True)
    flat_radius = (
        requested_radius * nonzero_alphas.view(1, -1)).reshape(-1)
    flat_active = (
        residual_active[:, None]
        .expand(-1, n_nonzero).reshape(-1)
    )
    projected, projected_ok = _project_position_dls(
        kin, flat_q_initial, flat_base, flat_p0, flat_radius,
        flat_active, config)
    projected = projected.reshape(batch_size, n_nonzero, 7)
    projected_ok = projected_ok.reshape(batch_size, n_nonzero)

    # Construct alpha-zero separately: no addition, projection, normalisation,
    # or dtype conversion may alter its bit pattern.
    q_candidates = torch.empty(
        (batch_size, n_alpha, 7), device=device, dtype=dtype)
    q_candidates[:, :-1] = projected
    q_candidates[:, -1] = base_q
    projection_ok_by_alpha = torch.cat([
        projected_ok,
        torch.ones((batch_size, 1), device=device, dtype=torch.bool),
    ], dim=1)

    checks = _evaluate_candidates(
        kin, collision, q_candidates, base_q, safe_p0, n_target,
        task_input_valid, config)
    hard_valid = (
        checks['finite']
        & checks['joint_limits']
        & checks['position']
        & checks['cone']
        & checks['collision_free']
        & checks['branch']
    )
    residual_allowed = (
        residual_active[:, None]
        .expand(-1, n_nonzero)
    )
    attempted_valid = hard_valid.clone()
    attempted_valid[:, :-1] &= residual_allowed & projected_ok
    # Alpha zero is a genuine safety candidate independent of the residual
    # latent.  It may still be invalid if the supplied base seed is unsafe.
    attempted_valid[:, -1] &= task_input_valid

    first_valid_index = attempted_valid.to(torch.int64).argmax(dim=1)
    has_valid = attempted_valid.any(dim=1)
    fallback_index = torch.full_like(first_valid_index, n_alpha - 1)
    selected_index = torch.where(has_valid, first_valid_index, fallback_index)
    # A disabled gate is semantically alpha zero even when a numerically equal
    # nonzero-alpha candidate would pass every check.
    selected_index = torch.where(enabled_tensor, selected_index, fallback_index)

    row = torch.arange(batch_size, device=device)
    selected_valid = attempted_valid[row, selected_index]
    accepted_alpha = alphas[selected_index]
    selected_q = q_candidates[row, selected_index]
    result_q = base_q.clone()
    accepted_nonzero = (selected_index != n_alpha - 1) & selected_valid
    if bool(accepted_nonzero.any().item()):
        result_q[accepted_nonzero] = selected_q[accepted_nonzero]

    diagnostics = ResidualSeedDiagnostics(
        position_error=checks['position_error'][row, selected_index],
        cone_cosine=checks['cone_cosine'][row, selected_index],
        collision_margin=checks['collision_margin'][row, selected_index],
        joint_margin=checks['joint_margin'][row, selected_index],
        branch_distance=checks['branch_distance'][row, selected_index],
        basis_fallback=basis_fallback,
        finite=checks['finite'][row, selected_index],
        joint_limits=checks['joint_limits'][row, selected_index],
        position=checks['position'][row, selected_index],
        cone=checks['cone'][row, selected_index],
        collision_free=checks['collision_free'][row, selected_index],
        branch=checks['branch'][row, selected_index],
        projection_ok=projection_ok_by_alpha[row, selected_index],
        input_finite=task_input_valid & ((~enabled_tensor) | latent_finite),
        basis_finite=basis_finite,
        attempted_valid=attempted_valid,
        projection_ok_by_alpha=projection_ok_by_alpha,
        selected_index=selected_index,
    )
    return ResidualSeedResult(
        q=result_q,
        accepted_alpha=accepted_alpha,
        valid=selected_valid,
        diagnostics=diagnostics,
    )


__all__ = [
    'ResidualSeedConfig',
    'ResidualSeedDiagnostics',
    'ResidualSeedResult',
    'apply_residual_seed',
]
