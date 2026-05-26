"""Module 4: task perturbation.

A task ``c = (p0, line_dir, n_target)`` is the line-following task spec used
by ``LineDistribution``. ``line_dir ⊥ n_target`` is a hard constraint of the
task definition — the perturbation must preserve it.

Rotation order (so d ⊥ n is preserved without re-projection drift):
    1. Sample ``θ_n ∈ [-perturb_n_deg, +perturb_n_deg]``; rotate ``n`` around
       the OLD ``d`` axis → ``n'``.
    2. Re-project ``d`` onto the plane ⊥ ``n'`` and renormalize → ``d_tmp``.
    3. Sample ``θ_d ∈ [-perturb_d_deg, +perturb_d_deg]``; rotate ``d_tmp``
       around ``n'`` → ``d'``  (stays ⊥ ``n'`` by construction).
    4. Sample ``Δp ∈ ball(perturb_p0_mm/1000)``  → ``p0' = p0 + Δp``.

Step 2 is what makes the order "n first, then d". If we rotated d first then
n, d would no longer be in the new normal plane and we'd need a second
re-projection. Doing n first cuts that to one projection.
"""
from __future__ import annotations

import torch


def _rodrigues(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotation matrix for rotating by ``theta`` around unit ``axis``.

    axis: (3,), unit. theta: scalar tensor. Returns (3, 3)."""
    a = axis / axis.norm().clamp_min(1e-12)
    c, s = torch.cos(theta), torch.sin(theta)
    K = torch.tensor([[0.0, -a[2], a[1]],
                      [a[2], 0.0, -a[0]],
                      [-a[1], a[0], 0.0]],
                     dtype=axis.dtype, device=axis.device)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + s * K + (1.0 - c) * (K @ K)


def _sample_in_ball(radius: float,
                    dtype: torch.dtype,
                    device: torch.device,
                    generator: torch.Generator | None) -> torch.Tensor:
    """Uniform sample inside a ball of radius ``radius`` (in same units)."""
    if radius <= 0.0:
        return torch.zeros(3, dtype=dtype, device=device)
    # Direction: unit-norm 3D Gaussian.
    v = torch.randn(3, dtype=dtype, device=device, generator=generator)
    v = v / v.norm().clamp_min(1e-12)
    # Radius: r = R * U^(1/3) for uniform-in-volume.
    u = torch.rand((), dtype=dtype, device=device, generator=generator)
    r = radius * u.pow(1.0 / 3.0)
    return r * v


def perturb_task(
    c: dict[str, torch.Tensor],
    *,
    perturb_d_deg: float = 0.0,
    perturb_n_deg: float = 0.0,
    perturb_p0_mm: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a perturbed task ``c' = (p0', line_dir', n_target')``.

    The three perturbation magnitudes are independent caps; each axis-aligned
    component is uniformly sampled in ``[-cap, +cap]`` (or ball(cap) for p0).

    Pass ``perturb_*=0`` on any axis to disable that perturbation.

    Args:
        c: dict with keys ``p0`` (3,), ``line_dir`` (3,), ``n_target`` (3,).
           Tensors must be unit-norm for line_dir / n_target and broadly
           ⊥ (small numerical drift OK; we re-orthogonalize).
        perturb_d_deg, perturb_n_deg: max rotation angles in DEGREES.
        perturb_p0_mm: max translation magnitude in MILLIMETERS.
        generator: torch.Generator for reproducibility.

    Returns:
        Dict with the same keys; tensors share dtype/device with the input.
    """
    p0 = c["p0"]
    d = c["line_dir"]
    n = c["n_target"]
    dtype, device = p0.dtype, p0.device

    # Step 1: rotate n around the OLD d axis.
    if perturb_n_deg > 0.0:
        cap_n = float(perturb_n_deg) * torch.pi / 180.0
        theta_n = (torch.rand((), dtype=dtype, device=device, generator=generator)
                   * 2.0 - 1.0) * cap_n
        R_n = _rodrigues(d, theta_n)
        n_new = R_n @ n
        n_new = n_new / n_new.norm().clamp_min(1e-12)
    else:
        n_new = n / n.norm().clamp_min(1e-12)

    # Step 2: re-orthogonalize d against n_new.
    d_tmp = d - (d * n_new).sum() * n_new
    d_tmp_norm = d_tmp.norm()
    if float(d_tmp_norm) < 1e-6:
        # n was rotated almost into d's span. Fall back to a stable perpendicular.
        # Pick world x or y, whichever is more ⊥ to n_new.
        wx = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        wy = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
        fallback = wx if abs(float((wx * n_new).sum())) < abs(float((wy * n_new).sum())) else wy
        d_tmp = fallback - (fallback * n_new).sum() * n_new
        d_tmp_norm = d_tmp.norm()
    d_tmp = d_tmp / d_tmp_norm.clamp_min(1e-12)

    # Step 3: rotate d_tmp around n_new.
    if perturb_d_deg > 0.0:
        cap_d = float(perturb_d_deg) * torch.pi / 180.0
        theta_d = (torch.rand((), dtype=dtype, device=device, generator=generator)
                   * 2.0 - 1.0) * cap_d
        R_d = _rodrigues(n_new, theta_d)
        d_new = R_d @ d_tmp
        d_new = d_new / d_new.norm().clamp_min(1e-12)
    else:
        d_new = d_tmp

    # Step 4: translate p0.
    delta = _sample_in_ball(perturb_p0_mm * 1e-3, dtype, device, generator)
    p0_new = p0 + delta

    return {"p0": p0_new, "line_dir": d_new, "n_target": n_new}
