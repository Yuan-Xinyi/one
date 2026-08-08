"""Cartesian path geometry for the extreme-motion-generation task.

The task path is given intrinsically: a start point ``p0``, an initial tangent
``d0``, a plane normal ``n`` (the same vector that anchors the tool-orientation
cone), and a signed curvature ``kappa`` [1/m]. ``kappa = 0`` is the straight
ray used in the ISRR submission; ``kappa != 0`` is a constant-curvature arc in
the plane through ``p0`` spanned by ``d0`` and ``n x d0``.

Both cases expose the same three quantities, which is all the environment
needs:

    tangent       unit vector along the direction of travel at the point of
                  the path closest to the current TCP;
    lateral_vec   vector from the current TCP to that closest point (always
                  orthogonal to ``tangent``, so feeding it back never
                  contributes along-path motion);
    lateral_dist  ``norm(lateral_vec)`` — the distance to the path, used by the
                  lateral safety net.

Anchoring the frame on ``n`` (a physical quantity: the workpiece-surface
normal) rather than on the Frenet normal keeps everything well defined at
``kappa = 0`` and free of the sign flip a Frenet frame suffers when the
curvature changes sign.

Sign convention (verified by ``_selftest`` below): with the arc centre placed
at ``c = p0 + (1/kappa) * (n x d0)``, the tangent at ``p0`` is ``+d0`` for both
signs of ``kappa``; ``kappa > 0`` turns towards ``n x d0``.
"""
from __future__ import annotations

import torch


KAPPA_EPS = 1e-6


def _serpentine_closest(u, v, amp, k, n_iter: int = 8):
    """Axis coordinate of the point of ``y = amp sin(k x)`` closest to (u, v).

    Newton on ``f'(x) = 0`` for ``f = (x-u)^2 + (amp sin(kx) - v)^2``, started
    at ``x = u``. The lateral safety net keeps the TCP within 2 cm of the path,
    so the minimum is unambiguous and the iteration has a single basin; the
    slope ``amp*k = tan(swing)`` stays below 1 for every sampled task.
    """
    x = u
    for _ in range(n_iter):
        s, c = torch.sin(k * x), torch.cos(k * x)
        r = amp * s - v
        f1 = 2.0 * (x - u) + 2.0 * amp * k * c * r
        f2 = 2.0 + 2.0 * ((amp * k * c) ** 2 - amp * k * k * s * r)
        x = x - f1 / f2.clamp_min(1e-6)
    return x


def path_frame(p: torch.Tensor,
               p0: torch.Tensor,
               d0: torch.Tensor,
               n_axis: torch.Tensor,
               kappa: torch.Tensor,
               amp: torch.Tensor | None = None,
               wavelen: torch.Tensor | None = None,
               eps: float = 1e-9
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Instantaneous tangent / lateral offset for a ray, an arc, or a wave.

    Args:
        p:       (B, 3) current TCP position.
        p0:      (B, 3) path origin.
        d0:      (B, 3) unit initial tangent, orthogonal to ``n_axis``.
        n_axis:  (B, 3) unit plane normal (the cone-constraint reference).
        kappa:   (B,)   signed curvature [1/m]; 0 = straight ray.
        amp:     (B,)   lateral amplitude [m] of the serpentine; 0 = not a wave.
        wavelen: (B,)   wavelength [m] of the serpentine.

    A task uses at most one of ``kappa`` and ``amp``. The serpentine is defined
    as a lateral offset from the straight axis rather than intrinsically by
    ``kappa(s)``, because that is how a weaving seam is specified and because it
    keeps the path non-self-intersecting and extending, so that arc length along
    the axis remains a well-defined and finite measure of how far the stroke
    reached.

    Returns:
        tangent (B, 3), lateral_vec (B, 3), lateral_dist (B,).
    """
    is_wave = (amp.abs() > KAPPA_EPS) if amp is not None else torch.zeros_like(
        kappa, dtype=torch.bool)
    is_line = (kappa.abs() < KAPPA_EPS) & ~is_wave

    # ---- straight ray branch -------------------------------------------
    delta = p - p0
    along = (delta * d0).sum(-1, keepdim=True)
    lat_line = (along * d0 - delta)          # = closest_point - p
    t_line = d0

    # ---- arc branch ------------------------------------------------------
    # Safe kappa so the 1/kappa never produces inf/nan on the unused branch.
    kappa_safe = torch.where(is_line, torch.ones_like(kappa), kappa)
    radius_signed = (1.0 / kappa_safe).unsqueeze(-1)          # (B, 1)
    m0 = torch.linalg.cross(n_axis, d0, dim=-1)               # (B, 3)
    centre = p0 + radius_signed * m0
    w = p - centre
    w_par = (w * n_axis).sum(-1, keepdim=True) * n_axis
    w_perp = w - w_par
    rho = w_perp.norm(dim=-1, keepdim=True).clamp_min(eps)
    u_hat = w_perp / rho
    lat_arc = (radius_signed.abs() - rho) * u_hat - w_par     # = closest - p
    t_arc = torch.sign(kappa_safe).unsqueeze(-1) * torch.linalg.cross(
        n_axis, u_hat, dim=-1)
    # Only the arc branch is renormalized: on the straight ray the tangent must
    # come through as the caller's d0, bit for bit, so that kappa = 0 is the
    # published pipeline and not a last-bit perturbation of it.
    t_arc = t_arc / t_arc.norm(dim=-1, keepdim=True).clamp_min(eps)

    # ---- serpentine branch ----------------------------------------------
    if amp is not None:
        amp_safe = torch.where(is_wave, amp, torch.zeros_like(amp))
        lam = wavelen.clamp_min(1e-3)
        k = (2.0 * torch.pi / lam)
        m0 = torch.linalg.cross(n_axis, d0, dim=-1)
        u = (delta * d0).sum(-1)
        v = (delta * m0).sum(-1)
        x = _serpentine_closest(u, v, amp_safe, k)
        s, c = torch.sin(k * x), torch.cos(k * x)
        closest = p0 + x.unsqueeze(-1) * d0 + (amp_safe * s).unsqueeze(-1) * m0
        lat_wave = closest - p
        yp = (amp_safe * k * c).unsqueeze(-1)
        t_wave = d0 + yp * m0
        t_wave = t_wave / t_wave.norm(dim=-1, keepdim=True).clamp_min(eps)
    else:
        lat_wave, t_wave = lat_line, t_line

    sel_line = is_line.unsqueeze(-1)
    sel_wave = is_wave.unsqueeze(-1)
    tangent = torch.where(sel_line, t_line, torch.where(sel_wave, t_wave, t_arc))
    lateral_vec = torch.where(sel_line, lat_line,
                              torch.where(sel_wave, lat_wave, lat_arc))
    return tangent, lateral_vec, lateral_vec.norm(dim=-1)


def serpentine_curvature(p: torch.Tensor, p0: torch.Tensor, d0: torch.Tensor,
                         n_axis: torch.Tensor, amp: torch.Tensor,
                         wavelen: torch.Tensor) -> torch.Tensor:
    """Signed curvature at the point of the wave closest to ``p``.

    This is the local rate at which the direction of travel rotates, which is
    what the policy needs at each step; the total turned angle is a property of
    the whole task and is used only for reporting.
    """
    lam = wavelen.clamp_min(1e-3)
    k = 2.0 * torch.pi / lam
    m0 = torch.linalg.cross(n_axis, d0, dim=-1)
    delta = p - p0
    x = _serpentine_closest((delta * d0).sum(-1), (delta * m0).sum(-1), amp, k)
    yp = amp * k * torch.cos(k * x)
    ypp = -amp * k * k * torch.sin(k * x)
    return ypp / (1.0 + yp * yp) ** 1.5


def serpentine_point(p0: torch.Tensor, d0: torch.Tensor, n_axis: torch.Tensor,
                     amp: torch.Tensor, wavelen: torch.Tensor,
                     x: torch.Tensor) -> torch.Tensor:
    """Point of the wave at axis coordinate ``x``."""
    k = 2.0 * torch.pi / wavelen.clamp_min(1e-3)
    m0 = torch.linalg.cross(n_axis, d0, dim=-1)
    return (p0 + x.unsqueeze(-1) * d0
            + (amp * torch.sin(k * x)).unsqueeze(-1) * m0)


def arc_point(p0: torch.Tensor, d0: torch.Tensor, n_axis: torch.Tensor,
              kappa: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Point on the path at arc length ``s`` (used for plotting / checks)."""
    is_line = kappa.abs() < KAPPA_EPS
    kappa_safe = torch.where(is_line, torch.ones_like(kappa), kappa)
    radius_signed = (1.0 / kappa_safe).unsqueeze(-1)
    m0 = torch.linalg.cross(n_axis, d0, dim=-1)
    centre = p0 + radius_signed * m0
    phi = (kappa_safe * s).unsqueeze(-1)
    # p0 - centre = -radius_signed * m0, rotated about n_axis by phi.
    r0 = -radius_signed * m0
    rot = (r0 * torch.cos(phi)
           + torch.linalg.cross(n_axis, r0, dim=-1) * torch.sin(phi))
    p_arc = centre + rot
    p_line = p0 + s.unsqueeze(-1) * d0
    return torch.where(is_line.unsqueeze(-1), p_line, p_arc)


def _selftest() -> None:  # pragma: no cover - run via __main__
    torch.manual_seed(0)
    B = 512
    d0 = torch.randn(B, 3, dtype=torch.float64)
    n_axis = torch.randn(B, 3, dtype=torch.float64)
    n_axis = n_axis / n_axis.norm(dim=-1, keepdim=True)
    d0 = d0 - (d0 * n_axis).sum(-1, keepdim=True) * n_axis
    d0 = d0 / d0.norm(dim=-1, keepdim=True)
    p0 = torch.randn(B, 3, dtype=torch.float64) * 0.5
    kappa = torch.cat([
        torch.zeros(B // 4, dtype=torch.float64),
        torch.linspace(0.2, 5.0, B // 4, dtype=torch.float64),
        -torch.linspace(0.2, 5.0, B // 4, dtype=torch.float64),
        torch.linspace(-4.0, 4.0, B - 3 * (B // 4), dtype=torch.float64),
    ])

    # 1. At p0 the tangent is d0 for either sign, and the lateral offset is 0.
    t, lat, dist = path_frame(p0, p0, d0, n_axis, kappa)
    assert (t - d0).abs().max() < 1e-9, (t - d0).abs().max()
    assert dist.max() < 1e-9, dist.max()

    # 2. arc_point is consistent with path_frame: walking arc length s along
    #    the path keeps lateral_dist at 0 and the tangent equal to ds/ds.
    s = torch.rand(B, dtype=torch.float64) * 0.8
    p_s = arc_point(p0, d0, n_axis, kappa, s)
    t_s, _, dist_s = path_frame(p_s, p0, d0, n_axis, kappa)
    assert dist_s.max() < 1e-8, dist_s.max()
    h = 1e-6
    fd = (arc_point(p0, d0, n_axis, kappa, s + h)
          - arc_point(p0, d0, n_axis, kappa, s - h)) / (2 * h)
    assert (fd - t_s).abs().max() < 1e-5, (fd - t_s).abs().max()

    # 3. lateral_vec is orthogonal to tangent everywhere (so lateral feedback
    #    injects no along-path velocity).
    p_off = p_s + torch.randn(B, 3, dtype=torch.float64) * 0.01
    t_o, lat_o, dist_o = path_frame(p_off, p0, d0, n_axis, kappa)
    assert (lat_o * t_o).sum(-1).abs().max() < 1e-9

    # 4. lateral_dist really is the distance to the path (compare against a
    #    dense arc-length search).
    s_grid = torch.linspace(-2.0, 2.0, 4001, dtype=torch.float64)
    idx = torch.randint(0, B, (32,))
    for i in idx.tolist():
        pts = arc_point(p0[i:i + 1].expand(4001, 3),
                        d0[i:i + 1].expand(4001, 3),
                        n_axis[i:i + 1].expand(4001, 3),
                        kappa[i:i + 1].expand(4001), s_grid)
        brute = (pts - p_off[i]).norm(dim=-1).min()
        assert abs(brute - dist_o[i]) < 2e-4, (i, brute.item(), dist_o[i].item())

    # 5. kappa -> 0 converges to the straight-ray branch. At |s| <= 2 m an arc
    #    of curvature kappa departs from its initial tangent by ~kappa*s^2/2,
    #    so the residual below is the genuine geometric difference, not error.
    for kap, tol in ((1e-5, 5e-5), (1e-7, 1e-6)):
        tiny = torch.full((B,), kap, dtype=torch.float64)
        t_t, _, d_t = path_frame(p_off, p0, d0, n_axis, tiny)
        t_l, _, d_l = path_frame(p_off, p0, d0, n_axis, torch.zeros_like(tiny))
        assert (t_t - t_l).abs().max() < tol, (kap, (t_t - t_l).abs().max())
        assert (d_t - d_l).abs().max() < tol, (kap, (d_t - d_l).abs().max())

    print("path_geometry selftest OK")


if __name__ == "__main__":
    _selftest()
