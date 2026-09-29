"""Table paths: generic curves on a surface, sampled every DS metres.

A task path is described intrinsically in the moving frame (t, e2, n) of the
surface it lies on: t is the tangent, n the surface normal (the cone axis)
and e2 = n x t. Along the arc length s the frame rotates with the angular
velocity

    omega(s) = kg(s) n + kn(s) e2 + tau(s) t

where kg is the geodesic curvature (turning within the surface: arcs,
circles, waves, spirals), kn the normal curvature (the surface bends: a
cylinder, a bowl) and tau the twist (the normal rotates about the path: the
rotating-axis family, helices). Corners are finite rotations about n at a
point. Straight lines, arcs, serpentines, polygons, spirals and helices are
all members, so a policy trained on the random process below sees every
primitive the paper evaluates as a special case.

The environment reads a table path by the commanded arc length (the
protocol of the one-stroke figures): reference point, tangent and normal at
s = arc_progress, linearly interpolated between samples.
"""
from __future__ import annotations

import math

import torch

DS = 0.01            # sample spacing [m]
M = 201              # samples per path: 2.0 m
# Look-ahead offsets [m] of the preview observation: dense near, coarse far.
PREVIEW_DS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00, 1.40)


def _rodrigues(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Rotate v (B,3) by the rotation vector w (B,3)."""
    th = w.norm(dim=-1, keepdim=True)
    k = w / th.clamp_min(1e-12)
    c, s = torch.cos(th), torch.sin(th)
    kxv = torch.linalg.cross(k, v, dim=-1)
    kdv = (k * v).sum(-1, keepdim=True)
    out = v * c + kxv * s + k * kdv * (1.0 - c)
    return torch.where(th > 1e-9, out, v)


@torch.no_grad()
def integrate_frames(p0, t0, n0, kg, kn, tau, corner, ds: float = DS):
    """Integrate the moving frame along the arc.

    p0, t0, n0: (B,3) start point, tangent, normal (t0 orthogonal to n0).
    kg, kn, tau: (B, M-1) per-step rates [1/m, 1/m, rad/m].
    corner: (B, M-1) corner angle [rad] applied at the START of step i
            (0 = no corner).
    Returns pts, tan, nrm each (B, M, 3).
    """
    B, Mm1 = kg.shape
    P, t, n = p0.clone(), t0.clone(), n0.clone()
    pts, tan, nrm = [P.clone()], [t.clone()], [n.clone()]
    for i in range(Mm1):
        c = corner[:, i]
        if bool((c != 0).any()):
            t = _rodrigues(t, n * c.unsqueeze(-1))
            tan[-1] = t.clone()           # the corner belongs to sample i
        e2 = torch.linalg.cross(n, t, dim=-1)
        w = (kg[:, i:i + 1] * n + kn[:, i:i + 1] * e2 + tau[:, i:i + 1] * t) * ds
        # midpoint tangent for the position update, exact rotation of the frame
        t_new = _rodrigues(t, w)
        n_new = _rodrigues(n, w)
        P = P + 0.5 * (t + t_new) * ds
        t, n = t_new, n_new
        # re-orthonormalise against drift
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        t = t - (t * n).sum(-1, keepdim=True) * n
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pts.append(P.clone()); tan.append(t.clone()); nrm.append(n.clone())
    return torch.stack(pts, 1), torch.stack(tan, 1), torch.stack(nrm, 1)


@torch.no_grad()
def sample_curve_params(B: int, gen: torch.Generator, device, cfg: dict | None = None):
    """Random piecewise-constant (kg, kn, tau) processes with corners.

    cfg keys (defaults in brackets): p_straight [0.3] share of pure straight
    rays; seg_len [(0.15, 0.8)] m; p_kg [0.6], kg_max [6.0] 1/m; p_kn [0.2],
    kn_max [2.0]; p_tau [0.2], tau_max [1.5] rad/m; p_corner [0.3],
    corner_deg [(30, 150)]; n_seg_max [6].
    Returns kg, kn, tau, corner each (B, M-1).
    """
    c = dict(p_straight=0.3, seg_len=(0.15, 0.8), p_kg=0.6, kg_max=6.0,
             p_kn=0.2, kn_max=2.0, p_tau=0.2, tau_max=1.5, p_corner=0.3,
             corner_deg=(30.0, 150.0), n_seg_max=6)
    if cfg:
        c.update(cfg)
    Mm1 = M - 1
    f = lambda *s: torch.rand(*s, generator=gen, device=device)
    kg = torch.zeros(B, Mm1, device=device); kn = torch.zeros_like(kg)
    tau = torch.zeros_like(kg); corner = torch.zeros_like(kg)
    n_seg = int(c['n_seg_max'])
    # segment boundaries (in samples) per task
    L = c['seg_len'][0] + f(B, n_seg) * (c['seg_len'][1] - c['seg_len'][0])
    ends = torch.cumsum((L / DS).round().long(), dim=1).clamp(max=Mm1)   # (B, n_seg)
    starts = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), ends[:, :-1]], 1)
    idx = torch.arange(Mm1, device=device).view(1, 1, -1)
    inseg = (idx >= starts.unsqueeze(-1)) & (idx < ends.unsqueeze(-1))       # (B, n_seg, M-1)
    sgn = lambda: torch.where(f(B, n_seg) < 0.5, -1.0, 1.0)
    kg_s = torch.where(f(B, n_seg) < c['p_kg'], sgn() * (0.5 + f(B, n_seg) * (c['kg_max'] - 0.5)), 0.0)
    kn_s = torch.where(f(B, n_seg) < c['p_kn'], sgn() * (0.3 + f(B, n_seg) * (c['kn_max'] - 0.3)), 0.0)
    ta_s = torch.where(f(B, n_seg) < c['p_tau'], sgn() * (0.3 + f(B, n_seg) * (c['tau_max'] - 0.3)), 0.0)
    kg = (inseg * kg_s.unsqueeze(-1)).sum(1)
    kn = (inseg * kn_s.unsqueeze(-1)).sum(1)
    tau = (inseg * ta_s.unsqueeze(-1)).sum(1)
    # corners at segment starts (never at s = 0: the start tangent is the task's d)
    lo, hi = math.radians(c['corner_deg'][0]), math.radians(c['corner_deg'][1])
    ang = torch.where(f(B, n_seg) < c['p_corner'], sgn() * (lo + f(B, n_seg) * (hi - lo)), 0.0)
    ang[:, 0] = 0.0
    corner.scatter_add_(1, starts.clamp(max=Mm1 - 1), ang)
    corner[:, 0] = 0.0
    straight = f(B) < c['p_straight']
    for x in (kg, kn, tau, corner):
        x[straight] = 0.0
    return kg, kn, tau, corner


@torch.no_grad()
def build_tables(p0, t0, n0, gen, cfg=None, chunk: int = 8192):
    """Random table paths for B tasks starting at (p0, t0, n0); returns
    (pts, tan, nrm) each (B, M, 3) on p0's device."""
    B = p0.shape[0]
    out = [[], [], []]
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        kg, kn, tau, corner = sample_curve_params(hi - lo, gen, p0.device, cfg)
        pts, tan, nrm = integrate_frames(p0[lo:hi], t0[lo:hi], n0[lo:hi],
                                         kg.to(p0.dtype), kn.to(p0.dtype),
                                         tau.to(p0.dtype), corner.to(p0.dtype))
        for o, v in zip(out, (pts, tan, nrm)):
            o.append(v)
    return tuple(torch.cat(o, 0) for o in out)


def table_at(pts, tan, nrm, s):
    """Linear interpolation of the table at arc length s (B,); s is clamped
    to the table. Returns point, unit tangent, unit normal, each (B,3)."""
    x = (s / DS).clamp(0.0, M - 1 - 1e-6)
    i0 = x.floor().long()
    w = (x - i0.to(x.dtype)).unsqueeze(-1)
    i1 = (i0 + 1).clamp(max=M - 1)
    ar = torch.arange(pts.shape[0], device=pts.device)
    P = pts[ar, i0] * (1 - w) + pts[ar, i1] * w
    t = tan[ar, i0] * (1 - w) + tan[ar, i1] * w
    n = nrm[ar, i0] * (1 - w) + nrm[ar, i1] * w
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return P, t, n


def preview_features(P_s, t_s, n_s, P_k, n_k):
    """Preview channels: future points and cone axes in the path frame at s.

    P_s, t_s, n_s: (B,3) reference point / tangent / normal at the current arc;
    P_k, n_k: (B,K,3) points and normals at s + PREVIEW_DS.
    Returns (B, 6K): [(P_k-P_s).t, .e2, .n, n_k.t, n_k.e2, n_k.n] per k.
    """
    e2 = torch.linalg.cross(n_s, t_s, dim=-1)
    frame = torch.stack([t_s, e2, n_s], 1)                 # (B,3,3) rows = axes
    dp = torch.einsum('bij,bkj->bki', frame, P_k - P_s.unsqueeze(1))
    dn = torch.einsum('bij,bkj->bki', frame, n_k)
    return torch.cat([dp, dn], -1).reshape(P_s.shape[0], -1)


def _selftest():  # pragma: no cover
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    B = 64
    p0 = torch.zeros(B, 3); n0 = torch.tensor([0., 0., 1.]).expand(B, 3).clone()
    t0 = torch.tensor([1., 0., 0.]).expand(B, 3).clone()
    # pure circle: kg = 2 pi / L over the table -> closes after 1/kg*2pi
    kg = torch.full((B, M - 1), 2 * math.pi / 1.0); z = torch.zeros_like(kg)
    pts, tan, nrm = integrate_frames(p0, t0, n0, kg, z, z, z)
    i = int(round(1.0 / DS))
    assert (pts[:, i] - p0).norm(dim=-1).max() < 2e-3, (pts[:, i] - p0).norm(dim=-1).max()
    # straight
    pts, tan, nrm = integrate_frames(p0, t0, n0, z, z, z, z)
    assert (pts[:, -1] - torch.tensor([2.0, 0, 0])).norm(dim=-1).max() < 1e-5
    # random tables: unit frames, orthogonality, arc-length spacing
    pts, tan, nrm = build_tables(p0, t0, n0, g)
    assert ((tan * nrm).sum(-1).abs().max() < 1e-4)
    seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)
    assert (seg - DS).abs().max() < 2e-3, (seg - DS).abs().max()
    P, t, n = table_at(pts, tan, nrm, torch.full((B,), 0.505))
    assert (P - 0.5 * (pts[:, 50] + pts[:, 51])).abs().max() < 1e-6
    # preview on a straight ray = constants
    pts, tan, nrm = integrate_frames(p0, t0, n0, z, z, z, z)
    s = torch.zeros(B)
    Pk = torch.stack([table_at(pts, tan, nrm, s + d)[0] for d in PREVIEW_DS], 1)
    nk = torch.stack([table_at(pts, tan, nrm, s + d)[2] for d in PREVIEW_DS], 1)
    f = preview_features(pts[:, 0], tan[:, 0], nrm[:, 0], Pk, nk).view(B, len(PREVIEW_DS), 6)
    assert (f[:, :, 0] - torch.tensor(PREVIEW_DS)).abs().max() < 1e-6
    assert f[:, :, 1:5].abs().max() < 1e-6 and (f[:, :, 5] - 1).abs().max() < 1e-6
    print('path_table selftest OK')


if __name__ == '__main__':
    _selftest()
