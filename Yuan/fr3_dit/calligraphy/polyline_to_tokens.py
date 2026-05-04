"""Convert a stroke polyline (world coords) to a 32-D token sequence + ancillary data
required for DiT inference and IK refine.

Mirrors ``stitch_composite_tasks.tokenize_subseg_path`` exactly. Output layout:

    tokens         (T, 32)   one START + per-segment + per-corner
    token_kinds    (T,) u8   0=start, 1=segment, 2=corner
    local_frame    (3, 3)    columns = (x̂=v̂₁, ŷ, ẑ=plane_normal)
    local_origin   (3,)      stroke start point in world frame (= polyline[0])
    seg_dirs_world list[(3,)] world-frame segment directions
    seg_lens       list[float] segment lengths in meters
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


TOKEN_DIM = 32
TOKEN_KIND_START   = 0
TOKEN_KIND_SEGMENT = 1
TOKEN_KIND_CORNER  = 2

# Field offsets (matches stitch_composite_tasks.TOKEN_LAYOUT)
OFF_KIND   = 0   # 3
OFF_DIR    = 3   # 3
OFF_LEN    = 6   # 1
OFF_DELTA  = 7   # 2 (sin, cos)
OFF_AXIS   = 9   # 3
OFF_BISECT = 12  # 3
OFF_PNL    = 15  # 3 (plane normal in local frame)
OFF_CUMLEN = 18  # 1
OFF_FOURIER= 19  # 8
# 27..32 = pad (zeros)

DEFAULT_LENGTH_REF = 0.30
DEFAULT_FOURIER_BANDS = 4


def fourier_features(t: float, bands: int) -> np.ndarray:
    """Replicate the Fourier feature encoding used at training time."""
    out = np.zeros(2 * bands, dtype=np.float32)
    for k in range(bands):
        omega = 2.0 * np.pi * (2 ** k)
        out[2 * k]     = np.sin(omega * t)
        out[2 * k + 1] = np.cos(omega * t)
    return out


def build_local_frame(first_seg_dir_world: np.ndarray, plane_normal_world: np.ndarray) -> np.ndarray:
    """x̂ = v̂₁ (first segment dir), ẑ = plane_normal after Gram-Schmidt vs x̂, ŷ = ẑ × x̂.
    Returns 3×3 with columns (x̂, ŷ, ẑ)."""
    x = first_seg_dir_world / max(float(np.linalg.norm(first_seg_dir_world)), 1e-12)
    n = np.asarray(plane_normal_world, dtype=np.float64)
    z = n - x * float(np.dot(n, x))
    z_norm = float(np.linalg.norm(z))
    if z_norm < 1e-9:
        # Degenerate: first seg is parallel to plane normal. Fall back to a perpendicular helper.
        helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        z = helper - x * float(np.dot(helper, x))
        z = z / max(float(np.linalg.norm(z)), 1e-12)
    else:
        z = z / z_norm
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float64)


def _token_fill(
    kind: int,
    dir_local: np.ndarray,
    length_m: float,
    delta_theta: float,
    axis_local: np.ndarray,
    bisector_local: np.ndarray,
    plane_normal_local: np.ndarray,
    cum_len_m: float,
    length_ref: float,
    fourier_bands: int,
) -> np.ndarray:
    tok = np.zeros(TOKEN_DIM, dtype=np.float32)
    tok[OFF_KIND + kind] = 1.0
    tok[OFF_DIR : OFF_DIR + 3] = dir_local
    tok[OFF_LEN] = length_m / length_ref
    tok[OFF_DELTA] = float(np.sin(delta_theta))
    tok[OFF_DELTA + 1] = float(np.cos(delta_theta))
    tok[OFF_AXIS : OFF_AXIS + 3] = axis_local
    tok[OFF_BISECT : OFF_BISECT + 3] = bisector_local
    tok[OFF_PNL : OFF_PNL + 3] = plane_normal_local
    tok[OFF_CUMLEN] = cum_len_m / length_ref
    tok[OFF_FOURIER : OFF_FOURIER + 2 * fourier_bands] = fourier_features(
        cum_len_m / length_ref, fourier_bands
    )
    return tok


@dataclass
class TokenizedStroke:
    tokens: np.ndarray            # (T, 32) float32
    token_kinds: np.ndarray       # (T,) uint8
    local_frame: np.ndarray       # (3, 3) float64, cols = (x̂, ŷ, ẑ)
    local_origin: np.ndarray      # (3,) float32 — stroke start world XYZ (path-start TCP target)
    seg_dirs_world: List[np.ndarray]
    seg_lens: List[float]
    n_tokens: int
    n_segments: int


def tokenize_stroke(
    polyline_world: np.ndarray,        # (n_vertices, 3) world coords
    plane_normal_world: np.ndarray,    # (3,) outward desk normal
    length_ref: float = DEFAULT_LENGTH_REF,
    fourier_bands: int = DEFAULT_FOURIER_BANDS,
    spatial_anchor_xy_norm: tuple[float, float] | None = None,
) -> TokenizedStroke:
    """Convert a single stroke polyline (vertex sequence in world frame) to the
    32-D token sequence used by the DiT model.

    If ``spatial_anchor_xy_norm`` is provided, the START token's ``dir_local`` slot
    is overwritten with ``(x_norm, y_norm, 0)`` to mirror ``add_spatial_anchor.py``.
    """
    pts = np.asarray(polyline_world, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError(f"polyline must be (n>=2, 3); got {pts.shape}")

    # Per-segment direction + length (world frame).
    seg_dirs_world: List[np.ndarray] = []
    seg_lens: List[float] = []
    for i in range(pts.shape[0] - 1):
        delta = pts[i + 1] - pts[i]
        L = float(np.linalg.norm(delta))
        if L < 1e-6:
            continue
        seg_dirs_world.append((delta / L).astype(np.float32))
        seg_lens.append(L)
    if not seg_dirs_world:
        raise ValueError("polyline has no non-degenerate segments")

    # Local frame: x̂ = first seg dir, ẑ = plane_normal (orthonormalized).
    R = build_local_frame(seg_dirs_world[0], plane_normal_world)
    R_T = R.T
    plane_normal_local = (R_T @ np.asarray(plane_normal_world, dtype=np.float64)).astype(np.float32)
    seg_dirs_local = [R_T @ d.astype(np.float64) for d in seg_dirs_world]

    tokens: List[np.ndarray] = []
    kinds: List[int] = []

    # START token
    start_dir_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if spatial_anchor_xy_norm is not None:
        start_dir_local = np.array(
            [float(spatial_anchor_xy_norm[0]), float(spatial_anchor_xy_norm[1]), 0.0],
            dtype=np.float32,
        )
    tokens.append(_token_fill(
        kind=TOKEN_KIND_START, dir_local=start_dir_local,
        length_m=0.0, delta_theta=0.0,
        axis_local=np.zeros(3, dtype=np.float32),
        bisector_local=np.zeros(3, dtype=np.float32),
        plane_normal_local=plane_normal_local,
        cum_len_m=0.0,
        length_ref=length_ref, fourier_bands=fourier_bands,
    ))
    kinds.append(TOKEN_KIND_START)

    cum_len = 0.0
    for k, (d_local, L) in enumerate(zip(seg_dirs_local, seg_lens)):
        # Segment token.
        tokens.append(_token_fill(
            kind=TOKEN_KIND_SEGMENT,
            dir_local=d_local.astype(np.float32),
            length_m=L, delta_theta=0.0,
            axis_local=np.zeros(3, dtype=np.float32),
            bisector_local=np.zeros(3, dtype=np.float32),
            plane_normal_local=plane_normal_local,
            cum_len_m=cum_len,
            length_ref=length_ref, fourier_bands=fourier_bands,
        ))
        kinds.append(TOKEN_KIND_SEGMENT)
        cum_len += L

        # Corner token (between consecutive segments).
        if k + 1 < len(seg_dirs_local):
            v1, v2 = d_local, seg_dirs_local[k + 1]
            cross = np.cross(v1, v2)
            sin_t = float(np.dot(cross, plane_normal_local.astype(np.float64)))
            cos_t = float(np.dot(v1, v2))
            delta_theta = float(np.arctan2(sin_t, cos_t))
            n_cross = float(np.linalg.norm(cross))
            axis = cross / max(n_cross, 1e-12)
            bisect = v1 + v2
            n_bisect = float(np.linalg.norm(bisect))
            bisect = bisect / max(n_bisect, 1e-12)
            tokens.append(_token_fill(
                kind=TOKEN_KIND_CORNER,
                dir_local=np.zeros(3, dtype=np.float32),
                length_m=0.0, delta_theta=delta_theta,
                axis_local=axis.astype(np.float32),
                bisector_local=bisect.astype(np.float32),
                plane_normal_local=plane_normal_local,
                cum_len_m=cum_len,
                length_ref=length_ref, fourier_bands=fourier_bands,
            ))
            kinds.append(TOKEN_KIND_CORNER)

    return TokenizedStroke(
        tokens=np.stack(tokens, axis=0).astype(np.float32),
        token_kinds=np.asarray(kinds, dtype=np.uint8),
        local_frame=R,
        local_origin=pts[0].astype(np.float32).copy(),
        seg_dirs_world=seg_dirs_world,
        seg_lens=seg_lens,
        n_tokens=len(tokens),
        n_segments=len(seg_dirs_world),
    )


if __name__ == "__main__":
    # Quick test: tokenize 中's 4 strokes at 8cm and report token shapes.
    from Yuan.fr3_dit.calligraphy.character_def import place_character

    desk_center = np.array([0.5, 0.0, -0.05], dtype=np.float32)
    desk_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    strokes_world = place_character("中", desk_center, desk_normal, size_m=0.08)

    print(f"中 @ 8cm — {len(strokes_world)} strokes:")
    for i, poly in enumerate(strokes_world):
        ts = tokenize_stroke(poly, desk_normal)
        print(f"  stroke {i+1}: vertices={poly.shape[0]}  segments={ts.n_segments}  "
              f"tokens={ts.n_tokens}  start_xyz={ts.local_origin.round(3).tolist()}  "
              f"seg_lens_cm={[round(L*100, 1) for L in ts.seg_lens]}")
