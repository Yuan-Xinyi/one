"""Canonical character definitions + scene-placement transform.

Each character is a list of polylines (one per pen-down stroke). Polylines are
expressed in **canonical 2-D coordinates** in [-1, 1] × [-1, 1] (origin at character
centre, +x → right, +y → up). The ``place_character`` helper maps canonical → world:

    world_xyz_i = desk_origin + R(θ) @ (size * canonical_xy_i, 0) + (lift_z=0 on desk plane)

Output of every helper is a list of ``np.ndarray`` of shape ``(n_vertices, 3)``
in world coordinates, ready to be fed to ``polyline_to_tokens``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


# ---- Canonical character library -----------------------------------------------

# Each entry: list of polylines; each polyline = list of (x, y) vertices in canonical [-1, 1].
# Stroke order matches standard Chinese stroke order; each stroke gets a single pen-down.
CANONICAL_CHARACTERS: Dict[str, List[List[tuple[float, float]]]] = {
    "一": [
        [(-0.7, 0.0), (0.7, 0.0)],                     # single horizontal stroke
    ],
    "二": [
        [(-0.5, 0.5), (0.5, 0.5)],                     # top short horizontal
        [(-0.7, -0.5), (0.7, -0.5)],                   # bottom long horizontal
    ],
    "十": [
        [(-0.7, 0.0), (0.7, 0.0)],                     # horizontal
        [(0.0, 0.7), (0.0, -0.7)],                     # vertical
    ],
    "万": [
        # 笔1 一 (top horizontal)
        [(-0.7, 0.7), (0.7, 0.7)],
        # 笔2 横折 (horizontal + vertical down). The character traditionally has the
        # horizontal portion overlap the top stroke; we offset slightly to give DiT
        # cleaner segments.
        [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.6)],
        # 笔3 撇 (leftward diagonal)
        [(0.2, 0.3), (-0.7, -0.7)],
    ],
    "日": [
        # outline: top→right→bottom→left→top (single closed loop)
        [(-0.5, 0.7), (0.5, 0.7), (0.5, -0.7), (-0.5, -0.7), (-0.5, 0.7)],
        # middle horizontal
        [(-0.5, 0.0), (0.5, 0.0)],
    ],
    "中": [
        # 笔1 竖 (left vertical of the box): A → D
        [(-0.5, 0.7), (-0.5, -0.4)],
        # 笔2 横折 (top horizontal + right vertical): A → B → C
        [(-0.5, 0.7), (0.5, 0.7), (0.5, -0.4)],
        # 笔3 横 (bottom horizontal of box): D → C
        [(-0.5, -0.4), (0.5, -0.4)],
        # 笔4 中央竖 (center vertical, extends above and below the box): E → F
        [(0.0, 1.0), (0.0, -1.0)],
    ],
}


def list_characters() -> list[str]:
    """Return all characters defined in the canonical library."""
    return sorted(CANONICAL_CHARACTERS.keys())


def get_canonical(name: str) -> List[List[tuple[float, float]]]:
    if name not in CANONICAL_CHARACTERS:
        raise KeyError(f"Character {name!r} not defined; known: {list_characters()}")
    return CANONICAL_CHARACTERS[name]


# ---- Scene placement -----------------------------------------------------------


def place_character(
    name: str,
    desk_center: np.ndarray,            # (3,) — world XYZ of canvas centre on desk plane
    desk_normal: np.ndarray,            # (3,) — outward desk normal (the side robot reaches from)
    size_m: float = 0.10,               # character bounding-box half-extent in meters
    theta_rad: float = 0.0,             # rotation around desk_normal in canvas plane (CCW)
    pen_into_desk_offset: float = 0.0,  # tiny offset along -desk_normal so pen tip is ON the desk
) -> List[np.ndarray]:
    """Map a named character's canonical polylines to world coordinates on the desk plane.

    Returns a list of polylines, each a (n_vertices, 3) np.ndarray in world frame.
    """
    polylines = get_canonical(name)
    desk_center = np.asarray(desk_center, dtype=np.float64).reshape(3)
    desk_normal = np.asarray(desk_normal, dtype=np.float64).reshape(3)
    desk_normal = desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)

    # Build an orthonormal basis on the desk plane.
    # u = direction in the desk plane that maps canonical x; v maps canonical y.
    helper = np.array([1.0, 0.0, 0.0]) if abs(desk_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u0 = helper - desk_normal * float(np.dot(helper, desk_normal))
    u0 = u0 / max(float(np.linalg.norm(u0)), 1e-12)
    v0 = np.cross(desk_normal, u0)
    v0 = v0 / max(float(np.linalg.norm(v0)), 1e-12)

    # In-plane rotation by theta_rad around desk_normal (CCW when viewed from +desk_normal).
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    u = c * u0 + s * v0
    v = -s * u0 + c * v0

    origin = desk_center - desk_normal * float(pen_into_desk_offset)
    out: List[np.ndarray] = []
    for poly in polylines:
        pts = np.zeros((len(poly), 3), dtype=np.float32)
        for i, (cx, cy) in enumerate(poly):
            pts[i] = origin + u * (size_m * float(cx)) + v * (size_m * float(cy))
        out.append(pts)
    return out


# ---- Stroke geometry helpers ---------------------------------------------------


def stroke_segments(polyline: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Decompose a polyline into (direction_world, length_m) per straight segment.

    Skips zero-length and colinear duplicates.
    """
    pts = np.asarray(polyline, dtype=np.float32)
    out = []
    for i in range(pts.shape[0] - 1):
        delta = pts[i + 1] - pts[i]
        L = float(np.linalg.norm(delta))
        if L < 1e-6:
            continue
        out.append(((delta / L).astype(np.float32), L))
    return out


def total_length(polyline: np.ndarray) -> float:
    return float(sum(L for _, L in stroke_segments(polyline)))


if __name__ == "__main__":
    # Quick sanity check
    desk_center = np.array([0.5, 0.0, -0.05], dtype=np.float32)
    desk_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for name in list_characters():
        strokes = place_character(name, desk_center, desk_normal, size_m=0.08, theta_rad=0.0)
        total_len_cm = sum(total_length(s) for s in strokes) * 100
        print(f"{name}: {len(strokes)} strokes, "
              f"per-stroke segs={[len(stroke_segments(s)) for s in strokes]}, "
              f"total {total_len_cm:.1f}cm")
