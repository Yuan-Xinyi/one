"""Adapters that present ``FR3SphereCollision`` in the legacy
``self_collision_fn(q) / sphere_positions_fn(q)`` shape.

The legacy ``wrs.SphereCollisionChecker`` API used in the fr3_dit pipeline was:

    self_collision_fn(q)            # (B,) cost; > 0 means colliding
    sphere_positions_fn(q)          # (B, n_spheres, 3) world sphere centers
    sphere_radii                    # (n_spheres,)
    sphere_link_indices             # (n_spheres,) which link each sphere belongs to

Both functions did FK internally (JAX-jitted, then bridged via ``jax2torch``).
The new ``FR3SphereCollision`` consumes pre-computed link transforms; this
module composes it with ``BatchedFR3Kinematics.link_transforms`` so call sites
keep their original signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics


@dataclass
class PenCollisionHelpers:
    self_collision_fn: Callable[[torch.Tensor], torch.Tensor]
    sphere_positions_fn: Callable[[torch.Tensor], torch.Tensor]
    sphere_radii: np.ndarray
    sphere_link_indices: np.ndarray
    checker: FR3SphereCollision


def make_pen_collision_helpers(robot: BatchedFR3Kinematics,
                               threshold: float = -0.005) -> PenCollisionHelpers:
    """Build (self_collision_fn, sphere_positions_fn, sphere_radii, sphere_link_indices).

    ``threshold`` is the allowed sphere-pair interpenetration: the cost is
    ``max(0, threshold - min_margin)`` so by default a 5 mm overlap is
    tolerated, mirroring the legacy ``cc.self_collision_cost(q, 1.0, -0.005)``.

    Note: the FR3 sphere set covers link0..link7 only — the original wrs URDF
    additionally encoded the Franka Hand and the pen as sphere links. Plane-
    clearance and self-collision checks here are therefore evaluated on the
    bare arm; downstream code that uses ``sphere_link_indices`` to derive
    ``protected_from_plane`` keeps working with the smaller set.
    """
    checker = FR3SphereCollision(device=robot.device, dtype=robot.dtype)

    def self_collision_fn(q: torch.Tensor) -> torch.Tensor:
        link_tfs = robot.link_transforms(q)
        min_m = checker.min_margin(link_tfs)
        return torch.clamp(threshold - min_m, min=0.0)

    def sphere_positions_fn(q: torch.Tensor) -> torch.Tensor:
        link_tfs = robot.link_transforms(q)
        return checker.sphere_positions(link_tfs)

    return PenCollisionHelpers(
        self_collision_fn=self_collision_fn,
        sphere_positions_fn=sphere_positions_fn,
        sphere_radii=np.asarray(checker.radii.detach().cpu().numpy(), dtype=np.float32),
        sphere_link_indices=np.asarray(checker.link_indices.detach().cpu().numpy(), dtype=np.int64),
        checker=checker,
    )
