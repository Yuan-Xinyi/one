"""Batched sphere self-collision for the FR3 arm.

The sphere sets are copied from ``franka_research_3/collision_spheres`` and
kept in this FR3 package so code using ``one.robots...fr3`` has a local,
tensor-friendly collision approximation. This checker is deliberately
lightweight: it consumes batched link transforms and evaluates all relevant
sphere-sphere distances in torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


FR3_LINK_NAMES = tuple(f'link{i}' for i in range(8))


def _default_sphere_dir() -> Path:
    return Path(__file__).resolve().parent / 'collision_spheres'


def load_link_spheres(sphere_dir: str | Path | None = None,
                      device=None,
                      dtype=torch.float32):
    """Load local FR3 link spheres.

    Returns:
        centers: ``(N, 3)`` local centers.
        radii: ``(N,)`` sphere radii.
        link_indices: ``(N,)`` owning link index, link0..link7.
    """
    sphere_dir = _default_sphere_dir() if sphere_dir is None else Path(sphere_dir)
    centers, radii, link_indices = [], [], []
    for link_idx, link_name in enumerate(FR3_LINK_NAMES):
        path = sphere_dir / f'{link_name}-spheres.json'
        with open(path, 'r') as f:
            records = json.load(f)
        # Existing files store one tuned record with a "spheres" list.
        spheres = records[0]['spheres'] if isinstance(records, list) else records['spheres']
        for sph in spheres:
            centers.append(sph['origin'])
            radii.append(float(sph['radius']))
            link_indices.append(link_idx)
    return (torch.as_tensor(centers, device=device, dtype=dtype),
            torch.as_tensor(radii, device=device, dtype=dtype),
            torch.as_tensor(link_indices, device=device, dtype=torch.long))


def fr3_self_collision_mask(link_indices: torch.Tensor,
                            ignore_same_or_adjacent: bool = True,
                            ignore_fr3_pairs: bool = True) -> torch.Tensor:
    """Build an upper-triangular sphere pair mask for FR3 self-collision."""
    li = link_indices[:, None]
    lj = link_indices[None, :]
    mask = li != lj
    if ignore_same_or_adjacent:
        mask = mask & ((li - lj).abs() > 1)
    if ignore_fr3_pairs:
        # Mirrors ignore_collision pairs in fr3.py. Connected links are already
        # covered by the adjacent-link rule above.
        ignored = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)]
        for a, b in ignored:
            pair = ((li == a) & (lj == b)) | ((li == b) & (lj == a))
            mask = mask & ~pair
    upper = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
    return mask & upper


class FR3SphereCollision:
    """Torch batched sphere collision approximation for bare FR3."""

    def __init__(self, sphere_dir: str | Path | None = None,
                 device=None,
                 dtype=torch.float32,
                 margin: float = 0.0):
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.margin = float(margin)
        self.centers, self.radii, self.link_indices = load_link_spheres(
            sphere_dir, device=self.device, dtype=self.dtype)
        self.mask = fr3_self_collision_mask(self.link_indices)

    def sphere_positions(self, link_tfs: torch.Tensor) -> torch.Tensor:
        """Return world sphere centers from link transforms.

        Args:
            link_tfs: ``(B, 8, 4, 4)`` transforms for link0..link7.
        """
        link_tfs = link_tfs.to(device=self.device, dtype=self.dtype)
        tfs = link_tfs[:, self.link_indices]
        centers_h = torch.cat([
            self.centers,
            torch.ones((self.centers.shape[0], 1),
                       device=self.device, dtype=self.dtype),
        ], dim=-1)
        return (tfs @ centers_h.view(1, -1, 4, 1)).squeeze(-1)[..., :3]

    def margins(self, link_tfs: torch.Tensor) -> torch.Tensor:
        """Return pairwise signed margins ``distance - radius_sum``."""
        centers = self.sphere_positions(link_tfs)
        diff = centers[:, :, None, :] - centers[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        radius_sum = self.radii[:, None] + self.radii[None, :]
        return dist - radius_sum

    def min_margin(self, link_tfs: torch.Tensor) -> torch.Tensor:
        margins = self.margins(link_tfs)
        masked = torch.where(self.mask.unsqueeze(0), margins,
                             torch.full_like(margins, 1e6))
        return masked.amin(dim=(1, 2))

    def is_collided(self, link_tfs: torch.Tensor,
                    margin: float | None = None) -> torch.Tensor:
        threshold = self.margin if margin is None else float(margin)
        return self.min_margin(link_tfs) < threshold
