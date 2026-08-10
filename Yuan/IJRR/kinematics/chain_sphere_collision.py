"""Batched sphere self-collision for any arm with generated sphere sets.

Same algorithm and interface as ``FR3SphereCollision`` --- consume batched
link transforms, evaluate masked pairwise sphere margins --- but loading the
sphere JSONs produced by ``gen_link_spheres.py`` and building the ignore
mask generically: same and adjacent links are ignored, and so are links two
apart, which matches both the FR3's hand-written ignore list and the
Cobotta model's (every listed pair there is a distance-2 pair; links that
meet at a joint overlap by construction and their contact is not a
collision).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _sphere_dir(robot: str) -> Path:
    return Path(__file__).resolve().parent / 'spheres' / robot


class ChainSphereCollision:
    """Torch batched sphere self-collision from generated sphere sets."""

    def __init__(self, robot: str, n_links: int, device=None,
                 dtype=torch.float32, margin: float = 0.0):
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.margin = float(margin)
        centers, radii, link_indices = [], [], []
        d = _sphere_dir(robot)
        for link_idx in range(n_links):
            records = json.loads((d / f'link{link_idx}-spheres.json')
                                 .read_text())
            spheres = (records[0]['spheres'] if isinstance(records, list)
                       else records['spheres'])
            for sph in spheres:
                centers.append(sph['origin'])
                radii.append(float(sph['radius']))
                link_indices.append(link_idx)
        self.centers = torch.as_tensor(centers, device=self.device,
                                       dtype=dtype)
        self.radii = torch.as_tensor(radii, device=self.device, dtype=dtype)
        self.link_indices = torch.as_tensor(link_indices, device=self.device,
                                            dtype=torch.long)
        li = self.link_indices[:, None]
        lj = self.link_indices[None, :]
        mask = (li - lj).abs() > 2
        upper = torch.triu(torch.ones_like(mask, dtype=torch.bool),
                           diagonal=1)
        self.mask = mask & upper

    def sphere_positions(self, link_tfs: torch.Tensor) -> torch.Tensor:
        link_tfs = link_tfs.to(device=self.device, dtype=self.dtype)
        tfs = link_tfs[:, self.link_indices]
        centers_h = torch.cat([
            self.centers,
            torch.ones((self.centers.shape[0], 1), device=self.device,
                       dtype=self.dtype),
        ], dim=-1)
        return (tfs @ centers_h.view(1, -1, 4, 1)).squeeze(-1)[..., :3]

    def margins(self, link_tfs: torch.Tensor) -> torch.Tensor:
        centers = self.sphere_positions(link_tfs)
        diff = centers[:, :, None, :] - centers[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        return dist - (self.radii[:, None] + self.radii[None, :])

    def min_margin(self, link_tfs: torch.Tensor) -> torch.Tensor:
        margins = self.margins(link_tfs)
        masked = torch.where(self.mask.unsqueeze(0), margins,
                             torch.full_like(margins, 1e6))
        return masked.amin(dim=(1, 2))

    def is_collided(self, link_tfs: torch.Tensor,
                    margin: float | None = None) -> torch.Tensor:
        threshold = self.margin if margin is None else float(margin)
        return self.min_margin(link_tfs) < threshold
