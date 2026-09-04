"""Sphere collision for the Nova2 drag env.

Subclasses IJRR's ChainSphereCollision to (a) load the Qling-generated
sphere sets (drag/data/spheres/nova2, 8 records: 0..6 arm links +
7 gripper) instead of the IJRR spheres directory, and (b) add a table
half-space margin. All pairwise-margin math is inherited unchanged.

Callers pass AUGMENTED link transforms (B, 8, 4, 4): the 7 chain frames
plus a copy of the flange frame (index 7) carrying the gripper spheres.
Assigning the gripper its own link index means gripper-vs-link4 pairs
are checked (distance 3 > 2) while gripper-vs-wrist stays ignored,
consistent with the generic distance-2 ignore rule.

Table margin uses arm links 2..6 only:
  - base/shoulder (0, 1) always straddle the table plane by construction;
  - the gripper legitimately operates millimetres above the table while
    grasping, and its table clearance is constant during an episode
    (z and tilt locked), so it is excluded on purpose.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .ijrr_root import add_ijrr_path
add_ijrr_path()
from Yuan.IJRR.kinematics.chain_sphere_collision import (  # noqa: E402
    ChainSphereCollision)

SPHERE_DIR = Path(__file__).resolve().parent / 'data' / 'spheres' / 'nova2'
N_RECORDS = 8


class Nova2DragCollision(ChainSphereCollision):

    def __init__(self, device=None, dtype=torch.float32,
                 margin: float = 0.0):
        # replicate the parent's loading from the Qling sphere dir
        # (the parent hardcodes the IJRR spheres path); mask building
        # and all margin math below are the inherited implementations.
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.margin = float(margin)
        centers, radii, link_indices = [], [], []
        for link_idx in range(N_RECORDS):
            records = json.loads(
                (SPHERE_DIR / f'link{link_idx}-spheres.json').read_text())
            for sph in records[0]['spheres']:
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
        self.table_sel = (self.link_indices >= 2) & (self.link_indices <= 6)

    @staticmethod
    def augment(link_tfs: torch.Tensor) -> torch.Tensor:
        """(B, 7, 4, 4) chain frames -> (B, 8, 4, 4) with the gripper
        record riding on a copy of the flange frame."""
        return torch.cat([link_tfs, link_tfs[:, 6:7]], dim=1)

    def table_margin(self, link_tfs_aug: torch.Tensor,
                     table_z: float) -> torch.Tensor:
        pos = self.sphere_positions(link_tfs_aug)
        m = pos[..., 2] - self.radii - table_z
        return m[:, self.table_sel].amin(dim=1)

    def combined_margin(self, link_tfs_aug: torch.Tensor,
                        table_z: float) -> torch.Tensor:
        return torch.minimum(self.min_margin(link_tfs_aug),
                             self.table_margin(link_tfs_aug, table_z))
