"""SeedSelectionDataset — random-label sampling for (c → q0) training.

Loads tasks from a pilot NPZ built by `dataset_builder.py` and exposes one
(c, q0) pair per `__getitem__`. For tasks with multiple SMM labels, q0 is
sampled uniformly at random from `labels_q0[:n_labels]` on every call, so
multi-modal targets stay multi-modal under SGD.

Filters tasks by `status` (default keeps {kept, edge, edge_seed_fallback}),
which drops infeasible (fallback q0 from line_distribution) and low_quality
tasks where labels exist but their L_clean is below acceptable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_KEEP_STATUSES = ('kept', 'edge', 'edge_seed_fallback')

# FR3 has a left-right symmetry across its xz plane (y → -y). Flipping these
# joints + flipping the y component of (p0, line_dir, n_target) gives an
# equivalent valid task with the same physics: free 2× data via reflection.
# Same convention as fr3_dit's StartQDataset / KeypointDataset.
_FLIP_MULT_Q = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float32)


class SeedSelectionDataset(Dataset):
    """One (c, q0) pair per __getitem__, q0 sampled at random from valid labels.

    c   : (9,) float32 — concat of (cs_p0, cs_line_dir, cs_n_target)
    q0  : (7,) float32 — one of `labels_q0[:n_labels]`, sampled uniformly

    Optional mirror augmentation: with probability ``mirror_prob`` each
    __getitem__ call reflects the entire task across the robot's xz plane —
    flipping y in (p0, line_dir, n_target) and applying ``_FLIP_MULT_Q`` to
    q0. Costs ~free and doubles the effective dataset size.
    """

    def __init__(
        self,
        npz_path: str | Path,
        keep_statuses: Iterable[str] = DEFAULT_KEEP_STATUSES,
        mirror_prob: float = 0.0,
        plane_collision_path: str | Path | None = 'auto',
        plane_collision_scope: str = 'labels_only',
    ):
        z = np.load(Path(npz_path), allow_pickle=False)
        status = z['status']
        n_labels = z['n_labels']
        keep = np.isin(status, list(keep_statuses)) & (n_labels >= 1)

        # Plane-collision filter: drop tasks where any q (q0_seed or any valid
        # label) penetrates the task plane. Use the side NPZ produced by
        # `check_plane_collision.py`. Pass `None` to disable.
        n_dropped_pc = 0
        if plane_collision_path is not None:
            if plane_collision_path == 'auto':
                pc_path = Path(npz_path).with_suffix('.plane_collision.npz')
            else:
                pc_path = Path(plane_collision_path)
            if pc_path.exists():
                pc = np.load(pc_path, allow_pickle=False)
                if plane_collision_scope == 'labels_only':
                    coll_flag = pc['any_label_collides'].astype(bool)
                elif plane_collision_scope == 'any_q':
                    coll_flag = pc['any_q_collides'].astype(bool)
                else:
                    raise ValueError(f"plane_collision_scope must be 'labels_only' or 'any_q', "
                                     f"got {plane_collision_scope!r}")
                if coll_flag.shape[0] != status.shape[0]:
                    raise RuntimeError(
                        f'plane_collision file {pc_path} has {coll_flag.shape[0]} '
                        f'entries but data NPZ has {status.shape[0]}')
                n_before = int(keep.sum())
                keep = keep & ~coll_flag
                n_dropped_pc = n_before - int(keep.sum())
                print(f'[SeedSelectionDataset] plane-collision filter ({plane_collision_scope}): '
                      f'dropped {n_dropped_pc} tasks (from {pc_path.name})')
            elif plane_collision_path != 'auto':
                # Explicit path that doesn't exist → fail loud.
                raise FileNotFoundError(f'plane_collision_path={pc_path} not found')

        idx = np.where(keep)[0]
        self.cs_p0 = z['cs_p0'][idx].astype(np.float32)
        self.cs_line_dir = z['cs_line_dir'][idx].astype(np.float32)
        self.cs_n_target = z['cs_n_target'][idx].astype(np.float32)
        self.labels_q0 = z['labels_q0'][idx].astype(np.float32)
        self.n_labels = n_labels[idx].astype(np.int32)
        self.mirror_prob = float(mirror_prob)
        self._npz_path = Path(npz_path)
        self._plane_collision_dropped = int(n_dropped_pc)

    def __len__(self) -> int:
        return int(self.n_labels.shape[0])

    def __getitem__(self, idx: int):
        n = int(self.n_labels[idx])
        label_idx = int(np.random.randint(0, n))
        c_p0 = self.cs_p0[idx].copy()
        c_d  = self.cs_line_dir[idx].copy()
        c_n  = self.cs_n_target[idx].copy()
        q0 = self.labels_q0[idx, label_idx].copy()
        if self.mirror_prob > 0.0 and np.random.rand() < self.mirror_prob:
            c_p0[1] = -c_p0[1]
            c_d[1]  = -c_d[1]
            c_n[1]  = -c_n[1]
            q0 = q0 * _FLIP_MULT_Q
        c = np.concatenate([c_p0, c_d, c_n])
        return torch.from_numpy(c.astype(np.float32)), torch.from_numpy(q0.astype(np.float32))
