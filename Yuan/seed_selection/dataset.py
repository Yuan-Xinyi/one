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


class SeedSelectionDataset(Dataset):
    """One (c, q0) pair per __getitem__, q0 sampled at random from valid labels.

    c   : (9,) float32 — concat of (cs_p0, cs_line_dir, cs_n_target)
    q0  : (7,) float32 — one of `labels_q0[:n_labels]`, sampled uniformly
    """

    def __init__(
        self,
        npz_path: str | Path,
        keep_statuses: Iterable[str] = DEFAULT_KEEP_STATUSES,
    ):
        z = np.load(Path(npz_path), allow_pickle=False)
        status = z['status']
        n_labels = z['n_labels']
        keep = np.isin(status, list(keep_statuses)) & (n_labels >= 1)
        idx = np.where(keep)[0]
        self.cs_p0 = z['cs_p0'][idx].astype(np.float32)
        self.cs_line_dir = z['cs_line_dir'][idx].astype(np.float32)
        self.cs_n_target = z['cs_n_target'][idx].astype(np.float32)
        self.labels_q0 = z['labels_q0'][idx].astype(np.float32)
        self.n_labels = n_labels[idx].astype(np.int32)
        self._npz_path = Path(npz_path)

    def __len__(self) -> int:
        return int(self.n_labels.shape[0])

    def __getitem__(self, idx: int):
        n = int(self.n_labels[idx])
        label_idx = int(np.random.randint(0, n))
        c = np.concatenate([self.cs_p0[idx], self.cs_line_dir[idx], self.cs_n_target[idx]])
        q0 = self.labels_q0[idx, label_idx]
        return torch.from_numpy(c.astype(np.float32)), torch.from_numpy(q0.astype(np.float32))
