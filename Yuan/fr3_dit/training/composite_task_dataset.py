#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetStats:
    token_dim: int
    num_tasks: int
    max_tokens: int
    max_qsteps: int
    length_ref: float
    fourier_bands: int


class CompositeTaskDataset(Dataset):
    """Loads composite task tokens + q-space trajectories from packed HDF5.

    Each sample returns:
        tokens       : (T_max, D_tok)   zero-padded, variable-length task description
        token_mask   : (T_max,)         1 where valid, 0 on padding
        token_kind   : (T_max,)         0=start, 1=segment, 2=corner (uint8)
        qtraj        : (Q_max, 7)       zero-padded joint-space trajectory
        qtraj_mask   : (Q_max,)         1 where valid
        start_q      : (7,)
        local_frame  : (3, 3)           columns = (x_hat, y_hat, z_hat) of local frame
        local_origin : (3,)             TCP position of first segment start
        plane_normal : (3,)             world-frame plane normal
        seg_count    : scalar
    """

    def __init__(
        self,
        h5_path: str | Path,
        max_tokens: int | None = None,
        max_qsteps: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.dtype = dtype
        self._f: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            meta = f["meta"]
            ts = f["tasks"]
            self.token_dim = int(meta.attrs["token_dim"])
            self.length_ref = float(meta.attrs["length_ref"])
            self.fourier_bands = int(meta.attrs["fourier_bands"])
            self.token_offset = np.asarray(ts["token_offset"][()], dtype=np.int64)
            self.qtraj_offset = np.asarray(ts["qtraj_offset"][()], dtype=np.int64)
            self.num_tasks = int(self.token_offset.shape[0] - 1)
            token_lens = np.diff(self.token_offset)
            q_lens = np.diff(self.qtraj_offset)
            self.max_tokens = int(max_tokens) if max_tokens is not None else int(token_lens.max())
            self.max_qsteps = int(max_qsteps) if max_qsteps is not None else int(q_lens.max())
            self.stats = DatasetStats(
                token_dim=self.token_dim,
                num_tasks=self.num_tasks,
                max_tokens=self.max_tokens,
                max_qsteps=self.max_qsteps,
                length_ref=self.length_ref,
                fourier_bands=self.fourier_bands,
            )

    def _ensure_open(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    def __len__(self) -> int:
        return self.num_tasks

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_f"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.num_tasks
        if idx < 0 or idx >= self.num_tasks:
            raise IndexError(idx)
        f = self._ensure_open()
        ts = f["tasks"]

        t_lo = int(self.token_offset[idx])
        t_hi = int(self.token_offset[idx + 1])
        q_lo = int(self.qtraj_offset[idx])
        q_hi = int(self.qtraj_offset[idx + 1])
        n_tok = t_hi - t_lo
        n_q = q_hi - q_lo

        if n_tok > self.max_tokens:
            raise ValueError(f"task {idx} tokens={n_tok} > max_tokens={self.max_tokens}")
        if n_q > self.max_qsteps:
            raise ValueError(f"task {idx} qsteps={n_q} > max_qsteps={self.max_qsteps}")

        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_kind = np.zeros((self.max_tokens,), dtype=np.int64)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_mask[:n_tok] = 1.0
        token_kind[:n_tok] = ts["token_kind"][t_lo:t_hi]

        qtraj = np.zeros((self.max_qsteps, 7), dtype=np.float32)
        qtraj_mask = np.zeros((self.max_qsteps,), dtype=np.float32)
        qtraj[:n_q] = ts["qtraj_flat"][q_lo:q_hi]
        qtraj_mask[:n_q] = 1.0

        start_q = np.asarray(ts["start_q"][idx], dtype=np.float32)
        local_frame = np.asarray(ts["local_frame"][idx], dtype=np.float32)
        local_origin = np.asarray(ts["local_origin"][idx], dtype=np.float32)
        plane_normal = np.asarray(ts["plane_normal"][idx], dtype=np.float32)
        seg_count = int(ts["seg_count"][idx])
        total_length = float(ts["total_length"][idx])

        return {
            "tokens": torch.from_numpy(tokens).to(self.dtype),
            "token_mask": torch.from_numpy(token_mask).to(self.dtype),
            "token_kind": torch.from_numpy(token_kind),
            "qtraj": torch.from_numpy(qtraj).to(self.dtype),
            "qtraj_mask": torch.from_numpy(qtraj_mask).to(self.dtype),
            "start_q": torch.from_numpy(start_q).to(self.dtype),
            "local_frame": torch.from_numpy(local_frame).to(self.dtype),
            "local_origin": torch.from_numpy(local_origin).to(self.dtype),
            "plane_normal": torch.from_numpy(plane_normal).to(self.dtype),
            "seg_count": torch.tensor(seg_count, dtype=torch.int32),
            "total_length": torch.tensor(total_length, dtype=self.dtype),
            "num_tokens": torch.tensor(n_tok, dtype=torch.int32),
            "num_qsteps": torch.tensor(n_q, dtype=torch.int32),
        }


def dit_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check composite task dataset.")
    parser.add_argument("--h5", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks.hdf5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()

    ds = CompositeTaskDataset(args.h5)
    print(
        f"[dataset] tasks={ds.num_tasks} token_dim={ds.token_dim} "
        f"max_tokens={ds.max_tokens} max_qsteps={ds.max_qsteps}"
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=dit_collate, num_workers=0
    )
    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        print(
            f"[batch {bi}] tokens={tuple(batch['tokens'].shape)} "
            f"qtraj={tuple(batch['qtraj'].shape)} "
            f"seg_count={batch['seg_count'].tolist()} "
            f"num_tokens={batch['num_tokens'].tolist()}"
        )
    ds.close()
