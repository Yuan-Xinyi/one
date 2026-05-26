"""Smoke test for SeedSelectionDataset:
  * filter drops infeasible / low_quality
  * __getitem__(idx) called N times returns q0 from labels_q0[:n_labels]
  * for multi-label tasks, sampled q0s are NOT all identical
  * shape and dtype checks
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from Yuan.seed_selection.dataset import SeedSelectionDataset


NPZ = Path('Yuan/seed_selection/runs/pilot_day5/pilot_1k.partial-800.npz')


def main():
    ds = SeedSelectionDataset(NPZ)
    print(f'[test] len(ds) = {len(ds)}')
    assert len(ds) > 0, 'empty dataset after filtering'

    # Shape + dtype
    c, q0 = ds[0]
    assert c.shape == (9,) and c.dtype == torch.float32, (c.shape, c.dtype)
    assert q0.shape == (7,) and q0.dtype == torch.float32, (q0.shape, q0.dtype)
    print(f'[test] ds[0] c.shape={tuple(c.shape)} q0.shape={tuple(q0.shape)} dtypes OK')

    # Find a multi-label task to test sampling variety.
    multi_idx = int(np.argmax(ds.n_labels))
    n = int(ds.n_labels[multi_idx])
    assert n >= 2, f'no multi-label task found in dataset (max n_labels = {n})'
    print(f'[test] testing task idx={multi_idx} with n_labels={n}')

    np.random.seed(0)
    sampled = [ds[multi_idx][1].numpy() for _ in range(8)]
    sampled_arr = np.stack(sampled, axis=0)

    # Each sampled q0 must equal one of labels_q0[:n] (no off-by-one / no padding).
    valid_labels = ds.labels_q0[multi_idx, :n]  # (n, 7)
    for k, q in enumerate(sampled):
        matches = np.all(np.isclose(valid_labels, q[None, :]), axis=1)
        assert matches.any(), f'sample {k} not in valid labels: {q}\nvalid={valid_labels}'
    print(f'[test] all 8 sampled q0 are in labels_q0[:{n}] ✓')

    # Not all identical (with n>=2, prob of 8 identical < (1/n)^7; for n=3 ≈ 5e-4).
    distinct = np.unique(sampled_arr.round(decimals=6), axis=0)
    assert distinct.shape[0] > 1, f'all 8 samples identical: {sampled_arr}'
    print(f'[test] saw {distinct.shape[0]} distinct q0 across 8 draws ✓')

    # Status filter sanity: filtered count should match status mask on raw NPZ.
    z = np.load(NPZ, allow_pickle=False)
    status = z['status']
    n_lab = z['n_labels']
    expected = int((np.isin(status, ['kept', 'edge', 'edge_seed_fallback']) & (n_lab >= 1)).sum())
    assert len(ds) == expected, f'len(ds)={len(ds)} != expected={expected}'
    print(f'[test] filter matches raw NPZ ({expected} kept/edge tasks of {len(status)} total) ✓')

    print('[test] PASS')


if __name__ == '__main__':
    main()
