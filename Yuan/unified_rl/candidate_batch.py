"""Typed task/candidate batches for the seed stage of the unified SMDP."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SeedSelection:
    """One selected feasible initial configuration per task."""

    q0: torch.Tensor
    p0: torch.Tensor
    line_dir: torch.Tensor
    n_target: torch.Tensor

    @property
    def n_tasks(self) -> int:
        return self.q0.shape[0]

    def specs(self) -> dict[str, torch.Tensor]:
        """Specs accepted by ``ScriptedLineDistribution``."""
        return {
            'q0': self.q0,
            'p0': self.p0,
            'line_dir': self.line_dir,
            'n_target': self.n_target,
        }


@dataclass(frozen=True)
class SeedCandidateBatch:
    """A batch of task-conditioned feasible seed action sets.

    Shapes:
        q0:        (B, K, 7)
        p0:        (B, 3)
        line_dir:  (B, 3)
        n_target:  (B, 3)
        valid:     (B, K)
    """

    q0: torch.Tensor
    p0: torch.Tensor
    line_dir: torch.Tensor
    n_target: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.q0.ndim != 3 or self.q0.shape[-1] != 7:
            raise ValueError(f'q0 must have shape (B,K,7), got {tuple(self.q0.shape)}')
        b, k, _ = self.q0.shape
        expected = {
            'p0': (b, 3),
            'line_dir': (b, 3),
            'n_target': (b, 3),
            'valid': (b, k),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}, got {tuple(value.shape)}')
            if value.device != self.q0.device:
                raise ValueError(f'{name} and q0 must be on the same device')
        if self.valid.dtype != torch.bool:
            raise ValueError('valid must be a bool tensor')
        if not bool(self.valid.any(dim=1).all().item()):
            raise ValueError('every task must have at least one valid seed candidate')
        for name in ('p0', 'line_dir', 'n_target'):
            if not bool(torch.isfinite(getattr(self, name)).all().item()):
                raise ValueError(f'{name} must be finite')
        for name in ('line_dir', 'n_target'):
            norms = getattr(self, name).norm(dim=-1)
            if not bool(torch.allclose(
                    norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3)):
                raise ValueError(f'{name} must contain unit vectors')

    @property
    def n_tasks(self) -> int:
        return self.q0.shape[0]

    @property
    def n_candidates(self) -> int:
        return self.q0.shape[1]

    @property
    def device(self) -> torch.device:
        return self.q0.device

    def to(self, device: torch.device | str,
           dtype: torch.dtype | None = None) -> 'SeedCandidateBatch':
        target_dtype = self.q0.dtype if dtype is None else dtype
        return SeedCandidateBatch(
            q0=self.q0.to(device=device, dtype=target_dtype),
            p0=self.p0.to(device=device, dtype=target_dtype),
            line_dir=self.line_dir.to(device=device, dtype=target_dtype),
            n_target=self.n_target.to(device=device, dtype=target_dtype),
            valid=self.valid.to(device=device),
        )

    def index_select(self, index: torch.Tensor) -> 'SeedCandidateBatch':
        index = index.to(device=self.device, dtype=torch.long)
        return SeedCandidateBatch(
            q0=self.q0[index],
            p0=self.p0[index],
            line_dir=self.line_dir[index],
            n_target=self.n_target[index],
            valid=self.valid[index],
        )

    def repeat_interleave(self, repeats: int) -> 'SeedCandidateBatch':
        if repeats < 1:
            raise ValueError('repeats must be positive')
        return SeedCandidateBatch(
            q0=self.q0.repeat_interleave(repeats, dim=0),
            p0=self.p0.repeat_interleave(repeats, dim=0),
            line_dir=self.line_dir.repeat_interleave(repeats, dim=0),
            n_target=self.n_target.repeat_interleave(repeats, dim=0),
            valid=self.valid.repeat_interleave(repeats, dim=0),
        )

    def with_valid(self, valid: torch.Tensor) -> 'SeedCandidateBatch':
        """Return the same tasks/actions with a stricter validity mask."""
        return SeedCandidateBatch(
            q0=self.q0,
            p0=self.p0,
            line_dir=self.line_dir,
            n_target=self.n_target,
            valid=valid.to(device=self.device, dtype=torch.bool),
        )

    def select(self, candidate_index: torch.Tensor) -> SeedSelection:
        candidate_index = candidate_index.to(device=self.device, dtype=torch.long)
        if candidate_index.shape != (self.n_tasks,):
            raise ValueError(
                f'candidate_index must have shape ({self.n_tasks},), '
                f'got {tuple(candidate_index.shape)}')
        if bool(((candidate_index < 0) | (candidate_index >= self.n_candidates)).any().item()):
            raise ValueError('candidate_index is out of range')
        row = torch.arange(self.n_tasks, device=self.device)
        if not bool(self.valid[row, candidate_index].all().item()):
            raise ValueError('candidate_index selected an invalid seed')
        return SeedSelection(
            q0=self.q0[row, candidate_index],
            p0=self.p0,
            line_dir=self.line_dir,
            n_target=self.n_target,
        )


class CachedSeedCandidateDataset:
    """In-memory adapter for existing ``candidates_*.npz`` artifacts."""

    def __init__(self, batch: SeedCandidateBatch,
                 task_indices: torch.Tensor | None = None,
                 fallback_index: int | None = None):
        if batch.device.type != 'cpu':
            raise ValueError('cached candidate data must be kept on CPU')
        if task_indices is None:
            task_indices = torch.arange(batch.n_tasks, dtype=torch.long)
        task_indices = task_indices.to(device='cpu', dtype=torch.long)
        if task_indices.shape != (batch.n_tasks,):
            raise ValueError(
                f'task_indices must have shape ({batch.n_tasks},), '
                f'got {tuple(task_indices.shape)}')
        if task_indices.unique().numel() != batch.n_tasks:
            raise ValueError('task_indices must be unique')
        if fallback_index is not None:
            fallback_index = int(fallback_index)
            if not 0 <= fallback_index < batch.n_candidates:
                raise ValueError('fallback_index is out of candidate range')
        self.batch = batch
        self.task_indices = task_indices
        self.fallback_index = fallback_index

    def __len__(self) -> int:
        return self.batch.n_tasks

    @property
    def task_fingerprints(self) -> tuple[str, ...]:
        """Stable SHA256 fingerprints of task geometry, in dataset order.

        Each digest covers the canonical little-endian float32 bytes of
        ``p0``, ``line_dir``, and ``n_target``. Positive and negative zero are
        normalized before hashing, so fingerprints are suitable for auditing
        train/validation geometry overlap.
        """
        geometry = torch.cat([
            self.batch.p0,
            self.batch.line_dir,
            self.batch.n_target,
        ], dim=-1).detach().to(device='cpu', dtype=torch.float32).contiguous()
        canonical = np.asarray(
            geometry.numpy(), dtype=np.dtype('<f4'), order='C').copy()
        if not np.isfinite(canonical).all():
            raise ValueError(
                'task geometry must be representable as finite float32 values')
        canonical[canonical == np.float32(0.0)] = np.float32(0.0)
        return tuple(
            hashlib.sha256(row.tobytes(order='C')).hexdigest()
            for row in canonical)

    @classmethod
    def from_npz(cls, path: str | Path, *, include_fallback: bool = True
                 ) -> 'CachedSeedCandidateDataset':
        with np.load(Path(path)) as data:
            required = ('seeds', 'p0', 'line_dir', 'n_target')
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f'candidate cache is missing keys: {missing}')
            q0 = torch.from_numpy(data['seeds'].astype(np.float32, copy=True))
            mask_keys = [key for key in ('ik_ok', 'ok') if key in data]
            if not mask_keys:
                raise ValueError(
                    'candidate cache must contain a validity mask named '
                    "'ik_ok' or 'ok'")

            masks = []
            expected_shape = tuple(q0.shape[:2])
            for key in mask_keys:
                raw_mask = np.asarray(data[key])
                if raw_mask.shape != expected_shape:
                    raise ValueError(
                        f"candidate mask '{key}' must have shape "
                        f'{expected_shape}, got {raw_mask.shape}')
                if raw_mask.dtype != np.bool_:
                    if (not np.issubdtype(raw_mask.dtype, np.number)
                            or not np.isfinite(raw_mask).all()
                            or not np.isin(raw_mask, (0, 1)).all()):
                        raise ValueError(
                            f"candidate mask '{key}' must be boolean or contain "
                            'only 0/1 values')
                masks.append(raw_mask.astype(bool, copy=True))
            if len(masks) == 2 and not np.array_equal(masks[0], masks[1]):
                raise ValueError(
                    "candidate cache contains conflicting 'ik_ok' and 'ok' masks")
            valid = torch.from_numpy(masks[0])
            source_keys = [key for key in ('task_indices', 'src_idx')
                           if key in data]
            task_indices = None
            if source_keys:
                source_arrays = [np.asarray(data[key]) for key in source_keys]
                expected_source_shape = (q0.shape[0],)
                for key, values in zip(source_keys, source_arrays):
                    if values.shape != expected_source_shape:
                        raise ValueError(
                            f"source index '{key}' must have shape "
                            f'{expected_source_shape}, got {values.shape}')
                    if (not np.issubdtype(values.dtype, np.integer)
                            and (not np.issubdtype(values.dtype, np.number)
                                 or not np.isfinite(values).all()
                                 or not np.equal(values, np.round(values)).all())):
                        raise ValueError(
                            f"source index '{key}' must contain finite integers")
                if (len(source_arrays) == 2
                        and not np.array_equal(source_arrays[0], source_arrays[1])):
                    raise ValueError(
                        "candidate cache contains conflicting 'task_indices' "
                        "and 'src_idx'")
                task_indices = torch.from_numpy(
                    source_arrays[0].astype(np.int64, copy=True))
            # Historical diffusion caches can contain tasks for which all K
            # Newton projections failed. Their native/pilot seed is a known
            # feasible fallback and was the ninth candidate of the strongest
            # ranker. Appending it keeps the RL action set non-empty without
            # silently dropping hard tasks.
            fallback_index = None
            if include_fallback and 'q0_pilot' in data:
                fallback_index = q0.shape[1]
                raw_fallback = np.asarray(data['q0_pilot'])
                expected_fallback_shape = (q0.shape[0], 7)
                if raw_fallback.shape != expected_fallback_shape:
                    raise ValueError(
                        'q0_pilot must have shape '
                        f'{expected_fallback_shape}, got {raw_fallback.shape}')
                if (not np.issubdtype(raw_fallback.dtype, np.number)
                        or not np.isfinite(raw_fallback).all()):
                    raise ValueError('q0_pilot must contain finite numeric values')
                fallback = torch.from_numpy(
                    raw_fallback.astype(np.float32, copy=True))[:, None, :]
                q0 = torch.cat([q0, fallback], dim=1)
                valid = torch.cat([
                    valid,
                    torch.ones((q0.shape[0], 1), dtype=torch.bool),
                ], dim=1)
            batch = SeedCandidateBatch(
                q0=q0,
                p0=torch.from_numpy(data['p0'].astype(np.float32, copy=True)),
                line_dir=torch.from_numpy(
                    data['line_dir'].astype(np.float32, copy=True)),
                n_target=torch.from_numpy(
                    data['n_target'].astype(np.float32, copy=True)),
                valid=valid,
            )
        return cls(
            batch, task_indices=task_indices,
            fallback_index=fallback_index)

    def sample(self, n: int, generator: torch.Generator | None = None,
               replace: bool = True) -> SeedCandidateBatch:
        if n < 1:
            raise ValueError('n must be positive')
        if replace:
            index = torch.randint(len(self), (n,), generator=generator)
        else:
            if n > len(self):
                raise ValueError('cannot sample more tasks than the dataset without replacement')
            index = torch.randperm(len(self), generator=generator)[:n]
        return self.batch.index_select(index)

    def with_valid(self, valid: torch.Tensor) -> 'CachedSeedCandidateDataset':
        return CachedSeedCandidateDataset(
            self.batch.with_valid(valid.cpu()), self.task_indices,
            self.fallback_index)

    def index_select(self, index: torch.Tensor) -> 'CachedSeedCandidateDataset':
        index = index.to(device='cpu', dtype=torch.long)
        return CachedSeedCandidateDataset(
            self.batch.index_select(index), self.task_indices[index],
            self.fallback_index)

    def select_source_tasks(
        self, task_indices: torch.Tensor,
    ) -> 'CachedSeedCandidateDataset':
        """Select source-cache task ids in the requested order."""
        requested = task_indices.to(device='cpu', dtype=torch.long).reshape(-1)
        lookup = {int(task_id): position for position, task_id
                  in enumerate(self.task_indices.tolist())}
        if len(lookup) != len(self.task_indices):
            raise ValueError('dataset contains duplicate source task indices')
        missing = [int(task_id) for task_id in requested.tolist()
                   if int(task_id) not in lookup]
        if missing:
            raise ValueError(
                f'source task indices are absent from dataset: {missing[:20]}')
        local = torch.tensor(
            [lookup[int(task_id)] for task_id in requested.tolist()],
            dtype=torch.long)
        return self.index_select(local)

    def train_validation_split(
        self, validation_fraction: float, seed: int,
    ) -> tuple['CachedSeedCandidateDataset', 'CachedSeedCandidateDataset',
               torch.Tensor, torch.Tensor]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError('validation_fraction must be in (0, 1)')
        if len(self) < 2:
            raise ValueError('at least two tasks are required for a train/validation split')

        groups_by_fingerprint: dict[str, list[int]] = {}
        for row, fingerprint in enumerate(self.task_fingerprints):
            groups_by_fingerprint.setdefault(fingerprint, []).append(row)
        groups = list(groups_by_fingerprint.values())
        if len(groups) < 2:
            raise ValueError(
                'at least two unique task-geometry groups are required for '
                'a train/validation split')

        generator = torch.Generator().manual_seed(int(seed))
        n_validation = min(
            len(self) - 1, max(1, round(len(self) * validation_fraction)))

        # Preserve the historical row-level split exactly when every task has
        # unique geometry. This keeps both the requested validation count and
        # the seeded index order unchanged for existing independent datasets.
        if len(groups) == len(self):
            order = torch.randperm(len(self), generator=generator)
            validation_index = order[:n_validation]
            train_index = order[n_validation:]
        else:
            group_order = torch.randperm(
                len(groups), generator=generator).tolist()
            validation_groups = self._closest_group_subset(
                groups, group_order, n_validation, len(self))
            validation_group_set = set(validation_groups)
            train_groups = [group for group in group_order
                            if group not in validation_group_set]
            validation_index = torch.tensor(
                [row for group in validation_groups for row in groups[group]],
                dtype=torch.long)
            train_index = torch.tensor(
                [row for group in train_groups for row in groups[group]],
                dtype=torch.long)
        return (
            self.index_select(train_index),
            self.index_select(validation_index),
            train_index,
            validation_index,
        )

    @staticmethod
    def _closest_group_subset(
        groups: list[list[int]], group_order: list[int], target_rows: int,
        total_rows: int,
    ) -> list[int]:
        """Choose whole groups with a row count closest to ``target_rows``.

        The seeded group order determines ties between equivalent subsets.
        Reachable row counts are represented as a Python integer bitset, so
        exact subset selection remains practical for large cached datasets.
        """
        reachable = 1
        row_mask = (1 << (total_rows + 1)) - 1
        parent_group = np.full(total_rows + 1, -1, dtype=np.int64)
        parent_sum = np.full(total_rows + 1, -1, dtype=np.int64)

        for group in group_order:
            size = len(groups[group])
            shifted = (reachable << size) & row_mask
            new_sums = shifted & ~reachable
            while new_sums:
                lowest_bit = new_sums & -new_sums
                row_sum = lowest_bit.bit_length() - 1
                parent_group[row_sum] = group
                parent_sum[row_sum] = row_sum - size
                new_sums ^= lowest_bit
            reachable |= shifted
            if reachable & (1 << target_rows):
                break

        selected_rows: int | None = None
        for distance in range(total_rows):
            lower = target_rows - distance
            upper = target_rows + distance
            if lower >= 1 and lower < total_rows and reachable & (1 << lower):
                selected_rows = lower
                break
            if upper >= 1 and upper < total_rows and reachable & (1 << upper):
                selected_rows = upper
                break
        if selected_rows is None:
            raise RuntimeError(
                'failed to construct a non-empty task-geometry group split')

        selected = set()
        row_sum = selected_rows
        while row_sum:
            group = int(parent_group[row_sum])
            previous = int(parent_sum[row_sum])
            if group < 0 or previous < 0:
                raise RuntimeError(
                    'task-geometry group split reconstruction failed')
            selected.add(group)
            row_sum = previous
        return [group for group in group_order if group in selected]
