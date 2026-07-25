"""Merge historical evaluation artifacts into the unified cache schema.

Historical ranking holdouts stored candidate joints separately from task
geometry. This utility only converts their representation; ``evaluate`` still
audits task-geometry overlap before treating the result as an external set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Yuan.unified_rl.provenance import file_fingerprint


def _read_alias(data, names: tuple[str, ...], label: str) -> np.ndarray:
    present = [name for name in names if name in data]
    if not present:
        raise ValueError(f'missing {label}; expected one of {names}')
    arrays = [np.asarray(data[name]) for name in present]
    if len(arrays) > 1 and any(
            not np.array_equal(arrays[0], value) for value in arrays[1:]):
        raise ValueError(f'conflicting aliases for {label}: {present}')
    return arrays[0]


def _strict_mask(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if value.shape != shape:
        raise ValueError(f'validity must have shape {shape}, got {value.shape}')
    if value.dtype == np.bool_:
        return value.astype(bool, copy=True)
    if (not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
            or not np.isin(value, (0, 1)).all()):
        raise ValueError('validity must be boolean or contain only 0/1 values')
    return value.astype(bool, copy=True)


def _strict_source_index(value: np.ndarray, n_tasks: int) -> np.ndarray:
    if value.shape != (n_tasks,):
        raise ValueError(
            f'source task index must have shape ({n_tasks},), got {value.shape}')
    if (not np.issubdtype(value.dtype, np.integer)
            and (not np.issubdtype(value.dtype, np.number)
                 or not np.isfinite(value).all()
                 or not np.equal(value, np.round(value)).all())):
        raise ValueError('source task index must contain finite integers')
    value = value.astype(np.int64, copy=True)
    if np.unique(value).size != n_tasks:
        raise ValueError('source task index must be unique')
    return value


def prepare_holdout_cache(
    candidate_path: str | Path,
    task_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Create a unified candidate cache without changing candidate order."""
    candidate_path = Path(candidate_path)
    task_path = Path(task_path)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f'refusing to overwrite holdout cache: {out_path}')

    with np.load(candidate_path) as candidates, np.load(task_path) as tasks:
        seeds = _read_alias(candidates, ('seeds',), 'candidate joints')
        valid = _read_alias(candidates, ('ik_ok', 'ok'), 'candidate validity')
        p0 = _read_alias(tasks, ('p0', 'cs_p0'), 'task ray origin')
        line_dir = _read_alias(
            tasks, ('line_dir', 'cs_line_dir'), 'task line direction')
        n_target = _read_alias(
            tasks, ('n_target', 'cs_n_target'), 'task target normal')
        q0_pilot = _read_alias(
            tasks, ('q0_pilot', 'q0_seed'), 'pilot seed')
        if seeds.ndim != 3 or seeds.shape[-1] != 7:
            raise ValueError(f'seeds must have shape (B,K,7), got {seeds.shape}')
        if not np.issubdtype(seeds.dtype, np.number):
            raise ValueError('seeds must contain numeric values')
        n_tasks = seeds.shape[0]
        expected = {
            'validity': (n_tasks, seeds.shape[1]),
            'p0': (n_tasks, 3),
            'line_dir': (n_tasks, 3),
            'n_target': (n_tasks, 3),
            'q0_pilot': (n_tasks, 7),
        }
        values = {
            'validity': valid,
            'p0': p0,
            'line_dir': line_dir,
            'n_target': n_target,
            'q0_pilot': q0_pilot,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(
                    f'{name} must have shape {shape}, got {values[name].shape}')
        valid = _strict_mask(valid, expected['validity'])
        for name, value in (
                ('p0', p0), ('line_dir', line_dir),
                ('n_target', n_target), ('q0_pilot', q0_pilot)):
            if (not np.issubdtype(value.dtype, np.number)
                    or not np.isfinite(value).all()):
                raise ValueError(f'{name} must contain finite numeric values')
        if not np.isfinite(seeds[valid]).all():
            raise ValueError('valid candidate seeds must be finite')
        source_index = (
            _read_alias(tasks, ('task_indices', 'src_idx'), 'source task index')
            if any(name in tasks for name in ('task_indices', 'src_idx'))
            else np.arange(n_tasks, dtype=np.int64))
        source_index = _strict_source_index(source_index, n_tasks)

        # The historical K25 artifact stored the corresponding task's pilot
        # once. Requiring one exact match per row both proves row alignment
        # between the two source files and prevents fallback multiplicity.
        pilot_match = np.equal(seeds, q0_pilot[:, None, :]).all(axis=-1)
        pilot_match_count = pilot_match.sum(axis=1)
        if not np.equal(pilot_match_count, 1).all():
            bad = np.flatnonzero(pilot_match_count != 1)
            raise ValueError(
                'candidate/task rows cannot be aligned: expected exactly '
                f'one q0_pilot match per row, bad rows={bad[:20].tolist()}')
        removed_pilot_slots = pilot_match.argmax(axis=1).astype(np.int64)
        seeds = seeds[~pilot_match].reshape(
            n_tasks, seeds.shape[1] - 1, 7)
        valid = valid[~pilot_match].reshape(
            n_tasks, valid.shape[1] - 1)

        payload = {
            'seeds': seeds.astype(np.float32, copy=True),
            'ok': valid,
            'p0': p0.astype(np.float32, copy=True),
            'line_dir': line_dir.astype(np.float32, copy=True),
            'n_target': n_target.astype(np.float32, copy=True),
            'q0_pilot': q0_pilot.astype(np.float32, copy=True),
            'src_idx': source_index,
            'source_candidate_sha256': np.asarray(
                file_fingerprint(candidate_path)['sha256']),
            'source_task_sha256': np.asarray(
                file_fingerprint(task_path)['sha256']),
            'removed_pilot_slots': removed_pilot_slots,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = prepare_holdout_cache(args.candidates, args.tasks, args.out)
    fingerprint = file_fingerprint(out)
    print(
        f'[unified-holdout] {out}  size={fingerprint["size"]}  '
        f'sha256={fingerprint["sha256"]}')


if __name__ == '__main__':
    main()
