"""Train a leakage-safe ensemble selector from exhaustive seed returns.

This is the high-capacity offline companion to ``offline_seed_train``.  The
controller is immutable: exhaustive progress labels train several independent
45-D candidate policies, while a held-out geometry split is used exactly once
to calibrate a conservative first-valid gate.  The resulting ``unified.pt``
is directly consumable by the ensemble-aware evaluator.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedCandidateBatch,
)
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_run_config,
    ppo_config_from_run,
    resolve_controller_dir,
)
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.offline_seed_train import (
    _assert_artifact_unchanged,
    _calibrate_one_head,
    _cpu_tree,
    _same_content,
    _validate_source_checkpoint,
    geometry_groups,
    load_return_cache,
)
from Yuan.unified_rl.provenance import (
    controller_fingerprint,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.reproducibility import (
    device_identity,
    global_rng_state,
    seed_global_rng,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    CandidateSeedPolicyEnsemble,
    SEED_ENSEMBLE_AGGREGATION,
    SEED_ENSEMBLE_FORMAT,
    infer_seed_policy_config,
    seed_policy_ensemble_states,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


_EXTERNAL_REQUIRED_KEYS = (
    'format', 'format_version', 'valid', 'discounted_return',
    'undiscounted_return', 'progress_m', 'episode_len', 'term_reason',
    'switch_count', 'q0', 'p0', 'line_dir', 'n_target', 'task_indices',
    'task_fingerprints', 'external_row_indices',
    'retained_source_task_indices', 'retained_task_fingerprints',
    'retained_external_row_indices', 'candidate_indices', 'fallback_index',
    'n_tasks', 'n_candidates', 'n_valid_rollouts',
    'source_validation_n_tasks', 'source_validation_unique_fingerprints',
    'source_validation_task_fingerprints',
    'validation_overlap_after_exclusion', 'source_checkpoint_size',
    'source_checkpoint_sha256', 'source_candidate_cache_size',
    'source_candidate_cache_sha256', 'candidate_cache_path',
    'candidate_cache_size', 'candidate_cache_sha256',
    'controller_agent_size', 'controller_agent_sha256',
    'controller_config_size', 'controller_config_sha256',
    'controller_state_sha256', 'controller_gamma', 'controller_kind',
    'source_phase', 'source_outer_round', 'seed_return_objective',
    'source_split_mode', 'split_mode', 'physical_validation',
)

_DEFAULT_FEASIBILITY_TARGET = 'progress-minus-first-valid-m-v1'
_ROBUST_FEASIBILITY_TARGET = (
    'min-primary-reference-progress-minus-own-first-valid-m-v1')


@dataclasses.dataclass(frozen=True)
class ExternalReturnData:
    dataset: CachedSeedCandidateDataset
    progress_m: torch.Tensor
    valid: torch.Tensor
    task_fingerprints: tuple[str, ...]
    cache_artifact: dict[str, str | int]
    candidate_artifact: dict[str, str | int]
    cache_path: Path
    candidate_path: Path
    physical_stats: dict[str, float | list[int]]


@dataclasses.dataclass(frozen=True)
class TrainingTable:
    features: torch.Tensor
    valid: torch.Tensor
    progress_m: torch.Tensor
    feasibility_target_m: torch.Tensor
    task_fingerprints: tuple[str, ...]
    n_source_rows: int
    n_external_rows: int


def _scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f'external return-cache {key!r} must be scalar')
    return value.item()


def _int_scalar(data: Any, key: str) -> int:
    value = np.asarray(data[key])
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise ValueError(
            f'external return-cache {key!r} must be an int64 scalar')
    return int(value.item())


def _str_scalar(data: Any, key: str) -> str:
    value = np.asarray(data[key])
    if value.shape != () or value.dtype.kind not in ('U', 'S'):
        raise ValueError(
            f'external return-cache {key!r} must be a string scalar')
    return str(value.item())


def _require_array(data: Any, key: str, shape: tuple[int, ...],
                   dtype: np.dtype) -> np.ndarray:
    value = np.asarray(data[key])
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(
            f'external return-cache {key!r} must have shape {shape} and '
            f'dtype {dtype}, got {value.shape}, {value.dtype}')
    return value.copy()


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _assert_paired_candidate_datasets(
    primary: CachedSeedCandidateDataset,
    reference: CachedSeedCandidateDataset,
    *,
    label: str,
) -> None:
    """Require two controller caches to describe exactly the same actions."""
    if primary.fallback_index != reference.fallback_index:
        raise ValueError(f'{label} fallback candidate differs')
    if not torch.equal(primary.task_indices, reference.task_indices):
        raise ValueError(f'{label} task indices differ')
    if primary.task_fingerprints != reference.task_fingerprints:
        raise ValueError(f'{label} task fingerprints differ')
    for name in ('q0', 'p0', 'line_dir', 'n_target', 'valid'):
        left = getattr(primary.batch, name)
        right = getattr(reference.batch, name)
        if left.shape != right.shape or left.dtype != right.dtype:
            same = False
        elif left.dtype.is_floating_point:
            same = torch.allclose(
                left, right, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            same = torch.equal(left, right)
        if not same:
            raise ValueError(f'{label} candidate {name} arrays differ')


def _relative_advantage_target(
    progress_m: torch.Tensor,
    valid: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if progress_m.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError(f'{label} progress/valid shapes or dtypes differ')
    if not valid.any(dim=1).all():
        raise ValueError(f'{label} has a row without a valid candidate')
    if not torch.isfinite(progress_m[valid]).all():
        raise ValueError(f'{label} valid progress values must be finite')
    if not torch.isnan(progress_m[~valid]).all():
        raise ValueError(f'{label} invalid progress values must be NaN')
    first = valid.float().argmax(dim=-1)
    row = torch.arange(progress_m.shape[0], device=progress_m.device)
    relative = progress_m - progress_m[row, first].unsqueeze(-1)
    return torch.where(valid, relative, torch.zeros_like(relative))


def _controller_robust_feasibility_target(
    primary_progress_m: torch.Tensor,
    reference_progress_m: torch.Tensor,
    primary_valid: torch.Tensor,
    reference_valid: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    """Worst-case relative advantage across the primary and reference controller."""
    if not torch.equal(primary_valid, reference_valid):
        raise ValueError(f'{label} controller valid masks differ')
    primary = _relative_advantage_target(
        primary_progress_m, primary_valid, label=f'{label} primary')
    reference = _relative_advantage_target(
        reference_progress_m, reference_valid, label=f'{label} reference')
    if (primary.shape != reference.shape
            or primary.dtype != reference.dtype
            or primary.device != reference.device):
        raise ValueError(f'{label} controller progress arrays differ')
    return torch.where(
        primary_valid, torch.minimum(primary, reference),
        torch.zeros_like(primary))


def _validate_result_slots(data: Any, valid: np.ndarray) -> np.ndarray:
    progress = None
    for key in ('discounted_return', 'undiscounted_return', 'progress_m'):
        value = _require_array(data, key, valid.shape, np.dtype(np.float32))
        if not np.isfinite(value[valid]).all():
            raise ValueError(f'valid external {key} values must be finite')
        if not np.isnan(value[~valid]).all():
            raise ValueError(f'invalid external {key} slots must be NaN')
        if key == 'progress_m':
            progress = value
    for key, dtype in (
            ('episode_len', np.dtype(np.int64)),
            ('term_reason', np.dtype(np.int32)),
            ('switch_count', np.dtype(np.int64))):
        value = _require_array(data, key, valid.shape, dtype)
        if np.any(value[valid] < 0) or not np.equal(value[~valid], -1).all():
            raise ValueError(
                f'external {key} must be non-negative on valid slots and -1 '
                'on invalid slots')
    assert progress is not None
    return progress


def _external_dataset_from_arrays(
    q0: np.ndarray,
    p0: np.ndarray,
    line_dir: np.ndarray,
    n_target: np.ndarray,
    valid: np.ndarray,
    task_indices: np.ndarray,
    fallback_index: int,
) -> CachedSeedCandidateDataset:
    return CachedSeedCandidateDataset(
        SeedCandidateBatch(
            q0=torch.from_numpy(q0),
            p0=torch.from_numpy(p0),
            line_dir=torch.from_numpy(line_dir),
            n_target=torch.from_numpy(n_target),
            valid=torch.from_numpy(valid),
        ),
        task_indices=torch.from_numpy(task_indices),
        fallback_index=None if fallback_index < 0 else fallback_index,
    )


def load_external_return_cache(
    path: str | Path,
    *,
    source: dict,
    source_artifact: dict,
    source_candidate_artifact: dict,
    controller_artifact: dict,
    controller_state_sha256: str,
    objective: str,
    gamma: float,
    source_validation: CachedSeedCandidateDataset,
    env,
    physical_chunk_size: int,
) -> ExternalReturnData:
    """Strictly bind an external cache to all immutable source artifacts."""
    cache_path = Path(path).expanduser().resolve(strict=True)
    cache_artifact = file_fingerprint(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        missing = [key for key in _EXTERNAL_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(
                f'external return cache is missing required fields: {missing}')
        if _str_scalar(data, 'format') != 'external-seed-return-cache-v1':
            raise ValueError(
                'external cache must use external-seed-return-cache-v1')
        if _int_scalar(data, 'format_version') != 1:
            raise ValueError('external cache format_version must equal 1')
        physical = np.asarray(data['physical_validation'])
        if (physical.shape != () or physical.dtype != np.bool_
                or not bool(physical.item())):
            raise ValueError('external cache must record physical validation')

        n = _int_scalar(data, 'n_tasks')
        k = _int_scalar(data, 'n_candidates')
        if n < 1 or k < 1:
            raise ValueError('external cache dimensions must be positive')
        valid = _require_array(data, 'valid', (n, k), np.dtype(np.bool_))
        if not valid.any(axis=1).all():
            raise ValueError('every external row must have a valid candidate')
        if _int_scalar(data, 'n_valid_rollouts') != int(valid.sum()):
            raise ValueError('external n_valid_rollouts is inconsistent')
        progress = _validate_result_slots(data, valid)

        q0 = _require_array(data, 'q0', (n, k, 7), np.dtype(np.float32))
        p0 = _require_array(data, 'p0', (n, 3), np.dtype(np.float32))
        line_dir = _require_array(
            data, 'line_dir', (n, 3), np.dtype(np.float32))
        n_target = _require_array(
            data, 'n_target', (n, 3), np.dtype(np.float32))
        task_indices = _require_array(
            data, 'task_indices', (n,), np.dtype(np.int64))
        if np.unique(task_indices).size != n:
            raise ValueError('external task_indices must be unique')
        candidate_indices = _require_array(
            data, 'candidate_indices', (k,), np.dtype(np.int64))
        if not np.array_equal(candidate_indices, np.arange(k, dtype=np.int64)):
            raise ValueError('external candidate_indices are not canonical')
        fallback_index = _int_scalar(data, 'fallback_index')
        if fallback_index != -1 and not 0 <= fallback_index < k:
            raise ValueError('external fallback_index is out of range')
        fingerprints = _require_array(
            data, 'task_fingerprints', (n,), np.dtype('<U64'))
        retained_fingerprints = _require_array(
            data, 'retained_task_fingerprints', (n,), np.dtype('<U64'))
        if not np.array_equal(fingerprints, retained_fingerprints):
            raise ValueError('external retained fingerprints disagree')
        external_rows = _require_array(
            data, 'external_row_indices', (n,), np.dtype(np.int64))
        retained_rows = _require_array(
            data, 'retained_external_row_indices', (n,), np.dtype(np.int64))
        retained_tasks = _require_array(
            data, 'retained_source_task_indices', (n,), np.dtype(np.int64))
        if (not np.array_equal(external_rows, retained_rows)
                or not np.array_equal(task_indices, retained_tasks)):
            raise ValueError('external retained row/task aliases disagree')
        if np.any(external_rows < 0) or np.unique(external_rows).size != n:
            raise ValueError('external retained rows must be unique and >= 0')

        if _str_scalar(data, 'source_phase') != source['phase']:
            raise ValueError('external cache source phase differs')
        if _int_scalar(data, 'source_outer_round') != int(source['outer_round']):
            raise ValueError('external cache source outer round differs')
        if _str_scalar(data, 'seed_return_objective') != objective:
            raise ValueError('external cache objective differs from source')
        if _str_scalar(data, 'source_split_mode') != source['split_mode']:
            raise ValueError('external cache source split mode differs')
        if (_str_scalar(data, 'split_mode')
                != 'external-validation-fingerprint-excluded-v1'):
            raise ValueError('external cache split mode is invalid')
        if _str_scalar(data, 'controller_kind') != 'pure':
            raise ValueError('external cache controller must be pure')
        cached_gamma = np.asarray(data['controller_gamma'])
        if cached_gamma.shape != () or cached_gamma.dtype != np.float64:
            raise ValueError('external controller_gamma must be float64')
        if float(cached_gamma.item()) != gamma:
            raise ValueError('external controller gamma differs from source')

        _same_content(
            _int_scalar(data, 'source_checkpoint_size'),
            _str_scalar(data, 'source_checkpoint_sha256'), source_artifact,
            label='external source checkpoint')
        _same_content(
            _int_scalar(data, 'source_candidate_cache_size'),
            _str_scalar(data, 'source_candidate_cache_sha256'),
            source_candidate_artifact, label='external source candidates')
        _same_content(
            _int_scalar(data, 'controller_agent_size'),
            _str_scalar(data, 'controller_agent_sha256'),
            controller_artifact['agent'], label='external controller agent')
        _same_content(
            _int_scalar(data, 'controller_config_size'),
            _str_scalar(data, 'controller_config_sha256'),
            controller_artifact['config'], label='external controller config')
        if (_str_scalar(data, 'controller_state_sha256')
                != controller_state_sha256):
            raise ValueError('external controller state hash differs')

        validation_fingerprints = np.asarray(
            source_validation.task_fingerprints, dtype='<U64')
        saved_validation = _require_array(
            data, 'source_validation_task_fingerprints',
            validation_fingerprints.shape, np.dtype('<U64'))
        if not np.array_equal(saved_validation, validation_fingerprints):
            raise ValueError(
                'external cache validation fingerprint exclusion set differs')
        if _int_scalar(data, 'source_validation_n_tasks') != len(
                validation_fingerprints):
            raise ValueError('external source_validation_n_tasks differs')
        if _int_scalar(data, 'source_validation_unique_fingerprints') != len(
                set(validation_fingerprints.tolist())):
            raise ValueError(
                'external source validation unique count is inconsistent')
        overlap_marker = np.asarray(data['validation_overlap_after_exclusion'])
        if (overlap_marker.shape != () or overlap_marker.dtype != np.bool_
                or bool(overlap_marker.item())):
            raise ValueError('external cache does not certify zero overlap')
        if set(fingerprints.tolist()) & set(validation_fingerprints.tolist()):
            raise ValueError(
                'external cache overlaps source validation geometry')

        candidate_path = Path(
            _str_scalar(data, 'candidate_cache_path')).expanduser().resolve(
                strict=True)
        candidate_artifact = file_fingerprint(candidate_path)
        _same_content(
            _int_scalar(data, 'candidate_cache_size'),
            _str_scalar(data, 'candidate_cache_sha256'), candidate_artifact,
            label='external candidate cache')
        if (candidate_artifact['size'] == source_candidate_artifact['size']
                and candidate_artifact['sha256']
                == source_candidate_artifact['sha256']):
            raise ValueError('external and source candidate artifacts coincide')

    cached_dataset = _external_dataset_from_arrays(
        q0, p0, line_dir, n_target, valid, task_indices, fallback_index)
    if tuple(fingerprints.tolist()) != cached_dataset.task_fingerprints:
        raise ValueError('external cached geometry fingerprints are invalid')

    # Recreate the retained rows from their immutable external candidate file,
    # then rerun physical filtering.  This binds both action tensors and masks
    # instead of trusting a self-reported cache flag.
    raw_external = CachedSeedCandidateDataset.from_npz(candidate_path)
    if np.any(external_rows >= len(raw_external)):
        raise ValueError('external retained row index is out of range')
    raw_retained = raw_external.index_select(torch.from_numpy(external_rows))
    validated, physical_stats = validate_cached_dataset(
        raw_retained, env.kin, env.collision,
        chunk_size=physical_chunk_size, cone_deg=env.cfg.cone_deg)
    if len(validated) != n:
        raise ValueError('external physical validation no longer retains all rows')
    expected_arrays = {
        'q0': cached_dataset.batch.q0.numpy(),
        'p0': cached_dataset.batch.p0.numpy(),
        'line_dir': cached_dataset.batch.line_dir.numpy(),
        'n_target': cached_dataset.batch.n_target.numpy(),
        'valid': cached_dataset.batch.valid.numpy(),
        'task_indices': cached_dataset.task_indices.numpy(),
    }
    actual_arrays = {
        'q0': validated.batch.q0.numpy(),
        'p0': validated.batch.p0.numpy(),
        'line_dir': validated.batch.line_dir.numpy(),
        'n_target': validated.batch.n_target.numpy(),
        'valid': validated.batch.valid.numpy(),
        'task_indices': validated.task_indices.numpy(),
    }
    for key, expected in expected_arrays.items():
        if not _same_array(expected, actual_arrays[key]):
            raise ValueError(
                f'external cached {key} differs from physically validated '
                'candidate content')
    if validated.fallback_index != cached_dataset.fallback_index:
        raise ValueError('external cached fallback metadata differs')

    if file_fingerprint(cache_path) != cache_artifact:
        raise RuntimeError('external return cache changed while loading')
    if file_fingerprint(candidate_path) != candidate_artifact:
        raise RuntimeError('external candidate cache changed while loading')
    return ExternalReturnData(
        dataset=cached_dataset,
        progress_m=torch.from_numpy(progress),
        valid=torch.from_numpy(valid),
        task_fingerprints=tuple(fingerprints.tolist()),
        cache_artifact=cache_artifact,
        candidate_artifact=candidate_artifact,
        cache_path=cache_path,
        candidate_path=candidate_path,
        physical_stats=physical_stats,
    )


def _build_features(kin, dataset: CachedSeedCandidateDataset,
                    chunk_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        batch = dataset.batch.index_select(torch.arange(start, end)).to(
            kin.device, dtype=kin.dtype)
        parts.append(initial_observation_features(
            kin, batch, include_ray_error=True, include_log_manip=True,
            include_directional_dynamics=True).cpu())
    result = torch.cat(parts, dim=0)
    if result.shape[-1] != 45:
        raise RuntimeError(f'expected 45-D ensemble features, got {result.shape[-1]}')
    return result


def _three_way_geometry_split(
    dataset: CachedSeedCandidateDataset,
    model_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remaining, _, remaining_index, calibration_index = (
        dataset.train_validation_split(calibration_fraction, seed + 1))
    adjusted_model_fraction = model_fraction / (1.0 - calibration_fraction)
    if not 0.0 < adjusted_model_fraction < 1.0:
        raise ValueError('model/calibration fractions leave no fit partition')
    _, _, fit_local, model_local = remaining.train_validation_split(
        adjusted_model_fraction, seed + 2)
    fit_index = remaining_index[fit_local]
    model_index = remaining_index[model_local]
    _assert_three_way_geometry_disjoint(
        (fit_index, model_index, calibration_index),
        dataset.task_fingerprints)
    return fit_index, model_index, calibration_index


def _assert_three_way_geometry_disjoint(
    indices: Sequence[torch.Tensor],
    fingerprints: Sequence[str],
) -> None:
    """Fail closed if one geometry appears in multiple selector partitions."""
    if len(indices) != 3:
        raise ValueError('a selector split must contain exactly three partitions')
    partitions = [
        {fingerprints[int(row)] for row in index.tolist()}
        for index in indices
    ]
    if any(partitions[i] & partitions[j]
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('three-way source geometry split overlaps')


def _pad_candidates(value: torch.Tensor, size: int, fill: float | bool) -> torch.Tensor:
    if value.shape[1] == size:
        return value
    if value.shape[1] > size:
        raise ValueError('cannot shrink candidate axis')
    shape = (value.shape[0], size - value.shape[1], *value.shape[2:])
    padding = torch.full(shape, fill, dtype=value.dtype)
    return torch.cat([value, padding], dim=1)


def _make_training_table(
    source_features: torch.Tensor,
    source_valid: torch.Tensor,
    source_progress: torch.Tensor,
    source_fingerprints: Sequence[str],
    fit_indices: torch.Tensor,
    external: Sequence[tuple[ExternalReturnData, torch.Tensor, torch.Tensor]],
    *,
    source_feasibility_target: torch.Tensor | None = None,
    external_feasibility_targets: Sequence[torch.Tensor] | None = None,
) -> TrainingTable:
    if source_feasibility_target is None:
        source_feasibility_target = _relative_advantage_target(
            source_progress, source_valid, label='source training table')
    if (source_feasibility_target.shape != source_progress.shape
            or source_feasibility_target.dtype != source_progress.dtype
            or not torch.isfinite(
                source_feasibility_target[source_valid]).all()):
        raise ValueError('source feasibility target is invalid')
    if not torch.equal(
            source_feasibility_target[~source_valid],
            torch.zeros_like(source_feasibility_target[~source_valid])):
        raise ValueError('source invalid feasibility targets must be zero')
    if external_feasibility_targets is None:
        external_feasibility_targets = tuple(
            _relative_advantage_target(
                cache.progress_m, cache.valid,
                label=f'external training table {index}')
            for index, (cache, _, _) in enumerate(external))
    if len(external_feasibility_targets) != len(external):
        raise ValueError(
            'external feasibility targets must align one-to-one with caches')
    feature_parts = [source_features[fit_indices]]
    valid_parts = [source_valid[fit_indices]]
    progress_parts = [source_progress[fit_indices]]
    feasibility_parts = [source_feasibility_target[fit_indices]]
    fingerprints = [source_fingerprints[int(row)] for row in fit_indices.tolist()]
    n_external = 0
    for external_index, (cache, features, keep) in enumerate(external):
        feasibility_target = external_feasibility_targets[external_index]
        if (feasibility_target.shape != cache.progress_m.shape
                or feasibility_target.dtype != cache.progress_m.dtype
                or not torch.isfinite(feasibility_target[cache.valid]).all()):
            raise ValueError(
                f'external feasibility target {external_index} is invalid')
        if not torch.equal(
                feasibility_target[~cache.valid],
                torch.zeros_like(feasibility_target[~cache.valid])):
            raise ValueError(
                f'external invalid feasibility target {external_index} '
                'must be zero')
        feature_parts.append(features[keep])
        valid_parts.append(cache.valid[keep])
        progress_parts.append(cache.progress_m[keep])
        feasibility_parts.append(feasibility_target[keep])
        fingerprints.extend(cache.task_fingerprints[int(row)]
                            for row in keep.tolist())
        n_external += int(keep.numel())
    max_candidates = max(part.shape[1] for part in feature_parts)
    features = torch.cat([
        _pad_candidates(part, max_candidates, 0.0) for part in feature_parts
    ], dim=0)
    valid = torch.cat([
        _pad_candidates(part, max_candidates, False) for part in valid_parts
    ], dim=0)
    progress = torch.cat([
        _pad_candidates(part, max_candidates, float('nan'))
        for part in progress_parts
    ], dim=0)
    feasibility_target = torch.cat([
        _pad_candidates(part, max_candidates, 0.0)
        for part in feasibility_parts
    ], dim=0)
    if not valid.any(dim=1).all() or not torch.isfinite(progress[valid]).all():
        raise ValueError('combined training table has invalid progress labels')
    if not torch.isnan(progress[~valid]).all():
        raise ValueError('combined invalid progress slots must be NaN')
    if (not torch.isfinite(feasibility_target[valid]).all()
            or not torch.equal(
                feasibility_target[~valid],
                torch.zeros_like(feasibility_target[~valid]))):
        raise ValueError('combined training table has invalid feasibility labels')
    return TrainingTable(
        features=features,
        valid=valid,
        progress_m=progress,
        feasibility_target_m=feasibility_target,
        task_fingerprints=tuple(fingerprints),
        n_source_rows=int(fit_indices.numel()),
        n_external_rows=n_external,
    )


def _normalization(features: torch.Tensor,
                   valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = features[valid].double()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError('valid-only normalization inputs must be finite')
    mean = values.mean(dim=0)
    std = (values.square().mean(dim=0) - mean.square()).clamp_min(
        0.0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


def _pairwise_actor_loss(logits: torch.Tensor, progress: torch.Tensor,
                         valid: torch.Tensor, min_delta: float) -> torch.Tensor:
    safe_progress = torch.where(valid, progress, torch.zeros_like(progress))
    target_delta = safe_progress.unsqueeze(2) - safe_progress.unsqueeze(1)
    score_delta = logits.unsqueeze(2) - logits.unsqueeze(1)
    better = (valid.unsqueeze(2) & valid.unsqueeze(1)
              & (target_delta >= min_delta))
    count = better.sum(dim=(1, 2))
    keep = count > 0
    if not bool(keep.any().item()):
        return logits.sum() * 0.0
    raw_weight = target_delta.clamp_min(0.0) * better
    mean_weight = raw_weight.sum(dim=(1, 2)) / count.clamp_min(1)
    weight = raw_weight / mean_weight.clamp_min(1e-8)[:, None, None]
    per_task = (F.softplus(-score_delta) * weight.detach() * better).sum(
        dim=(1, 2)) / count.clamp_min(1)
    return per_task[keep].mean()


def _geometry_epoch_indices(
    groups: Sequence[Sequence[int]], generator: torch.Generator,
) -> torch.Tensor:
    """One random row per geometry, avoiding RNG calls for singleton groups."""
    rows = [
        int(group[0]) if len(group) == 1 else int(group[int(torch.randint(
            len(group), (1,), generator=generator).item())])
        for group in groups
    ]
    row_tensor = torch.tensor(rows, dtype=torch.long)
    return row_tensor[torch.randperm(len(rows), generator=generator)]


def _warm_retention_losses(
    student_logits: torch.Tensor,
    student_feasibility: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_feasibility: torch.Tensor,
    valid: torch.Tensor,
    first: torch.Tensor,
    *,
    beta_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain a warm selector's proposal distribution and deployed Q margin.

    The conservative gate uses feasibility differences relative to the first
    valid candidate, so retaining absolute Q values would constrain an
    irrelevant per-task offset.  Actor retention is the forward KL
    ``KL(teacher || student)`` over valid candidates only.
    """
    if student_logits.shape != student_feasibility.shape:
        raise ValueError('student logits and feasibility must have equal shape')
    if (teacher_logits.shape != student_logits.shape
            or teacher_feasibility.shape != student_logits.shape
            or valid.shape != student_logits.shape):
        raise ValueError('warm-retention tensors must have equal (B,K) shape')
    if valid.dtype != torch.bool:
        raise ValueError('warm-retention valid mask must be bool')
    if first.shape != student_logits.shape[:1]:
        raise ValueError('warm-retention first index must have shape (B,)')
    if not math.isfinite(beta_m) or beta_m <= 0.0:
        raise ValueError('warm-retention beta must be positive')
    row = torch.arange(student_logits.shape[0], device=student_logits.device)
    student_margin = (
        student_feasibility
        - student_feasibility[row, first].unsqueeze(-1))
    teacher_margin = (
        teacher_feasibility
        - teacher_feasibility[row, first].unsqueeze(-1))
    margin_error = F.smooth_l1_loss(
        student_margin, teacher_margin,
        reduction='none', beta=beta_m)
    feasibility_loss = (
        torch.where(valid, margin_error, torch.zeros_like(margin_error))
        .sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()

    teacher_probability = teacher_logits.exp()
    actor_kl_terms = teacher_probability * (
        teacher_logits - student_logits)
    actor_kl = torch.where(
        valid, actor_kl_terms, torch.zeros_like(actor_kl_terms)
    ).sum(dim=-1).mean()
    return feasibility_loss, actor_kl


def _configure_train_scope(
    policy: CandidateSeedActorCritic,
    scope: str,
) -> tuple[str, ...]:
    """Freeze a warm selector outside the requested backward-update scope."""
    if scope not in ('full', 'actor-feasibility', 'output-heads'):
        raise ValueError(f'unknown selector train scope: {scope!r}')
    trainable = []
    for name, parameter in policy.named_parameters():
        enabled = (
            scope == 'full'
            or (scope == 'actor-feasibility'
                and name.startswith(('actor.', 'feasibility.')))
            or (scope == 'output-heads'
                and name.startswith(('actor.2.', 'feasibility.2.')))
        )
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError('selector train scope contains no parameters')
    return tuple(trainable)


def _train_member(
    member_index: int,
    table: TrainingTable,
    mean: torch.Tensor,
    std: torch.Tensor,
    groups: Sequence[Sequence[int]],
    args: argparse.Namespace,
    device: torch.device,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[CandidateSeedActorCritic, torch.optim.Optimizer, dict[str, Any]]:
    member_seed = int(args.seed + 1009 * (member_index + 1))
    seed_global_rng(member_seed)
    policy = CandidateSeedActorCritic(
        45, hidden_dim=args.hidden, encoder_type='mean').to(device)
    if initial_state is None:
        policy.set_feature_normalization(mean.to(device), std.to(device))
    else:
        policy.load_state_dict(initial_state)
    trainable_names = _configure_train_scope(policy, args.train_scope)
    warm_teacher = None
    if initial_state is not None and (
            args.warm_feasibility_retain_coef > 0.0
            or args.warm_actor_kl_coef > 0.0):
        warm_teacher = copy.deepcopy(policy).eval()
        for parameter in warm_teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy.parameters()
         if parameter.requires_grad), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    generator = torch.Generator(device='cpu').manual_seed(member_seed + 1)
    bootstrap = torch.randint(
        len(groups), (len(groups),), generator=generator).tolist()
    bootstrap_groups = [groups[index] for index in bootstrap]
    unique_bootstrap = len(set(bootstrap))
    optimizer_steps = 0
    final_stats: dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        policy.train()
        order = _geometry_epoch_indices(bootstrap_groups, generator)
        totals = {key: 0.0 for key in (
            'loss', 'listnet', 'multi_positive', 'actor_pair',
            'feasibility', 'value', 'warm_feasibility', 'warm_actor_kl')}
        seen = 0
        for start in range(0, order.numel(), args.batch_size):
            index = order[start:start + args.batch_size]
            features = table.features[index].to(device)
            valid = table.valid[index].to(device)
            progress = table.progress_m[index].to(device)
            feasibility_target = table.feasibility_target_m[index].to(device)
            dist, value, feasibility = policy.distribution_and_values(
                features, valid)

            safe_min = progress.masked_fill(~valid, torch.inf).min(
                dim=-1).values
            safe_max = progress.masked_fill(~valid, -torch.inf).max(
                dim=-1).values
            progress_range = (safe_max - safe_min).clamp_min(1e-8)
            if args.actor_target == 'range-normalized':
                normalized = (
                    (progress - safe_min.unsqueeze(-1))
                    / progress_range.unsqueeze(-1))
                target_score = normalized / args.temperature
            else:
                # Absolute regret preserves the physical meaning of a tie:
                # 1 mm stays small on both easy and difficult tasks instead
                # of being stretched to [0, 1] by a tiny task range.
                target_score = (
                    (progress - safe_max.unsqueeze(-1))
                    / args.regret_temperature_m)
            target = torch.softmax(
                target_score.masked_fill(~valid, -torch.inf), dim=-1)
            listnet_per_task = -(target.detach() * dist.logits).sum(dim=-1)
            first = valid.float().argmax(dim=-1)
            row = torch.arange(index.numel(), device=device)
            first_progress = progress[row, first]
            headroom = (safe_max - first_progress).clamp_min(0.0)
            task_weight = 1.0 + args.headroom_weight * (
                headroom / args.headroom_reference_m).clamp_max(10.0)
            listnet_loss = (
                (listnet_per_task * task_weight.detach()).sum()
                / task_weight.sum().clamp_min(1e-8))
            near_oracle = (
                valid
                & (progress >= (safe_max - args.multi_positive_tolerance_m)
                   .unsqueeze(-1)))
            multi_target = (
                near_oracle.float()
                / near_oracle.sum(dim=-1, keepdim=True).clamp_min(1))
            multi_positive_per_task = -(
                multi_target.detach() * dist.logits).sum(dim=-1)
            multi_positive_loss = (
                (multi_positive_per_task * task_weight.detach()).sum()
                / task_weight.sum().clamp_min(1e-8))
            actor_pair_loss = _pairwise_actor_loss(
                dist.logits, progress, valid, args.pair_delta_m)

            predicted_advantage = (
                feasibility - feasibility[row, first].unsqueeze(-1))
            feasibility_error = F.smooth_l1_loss(
                predicted_advantage, feasibility_target,
                reduction='none', beta=args.feasibility_beta_m)
            feasibility_loss = (
                torch.where(valid, feasibility_error,
                            torch.zeros_like(feasibility_error)).sum(dim=-1)
                / valid.sum(dim=-1).clamp_min(1)).mean()
            value_loss = F.smooth_l1_loss(
                value, safe_max, beta=args.value_beta_m)
            warm_feasibility_loss = loss_zero = value.sum() * 0.0
            warm_actor_kl_loss = loss_zero
            if warm_teacher is not None:
                with torch.no_grad():
                    teacher_dist, _, teacher_feasibility = (
                        warm_teacher.distribution_and_values(features, valid))
                warm_feasibility_loss, warm_actor_kl_loss = (
                    _warm_retention_losses(
                        dist.logits, feasibility,
                        teacher_dist.logits, teacher_feasibility,
                        valid, first, beta_m=args.warm_retain_beta_m))
            loss = (listnet_loss
                    + args.multi_positive_coef * multi_positive_loss
                    + args.actor_pair_coef * actor_pair_loss
                    + args.feasibility_coef * feasibility_loss
                    + args.value_coef * value_loss
                    + args.warm_feasibility_retain_coef
                    * warm_feasibility_loss
                    + args.warm_actor_kl_coef * warm_actor_kl_loss)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError('ensemble seed loss became non-finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), args.max_grad_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError('ensemble seed gradient is non-finite')
            optimizer.step()
            optimizer_steps += 1
            count = int(index.numel())
            seen += count
            for name, scalar in (
                    ('loss', loss), ('listnet', listnet_loss),
                    ('multi_positive', multi_positive_loss),
                    ('actor_pair', actor_pair_loss),
                    ('feasibility', feasibility_loss), ('value', value_loss),
                    ('warm_feasibility', warm_feasibility_loss),
                    ('warm_actor_kl', warm_actor_kl_loss)):
                totals[name] += float(scalar.item()) * count
        final_stats = {key: value / max(seen, 1)
                       for key, value in totals.items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f'[offline-seed-ensemble] member {member_index + 1}/'
                f'{args.members} epoch {epoch:>3}/{args.epochs}  '
                f'loss={final_stats["loss"]:.4f}  '
                f'list={final_stats["listnet"]:.4f}  '
                f'multi={final_stats["multi_positive"]:.4f}  '
                f'pair={final_stats["actor_pair"]:.4f}  '
                f'q={final_stats["feasibility"]:.4f}  '
                f'retain-q={final_stats["warm_feasibility"]:.4f}  '
                f'retain-kl={final_stats["warm_actor_kl"]:.4f}', flush=True)
    policy.eval()
    return policy, optimizer, {
        'member': member_index,
        'seed': member_seed,
        'bootstrap_geometry_draws': len(bootstrap),
        'bootstrap_unique_geometries': unique_bootstrap,
        'optimizer_steps': optimizer_steps,
        'trainable_parameters': trainable_names,
        'final_epoch': final_stats,
        'sampler_state': generator.get_state(),
    }


@torch.no_grad()
def _ensemble_predictions(
    members: Sequence[CandidateSeedActorCritic],
    features: torch.Tensor,
    valid: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposals = []
    margins = []
    firsts = []
    ensemble = CandidateSeedPolicyEnsemble(list(members)).to(device).eval()
    for start in range(0, indices.numel(), batch_size):
        local = indices[start:start + batch_size]
        batch_features = features[local].to(device)
        batch_valid = valid[local].to(device)
        dist, _, mean_feasibility = ensemble.distribution_and_values(
            batch_features, batch_valid)
        mean_log_prob = dist.logits
        proposal = mean_log_prob.argmax(dim=-1)
        first = batch_valid.float().argmax(dim=-1)
        row = torch.arange(local.numel(), device=device)
        margin = (mean_feasibility[row, proposal]
                  - mean_feasibility[row, first])
        proposals.append(proposal.cpu())
        margins.append(margin.cpu())
        firsts.append(first.cpu())
    return tuple(torch.cat(values).numpy()
                 for values in (proposals, margins, firsts))


def _geometry_macro(values: np.ndarray,
                    fingerprints: Sequence[str]) -> float:
    sums: dict[str, list[float]] = {}
    for fingerprint, value in zip(fingerprints, values.tolist()):
        sums.setdefault(fingerprint, []).append(float(value))
    return float(np.mean([np.mean(group) for group in sums.values()]))


def _report_split(
    members: Sequence[CandidateSeedActorCritic],
    features: torch.Tensor,
    valid: torch.Tensor,
    progress: torch.Tensor,
    indices: torch.Tensor,
    fingerprints: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    proposal, _, first = _ensemble_predictions(
        members, features, valid, indices,
        batch_size=batch_size, device=device)
    progress_np = progress[indices].numpy()
    valid_np = valid[indices].numpy()
    row = np.arange(indices.numel())
    proposal_value = progress_np[row, proposal]
    first_value = progress_np[row, first]
    oracle_value = np.where(valid_np, progress_np, -np.inf).max(axis=1)
    selected_fingerprints = [fingerprints[int(index)]
                             for index in indices.tolist()]
    gain = proposal_value - first_value
    headroom = oracle_value - first_value
    macro_gain = _geometry_macro(gain, selected_fingerprints)
    macro_headroom = _geometry_macro(headroom, selected_fingerprints)
    return {
        'n_rows': int(indices.numel()),
        'n_geometry_groups': len(set(selected_fingerprints)),
        'mean_progress_m': _geometry_macro(
            proposal_value, selected_fingerprints),
        'first_valid_progress_m': _geometry_macro(
            first_value, selected_fingerprints),
        'oracle_progress_m': _geometry_macro(
            oracle_value, selected_fingerprints),
        'mean_gain_m': macro_gain,
        'oracle_capture': (
            macro_gain / macro_headroom if macro_headroom > 0.0 else 0.0),
        'oracle_hit_rate': _geometry_macro(
            (proposal_value == oracle_value).astype(np.float64),
            selected_fingerprints),
        'worse_rate': _geometry_macro(
            (gain < 0.0).astype(np.float64), selected_fingerprints),
    }


def _warm_refit_is_promotable(
    attempt: dict[str, float | int],
    warm: dict[str, float | int],
    *,
    minimum_gain_m: float,
    worse_tolerance: float,
) -> bool:
    """Binary model-split gate with the warm checkpoint as block zero."""
    if (not math.isfinite(minimum_gain_m) or minimum_gain_m < 0.0
            or not math.isfinite(worse_tolerance)
            or not 0.0 <= worse_tolerance <= 1.0):
        raise ValueError('warm promotion tolerances are invalid')
    return bool(
        float(attempt['mean_gain_m'])
        >= float(warm['mean_gain_m']) + minimum_gain_m
        and float(attempt['worse_rate'])
        <= float(warm['worse_rate']) + worse_tolerance)


def _calibrate_final(
    members: Sequence[CandidateSeedActorCritic],
    features: torch.Tensor,
    valid: torch.Tensor,
    progress: torch.Tensor,
    threshold_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    fingerprints: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    confidence_z: float,
    feasibility_target: str = _DEFAULT_FEASIBILITY_TARGET,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(feasibility_target, str) or not feasibility_target:
        raise ValueError('feasibility target name must be a non-empty string')
    threshold_proposal, threshold_margin, threshold_first = (
        _ensemble_predictions(
            members, features, valid, threshold_indices,
            batch_size=batch_size, device=device))
    threshold_fingerprints = [fingerprints[int(index)]
                              for index in threshold_indices.tolist()]
    threshold_selection = _calibrate_one_head(
        'actor', threshold_proposal, threshold_margin.astype(np.float64),
        threshold_first, progress[threshold_indices].numpy(),
        valid[threshold_indices].numpy(), threshold_fingerprints,
        confidence_z)

    proposal, margin, first = _ensemble_predictions(
        members, features, valid, calibration_indices,
        batch_size=batch_size, device=device)
    selected_fingerprints = [fingerprints[int(index)]
                             for index in calibration_indices.tolist()]
    progress_np = progress[calibration_indices].numpy()
    valid_np = valid[calibration_indices].numpy()

    def fixed_metrics(threshold: float) -> dict[str, Any]:
        row = np.arange(calibration_indices.numel())
        proposal_gain = (
            progress_np[row, proposal] - progress_np[row, first])
        deployed = np.where(margin.astype(np.float64) >= threshold,
                            proposal_gain, 0.0)
        grouped: dict[str, list[int]] = {}
        for local, fingerprint in enumerate(selected_fingerprints):
            grouped.setdefault(fingerprint, []).append(local)
        group_gain = np.asarray([
            deployed[index].mean() for index in grouped.values()],
            dtype=np.float64)
        selected_mask = margin.astype(np.float64) >= threshold
        group_selected = np.asarray([
            selected_mask[index].mean() for index in grouped.values()],
            dtype=np.float64)
        worse_mask = selected_mask & (proposal_gain < 0.0)
        group_worse = np.asarray([
            worse_mask[index].mean() for index in grouped.values()],
            dtype=np.float64)
        first_value = progress_np[row, first]
        oracle_value = np.where(valid_np, progress_np, -np.inf).max(axis=1)
        group_first = np.asarray([
            first_value[index].mean() for index in grouped.values()])
        group_headroom = np.asarray([
            (oracle_value - first_value)[index].mean()
            for index in grouped.values()])
        mean_gain = float(group_gain.mean())
        standard_error = float(
            group_gain.std(ddof=1) / math.sqrt(len(group_gain))
            if len(group_gain) > 1 else 0.0)
        headroom = float(group_headroom.mean())
        return {
            'proposal_head': 'actor',
            'threshold': float(threshold),
            'mean_gain': mean_gain,
            'mean_return': float(group_first.mean()) + mean_gain,
            'first_valid_mean_return': float(group_first.mean()),
            'oracle_headroom': headroom,
            'oracle_capture': mean_gain / headroom if headroom > 0.0 else 0.0,
            'lower_bound': mean_gain - confidence_z * standard_error,
            'standard_error': standard_error,
            'selection_rate': float(group_selected.mean()),
            'worse_rate': float(group_worse.mean()),
        }

    selected = fixed_metrics(float(threshold_selection['threshold']))
    certified = selected['lower_bound'] >= 0.0
    if not certified:
        # No finite float32 feasibility difference can reach this threshold.
        selected = fixed_metrics(float(np.finfo(np.float64).max))
    deployment = {
        'mode': 'conservative',
        'proposal_head': 'actor',
        'threshold': float(selected['threshold']),
        'comparison': 'ge',
    }
    calibration = {
        'format': 'seed-ensemble-deployment-calibration-v1',
        'aggregation': SEED_ENSEMBLE_AGGREGATION,
        'proposal': 'mean-member-log-probability-argmax',
        'gate_score': 'mean-feasibility-proposal-minus-first-valid',
        'fallback': 'first_valid',
        'feasibility_target': feasibility_target,
        'confidence_z': float(confidence_z),
        'threshold_selection_rows': int(threshold_indices.numel()),
        'threshold_selection_geometry_groups': len(
            set(threshold_fingerprints)),
        'threshold_selection': threshold_selection,
        'n_rows': int(calibration_indices.numel()),
        'n_geometry_groups': len(set(selected_fingerprints)),
        'selected': selected,
        'fixed_threshold_certified': bool(certified),
        'final_calibration_used_once': True,
    }
    return deployment, calibration


def _publish(
    out_dir: Path,
    *,
    source: dict,
    controller_dir: Path,
    members: Sequence[CandidateSeedActorCritic],
    optimizers: Sequence[torch.optim.Optimizer],
    deployment: dict,
    calibration: dict,
    training: dict,
    provenance: dict,
    fit_indices: torch.Tensor,
    model_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    training_fingerprints: Sequence[str],
    device: torch.device,
) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(mode=0o755)
    except FileExistsError as error:
        raise FileExistsError(
            f'refusing to overwrite output directory: {out_dir}') from error
    states = [_cpu_tree(member.state_dict()) for member in members]
    architecture = members[0].architecture
    ensemble_metadata = {
        'format': SEED_ENSEMBLE_FORMAT,
        'aggregation': SEED_ENSEMBLE_AGGREGATION,
        'size': len(states),
    }
    with open(controller_dir / 'agent.pt', 'rb') as source_stream:
        with open(out_dir / 'agent.pt', 'xb') as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    output_config = copy.deepcopy(load_run_config(controller_dir))
    output_unified = output_config.setdefault('unified', {})
    output_unified.update({
        'seed_architecture': copy.deepcopy(architecture),
        'seed_feature_schema': (
            'initial-observation-ray-logmanip-directional-45d-v1'),
        'seed_ensemble': copy.deepcopy(ensemble_metadata),
        'seed_deployment': copy.deepcopy(deployment),
        'offline_seed_selector': 'offline-seed-ensemble-v1',
    })
    with open(out_dir / 'config.yaml', 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    copied_agent = file_fingerprint(out_dir / 'agent.pt')
    expected_agent = provenance['controller']['agent']
    if (copied_agent['size'] != expected_agent['size']
            or copied_agent['sha256'] != expected_agent['sha256']):
        raise RuntimeError(
            'controller agent changed while publishing ensemble checkpoint')
    current_source_controller = controller_fingerprint(controller_dir)
    for kind in ('agent', 'config'):
        expected = provenance['controller'][kind]
        current = current_source_controller[kind]
        if (expected['size'] != current['size']
                or expected['sha256'] != current['sha256']):
            raise RuntimeError(
                f'controller {kind} changed while publishing ensemble checkpoint')

    state = copy.deepcopy(source)
    combined_provenance = copy.deepcopy(source['provenance'])
    combined_provenance['offline_seed_ensemble'] = copy.deepcopy(provenance)
    unique_training_fingerprints = tuple(sorted(set(training_fingerprints)))
    state.update({
        'phase': 'offline_seed_ensemble_complete',
        'seed_policy': copy.deepcopy(states[0]),
        'seed_policy_ensemble': copy.deepcopy(states),
        'seed_ensemble': copy.deepcopy(ensemble_metadata),
        'seed_ensemble_inference_only': True,
        'seed_architecture': copy.deepcopy(architecture),
        'feature_dim': 45,
        'hidden_dim': int(architecture['hidden_dim']),
        'seed_include_ray_error': True,
        'seed_include_log_manip': True,
        'seed_include_directional_dynamics': True,
        'seed_policy_feature_schema': 'initial-observation-ray-logmanip-directional-45d-v1',
        'seed_selector_objective': 'progress_m',
        'seed_optimizer': _cpu_tree(optimizers[0].state_dict()),
        'offline_seed_ensemble_optimizers': [
            _cpu_tree(optimizer.state_dict()) for optimizer in optimizers],
        'seed_deployment': copy.deepcopy(deployment),
        'seed_deployment_calibration': copy.deepcopy(calibration),
        'seed_feasibility_target': training['deployed_feasibility_target'],
        'offline_seed_ensemble_training': copy.deepcopy(training),
        'offline_seed_ensemble_provenance': copy.deepcopy(provenance),
        'offline_seed_ensemble_fit_task_fingerprints': (
            unique_training_fingerprints),
        'controller_run_config_sha256': file_fingerprint(
            out_dir / 'config.yaml')['sha256'],
        'provenance': combined_provenance,
        'offline_ensemble_fit_local_indices': fit_indices.cpu(),
        'offline_ensemble_model_select_local_indices': model_indices.cpu(),
        'offline_ensemble_calibration_local_indices': calibration_indices.cpu(),
        'offline_ensemble_fit_task_indices': torch.as_tensor(
            source['train_task_indices']).cpu()[fit_indices],
        'offline_ensemble_model_select_task_indices': torch.as_tensor(
            source['train_task_indices']).cpu()[model_indices],
        'offline_ensemble_calibration_task_indices': torch.as_tensor(
            source['train_task_indices']).cpu()[calibration_indices],
    })
    state.update(global_rng_state(device))
    checkpoint_path = out_dir / 'unified.pt'
    with open(checkpoint_path, 'xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Train an independently bootstrapped 45-D seed-policy ensemble '
            'from strict exhaustive source/external return caches.'))
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--return-cache', required=True)
    parser.add_argument(
        '--external-return-cache', action='append', default=[],
        help='repeatable external-seed-return-cache-v1 artifact')
    parser.add_argument(
        '--reference-source-checkpoint', default=None,
        help=(
            'optional S0 unified.pt that binds the reference controller for '
            'a controller-robust feasibility target'))
    parser.add_argument(
        '--reference-return-cache', default=None,
        help='source seed-return cache evaluated by the reference controller')
    parser.add_argument(
        '--reference-external-return-cache', action='append', default=[],
        help=(
            'reference-controller external cache paired positionally with '
            'each --external-return-cache'))
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--members', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--seed', type=int, default=31000)
    parser.add_argument('--feature-chunk-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument(
        '--train-scope',
        choices=('full', 'actor-feasibility', 'output-heads'),
        default='full',
        help=(
            'parameter scope for a backward selector update; output-heads '
            'keeps the learned representation fixed'))
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument(
        '--actor-target',
        choices=('range-normalized', 'absolute-regret'),
        default='range-normalized')
    parser.add_argument('--regret-temperature-m', type=float, default=0.005)
    parser.add_argument(
        '--multi-positive-tolerance-m', type=float, default=0.001)
    parser.add_argument('--multi-positive-coef', type=float, default=0.0)
    parser.add_argument(
        '--warm-start-checkpoint', default=None,
        help='optional compatible ensemble whose member weights initialize training')
    parser.add_argument(
        '--allow-controller-transfer-warm-start', action='store_true',
        help='explicitly warm-start S0 while relabeling under a new controller')
    parser.add_argument(
        '--warm-feasibility-retain-coef', type=float, default=0.0,
        help='retain the warm selector feasibility margin used by deployment')
    parser.add_argument(
        '--warm-actor-kl-coef', type=float, default=0.0,
        help='forward-KL trust region to the warm selector actor')
    parser.add_argument('--warm-retain-beta-m', type=float, default=0.001)
    parser.add_argument(
        '--warm-promotion-min-gain-m', type=float, default=0.001,
        help='minimum model-split gain over warm block zero before promotion')
    parser.add_argument(
        '--warm-promotion-worse-tolerance', type=float, default=0.0)
    parser.add_argument('--model-select-fraction', type=float, default=0.15)
    parser.add_argument('--calibration-fraction', type=float, default=0.15)
    parser.add_argument('--calibration-z', type=float, default=1.96)
    parser.add_argument('--headroom-weight', type=float, default=1.0)
    parser.add_argument('--headroom-reference-m', type=float, default=0.05)
    parser.add_argument('--pair-delta-m', type=float, default=0.001)
    parser.add_argument('--actor-pair-coef', type=float, default=0.25)
    parser.add_argument('--feasibility-coef', type=float, default=2.0)
    parser.add_argument('--feasibility-beta-m', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--value-beta-m', type=float, default=0.05)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--log-every', type=int, default=10)
    args = parser.parse_args()

    for name in ('members', 'epochs', 'batch_size', 'hidden',
                 'feature_chunk_size', 'log_every'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    positive = (
        'learning_rate', 'temperature', 'regret_temperature_m',
        'calibration_z', 'warm_retain_beta_m',
        'headroom_reference_m', 'pair_delta_m', 'feasibility_beta_m',
        'value_beta_m', 'max_grad_norm',
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    for name in ('weight_decay', 'headroom_weight', 'actor_pair_coef',
                 'multi_positive_tolerance_m', 'multi_positive_coef',
                 'feasibility_coef', 'value_coef',
                 'warm_feasibility_retain_coef', 'warm_actor_kl_coef',
                 'warm_promotion_min_gain_m',
                 'warm_promotion_worse_tolerance'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{name.replace("_", "-")} must be non-negative')
    if (not 0.0 < args.model_select_fraction < 1.0
            or not 0.0 < args.calibration_fraction < 1.0
            or args.model_select_fraction + args.calibration_fraction >= 1.0):
        raise ValueError('model-select/calibration fractions must leave fit data')
    if args.warm_promotion_worse_tolerance > 1.0:
        raise ValueError('--warm-promotion-worse-tolerance must be <= 1')

    source_path = Path(args.source_checkpoint).expanduser().resolve(strict=True)
    return_path = Path(args.return_cache).expanduser().resolve(strict=True)
    controller_dir = resolve_controller_dir(args.controller_ckpt)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if not source_path.is_file() or source_path.name != 'unified.pt':
        raise ValueError('--source-checkpoint must name unified.pt')
    if not return_path.is_file() or return_path.suffix.lower() != '.npz':
        raise ValueError('--return-cache must name an NPZ')
    if os.path.lexists(out_dir):
        raise FileExistsError(f'refusing to overwrite output: {out_dir}')
    external_paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in args.external_return_cache]
    if len(set(external_paths)) != len(external_paths):
        raise ValueError('duplicate --external-return-cache paths are forbidden')
    reference_mode = (
        args.reference_source_checkpoint is not None
        or args.reference_return_cache is not None
        or bool(args.reference_external_return_cache))
    if reference_mode and (
            args.reference_source_checkpoint is None
            or args.reference_return_cache is None):
        raise ValueError(
            'controller-robust feasibility requires both '
            '--reference-source-checkpoint and --reference-return-cache')
    if (not reference_mode and args.reference_external_return_cache):
        raise ValueError(
            '--reference-external-return-cache requires reference mode')
    reference_source_path = (
        Path(args.reference_source_checkpoint).expanduser().resolve(strict=True)
        if reference_mode else None)
    reference_return_path = (
        Path(args.reference_return_cache).expanduser().resolve(strict=True)
        if reference_mode else None)
    reference_external_paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in args.reference_external_return_cache]
    if len(set(reference_external_paths)) != len(reference_external_paths):
        raise ValueError(
            'duplicate --reference-external-return-cache paths are forbidden')
    if reference_mode and len(reference_external_paths) != len(external_paths):
        raise ValueError(
            'reference external caches must pair one-to-one with primary '
            'external caches')
    if reference_mode:
        assert reference_source_path is not None
        assert reference_return_path is not None
        if (not reference_source_path.is_file()
                or reference_source_path.name != 'unified.pt'):
            raise ValueError(
                '--reference-source-checkpoint must name unified.pt')
        if (not reference_return_path.is_file()
                or reference_return_path.suffix.lower() != '.npz'):
            raise ValueError('--reference-return-cache must name an NPZ')
    warm_path = (
        Path(args.warm_start_checkpoint).expanduser().resolve(strict=True)
        if args.warm_start_checkpoint is not None else None)
    if warm_path is None and (
            args.warm_feasibility_retain_coef > 0.0
            or args.warm_actor_kl_coef > 0.0):
        raise ValueError(
            'warm retention coefficients require --warm-start-checkpoint')
    if args.allow_controller_transfer_warm_start and warm_path is None:
        raise ValueError(
            '--allow-controller-transfer-warm-start requires '
            '--warm-start-checkpoint')
    if args.train_scope != 'full' and warm_path is None:
        raise ValueError(
            'a restricted --train-scope requires --warm-start-checkpoint')

    source_artifact = file_fingerprint(source_path)
    return_artifact = file_fingerprint(return_path)
    controller_artifact = controller_fingerprint(controller_dir)
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if not isinstance(source, dict):
        raise ValueError('source checkpoint must contain a dictionary')
    candidate_record = source.get('provenance', {}).get('candidate_cache')
    if not isinstance(candidate_record, dict) or 'path' not in candidate_record:
        raise ValueError('source provenance lacks a candidate cache path')
    candidate_path = Path(candidate_record['path']).expanduser().resolve(
        strict=True)
    candidate_artifact = file_fingerprint(candidate_path)
    agent_state = torch.load(
        controller_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if not isinstance(agent_state, dict):
        raise ValueError('controller agent.pt must contain a state dictionary')
    controller_state_sha256 = state_dict_fingerprint(agent_state)
    controller_run_config = load_run_config(controller_dir)
    effective_config = dataclasses.asdict(
        ppo_config_from_run(controller_run_config))
    objective, gamma = _validate_source_checkpoint(
        source, source_artifact, candidate_artifact, controller_artifact,
        controller_state_sha256, effective_config)

    reference_source = None
    reference_source_artifact = None
    reference_return_artifact = None
    reference_candidate_path = None
    reference_candidate_artifact = None
    reference_controller_dir = None
    reference_controller_artifact = None
    reference_controller_state_sha256 = None
    reference_objective = None
    reference_gamma = None
    if reference_mode:
        assert reference_source_path is not None
        assert reference_return_path is not None
        reference_source_artifact = file_fingerprint(reference_source_path)
        reference_return_artifact = file_fingerprint(reference_return_path)
        reference_source = torch.load(
            reference_source_path, map_location='cpu', weights_only=False)
        if not isinstance(reference_source, dict):
            raise ValueError(
                '--reference-source-checkpoint must contain a dictionary')
        reference_candidate_record = reference_source.get(
            'provenance', {}).get('candidate_cache')
        if (not isinstance(reference_candidate_record, dict)
                or 'path' not in reference_candidate_record):
            raise ValueError(
                'reference source provenance lacks a candidate cache path')
        reference_candidate_path = Path(
            reference_candidate_record['path']).expanduser().resolve(
                strict=True)
        reference_candidate_artifact = file_fingerprint(
            reference_candidate_path)
        _same_content(
            int(reference_candidate_artifact['size']),
            str(reference_candidate_artifact['sha256']), candidate_artifact,
            label='primary/reference source candidate cache')

        # The reference controller is deliberately derived from S0's source
        # directory.  _validate_source_checkpoint then binds agent.pt,
        # config.yaml, and the embedded controller tensors fail-closed.
        reference_controller_dir = resolve_controller_dir(
            reference_source_path.parent)
        reference_controller_artifact = controller_fingerprint(
            reference_controller_dir)
        reference_agent_state = torch.load(
            reference_controller_dir / 'agent.pt', map_location='cpu',
            weights_only=True)
        if not isinstance(reference_agent_state, dict):
            raise ValueError(
                'reference controller agent.pt must contain a state dictionary')
        reference_controller_state_sha256 = state_dict_fingerprint(
            reference_agent_state)
        reference_run_config = load_run_config(reference_controller_dir)
        reference_effective_config = dataclasses.asdict(
            ppo_config_from_run(reference_run_config))
        reference_objective, reference_gamma = _validate_source_checkpoint(
            reference_source, reference_source_artifact,
            reference_candidate_artifact, reference_controller_artifact,
            reference_controller_state_sha256, reference_effective_config)
        if reference_controller_state_sha256 == controller_state_sha256:
            raise ValueError(
                'primary/reference controller states must be different')
        if reference_effective_config != effective_config:
            raise ValueError(
                'primary/reference controller PPO semantics differ')
        if reference_run_config.get('env') != controller_run_config.get('env'):
            raise ValueError(
                'primary/reference controller environment semantics differ')
        if reference_objective != objective:
            raise ValueError(
                'primary/reference controller seed objectives differ')
        if reference_gamma != gamma:
            raise ValueError('primary/reference controller gamma differs')

    warm_source = None
    warm_states = None
    warm_artifact = None
    if warm_path is not None:
        warm_source = torch.load(
            warm_path, map_location='cpu', weights_only=False)
        if not isinstance(warm_source, dict):
            raise ValueError('--warm-start-checkpoint must contain a dictionary')
        ensemble_data = seed_policy_ensemble_states(warm_source)
        if ensemble_data is None:
            raise ValueError('--warm-start-checkpoint has no seed ensemble')
        warm_states, warm_metadata = ensemble_data
        warm_config = infer_seed_policy_config(warm_source)
        if (len(warm_states) != args.members
                or warm_config.feature_dim != 45
                or warm_config.hidden_dim != args.hidden
                or warm_config.encoder_type != 'mean'):
            raise ValueError(
                'warm-start ensemble size/architecture differs from requested '
                'training configuration')
        if warm_metadata['aggregation'] != SEED_ENSEMBLE_AGGREGATION:
            raise ValueError('warm-start ensemble aggregation is incompatible')
        warm_controller_transfer = (
            warm_source.get('controller_state_sha256')
            != controller_state_sha256)
        if (warm_controller_transfer
                and not args.allow_controller_transfer_warm_start):
            raise ValueError(
                'warm-start ensemble was trained against a different '
                'controller; pass --allow-controller-transfer-warm-start '
                'only for an explicit bidirectional backward phase')
        warm_candidate = warm_source.get('provenance', {}).get(
            'candidate_cache')
        if (not isinstance(warm_candidate, dict)
                or warm_candidate.get('size') != candidate_artifact['size']
                or warm_candidate.get('sha256')
                != candidate_artifact['sha256']):
            raise ValueError(
                'warm-start ensemble uses a different candidate cache')
        for key in ('train_task_indices', 'validation_task_indices',
                    'train_valid_mask', 'validation_valid_mask'):
            if key not in warm_source or key not in source:
                raise ValueError(
                    f'warm/source checkpoint is missing {key!r}')
            warm_value = torch.as_tensor(warm_source[key]).cpu()
            source_value = torch.as_tensor(source[key]).cpu()
            if not torch.equal(warm_value, source_value):
                raise ValueError(
                    f'warm-start ensemble differs in {key!r}')
        if (not bool(warm_source.get(
                'seed_include_directional_dynamics', False))
                or int(warm_source.get('feature_dim', -1)) != 45):
            raise ValueError(
                'warm-start ensemble does not use the 45-D directional schema')
        warm_artifact = file_fingerprint(warm_path)

    seed_global_rng(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    env = build_env_from_run(controller_dir, 1, device)
    source_dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    source_dataset, source_physical_stats = validate_cached_dataset(
        source_dataset, env.kin, env.collision,
        chunk_size=args.feature_chunk_size, cone_deg=env.cfg.cone_deg)
    train_dataset = source_dataset.select_source_tasks(
        torch.as_tensor(source['train_task_indices']).cpu())
    validation_dataset = source_dataset.select_source_tasks(
        torch.as_tensor(source['validation_task_indices']).cpu())
    assert_same_valid_mask(
        train_dataset, source['train_valid_mask'], label='training')
    assert_same_valid_mask(
        validation_dataset, source['validation_valid_mask'],
        label='validation')
    if set(train_dataset.task_fingerprints) & set(
            validation_dataset.task_fingerprints):
        raise ValueError('source train/validation geometry overlap')
    cached = load_return_cache(
        return_path, source=source, source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        controller_artifact=controller_artifact,
        controller_state_sha256=controller_state_sha256,
        objective=objective, gamma=gamma, train_dataset=train_dataset)
    if cached.artifact != return_artifact:
        raise RuntimeError('source return cache changed during loading')

    reference_env = None
    reference_cached = None
    reference_train_dataset = None
    reference_validation_dataset = None
    reference_source_physical_stats = None
    source_feasibility_target = None
    if reference_mode:
        assert reference_source is not None
        assert reference_source_artifact is not None
        assert reference_return_path is not None
        assert reference_return_artifact is not None
        assert reference_candidate_path is not None
        assert reference_candidate_artifact is not None
        assert reference_controller_dir is not None
        assert reference_controller_artifact is not None
        assert reference_controller_state_sha256 is not None
        assert reference_objective is not None
        assert reference_gamma is not None
        reference_env = build_env_from_run(reference_controller_dir, 1, device)
        reference_source_dataset = CachedSeedCandidateDataset.from_npz(
            reference_candidate_path)
        reference_source_dataset, reference_source_physical_stats = (
            validate_cached_dataset(
                reference_source_dataset, reference_env.kin,
                reference_env.collision, chunk_size=args.feature_chunk_size,
                cone_deg=reference_env.cfg.cone_deg))
        reference_train_dataset = reference_source_dataset.select_source_tasks(
            torch.as_tensor(reference_source['train_task_indices']).cpu())
        reference_validation_dataset = (
            reference_source_dataset.select_source_tasks(
                torch.as_tensor(
                    reference_source['validation_task_indices']).cpu()))
        assert_same_valid_mask(
            reference_train_dataset, reference_source['train_valid_mask'],
            label='reference training')
        assert_same_valid_mask(
            reference_validation_dataset,
            reference_source['validation_valid_mask'],
            label='reference validation')
        _assert_paired_candidate_datasets(
            train_dataset, reference_train_dataset,
            label='primary/reference source training')
        _assert_paired_candidate_datasets(
            validation_dataset, reference_validation_dataset,
            label='primary/reference source validation')
        reference_cached = load_return_cache(
            reference_return_path, source=reference_source,
            source_artifact=reference_source_artifact,
            candidate_artifact=reference_candidate_artifact,
            controller_artifact=reference_controller_artifact,
            controller_state_sha256=reference_controller_state_sha256,
            objective=reference_objective, gamma=reference_gamma,
            train_dataset=reference_train_dataset)
        if reference_cached.artifact != reference_return_artifact:
            raise RuntimeError(
                'reference source return cache changed during loading')
        if cached.task_fingerprints != reference_cached.task_fingerprints:
            raise ValueError(
                'primary/reference source cache task fingerprints differ')
        source_feasibility_target = _controller_robust_feasibility_target(
            cached.progress_m, reference_cached.progress_m,
            cached.valid, reference_cached.valid,
            label='source training cache')

    if warm_source is None:
        fit_indices, model_indices, calibration_indices = (
            _three_way_geometry_split(
                train_dataset, args.model_select_fraction,
                args.calibration_fraction, args.seed))
    else:
        required_split_keys = (
            'offline_ensemble_fit_local_indices',
            'offline_ensemble_model_select_local_indices',
            'offline_ensemble_calibration_local_indices',
        )
        if any(key not in warm_source for key in required_split_keys):
            raise ValueError(
                'warm-start checkpoint lacks its leakage-safe three-way split')
        fit_indices, model_indices, calibration_indices = tuple(
            torch.as_tensor(warm_source[key]).long().cpu()
            for key in required_split_keys)
        combined = torch.cat([
            fit_indices, model_indices, calibration_indices])
        if (combined.numel() != len(train_dataset)
                or not torch.equal(
                    combined.sort().values,
                    torch.arange(len(train_dataset)))):
            raise ValueError(
                'warm-start three-way split does not partition source training rows')
        _assert_three_way_geometry_disjoint(
            (fit_indices, model_indices, calibration_indices),
            train_dataset.task_fingerprints)
    heldout_fingerprints = {
        cached.task_fingerprints[int(index)]
        for index in torch.cat([model_indices, calibration_indices]).tolist()
    }

    external_data = []
    external_artifact_ids: set[tuple[int, str]] = set()
    reference_external_data = []
    reference_external_artifact_ids: set[tuple[int, str]] = set()
    external_feasibility_targets = [] if reference_mode else None
    for external_index, external_path in enumerate(external_paths):
        cache = load_external_return_cache(
            external_path, source=source, source_artifact=source_artifact,
            source_candidate_artifact=candidate_artifact,
            controller_artifact=controller_artifact,
            controller_state_sha256=controller_state_sha256,
            objective=objective, gamma=gamma,
            source_validation=validation_dataset, env=env,
            physical_chunk_size=args.feature_chunk_size)
        identity = (int(cache.cache_artifact['size']),
                    str(cache.cache_artifact['sha256']))
        if identity in external_artifact_ids:
            raise ValueError('duplicate external return-cache content')
        external_artifact_ids.add(identity)
        keep = torch.tensor([
            fingerprint not in heldout_fingerprints
            for fingerprint in cache.task_fingerprints], dtype=torch.bool)
        keep_index = torch.nonzero(keep, as_tuple=False).flatten()
        if keep_index.numel() == 0:
            print(
                f'[offline-seed-ensemble] external {external_path.name}: '
                'all rows excluded by source model/calibration geometry',
                flush=True)
        external_data.append((cache, keep_index))
        if reference_mode:
            assert reference_source is not None
            assert reference_source_artifact is not None
            assert reference_candidate_artifact is not None
            assert reference_controller_artifact is not None
            assert reference_controller_state_sha256 is not None
            assert reference_objective is not None
            assert reference_gamma is not None
            assert reference_validation_dataset is not None
            assert reference_env is not None
            reference_external_path = reference_external_paths[external_index]
            reference_cache = load_external_return_cache(
                reference_external_path, source=reference_source,
                source_artifact=reference_source_artifact,
                source_candidate_artifact=reference_candidate_artifact,
                controller_artifact=reference_controller_artifact,
                controller_state_sha256=reference_controller_state_sha256,
                objective=reference_objective, gamma=reference_gamma,
                source_validation=reference_validation_dataset,
                env=reference_env,
                physical_chunk_size=args.feature_chunk_size)
            reference_identity = (
                int(reference_cache.cache_artifact['size']),
                str(reference_cache.cache_artifact['sha256']))
            if reference_identity in reference_external_artifact_ids:
                raise ValueError(
                    'duplicate reference external return-cache content')
            reference_external_artifact_ids.add(reference_identity)
            _same_content(
                int(reference_cache.candidate_artifact['size']),
                str(reference_cache.candidate_artifact['sha256']),
                cache.candidate_artifact,
                label=(
                    f'primary/reference external candidate cache '
                    f'{external_index}'))
            _assert_paired_candidate_datasets(
                cache.dataset, reference_cache.dataset,
                label=f'primary/reference external cache {external_index}')
            if cache.task_fingerprints != reference_cache.task_fingerprints:
                raise ValueError(
                    f'primary/reference external cache {external_index} task '
                    'fingerprints differ')
            reference_keep = torch.tensor([
                fingerprint not in heldout_fingerprints
                for fingerprint in reference_cache.task_fingerprints],
                dtype=torch.bool)
            reference_keep_index = torch.nonzero(
                reference_keep, as_tuple=False).flatten()
            if not torch.equal(keep_index, reference_keep_index):
                raise ValueError(
                    f'primary/reference external cache {external_index} '
                    'retained rows differ')
            reference_external_data.append(
                (reference_cache, reference_keep_index))
            assert external_feasibility_targets is not None
            external_feasibility_targets.append(
                _controller_robust_feasibility_target(
                    cache.progress_m, reference_cache.progress_m,
                    cache.valid, reference_cache.valid,
                    label=f'external cache {external_index}'))

    print(
        '[offline-seed-ensemble] building 45-D directional features: '
        f'source={len(train_dataset)}, external='
        f'{sum(len(cache.dataset) for cache, _ in external_data)}', flush=True)
    source_features = _build_features(
        env.kin, train_dataset, args.feature_chunk_size)
    external_with_features = [
        (cache, _build_features(
            env.kin, cache.dataset, args.feature_chunk_size), keep)
        for cache, keep in external_data if keep.numel() > 0
    ]
    table_external_feasibility_targets = (
        [
            external_feasibility_targets[index]
            for index, (_, keep) in enumerate(external_data)
            if keep.numel() > 0
        ]
        if external_feasibility_targets is not None else None)
    table = _make_training_table(
        source_features, cached.valid, cached.progress_m,
        cached.task_fingerprints, fit_indices, external_with_features,
        source_feasibility_target=source_feasibility_target,
        external_feasibility_targets=table_external_feasibility_targets)
    groups = geometry_groups(table.task_fingerprints)
    mean, std = _normalization(table.features, table.valid)
    print(
        '[offline-seed-ensemble] fit rows='
        f'{table.features.shape[0]} (source={table.n_source_rows}, '
        f'external={table.n_external_rows}), geometries={len(groups)}, '
        f'model-select={model_indices.numel()}, '
        f'final-calibration={calibration_indices.numel()}', flush=True)

    members = []
    optimizers = []
    member_training = []
    for member_index in range(args.members):
        member, optimizer, member_stats = _train_member(
            member_index, table, mean, std, groups, args, device,
            initial_state=(None if warm_states is None
                           else dict(warm_states[member_index])))
        members.append(member)
        optimizers.append(optimizer)
        member_training.append(member_stats)

    # Fixed epochs are intentional.  For a warm refit, the immutable warm
    # checkpoint is block zero and the model-selection split may only answer
    # the binary promotion question.  Final calibration remains untouched
    # until that decision has been made.
    feasibility_target_name = (
        _ROBUST_FEASIBILITY_TARGET
        if reference_mode else _DEFAULT_FEASIBILITY_TARGET)
    deployed_feasibility_target = feasibility_target_name
    attempt_model_report = _report_split(
        members, source_features, cached.valid, cached.progress_m,
        model_indices, cached.task_fingerprints,
        batch_size=args.batch_size, device=device)
    warm_model_report = None
    warm_promoted = None
    if warm_states is not None:
        warm_members = []
        for state in warm_states:
            warm_member = CandidateSeedActorCritic(
                45, hidden_dim=args.hidden, encoder_type='mean').to(device)
            warm_member.load_state_dict(state)
            warm_member.eval()
            warm_members.append(warm_member)
        warm_model_report = _report_split(
            warm_members, source_features, cached.valid, cached.progress_m,
            model_indices, cached.task_fingerprints,
            batch_size=args.batch_size, device=device)
        warm_promoted = _warm_refit_is_promotable(
            attempt_model_report, warm_model_report,
            minimum_gain_m=args.warm_promotion_min_gain_m,
            worse_tolerance=args.warm_promotion_worse_tolerance)
        print(
            '[offline-seed-ensemble] warm promotion: '
            f'block0 gain={warm_model_report["mean_gain_m"]:+.4f}m, '
            f'attempt={attempt_model_report["mean_gain_m"]:+.4f}m, '
            f'promoted={warm_promoted}', flush=True)
        if not warm_promoted:
            members = warm_members
            warm_optimizer_states = warm_source.get(
                'offline_seed_ensemble_optimizers')
            optimizers = []
            for index, member in enumerate(members):
                optimizer = torch.optim.AdamW(
                    member.parameters(), lr=args.learning_rate,
                    weight_decay=args.weight_decay)
                if (isinstance(warm_optimizer_states, (list, tuple))
                        and len(warm_optimizer_states) == len(members)):
                    optimizer.load_state_dict(warm_optimizer_states[index])
                optimizers.append(optimizer)
            model_report = warm_model_report
            deployment = copy.deepcopy(warm_source['seed_deployment'])
            calibration = copy.deepcopy(
                warm_source['seed_deployment_calibration'])
            deployed_feasibility_target = warm_source.get(
                'seed_feasibility_target',
                calibration.get('feasibility_target'))
            if (not isinstance(deployed_feasibility_target, str)
                    or not deployed_feasibility_target):
                raise ValueError(
                    'warm-start checkpoint lacks its feasibility-target identity')
        else:
            model_report = attempt_model_report
    else:
        model_report = attempt_model_report

    if warm_promoted is not False:
        deployment, calibration = _calibrate_final(
            members, source_features, cached.valid, cached.progress_m,
            model_indices, calibration_indices, cached.task_fingerprints,
            batch_size=args.batch_size, device=device,
            confidence_z=args.calibration_z,
            feasibility_target=feasibility_target_name)
    chosen = calibration['selected']
    print(
        '[offline-seed-ensemble] model-select report: '
        f'gain={model_report["mean_gain_m"]:+.4f}m, '
        f'capture={100 * model_report["oracle_capture"]:.1f}%; '
        f'final gate threshold={deployment["threshold"]:.6g}, '
        f'gain={chosen["mean_gain"]:+.4f}m, '
        f'LCB={chosen["lower_bound"]:+.4f}m, '
        f'select={100 * chosen["selection_rate"]:.1f}%', flush=True)

    settings = {
        'members': args.members,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'train_scope': args.train_scope,
        'temperature': args.temperature,
        'actor_target_mode': args.actor_target,
        'regret_temperature_m': args.regret_temperature_m,
        'multi_positive_tolerance_m': args.multi_positive_tolerance_m,
        'multi_positive_coef': args.multi_positive_coef,
        'warm_feasibility_retain_coef': (
            args.warm_feasibility_retain_coef),
        'warm_actor_kl_coef': args.warm_actor_kl_coef,
        'warm_retain_beta_m': args.warm_retain_beta_m,
        'warm_promotion_min_gain_m': args.warm_promotion_min_gain_m,
        'warm_promotion_worse_tolerance': (
            args.warm_promotion_worse_tolerance),
        'warm_promoted': warm_promoted,
        'deployed_feasibility_target': deployed_feasibility_target,
        'model_select_fraction': args.model_select_fraction,
        'calibration_fraction': args.calibration_fraction,
        'calibration_z': args.calibration_z,
        'headroom_weight': args.headroom_weight,
        'headroom_reference_m': args.headroom_reference_m,
        'pair_delta_m': args.pair_delta_m,
        'actor_pair_coef': args.actor_pair_coef,
        'feasibility_coef': args.feasibility_coef,
        'feasibility_beta_m': args.feasibility_beta_m,
        'value_coef': args.value_coef,
        'value_beta_m': args.value_beta_m,
        'max_grad_norm': args.max_grad_norm,
        'seed': args.seed,
        'device': device_identity(device),
        'feature_schema': 'initial-observation-ray-logmanip-directional-45d-v1',
        'selector_objective': 'progress_m',
        'actor_target': (
            'range-normalized-listnet-v1'
            if args.actor_target == 'range-normalized'
            else 'absolute-regret-listnet-v1'),
        'warm_start_checkpoint': warm_artifact,
        'warm_controller_transfer': bool(
            warm_path is not None and warm_source.get(
                'controller_state_sha256') != controller_state_sha256),
        'controller_robust_feasibility': bool(reference_mode),
        'feasibility_target': feasibility_target_name,
        'geometry_epoch': 'bootstrap-one-row-per-geometry-v1',
        'epoch_selection': 'fixed-warm-block0-model-promotion-v1',
    }
    external_provenance = [
        {
            'return_cache': cache.cache_artifact,
            'candidate_cache': cache.candidate_artifact,
            'n_rows': len(cache.dataset),
            'n_rows_used': int(keep.numel()),
            'n_rows_excluded_model_or_calibration': int(
                len(cache.dataset) - keep.numel()),
            'physical_validation': cache.physical_stats,
        }
        for cache, keep in external_data
    ]
    reference_external_provenance = [
        {
            'paired_primary_index': index,
            'return_cache': cache.cache_artifact,
            'candidate_cache': cache.candidate_artifact,
            'n_rows': len(cache.dataset),
            'n_rows_used': int(keep.numel()),
            'n_rows_excluded_model_or_calibration': int(
                len(cache.dataset) - keep.numel()),
            'physical_validation': cache.physical_stats,
        }
        for index, (cache, keep) in enumerate(reference_external_data)
    ]
    reference_provenance = None
    if reference_mode:
        reference_provenance = {
            'format': 'controller-robust-feasibility-reference-v1',
            'source_checkpoint': reference_source_artifact,
            'source_return_cache': reference_return_artifact,
            'source_candidate_cache': reference_candidate_artifact,
            'external': reference_external_provenance,
            'controller': reference_controller_artifact,
            'controller_state_sha256': reference_controller_state_sha256,
            'objective': reference_objective,
            'gamma': reference_gamma,
        }
    provenance = {
        'format': 'offline-seed-ensemble-v1',
        'source_checkpoint': source_artifact,
        'source_return_cache': return_artifact,
        'source_candidate_cache': candidate_artifact,
        'external': external_provenance,
        'controller': controller_artifact,
        'controller_state_sha256': controller_state_sha256,
        'feasibility_reference': reference_provenance,
        'settings': settings,
    }
    training = {
        'format': 'offline-seed-ensemble-training-v1',
        'settings': settings,
        'deployed_feasibility_target': deployed_feasibility_target,
        'members': member_training,
        'n_fit_rows': int(fit_indices.numel()),
        'n_fit_geometry_groups': len(set(
            cached.task_fingerprints[int(i)] for i in fit_indices.tolist())),
        'n_model_select_rows': int(model_indices.numel()),
        'n_model_select_geometry_groups': len(set(
            cached.task_fingerprints[int(i)] for i in model_indices.tolist())),
        'n_final_calibration_rows': int(calibration_indices.numel()),
        'n_final_calibration_geometry_groups': len(set(
            cached.task_fingerprints[int(i)]
            for i in calibration_indices.tolist())),
        'n_external_rows_used': table.n_external_rows,
        'model_select_report_only': model_report,
        'attempt_model_select_report': attempt_model_report,
        'warm_model_select_report': warm_model_report,
        'warm_promoted': warm_promoted,
        'final_calibration': calibration,
        'source_physical_validation': source_physical_stats,
        'reference_source_physical_validation': (
            reference_source_physical_stats),
    }

    # Long training may only publish if every parsed byte source is unchanged.
    _assert_artifact_unchanged(
        'source checkpoint', source_artifact, file_fingerprint(source_path))
    _assert_artifact_unchanged(
        'source return cache', return_artifact, file_fingerprint(return_path))
    _assert_artifact_unchanged(
        'source candidate cache', candidate_artifact,
        file_fingerprint(candidate_path))
    if warm_path is not None:
        assert warm_artifact is not None
        _assert_artifact_unchanged(
            'warm-start checkpoint', warm_artifact,
            file_fingerprint(warm_path))
    current_controller = controller_fingerprint(controller_dir)
    _assert_artifact_unchanged(
        'controller agent', controller_artifact['agent'],
        current_controller['agent'])
    _assert_artifact_unchanged(
        'controller config', controller_artifact['config'],
        current_controller['config'])
    for cache, _ in external_data:
        _assert_artifact_unchanged(
            'external return cache', cache.cache_artifact,
            file_fingerprint(cache.cache_path))
        _assert_artifact_unchanged(
            'external candidate cache', cache.candidate_artifact,
            file_fingerprint(cache.candidate_path))
    if reference_mode:
        assert reference_source_path is not None
        assert reference_source_artifact is not None
        assert reference_return_path is not None
        assert reference_return_artifact is not None
        assert reference_candidate_path is not None
        assert reference_candidate_artifact is not None
        assert reference_controller_dir is not None
        assert reference_controller_artifact is not None
        _assert_artifact_unchanged(
            'reference source checkpoint', reference_source_artifact,
            file_fingerprint(reference_source_path))
        _assert_artifact_unchanged(
            'reference source return cache', reference_return_artifact,
            file_fingerprint(reference_return_path))
        _assert_artifact_unchanged(
            'reference source candidate cache', reference_candidate_artifact,
            file_fingerprint(reference_candidate_path))
        current_reference_controller = controller_fingerprint(
            reference_controller_dir)
        _assert_artifact_unchanged(
            'reference controller agent',
            reference_controller_artifact['agent'],
            current_reference_controller['agent'])
        _assert_artifact_unchanged(
            'reference controller config',
            reference_controller_artifact['config'],
            current_reference_controller['config'])
        for cache, _ in reference_external_data:
            _assert_artifact_unchanged(
                'reference external return cache', cache.cache_artifact,
                file_fingerprint(cache.cache_path))
            _assert_artifact_unchanged(
                'reference external candidate cache',
                cache.candidate_artifact,
                file_fingerprint(cache.candidate_path))
    if os.path.lexists(out_dir):
        raise FileExistsError(f'refusing to overwrite output: {out_dir}')
    _publish(
        out_dir, source=source, controller_dir=controller_dir,
        members=members, optimizers=optimizers, deployment=deployment,
        calibration=calibration, training=training, provenance=provenance,
        fit_indices=fit_indices, model_indices=model_indices,
        calibration_indices=calibration_indices,
        training_fingerprints=table.task_fingerprints, device=device)
    print(f'[offline-seed-ensemble] saved -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
