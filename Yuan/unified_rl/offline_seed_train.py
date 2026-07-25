"""Offline full-information training for the unified seed selector.

The expensive controller rollouts are read from ``seed-return-cache-v1``.
Only the training split of an immutable ``unified-bidirectional-v4`` source is
used: task-geometry groups are split again into fit and calibration subsets,
the source seed policy is fine-tuned listwise, and a conservative first-valid
abstention rule is selected on calibration geometries.

Example::

    python -m Yuan.unified_rl.offline_seed_train \
        --source-checkpoint Yuan/unified_rl/runs/r2_grouped_best/unified.pt \
        --return-cache /tmp/r2_train_seed_returns.npz \
        --out-dir Yuan/unified_rl/runs/r2_offline_selector \
        --device cuda

The output is a self-contained controller directory containing
``unified.pt``, ``agent.pt``, and ``config.yaml``.  Existing output paths are
never replaced.  The controller is copied byte-for-byte; only the seed policy
and its explicitly versioned deployment metadata change.
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

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    require_checkpoint_keys,
    resolve_controller_dir,
)
from Yuan.unified_rl.features import initial_observation_features
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
    infer_seed_policy_config,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


_SOURCE_REQUIRED_KEYS = (
    'format_version',
    'outer_round',
    'phase',
    'global_seed_update',
    'seed_policy',
    'controller',
    'controller_state_sha256',
    'seed_optimizer',
    'controller_config',
    'controller_run_config_sha256',
    'feature_dim',
    'seed_architecture',
    'seed_include_ray_error',
    'seed_include_log_manip',
    'seed_config',
    'train_indices',
    'validation_indices',
    'train_task_indices',
    'validation_task_indices',
    'train_valid_mask',
    'validation_valid_mask',
    'split_mode',
    'args',
    'provenance',
)

_CACHE_REQUIRED_KEYS = (
    'format',
    'format_version',
    'valid',
    'discounted_return',
    'undiscounted_return',
    'progress_m',
    'episode_len',
    'term_reason',
    'switch_count',
    'q0',
    'p0',
    'line_dir',
    'n_target',
    'task_indices',
    'task_fingerprints',
    'train_row_indices',
    'candidate_indices',
    'fallback_index',
    'n_tasks',
    'n_candidates',
    'n_valid_rollouts',
    'source_checkpoint_size',
    'source_checkpoint_sha256',
    'candidate_cache_size',
    'candidate_cache_sha256',
    'controller_agent_size',
    'controller_agent_sha256',
    'controller_config_size',
    'controller_config_sha256',
    'controller_state_sha256',
    'controller_gamma',
    'controller_kind',
    'source_phase',
    'source_outer_round',
    'seed_return_objective',
    'split_mode',
    'physical_validation',
)


@dataclasses.dataclass(frozen=True)
class OfflineReturnData:
    """Validated controller labels aligned with the source training split."""

    returns: torch.Tensor
    progress_m: torch.Tensor
    valid: torch.Tensor
    task_fingerprints: tuple[str, ...]
    artifact: dict[str, str | int]
    objective: str
    gamma: float


def _scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f'return-cache field {key!r} must be a scalar')
    return value.item()


def _integer_scalar(data: Any, key: str) -> int:
    value = np.asarray(data[key])
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise ValueError(
            f'return-cache field {key!r} must be an int64 scalar')
    return int(value.item())


def _string_scalar(data: Any, key: str) -> str:
    value = np.asarray(data[key])
    if value.shape != () or value.dtype.kind not in ('U', 'S'):
        raise ValueError(
            f'return-cache field {key!r} must be a string scalar')
    return str(value.item())


def _same_content(saved_size: int, saved_sha256: str,
                  artifact: dict[str, str | int], *, label: str) -> None:
    if (saved_size != artifact['size']
            or saved_sha256 != artifact['sha256']):
        raise ValueError(
            f'{label} content differs from return-cache provenance: '
            f'expected sha256={saved_sha256}, size={saved_size}; '
            f'got sha256={artifact["sha256"]}, size={artifact["size"]}')


def _assert_artifact_unchanged(label: str, before: dict,
                               after: dict) -> None:
    if (before.get('size') != after.get('size')
            or before.get('sha256') != after.get('sha256')):
        raise RuntimeError(
            f'{label} changed during offline training; refusing to publish')


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _validate_source_checkpoint(
    source: dict,
    source_artifact: dict,
    candidate_artifact: dict,
    controller_artifact: dict,
    controller_state_sha256: str,
    effective_controller_config: dict,
) -> tuple[str, float]:
    """Fail closed unless all source/controller/candidate identities agree."""
    require_checkpoint_keys(
        source, _SOURCE_REQUIRED_KEYS,
        kind='offline selector unified-bidirectional-v4 source')
    require_checkpoint_format_version(
        source, 4, kind='offline selector unified-bidirectional-v4 source')
    provenance = source['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('source provenance must be a dictionary')
    require_checkpoint_keys(
        provenance, ('format', 'candidate_cache', 'settings'),
        kind='source provenance')
    if provenance['format'] != 'unified-bidirectional-v4':
        raise ValueError(
            'offline selector requires unified-bidirectional-v4 provenance')
    if source['phase'] != 'round_complete':
        raise ValueError(
            'offline selector requires an immutable round_complete source')
    if source['split_mode'] != 'task-geometry-grouped-v1':
        raise ValueError(
            'offline selector requires task-geometry-grouped-v1 source split')
    settings = provenance['settings']
    if not isinstance(settings, dict):
        raise ValueError('source provenance settings must be a dictionary')
    if settings.get('split_mode') != source['split_mode']:
        raise ValueError('source split mode disagrees with provenance')

    saved_candidate = provenance['candidate_cache']
    if not isinstance(saved_candidate, dict):
        raise ValueError('source candidate provenance must be a dictionary')
    for key in ('size', 'sha256'):
        if key not in saved_candidate:
            raise ValueError(
                f'source candidate provenance is missing {key!r}')
    _same_content(
        int(saved_candidate['size']), str(saved_candidate['sha256']),
        candidate_artifact, label='candidate cache')

    embedded_hash = state_dict_fingerprint(source['controller'])
    if embedded_hash != source['controller_state_sha256']:
        raise ValueError(
            'source controller tensors disagree with their recorded hash')
    if controller_state_sha256 != embedded_hash:
        raise ValueError(
            'source agent.pt differs from its embedded controller tensors')
    if (controller_artifact['config']['sha256']
            != source['controller_run_config_sha256']):
        raise ValueError('source config.yaml differs from the unified source')
    if source['controller_config'] != effective_controller_config:
        raise ValueError(
            'source config.yaml PPO semantics differ from controller_config')

    args = source['args']
    if not isinstance(args, dict):
        raise ValueError('source args must be a dictionary')
    objective = args.get('seed_return')
    if objective not in ('discounted', 'undiscounted'):
        raise ValueError(f'invalid source seed objective: {objective!r}')
    if settings.get('seed_return') != objective:
        raise ValueError('source seed objective disagrees with provenance')
    controller_kind = source.get(
        'controller_kind', settings.get('controller_kind', 'pure'))
    if controller_kind != 'pure':
        raise ValueError('offline selector v1 supports only a pure controller')

    gamma = effective_controller_config.get('gamma')
    if (isinstance(gamma, bool) or not isinstance(gamma, (int, float))
            or not math.isfinite(float(gamma))
            or not 0.0 <= float(gamma) <= 1.0):
        raise ValueError('source controller gamma must be finite and in [0,1]')
    if not isinstance(source_artifact.get('sha256'), str):
        raise ValueError('source checkpoint fingerprint is invalid')
    return objective, float(gamma)


def _validate_result_array(
    data: Any, key: str, valid: np.ndarray, dtype: np.dtype,
    *, floating: bool,
) -> np.ndarray:
    values = np.asarray(data[key])
    if values.shape != valid.shape or values.dtype != dtype:
        raise ValueError(
            f'return-cache {key!r} must have shape {valid.shape} and '
            f'dtype {dtype}, got shape={values.shape}, dtype={values.dtype}')
    if floating:
        if not np.isfinite(values[valid]).all():
            raise ValueError(f'valid return-cache {key!r} values must be finite')
        if not np.isnan(values[~valid]).all():
            raise ValueError(f'invalid return-cache {key!r} slots must be NaN')
    else:
        if np.any(values[valid] < 0):
            raise ValueError(
                f'valid return-cache {key!r} values must be non-negative')
        if not np.equal(values[~valid], -1).all():
            raise ValueError(
                f'invalid return-cache {key!r} slots must equal -1')
    return values.copy()


def load_return_cache(
    path: str | Path,
    *,
    source: dict,
    source_artifact: dict,
    candidate_artifact: dict,
    controller_artifact: dict,
    controller_state_sha256: str,
    objective: str,
    gamma: float,
    train_dataset: CachedSeedCandidateDataset,
) -> OfflineReturnData:
    """Load and exhaustively validate ``seed-return-cache-v1``."""
    artifact = file_fingerprint(path)
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [key for key in _CACHE_REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f'return cache is missing required fields: {missing}')
        if _string_scalar(data, 'format') != 'seed-return-cache-v1':
            raise ValueError('return cache must use seed-return-cache-v1')
        if _integer_scalar(data, 'format_version') != 1:
            raise ValueError('return cache format_version must be 1')
        if not bool(_scalar(data, 'physical_validation')):
            raise ValueError('return cache must record physical validation')

        n_tasks = _integer_scalar(data, 'n_tasks')
        n_candidates = _integer_scalar(data, 'n_candidates')
        expected_shape = (
            train_dataset.batch.n_tasks,
            train_dataset.batch.n_candidates,
        )
        if (n_tasks, n_candidates) != expected_shape:
            raise ValueError(
                'return-cache dimensions disagree with source training split: '
                f'cache={(n_tasks, n_candidates)}, source={expected_shape}')

        valid = np.asarray(data['valid'])
        if valid.shape != expected_shape or valid.dtype != np.bool_:
            raise ValueError(
                f'return-cache valid must be bool with shape {expected_shape}')
        expected_valid = train_dataset.batch.valid.numpy()
        if not np.array_equal(valid, expected_valid):
            raise ValueError(
                'return-cache valid mask differs from the source train mask')
        if _integer_scalar(data, 'n_valid_rollouts') != int(valid.sum()):
            raise ValueError('return-cache n_valid_rollouts is inconsistent')

        float_results = {
            key: _validate_result_array(
                data, key, valid, np.dtype(np.float32), floating=True)
            for key in (
                'discounted_return', 'undiscounted_return', 'progress_m')
        }
        _validate_result_array(
            data, 'episode_len', valid, np.dtype(np.int64), floating=False)
        _validate_result_array(
            data, 'term_reason', valid, np.dtype(np.int32), floating=False)
        _validate_result_array(
            data, 'switch_count', valid, np.dtype(np.int64), floating=False)

        expected_arrays = {
            'q0': train_dataset.batch.q0.numpy().astype(np.float32, copy=False),
            'p0': train_dataset.batch.p0.numpy().astype(np.float32, copy=False),
            'line_dir': train_dataset.batch.line_dir.numpy().astype(
                np.float32, copy=False),
            'n_target': train_dataset.batch.n_target.numpy().astype(
                np.float32, copy=False),
            'task_indices': train_dataset.task_indices.numpy(),
            'train_row_indices': torch.as_tensor(
                source['train_indices'], dtype=torch.long).cpu().numpy(),
            'candidate_indices': np.arange(n_candidates, dtype=np.int64),
            'task_fingerprints': np.asarray(
                train_dataset.task_fingerprints, dtype='<U64'),
        }
        expected_dtypes = {
            'q0': np.dtype(np.float32),
            'p0': np.dtype(np.float32),
            'line_dir': np.dtype(np.float32),
            'n_target': np.dtype(np.float32),
            'task_indices': np.dtype(np.int64),
            'train_row_indices': np.dtype(np.int64),
            'candidate_indices': np.dtype(np.int64),
            'task_fingerprints': np.dtype('<U64'),
        }
        for key, expected in expected_arrays.items():
            cached = np.asarray(data[key])
            if cached.dtype != expected_dtypes[key] or not _arrays_equal(
                    cached, expected):
                raise ValueError(
                    f'return-cache {key!r} differs from the source train split')

        expected_fallback = (
            -1 if train_dataset.fallback_index is None
            else train_dataset.fallback_index)
        if _integer_scalar(data, 'fallback_index') != expected_fallback:
            raise ValueError('return-cache fallback_index is inconsistent')
        if _string_scalar(data, 'seed_return_objective') != objective:
            raise ValueError('return-cache seed objective differs from source')
        if _string_scalar(data, 'split_mode') != source['split_mode']:
            raise ValueError('return-cache split mode differs from source')
        if _string_scalar(data, 'controller_kind') != 'pure':
            raise ValueError('return-cache controller kind must be pure')
        if _string_scalar(data, 'source_phase') != source['phase']:
            raise ValueError('return-cache source phase differs from source')
        if _integer_scalar(data, 'source_outer_round') != int(
                source['outer_round']):
            raise ValueError('return-cache outer round differs from source')
        cached_gamma = np.asarray(data['controller_gamma'])
        if cached_gamma.shape != () or cached_gamma.dtype != np.float64:
            raise ValueError('return-cache controller_gamma must be float64')
        if float(cached_gamma.item()) != gamma:
            raise ValueError('return-cache controller gamma differs from source')

        _same_content(
            _integer_scalar(data, 'source_checkpoint_size'),
            _string_scalar(data, 'source_checkpoint_sha256'),
            source_artifact, label='source checkpoint')
        _same_content(
            _integer_scalar(data, 'candidate_cache_size'),
            _string_scalar(data, 'candidate_cache_sha256'),
            candidate_artifact, label='candidate cache')
        _same_content(
            _integer_scalar(data, 'controller_agent_size'),
            _string_scalar(data, 'controller_agent_sha256'),
            controller_artifact['agent'], label='controller agent')
        _same_content(
            _integer_scalar(data, 'controller_config_size'),
            _string_scalar(data, 'controller_config_sha256'),
            controller_artifact['config'], label='controller config')
        if (_string_scalar(data, 'controller_state_sha256')
                != controller_state_sha256):
            raise ValueError(
                'return-cache controller state hash differs from source')

        returns = float_results[f'{objective}_return']
        fingerprints = tuple(str(value) for value in expected_arrays[
            'task_fingerprints'].tolist())
    return OfflineReturnData(
        returns=torch.from_numpy(returns),
        progress_m=torch.from_numpy(float_results['progress_m']),
        valid=torch.from_numpy(valid.copy()),
        task_fingerprints=fingerprints,
        artifact=artifact,
        objective=objective,
        gamma=gamma,
    )


def geometry_groups(fingerprints: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Return deterministic row groups for identical task geometries."""
    if len(fingerprints) < 1:
        raise ValueError('at least one task fingerprint is required')
    grouped: dict[str, list[int]] = {}
    for row, fingerprint in enumerate(fingerprints):
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in '0123456789abcdef' for char in fingerprint)):
            raise ValueError('task fingerprints must be lowercase SHA-256 strings')
        grouped.setdefault(fingerprint, []).append(row)
    return tuple(tuple(rows) for rows in grouped.values())


def geometry_balanced_epoch_indices(
    groups: Sequence[Sequence[int]], generator: torch.Generator,
) -> torch.Tensor:
    """Sample one row per geometry group, then shuffle the groups."""
    if len(groups) < 1:
        raise ValueError('at least one geometry group is required')
    rows = []
    for group in groups:
        if len(group) < 1:
            raise ValueError('geometry groups must be non-empty')
        offset = int(torch.randint(
            len(group), (1,), generator=generator).item())
        rows.append(int(group[offset]))
    row_tensor = torch.tensor(rows, dtype=torch.long)
    return row_tensor[torch.randperm(len(rows), generator=generator)]


def _rank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    min_delta: float,
) -> torch.Tensor:
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    target_delta = safe_targets.unsqueeze(2) - safe_targets.unsqueeze(1)
    score_delta = safe_scores.unsqueeze(2) - safe_scores.unsqueeze(1)
    better = (valid.unsqueeze(2) & valid.unsqueeze(1)
              & (target_delta > min_delta))
    pairs_per_task = better.sum(dim=(1, 2))
    if not bool((pairs_per_task > 0).any().item()):
        return scores.sum() * 0.0
    raw_weight = target_delta.clamp_min(0.0) * better
    mean_weight = (
        raw_weight.sum(dim=(1, 2)) / pairs_per_task.clamp_min(1))
    weights = raw_weight / mean_weight.clamp_min(1e-8)[:, None, None]
    per_pair = F.softplus(-score_delta) * weights.detach() * better
    per_task = per_pair.sum(dim=(1, 2)) / pairs_per_task.clamp_min(1)
    return per_task[pairs_per_task > 0].mean()


def _build_features(
    kin,
    dataset: CachedSeedCandidateDataset,
    *,
    include_log_manip: bool,
    include_ray_error: bool,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        batch = dataset.batch.index_select(
            torch.arange(start, end)).to(kin.device, dtype=kin.dtype)
        chunk = initial_observation_features(
            kin, batch,
            include_log_manip=include_log_manip,
            include_ray_error=include_ray_error)
        chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0)


def _proposal_predictions(
    policy: CandidateSeedActorCritic,
    features: torch.Tensor,
    valid: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actor_parts = []
    feasibility_parts = []
    actor_margin_parts = []
    feasibility_margin_parts = []
    first_parts = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            local = indices[start:start + batch_size]
            batch_features = features[local].to(device)
            batch_valid = valid[local].to(device)
            dist, _, scores = policy.distribution_and_values(
                batch_features, batch_valid)
            actor = dist.logits.argmax(dim=-1)
            feasibility = scores.masked_fill(
                ~batch_valid, -torch.inf).argmax(dim=-1)
            first = batch_valid.float().argmax(dim=-1)
            row = torch.arange(local.numel(), device=device)
            first_score = scores[row, first]
            actor_parts.append(actor.cpu())
            feasibility_parts.append(feasibility.cpu())
            actor_margin_parts.append((scores[row, actor] - first_score).cpu())
            feasibility_margin_parts.append(
                (scores[row, feasibility] - first_score).cpu())
            first_parts.append(first.cpu())
    return tuple(
        torch.cat(parts).numpy()
        for parts in (
            actor_parts, feasibility_parts, actor_margin_parts,
            feasibility_margin_parts, first_parts)
    )


def _geometry_ids(fingerprints: Sequence[str]) -> tuple[np.ndarray, int]:
    lookup: dict[str, int] = {}
    ids = np.empty(len(fingerprints), dtype=np.int64)
    for row, fingerprint in enumerate(fingerprints):
        ids[row] = lookup.setdefault(fingerprint, len(lookup))
    return ids, len(lookup)


def _calibrate_one_head(
    proposal_head: str,
    proposal: np.ndarray,
    margin: np.ndarray,
    first: np.ndarray,
    returns: np.ndarray,
    valid: np.ndarray,
    fingerprints: Sequence[str],
    confidence_z: float,
) -> dict[str, float | int | str]:
    n = len(proposal)
    if not (proposal.shape == margin.shape == first.shape == (n,)):
        raise ValueError('calibration proposal arrays have inconsistent shapes')
    if not np.isfinite(margin).all():
        raise ValueError('calibration feasibility margins must be finite')
    row = np.arange(n)
    if (not valid[row, proposal].all()
            or not valid[row, first].all()):
        raise ValueError('calibration selected an invalid candidate')
    proposal_gain = returns[row, proposal] - returns[row, first]
    if not np.isfinite(proposal_gain).all():
        raise ValueError('calibration gains must be finite')
    group_ids, n_groups = _geometry_ids(fingerprints)
    group_count = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    first_values = returns[row, first]
    oracle_values = np.where(valid, returns, -np.inf).max(axis=1)
    oracle_headroom = oracle_values - first_values
    group_first = (
        np.bincount(group_ids, weights=first_values, minlength=n_groups)
        / group_count)
    group_oracle_headroom = (
        np.bincount(
            group_ids, weights=oracle_headroom, minlength=n_groups)
        / group_count)
    first_mean = float(group_first.mean())
    oracle_headroom_mean = float(group_oracle_headroom.mean())

    maximum = float(margin.max())
    reject_all = float(np.nextafter(maximum, math.inf))
    if not math.isfinite(reject_all) or reject_all <= maximum:
        raise ValueError('could not construct a finite reject-all threshold')
    # A negative threshold would authorize a switch that the feasibility head
    # itself predicts to be worse than first-valid.  Besides being an unsafe
    # interpretation of abstention, deployment schema v1 deliberately rejects
    # it.  Zero remains a candidate even when no observed margin is exactly 0.
    nonnegative_margin = margin[margin >= 0.0].astype(
        np.float64, copy=False)
    thresholds = np.concatenate([
        np.unique(nonnegative_margin),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([reject_all], dtype=np.float64),
    ])
    candidates = []
    for threshold in thresholds.tolist():
        selected = margin >= threshold
        deployed_gain = np.where(selected, proposal_gain, 0.0)
        group_gain = (
            np.bincount(
                group_ids, weights=deployed_gain, minlength=n_groups)
            / group_count)
        mean_gain = float(group_gain.mean())
        standard_error = float(
            group_gain.std(ddof=1) / math.sqrt(n_groups)
            if n_groups > 1 else 0.0)
        lower_bound = mean_gain - confidence_z * standard_error
        group_selection = (
            np.bincount(
                group_ids, weights=selected.astype(np.float64),
                minlength=n_groups)
            / group_count)
        worse = selected & (proposal_gain < 0.0)
        group_worse = (
            np.bincount(
                group_ids, weights=worse.astype(np.float64),
                minlength=n_groups)
            / group_count)
        candidates.append({
            'proposal_head': proposal_head,
            'threshold': float(threshold),
            'mean_gain': mean_gain,
            'mean_return': first_mean + mean_gain,
            'first_valid_mean_return': first_mean,
            'oracle_headroom': oracle_headroom_mean,
            'oracle_capture': (
                mean_gain / oracle_headroom_mean
                if oracle_headroom_mean > 0.0 else 0.0),
            'lower_bound': float(lower_bound),
            'standard_error': standard_error,
            'selection_rate': float(group_selection.mean()),
            'worse_rate': float(group_worse.mean()),
        })

    # Reject-all has exactly zero gain and is always eligible.  The LCB gate
    # therefore prevents publishing a switch rule whose calibrated gain is
    # statistically indistinguishable from harm.  Within the eligible set we
    # follow the requested mean-return objective; conservative threshold and
    # lower worse-rate resolve numerical ties.
    eligible = [candidate for candidate in candidates
                if candidate['lower_bound'] >= 0.0]
    return max(
        eligible,
        key=lambda item: (
            item['mean_gain'], item['lower_bound'], -item['worse_rate'],
            item['threshold']),
    )


def calibrate_deployment(
    policy: CandidateSeedActorCritic,
    features: torch.Tensor,
    valid: torch.Tensor,
    returns: torch.Tensor,
    calibration_indices: torch.Tensor,
    task_fingerprints: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    confidence_z: float,
) -> tuple[dict[str, str | float], dict[str, Any]]:
    """Choose proposal head and conservative first-valid threshold."""
    if calibration_indices.numel() < 1:
        raise ValueError('calibration split must be non-empty')
    selected_fingerprints = [task_fingerprints[int(index)]
                             for index in calibration_indices.tolist()]
    actor, feasibility, actor_margin, feasibility_margin, first = (
        _proposal_predictions(
            policy, features, valid, calibration_indices,
            device=device, batch_size=batch_size))
    return_np = returns[calibration_indices].numpy()
    valid_np = valid[calibration_indices].numpy()
    alternatives = {
        'actor': _calibrate_one_head(
            'actor', actor, actor_margin, first, return_np, valid_np,
            selected_fingerprints, confidence_z),
        'feasibility': _calibrate_one_head(
            'feasibility', feasibility, feasibility_margin, first,
            return_np, valid_np, selected_fingerprints, confidence_z),
    }
    chosen = max(
        alternatives.values(),
        key=lambda item: (
            item['mean_gain'], item['lower_bound'], -item['worse_rate'],
            item['threshold']),
    )
    deployment: dict[str, str | float] = {
        'mode': 'conservative',
        'proposal_head': str(chosen['proposal_head']),
        'threshold': float(chosen['threshold']),
        'comparison': 'ge',
    }
    metrics = {
        'format': 'seed-deployment-calibration-v1',
        'gate_score': 'feasibility_proposal_minus_first_valid',
        'fallback': 'first_valid',
        'feasibility_target': 'first-valid-advantage-v1',
        'confidence_z': float(confidence_z),
        'n_rows': int(calibration_indices.numel()),
        'n_geometry_groups': len(set(selected_fingerprints)),
        'selected': copy.deepcopy(chosen),
        'alternatives': alternatives,
    }
    return deployment, metrics


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _write_new_run(
    out_dir: Path,
    *,
    source: dict,
    source_controller_dir: Path,
    output_config: dict,
    seed_policy: CandidateSeedActorCritic,
    optimizer: torch.optim.Optimizer,
    deployment: dict,
    calibration: dict,
    training: dict,
    offline_provenance: dict,
    fit_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    sampler_generator: torch.Generator,
    device: torch.device,
) -> None:
    """Publish a new self-contained run without replacing any path."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(mode=0o755)
    except FileExistsError as error:
        raise FileExistsError(
            f'refusing to overwrite existing output directory: {out_dir}'
        ) from error

    config_path = out_dir / 'config.yaml'
    agent_path = out_dir / 'agent.pt'
    checkpoint_path = out_dir / 'unified.pt'
    with open(config_path, 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    with open(source_controller_dir / 'agent.pt', 'rb') as source_stream:
        with open(agent_path, 'xb') as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())

    state = copy.deepcopy(source)
    combined_provenance = copy.deepcopy(source['provenance'])
    combined_provenance['offline_seed'] = copy.deepcopy(offline_provenance)
    state.update({
        'phase': 'offline_seed_complete',
        'seed_policy': _cpu_tree(seed_policy.state_dict()),
        'seed_optimizer': _cpu_tree(optimizer.state_dict()),
        'controller_run_config_sha256': file_fingerprint(
            config_path)['sha256'],
        'seed_deployment': copy.deepcopy(deployment),
        'seed_deployment_calibration': copy.deepcopy(calibration),
        'seed_feasibility_target': 'first-valid-advantage-v1',
        'offline_seed_training': copy.deepcopy(training),
        'offline_seed_provenance': copy.deepcopy(offline_provenance),
        'provenance': combined_provenance,
        'offline_fit_local_indices': fit_indices.cpu(),
        'offline_calibration_local_indices': calibration_indices.cpu(),
        'offline_fit_task_indices': torch.as_tensor(
            source['train_task_indices']).cpu()[fit_indices],
        'offline_calibration_task_indices': torch.as_tensor(
            source['train_task_indices']).cpu()[calibration_indices],
        'offline_seed_optimizer': _cpu_tree(optimizer.state_dict()),
        'offline_sampler_generator_state': sampler_generator.get_state(),
    })
    state.update(global_rng_state(device))
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
            'Train a geometry-balanced seed selector from a strict exhaustive '
            'return cache and calibrate first-valid abstention.'))
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--return-cache', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--candidates', default=None,
        help='candidate cache; defaults to the immutable path in source provenance')
    parser.add_argument(
        '--controller-ckpt', default=None,
        help='controller directory; defaults to the source checkpoint directory')
    parser.add_argument('--device', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--feature-chunk-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--calibration-fraction', type=float, default=0.15)
    parser.add_argument('--calibration-z', type=float, default=1.96)
    parser.add_argument('--target-temperature', type=float, default=0.01)
    parser.add_argument('--oracle-target-mix', type=float, default=0.5)
    parser.add_argument('--regret-weight', type=float, default=1.0)
    parser.add_argument('--feasibility-coef', type=float, default=1.0)
    parser.add_argument('--rank-coef', type=float, default=0.25)
    parser.add_argument('--value-coef', type=float, default=0.1)
    parser.add_argument('--feasibility-beta', type=float, default=0.02)
    parser.add_argument('--rank-delta', type=float, default=1e-6)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=23000)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument(
        '--calibrate-every', type=int, default=5,
        help='calibration/model-selection cadence in epochs')
    args = parser.parse_args()

    positive_ints = {
        '--epochs': args.epochs,
        '--batch-size': args.batch_size,
        '--feature-chunk-size': args.feature_chunk_size,
        '--log-every': args.log_every,
        '--calibrate-every': args.calibrate_every,
    }
    for name, value in positive_ints.items():
        if value < 1:
            raise ValueError(f'{name} must be positive')
    positive_floats = {
        '--learning-rate': args.learning_rate,
        '--calibration-z': args.calibration_z,
        '--target-temperature': args.target_temperature,
        '--feasibility-beta': args.feasibility_beta,
        '--max-grad-norm': args.max_grad_norm,
    }
    for name, value in positive_floats.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
    nonnegative_floats = {
        '--regret-weight': args.regret_weight,
        '--feasibility-coef': args.feasibility_coef,
        '--rank-coef': args.rank_coef,
        '--value-coef': args.value_coef,
        '--rank-delta': args.rank_delta,
    }
    for name, value in nonnegative_floats.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')
    if (not math.isfinite(args.calibration_fraction)
            or not 0.0 < args.calibration_fraction < 1.0):
        raise ValueError('--calibration-fraction must be in (0,1)')
    if (not math.isfinite(args.oracle_target_mix)
            or not 0.0 <= args.oracle_target_mix <= 1.0):
        raise ValueError('--oracle-target-mix must be in [0,1]')

    source_path = Path(args.source_checkpoint).expanduser().resolve(strict=True)
    cache_path = Path(args.return_cache).expanduser().resolve(strict=True)
    if not source_path.is_file() or source_path.name != 'unified.pt':
        raise ValueError('--source-checkpoint must name an existing unified.pt')
    if not cache_path.is_file() or cache_path.suffix.lower() != '.npz':
        raise ValueError('--return-cache must name an existing .npz file')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if not isinstance(source, dict):
        raise ValueError('source checkpoint must contain a dictionary')

    provenance = source.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('source checkpoint has no valid provenance')
    saved_candidate = provenance.get('candidate_cache')
    if not isinstance(saved_candidate, dict) or 'path' not in saved_candidate:
        raise ValueError('source provenance has no candidate cache path')
    candidate_path = Path(
        args.candidates if args.candidates is not None
        else saved_candidate['path']).expanduser().resolve(strict=True)
    controller_dir = resolve_controller_dir(
        args.controller_ckpt if args.controller_ckpt is not None
        else source_path.parent)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if out_dir.exists() or os.path.lexists(out_dir):
        raise FileExistsError(
            f'refusing to overwrite existing output directory: {out_dir}')

    source_artifact = file_fingerprint(source_path)
    cache_artifact = file_fingerprint(cache_path)
    candidate_artifact = file_fingerprint(candidate_path)
    controller_artifact = controller_fingerprint(controller_dir)
    agent_state = torch.load(
        controller_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if not isinstance(agent_state, dict):
        raise ValueError('controller agent.pt must contain a state dictionary')
    controller_state_sha256 = state_dict_fingerprint(agent_state)
    effective_controller_config = dataclasses.asdict(
        ppo_config_from_run(load_run_config(controller_dir)))
    objective, gamma = _validate_source_checkpoint(
        source, source_artifact, candidate_artifact, controller_artifact,
        controller_state_sha256, effective_controller_config)

    seed_global_rng(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    env = build_env_from_run(controller_dir, 1, device)
    source_dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    source_dataset, validity_stats = validate_cached_dataset(
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
        raise ValueError('source train and validation geometries overlap')

    cached = load_return_cache(
        cache_path,
        source=source,
        source_artifact=source_artifact,
        candidate_artifact=candidate_artifact,
        controller_artifact=controller_artifact,
        controller_state_sha256=controller_state_sha256,
        objective=objective,
        gamma=gamma,
        train_dataset=train_dataset,
    )
    if cached.artifact != cache_artifact:
        raise RuntimeError('return cache changed while it was being loaded')

    _, _, fit_indices, calibration_indices = (
        train_dataset.train_validation_split(
            args.calibration_fraction, args.seed + 1))
    fit_fingerprints = [cached.task_fingerprints[int(index)]
                        for index in fit_indices.tolist()]
    calibration_fingerprints = [cached.task_fingerprints[int(index)]
                                for index in calibration_indices.tolist()]
    if set(fit_fingerprints) & set(calibration_fingerprints):
        raise RuntimeError('offline fit/calibration geometry split overlaps')
    fit_groups = geometry_groups(fit_fingerprints)
    # Convert group-local rows to rows in the complete source training split.
    fit_groups_global = tuple(
        tuple(int(fit_indices[row]) for row in group)
        for group in fit_groups)

    include_ray_error = bool(source['seed_include_ray_error'])
    include_log_manip = bool(source['seed_include_log_manip'])
    print(
        f'[offline-seed] building features for {len(train_dataset)} rows; '
        f'fit geometries={len(fit_groups_global)}, '
        f'calibration geometries={len(set(calibration_fingerprints))}',
        flush=True)
    features = _build_features(
        env.kin, train_dataset,
        include_log_manip=include_log_manip,
        include_ray_error=include_ray_error,
        chunk_size=args.feature_chunk_size)
    policy_config = infer_seed_policy_config(source)
    if features.shape[-1] != policy_config.feature_dim:
        raise ValueError(
            'generated feature dimension disagrees with source seed policy')
    policy = CandidateSeedActorCritic(**policy_config.to_dict()).to(device)
    policy.load_state_dict(source['seed_policy'])
    policy.train()

    return_scale = source['seed_config'].get('return_scale')
    if (isinstance(return_scale, bool)
            or not isinstance(return_scale, (int, float))
            or not math.isfinite(float(return_scale))
            or float(return_scale) <= 0.0):
        raise ValueError('source seed return_scale must be finite and positive')
    return_scale = float(return_scale)
    raw_returns = cached.returns
    valid = cached.valid
    first_index = valid.float().argmax(dim=-1)
    row = torch.arange(valid.shape[0])
    first_return = raw_returns[row, first_index]
    advantages = (raw_returns - first_return.unsqueeze(-1)) / return_scale
    advantages = torch.where(valid, advantages, torch.zeros_like(advantages))
    scaled_returns = torch.where(
        valid, raw_returns / return_scale, torch.zeros_like(raw_returns))

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    sampler = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    optimizer_steps = 0
    selected_optimizer_steps = 0
    final_epoch_stats: dict[str, float] = {}
    best_epoch = 0
    best_deployment, best_calibration = calibrate_deployment(
        policy, features, valid, raw_returns, calibration_indices,
        cached.task_fingerprints,
        device=device, batch_size=args.batch_size,
        confidence_z=args.calibration_z)
    best_policy_state = copy.deepcopy(_cpu_tree(policy.state_dict()))
    best_optimizer_state = copy.deepcopy(_cpu_tree(optimizer.state_dict()))
    best_sampler_state = sampler.get_state().clone()
    best_epoch_stats: dict[str, float] = {}

    def calibration_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
        selected = metrics['selected']
        return (
            float(selected['mean_gain']),
            float(selected['lower_bound']),
            -float(selected['worse_rate']),
        )

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_indices = geometry_balanced_epoch_indices(
            fit_groups_global, sampler)
        sums = {
            'loss': 0.0,
            'actor': 0.0,
            'feasibility': 0.0,
            'rank': 0.0,
            'value': 0.0,
        }
        n_seen = 0
        for start in range(0, epoch_indices.numel(), args.batch_size):
            index = epoch_indices[start:start + args.batch_size]
            batch_features = features[index].to(device)
            batch_valid = valid[index].to(device)
            batch_advantage = advantages[index].to(device)
            batch_returns = scaled_returns[index].to(device)
            dist, state_value, feasibility = policy.distribution_and_values(
                batch_features, batch_valid)

            target_logits = (batch_advantage / args.target_temperature).masked_fill(
                ~batch_valid, -torch.inf)
            soft_target = torch.softmax(target_logits, dim=-1)
            oracle = batch_advantage.masked_fill(
                ~batch_valid, -torch.inf).argmax(dim=-1)
            oracle_target = F.one_hot(
                oracle, num_classes=batch_valid.shape[1]).to(
                    batch_advantage.dtype)
            target = ((1.0 - args.oracle_target_mix) * soft_target
                      + args.oracle_target_mix * oracle_target)
            actor_per_task = -(target.detach() * dist.logits).sum(dim=-1)
            oracle_advantage = batch_advantage.gather(
                1, oracle.unsqueeze(1)).squeeze(1).clamp_min(0.0)
            positive_mean = oracle_advantage[oracle_advantage > 0].mean()
            if not bool(torch.isfinite(positive_mean).item()):
                positive_mean = torch.ones_like(positive_mean)
            relative_regret = (
                oracle_advantage / positive_mean.clamp_min(1e-8)).clamp_max(10.0)
            task_weight = 1.0 + args.regret_weight * relative_regret
            actor_loss = (
                (actor_per_task * task_weight.detach()).sum()
                / task_weight.sum().clamp_min(1e-8))

            batch_row = torch.arange(index.numel(), device=device)
            batch_first = batch_valid.float().argmax(dim=-1)
            predicted_advantage = (
                feasibility
                - feasibility[batch_row, batch_first].unsqueeze(-1))
            q_error = F.smooth_l1_loss(
                predicted_advantage, batch_advantage,
                reduction='none', beta=args.feasibility_beta)
            q_error = torch.where(
                batch_valid, q_error, torch.zeros_like(q_error))
            feasibility_loss = (
                q_error.sum(dim=-1)
                / batch_valid.sum(dim=-1).clamp_min(1)).mean()
            rank_loss = _rank_loss(
                feasibility, batch_advantage, batch_valid, args.rank_delta)
            value_target = batch_returns.masked_fill(
                ~batch_valid, -torch.inf).max(dim=-1).values
            value_loss = F.smooth_l1_loss(state_value, value_target)
            loss = (
                actor_loss
                + args.feasibility_coef * feasibility_loss
                + args.rank_coef * rank_loss
                + args.value_coef * value_loss)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError('offline seed loss became non-finite')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), args.max_grad_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError(
                    'offline seed gradient norm became non-finite')
            optimizer.step()
            optimizer_steps += 1

            count = int(index.numel())
            n_seen += count
            for key, value in (
                ('loss', loss),
                ('actor', actor_loss),
                ('feasibility', feasibility_loss),
                ('rank', rank_loss),
                ('value', value_loss)):
                sums[key] += float(value.item()) * count
        final_epoch_stats = {
            key: value / max(n_seen, 1) for key, value in sums.items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f'[offline-seed] epoch {epoch:>3}/{args.epochs}  '
                f'loss {final_epoch_stats["loss"]:.4f}  '
                f'actor {final_epoch_stats["actor"]:.4f}  '
                f'q {final_epoch_stats["feasibility"]:.4f}  '
                f'rank {final_epoch_stats["rank"]:.4f}',
                flush=True)
        if (epoch == 1 or epoch % args.calibrate_every == 0
                or epoch == args.epochs):
            candidate_deployment, candidate_calibration = calibrate_deployment(
                policy, features, valid, raw_returns, calibration_indices,
                cached.task_fingerprints,
                device=device, batch_size=args.batch_size,
                confidence_z=args.calibration_z)
            if (calibration_key(candidate_calibration)
                    > calibration_key(best_calibration)):
                best_epoch = epoch
                best_deployment = candidate_deployment
                best_calibration = candidate_calibration
                best_policy_state = copy.deepcopy(
                    _cpu_tree(policy.state_dict()))
                best_optimizer_state = copy.deepcopy(
                    _cpu_tree(optimizer.state_dict()))
                best_sampler_state = sampler.get_state().clone()
                best_epoch_stats = copy.deepcopy(final_epoch_stats)
                selected_optimizer_steps = optimizer_steps

    policy.load_state_dict(best_policy_state)
    optimizer.load_state_dict(best_optimizer_state)
    sampler.set_state(best_sampler_state)
    deployment = best_deployment
    calibration = best_calibration
    chosen = calibration['selected']
    print(
        f'[offline-seed] deployment proposal={deployment["proposal_head"]}  '
        f'threshold={deployment["threshold"]:.6g}  '
        f'cal gain={chosen["mean_gain"]:+.4f}  '
        f'LCB={chosen["lower_bound"]:+.4f}  '
        f'select={100 * chosen["selection_rate"]:.1f}%',
        flush=True)

    settings = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'calibration_fraction': args.calibration_fraction,
        'calibration_z': args.calibration_z,
        'target_temperature': args.target_temperature,
        'oracle_target_mix': args.oracle_target_mix,
        'regret_weight': args.regret_weight,
        'feasibility_coef': args.feasibility_coef,
        'rank_coef': args.rank_coef,
        'value_coef': args.value_coef,
        'feasibility_beta': args.feasibility_beta,
        'rank_delta': args.rank_delta,
        'max_grad_norm': args.max_grad_norm,
        'return_scale': return_scale,
        'seed': args.seed,
        'calibrate_every': args.calibrate_every,
        'device': device_identity(device),
        'geometry_epoch': 'one-row-per-geometry-v1',
        'fit_calibration_split': 'task-geometry-grouped-v1',
        'feasibility_target': 'first-valid-advantage-v1',
        'actor_target': 'regret-weighted-listwise-v1',
    }
    offline_provenance = {
        'format': 'offline-seed-selector-v1',
        'source_checkpoint': source_artifact,
        'return_cache': cache_artifact,
        'candidate_cache': candidate_artifact,
        'controller': controller_artifact,
        'controller_state_sha256': controller_state_sha256,
        'settings': settings,
    }
    training = {
        'format': 'offline-seed-training-v1',
        'settings': settings,
        'optimizer_steps': optimizer_steps,
        'selected_optimizer_steps': selected_optimizer_steps,
        'selected_epoch': best_epoch,
        'n_fit_rows': int(fit_indices.numel()),
        'n_fit_geometry_groups': len(fit_groups_global),
        'n_calibration_rows': int(calibration_indices.numel()),
        'n_calibration_geometry_groups': len(set(calibration_fingerprints)),
        'objective': objective,
        'final_epoch': final_epoch_stats,
        'selected_epoch_stats': best_epoch_stats,
        'physical_validation': validity_stats,
    }
    output_config = copy.deepcopy(load_run_config(controller_dir))
    output_config.setdefault('unified', {})['seed_deployment'] = copy.deepcopy(
        deployment)
    output_config['unified']['offline_seed_selector'] = {
        'format': 'offline-seed-selector-v1',
        'objective': objective,
        'feasibility_target': 'first-valid-advantage-v1',
        'source_checkpoint_sha256': source_artifact['sha256'],
        'return_cache_sha256': cache_artifact['sha256'],
    }

    # Long training must never publish against inputs that changed midway.
    _assert_artifact_unchanged(
        'source checkpoint', source_artifact, file_fingerprint(source_path))
    _assert_artifact_unchanged(
        'return cache', cache_artifact, file_fingerprint(cache_path))
    _assert_artifact_unchanged(
        'candidate cache', candidate_artifact,
        file_fingerprint(candidate_path))
    current_controller = controller_fingerprint(controller_dir)
    _assert_artifact_unchanged(
        'controller agent', controller_artifact['agent'],
        current_controller['agent'])
    _assert_artifact_unchanged(
        'controller config', controller_artifact['config'],
        current_controller['config'])

    _write_new_run(
        out_dir,
        source=source,
        source_controller_dir=controller_dir,
        output_config=output_config,
        seed_policy=policy,
        optimizer=optimizer,
        deployment=deployment,
        calibration=calibration,
        training=training,
        offline_provenance=offline_provenance,
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        sampler_generator=sampler,
        device=device,
    )
    print(f'[offline-seed] saved -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
