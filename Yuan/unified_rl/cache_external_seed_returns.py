"""Cache exhaustive returns for an external seed-candidate dataset.

This is the leakage-safe companion to :mod:`cache_seed_returns`.  It labels
every physically valid candidate in an external cache with the exact frozen
controller embedded in a ``unified-bidirectional-v4`` source checkpoint.  Any
external row whose task-geometry fingerprint occurs in the source validation
split is removed before physical validation or rollout.

The output schema is ``external-seed-return-cache-v1``.  In addition to the
candidate-shaped rollout arrays it records the retained external source-task
indices/fingerprints, the excluded overlap rows, the complete source
validation fingerprint set, physical-validation statistics, and content
identities for every immutable input.  Publication is atomic and never
overwrites an existing path.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import torch

from Yuan.unified_rl.cache_seed_returns import (
    _SOURCE_REQUIRED_KEYS,
    _assert_unchanged,
    _atomic_savez_new,
    _prepare_output_path,
    _same_content,
    _validate_source,
)
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_controller_agent,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    require_checkpoint_keys,
    resolve_controller_dir,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    rollout_selected_seeds,
)
from Yuan.unified_rl.provenance import (
    controller_fingerprint,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


def _recorded_source_candidate_path(checkpoint: dict) -> Path:
    """Resolve the source candidate artifact needed for leakage auditing."""
    require_checkpoint_keys(
        checkpoint, _SOURCE_REQUIRED_KEYS,
        kind='external-cache unified-bidirectional-v4 source')
    require_checkpoint_format_version(
        checkpoint, 4,
        kind='external-cache unified-bidirectional-v4 source')
    provenance = checkpoint['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('source provenance must be a dictionary')
    require_checkpoint_keys(
        provenance, ('format', 'candidate_cache', 'settings'),
        kind='external-cache source provenance')
    if provenance['format'] != 'unified-bidirectional-v4':
        raise ValueError(
            'source checkpoint must use unified-bidirectional-v4 provenance')
    candidate = provenance['candidate_cache']
    if not isinstance(candidate, dict):
        raise ValueError('source candidate provenance must be a dictionary')
    require_checkpoint_keys(
        candidate, ('path', 'size', 'sha256'),
        kind='external-cache source candidate provenance')
    path = Path(candidate['path']).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(
            'recorded source candidate cache must be an available file')
    return path


def _different_content(left: dict, right: dict) -> bool:
    """Return whether two file fingerprints identify different content."""
    return (left['size'] != right['size']
            or left['sha256'] != right['sha256'])


def _validated_source_fingerprints(
    checkpoint: dict,
    source_dataset: CachedSeedCandidateDataset,
    env,
    *,
    chunk_size: int,
) -> tuple[CachedSeedCandidateDataset, dict[str, float | list[int]]]:
    """Recreate and verify the checkpoint's source validation action set."""
    validation = source_dataset.select_source_tasks(
        torch.as_tensor(
            checkpoint['validation_task_indices'],
            device='cpu', dtype=torch.long))
    validated, stats = validate_cached_dataset(
        validation,
        env.kin,
        env.collision,
        chunk_size=chunk_size,
        cone_deg=env.cfg.cone_deg,
    )
    if len(validated) != len(validation):
        raise ValueError(
            'current physical validation rejects source validation tasks; '
            'refusing to construct a potentially incomplete exclusion set')
    assert_same_valid_mask(
        validated, checkpoint['validation_valid_mask'],
        label='source validation')
    return validated, stats


def _exclude_validation_overlap(
    external: CachedSeedCandidateDataset,
    source_validation_fingerprints: set[str],
) -> tuple[CachedSeedCandidateDataset, torch.Tensor, torch.Tensor]:
    """Remove every external row matching a source-validation geometry."""
    fingerprints = external.task_fingerprints
    excluded = torch.tensor(
        [fingerprint in source_validation_fingerprints
         for fingerprint in fingerprints],
        dtype=torch.bool)
    kept_rows = torch.nonzero(~excluded, as_tuple=False).flatten()
    excluded_rows = torch.nonzero(excluded, as_tuple=False).flatten()
    if kept_rows.numel() == 0:
        raise ValueError(
            'source validation fingerprint exclusion removed every external '
            'task row')
    retained = external.index_select(kept_rows)
    if set(retained.task_fingerprints) & source_validation_fingerprints:
        raise RuntimeError('validation fingerprint exclusion failed closed')
    return retained, kept_rows, excluded_rows


def _retained_external_rows(
    before_physical: CachedSeedCandidateDataset,
    after_physical: CachedSeedCandidateDataset,
    external_rows_before_physical: torch.Tensor,
) -> torch.Tensor:
    """Map physically retained task ids back to original external row ids."""
    position = {
        int(task_id): row
        for row, task_id in enumerate(before_physical.task_indices.tolist())
    }
    if len(position) != len(before_physical):
        raise ValueError('external source task indices must be unique')
    local = torch.tensor(
        [position[int(task_id)]
         for task_id in after_physical.task_indices.tolist()],
        dtype=torch.long)
    return external_rows_before_physical[local]


@torch.no_grad()
def _rollout_all_valid_candidates(
    dataset: CachedSeedCandidateDataset,
    env,
    controller_agent,
    *,
    gamma: float,
    chunk_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Roll out all valid pairs, padding the final pair block exactly once."""
    n_tasks = len(dataset)
    n_candidates = dataset.batch.n_candidates
    valid_pairs = torch.nonzero(
        dataset.batch.valid, as_tuple=False).to(dtype=torch.long)
    if valid_pairs.numel() == 0:
        raise ValueError('retained external dataset has no valid candidates')

    float_outputs = {
        name: np.full((n_tasks, n_candidates), np.nan, dtype=np.float32)
        for name in (
            'discounted_return', 'undiscounted_return', 'progress_m')
    }
    integer_outputs = {
        'episode_len': np.full(
            (n_tasks, n_candidates), -1, dtype=np.int64),
        'term_reason': np.full(
            (n_tasks, n_candidates), -1, dtype=np.int32),
        'switch_count': np.full(
            (n_tasks, n_candidates), -1, dtype=np.int64),
    }
    frozen_controller = FrozenRLController(controller_agent)
    n_pairs = int(valid_pairs.shape[0])
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        real_pairs = valid_pairs[start:end]
        n_real = end - start

        # The environment has exactly ``chunk_size`` instances.  Construct a
        # single padded view only for the rollout; indexing and publication
        # below always use the untouched real pairs.
        rollout_pairs = real_pairs
        if n_real < chunk_size:
            rollout_pairs = torch.cat([
                real_pairs,
                real_pairs[-1:].expand(chunk_size - n_real, -1),
            ], dim=0)

        candidates = dataset.batch.index_select(
            rollout_pairs[:, 0]).to(
                device=device, dtype=env.kin.dtype)
        candidate_index = rollout_pairs[:, 1].to(device=device)
        result = rollout_selected_seeds(
            env,
            candidates,
            candidate_index,
            frozen_controller,
            gamma=gamma,
        )

        real_float_results = {
            'discounted_return': result.discounted_return[:n_real],
            'undiscounted_return': result.undiscounted_return[:n_real],
            'progress_m': result.progress_m[:n_real],
        }
        for name, value in real_float_results.items():
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f'controller produced non-finite {name} for a valid seed')
        row_np = real_pairs[:, 0].numpy()
        slot_np = real_pairs[:, 1].numpy()
        for name, value in real_float_results.items():
            float_outputs[name][row_np, slot_np] = (
                value.detach().cpu().to(torch.float32).numpy())
        for name, value in (
                ('episode_len', result.episode_len[:n_real]),
                ('term_reason', result.term_reason[:n_real]),
                ('switch_count', result.switch_count[:n_real])):
            values = value.detach().cpu().numpy()
            if np.any(values < 0):
                raise ValueError(
                    f'controller produced invalid {name} for a valid seed')
            integer_outputs[name][row_np, slot_np] = values
        print(
            '[external-seed-return-cache] '
            f'{end}/{n_pairs} valid candidate episodes',
            flush=True)
    return float_outputs, integer_outputs, n_pairs


def _physical_stats_payload(
    prefix: str,
    stats: dict[str, float | list[int]],
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in stats.items():
        name = f'{prefix}_{key}'
        if key.startswith('frac_'):
            payload[name] = np.asarray(value, dtype=np.float64)
        elif key == 'rejected_task_indices':
            payload[name] = np.asarray(value, dtype=np.int64)
        elif key.startswith('n_'):
            payload[name] = np.asarray(value, dtype=np.int64)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Label an external candidate cache with the exact frozen '
            'unified-v4 controller after excluding every source-validation '
            'task-geometry fingerprint.'))
    parser.add_argument(
        '--source-checkpoint', required=True,
        help='unified-bidirectional-v4 unified.pt defining validation leakage')
    parser.add_argument(
        '--candidates', required=True,
        help='external candidate NPZ; content must differ from source cache')
    parser.add_argument(
        '--controller-ckpt', required=True,
        help='controller run directory (agent.pt + config.yaml) matched to source')
    parser.add_argument(
        '--out', required=True,
        help='new .npz path; existing files are never overwritten')
    parser.add_argument(
        '--device', default=None,
        help='Torch device (default: cuda when available, otherwise cpu)')
    parser.add_argument(
        '--chunk-size', type=int, default=1024,
        help='fixed number of controller episodes per rollout chunk')
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be positive')

    source_path = Path(args.source_checkpoint).expanduser().resolve(strict=True)
    external_path = Path(args.candidates).expanduser().resolve(strict=True)
    controller_dir = resolve_controller_dir(args.controller_ckpt)
    if not source_path.is_file():
        raise ValueError('--source-checkpoint must be a file')
    if not external_path.is_file():
        raise ValueError('--candidates must be a file')

    # Fingerprint before parsing so the final TOCTOU check also binds the
    # bytes from which the in-memory checkpoint was constructed.
    source_artifact = file_fingerprint(source_path)
    external_artifact = file_fingerprint(external_path)
    controller_artifact = controller_fingerprint(controller_dir)
    checkpoint = torch.load(
        source_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('source checkpoint must contain a dictionary')
    source_candidate_path = _recorded_source_candidate_path(checkpoint)
    output_path = _prepare_output_path(
        args.out,
        (source_path, source_candidate_path, external_path,
         controller_dir / 'agent.pt', controller_dir / 'config.yaml'))

    source_candidate_artifact = file_fingerprint(source_candidate_path)
    _same_content(
        checkpoint['provenance']['candidate_cache'],
        source_candidate_artifact,
        label='recorded source candidate cache')
    if not _different_content(source_candidate_artifact, external_artifact):
        raise ValueError(
            'external candidate cache has the same content as the source '
            'candidate cache')

    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    env = build_env_from_run(controller_dir, args.chunk_size, device)
    controller_agent = load_controller_agent(
        controller_dir, env, device).eval()
    controller_state_sha256 = state_dict_fingerprint(
        controller_agent.state_dict())
    controller_cfg = ppo_config_from_run(load_run_config(controller_dir))
    effective_controller_config = dataclasses.asdict(controller_cfg)
    seed_return, split_mode = _validate_source(
        checkpoint,
        source_candidate_artifact,
        controller_artifact,
        controller_state_sha256,
        effective_controller_config,
    )
    if (not math.isfinite(controller_cfg.gamma)
            or not 0.0 <= controller_cfg.gamma <= 1.0):
        raise ValueError('controller gamma must be finite and in [0, 1]')

    source_dataset = CachedSeedCandidateDataset.from_npz(
        source_candidate_path)
    source_validation, source_validation_stats = (
        _validated_source_fingerprints(
            checkpoint,
            source_dataset,
            env,
            chunk_size=args.chunk_size,
        ))
    source_validation_fingerprints = tuple(
        source_validation.task_fingerprints)
    source_validation_set = set(source_validation_fingerprints)
    source_validation_split_rows = torch.as_tensor(
        checkpoint['validation_indices'],
        device='cpu', dtype=torch.long)
    if source_validation_split_rows.shape != (len(source_validation),):
        raise ValueError(
            'source validation_indices shape disagrees with '
            'validation_task_indices')

    external_input = CachedSeedCandidateDataset.from_npz(external_path)
    external_input_fingerprints = external_input.task_fingerprints
    after_exclusion, rows_after_exclusion, excluded_rows = (
        _exclude_validation_overlap(
            external_input, source_validation_set))
    validated_external, external_valid_stats = validate_cached_dataset(
        after_exclusion,
        env.kin,
        env.collision,
        chunk_size=args.chunk_size,
        cone_deg=env.cfg.cone_deg,
    )
    retained_rows = _retained_external_rows(
        after_exclusion, validated_external, rows_after_exclusion)
    retained_fingerprints = tuple(validated_external.task_fingerprints)
    if set(retained_fingerprints) & source_validation_set:
        raise RuntimeError(
            'retained external data overlaps source validation geometry')

    float_outputs, integer_outputs, n_pairs = (
        _rollout_all_valid_candidates(
            validated_external,
            env,
            controller_agent,
            gamma=float(controller_cfg.gamma),
            chunk_size=args.chunk_size,
            device=device,
        ))

    n_tasks = len(validated_external)
    n_candidates = validated_external.batch.n_candidates
    valid = validated_external.batch.valid.numpy().copy()
    excluded_fingerprints = np.asarray(
        [external_input_fingerprints[int(row)]
         for row in excluded_rows.tolist()],
        dtype='<U64')
    payload: dict[str, np.ndarray] = {
        'format': np.asarray('external-seed-return-cache-v1'),
        'format_version': np.asarray(1, dtype=np.int64),
        'valid': valid,
        **float_outputs,
        **integer_outputs,
        'q0': validated_external.batch.q0.numpy().astype(
            np.float32, copy=True),
        'p0': validated_external.batch.p0.numpy().astype(
            np.float32, copy=True),
        'line_dir': validated_external.batch.line_dir.numpy().astype(
            np.float32, copy=True),
        'n_target': validated_external.batch.n_target.numpy().astype(
            np.float32, copy=True),
        'task_indices': validated_external.task_indices.numpy().copy(),
        'task_fingerprints': np.asarray(
            retained_fingerprints, dtype='<U64'),
        'external_row_indices': retained_rows.numpy().copy(),
        'retained_source_task_indices': (
            validated_external.task_indices.numpy().copy()),
        'retained_task_fingerprints': np.asarray(
            retained_fingerprints, dtype='<U64'),
        'retained_external_row_indices': retained_rows.numpy().copy(),
        'candidate_indices': np.arange(n_candidates, dtype=np.int64),
        'fallback_index': np.asarray(
            -1 if validated_external.fallback_index is None
            else validated_external.fallback_index,
            dtype=np.int64),
        'n_tasks': np.asarray(n_tasks, dtype=np.int64),
        'n_candidates': np.asarray(n_candidates, dtype=np.int64),
        'n_valid_rollouts': np.asarray(n_pairs, dtype=np.int64),
        'external_input_n_tasks': np.asarray(
            len(external_input), dtype=np.int64),
        'external_input_unique_fingerprints': np.asarray(
            len(set(external_input_fingerprints)), dtype=np.int64),
        'n_tasks_after_validation_exclusion': np.asarray(
            len(after_exclusion), dtype=np.int64),
        'excluded_validation_overlap_rows': np.asarray(
            excluded_rows.numel(), dtype=np.int64),
        'excluded_validation_overlap_unique_fingerprints': np.asarray(
            len(set(excluded_fingerprints.tolist())), dtype=np.int64),
        'excluded_external_row_indices': excluded_rows.numpy().copy(),
        'excluded_task_indices': external_input.task_indices[
            excluded_rows].numpy().copy(),
        'excluded_task_fingerprints': excluded_fingerprints,
        'source_validation_n_tasks': np.asarray(
            len(source_validation), dtype=np.int64),
        'source_validation_unique_fingerprints': np.asarray(
            len(source_validation_set), dtype=np.int64),
        'source_validation_task_indices': (
            source_validation.task_indices.numpy().copy()),
        'source_validation_split_row_indices': (
            source_validation_split_rows.numpy().copy()),
        'source_validation_task_fingerprints': np.asarray(
            source_validation_fingerprints, dtype='<U64'),
        'validation_overlap_after_exclusion': np.asarray(
            False, dtype=np.bool_),
        'source_checkpoint_path': np.asarray(source_artifact['path']),
        'source_checkpoint_size': np.asarray(
            source_artifact['size'], dtype=np.int64),
        'source_checkpoint_sha256': np.asarray(
            source_artifact['sha256']),
        'source_candidate_cache_path': np.asarray(
            source_candidate_artifact['path']),
        'source_candidate_cache_size': np.asarray(
            source_candidate_artifact['size'], dtype=np.int64),
        'source_candidate_cache_sha256': np.asarray(
            source_candidate_artifact['sha256']),
        'candidate_cache_path': np.asarray(external_artifact['path']),
        'candidate_cache_size': np.asarray(
            external_artifact['size'], dtype=np.int64),
        'candidate_cache_sha256': np.asarray(
            external_artifact['sha256']),
        'controller_path': np.asarray(controller_artifact['path']),
        'controller_agent_size': np.asarray(
            controller_artifact['agent']['size'], dtype=np.int64),
        'controller_agent_sha256': np.asarray(
            controller_artifact['agent']['sha256']),
        'controller_config_size': np.asarray(
            controller_artifact['config']['size'], dtype=np.int64),
        'controller_config_sha256': np.asarray(
            controller_artifact['config']['sha256']),
        'controller_state_sha256': np.asarray(controller_state_sha256),
        'controller_gamma': np.asarray(
            controller_cfg.gamma, dtype=np.float64),
        'controller_kind': np.asarray('pure'),
        'source_phase': np.asarray(checkpoint['phase']),
        'source_outer_round': np.asarray(
            checkpoint['outer_round'], dtype=np.int64),
        'seed_return_objective': np.asarray(seed_return),
        'source_split_mode': np.asarray(split_mode),
        'split_mode': np.asarray(
            'external-validation-fingerprint-excluded-v1'),
        'physical_validation': np.asarray(True, dtype=np.bool_),
    }
    payload.update(_physical_stats_payload(
        'physical', external_valid_stats))
    payload.update(_physical_stats_payload(
        'source_validation_physical', source_validation_stats))

    # Repeat all content hashes immediately before publication.  This makes a
    # long exhaustive sweep fail rather than publish a mixed-generation cache.
    _assert_unchanged(
        'source checkpoint', source_artifact,
        file_fingerprint(source_path))
    _assert_unchanged(
        'source candidate cache', source_candidate_artifact,
        file_fingerprint(source_candidate_path))
    _assert_unchanged(
        'external candidate cache', external_artifact,
        file_fingerprint(external_path))
    _assert_unchanged(
        'controller', controller_artifact,
        controller_fingerprint(controller_dir))
    _atomic_savez_new(output_path, payload)
    print(
        '[external-seed-return-cache] '
        f'excluded={excluded_rows.numel()} retained={n_tasks} '
        f'saved -> {output_path}')


if __name__ == '__main__':
    main()
