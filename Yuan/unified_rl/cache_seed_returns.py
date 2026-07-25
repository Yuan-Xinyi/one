"""Cache exhaustive seed returns under one frozen unified-v4 controller.

This utility materializes the expensive backward sweep once so later seed
policy experiments can reuse exactly the same downstream evidence.  Only the
training split recorded by a ``unified-bidirectional-v4`` checkpoint is
accepted, and every physically valid candidate in that split is rolled out.

Example::

    python -m Yuan.unified_rl.cache_seed_returns \
        --source-checkpoint Yuan/unified_rl/runs/r2_grouped_best/unified.pt \
        --candidates /path/to/candidates.npz \
        --controller-ckpt Yuan/unified_rl/runs/r2_grouped_best \
        --out Yuan/unified_rl/runs/r2_grouped_best/train_seed_returns.npz \
        --device cuda --chunk-size 1024

The output uses schema ``seed-return-cache-v1``.  Candidate-shaped arrays are
``valid``, ``discounted_return``, ``undiscounted_return``, ``progress_m``,
``episode_len``, ``term_reason``, and ``switch_count``.  Invalid candidate
slots contain NaN for floating-point results and -1 for integer results.  The
cache also carries the exact seed inputs, task identifiers/fingerprints, and
content hashes for the source checkpoint, candidate cache, and controller.
Existing output paths are never replaced; publication is an atomic hard-link
within the output directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

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


_SOURCE_REQUIRED_KEYS = (
    'format_version',
    'outer_round',
    'phase',
    'seed_policy',
    'controller',
    'controller_state_sha256',
    'controller_config',
    'controller_run_config_sha256',
    'train_indices',
    'validation_indices',
    'train_task_indices',
    'validation_task_indices',
    'train_valid_mask',
    'validation_valid_mask',
    'seed_include_ray_error',
    'seed_include_log_manip',
    'seed_architecture',
    'split_mode',
    'args',
    'provenance',
)


def _same_content(saved: dict, current: dict, *, label: str) -> None:
    """Require matching SHA-256 and byte size while allowing file relocation."""
    for key in ('size', 'sha256'):
        if key not in saved:
            raise ValueError(f'{label} provenance is missing {key!r}')
    if (saved['size'] != current['size']
            or saved['sha256'] != current['sha256']):
        raise ValueError(
            f'{label} differs from unified source provenance: '
            f'expected sha256={saved["sha256"]}, size={saved["size"]}; '
            f'got sha256={current["sha256"]}, size={current["size"]}')


def _prepare_output_path(value: str | Path,
                         protected_inputs: tuple[Path, ...]) -> Path:
    """Resolve a new NPZ target and reject aliases of immutable inputs."""
    raw = Path(value).expanduser()
    if raw.suffix.lower() != '.npz':
        raise ValueError('--out must have a .npz suffix')
    raw.parent.mkdir(parents=True, exist_ok=True)
    out = raw.resolve(strict=False)
    protected = {path.expanduser().resolve(strict=True)
                 for path in protected_inputs}
    if out in protected:
        raise ValueError('output path aliases an immutable input artifact')
    if os.path.lexists(out):
        raise FileExistsError(f'refusing to overwrite existing output: {out}')
    return out


def _atomic_savez_new(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically publish a new NPZ without a check-then-replace race."""
    temporary = path.with_name(
        f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link creation fails atomically if another writer won the
            # output name.  Unlike os.replace(), it can never overwrite it.
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f'refusing to overwrite concurrently created output: {path}'
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_source(
    checkpoint: dict,
    candidate_artifact: dict,
    controller_artifact: dict,
    controller_state_sha256: str,
    effective_controller_config: dict,
) -> tuple[str, str]:
    """Validate the immutable candidate/controller bindings of unified v4."""
    require_checkpoint_keys(
        checkpoint, _SOURCE_REQUIRED_KEYS,
        kind='unified-bidirectional-v4 source')
    require_checkpoint_format_version(
        checkpoint, 4, kind='unified-bidirectional-v4 source')

    provenance = checkpoint['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('source provenance must be a dictionary')
    require_checkpoint_keys(
        provenance, ('format', 'candidate_cache', 'settings'),
        kind='unified-bidirectional-v4 provenance')
    if provenance['format'] != 'unified-bidirectional-v4':
        raise ValueError(
            'source checkpoint must have provenance format '
            f"'unified-bidirectional-v4', got {provenance['format']!r}")
    _same_content(
        provenance['candidate_cache'], candidate_artifact,
        label='candidate cache')

    source_state_sha256 = state_dict_fingerprint(checkpoint['controller'])
    if source_state_sha256 != checkpoint['controller_state_sha256']:
        raise ValueError(
            'source controller tensors disagree with '
            'controller_state_sha256')
    if controller_state_sha256 != source_state_sha256:
        raise ValueError(
            'controller weights differ from the frozen controller embedded '
            'in the unified source')
    if (controller_artifact['config']['sha256']
            != checkpoint['controller_run_config_sha256']):
        raise ValueError(
            'controller config differs from the config fingerprint embedded '
            'in the unified source')
    if checkpoint['controller_config'] != effective_controller_config:
        raise ValueError(
            'controller config semantics differ from controller_config '
            'embedded in the unified source')

    settings = provenance['settings']
    if not isinstance(settings, dict):
        raise ValueError('source provenance settings must be a dictionary')
    split_mode = checkpoint['split_mode']
    if settings.get('split_mode') != split_mode:
        raise ValueError(
            'source split_mode disagrees with provenance settings')
    source_args = checkpoint['args']
    if not isinstance(source_args, dict):
        raise ValueError('source args must be a dictionary')
    seed_return = source_args.get('seed_return')
    if seed_return not in ('discounted', 'undiscounted'):
        raise ValueError(
            f'source seed return objective is invalid: {seed_return!r}')
    if settings.get('seed_return') != seed_return:
        raise ValueError(
            'source seed return objective disagrees with provenance settings')
    recorded_kind = checkpoint.get(
        'controller_kind', settings.get('controller_kind', 'pure'))
    if recorded_kind != 'pure':
        raise ValueError(
            'return-cache v1 supports only the deterministic pure RL '
            f'controller, got {recorded_kind!r}')
    return seed_return, split_mode


def _assert_unchanged(label: str, before: dict, after: dict) -> None:
    if before != after:
        raise RuntimeError(
            f'{label} changed while returns were being cached; refusing to '
            'publish a mixed-provenance artifact')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Roll out every valid candidate in the unified-v4 training split '
            'under its exact frozen controller and atomically save an NPZ.'))
    parser.add_argument(
        '--source-checkpoint', required=True,
        help='unified-bidirectional-v4 unified.pt defining the train split')
    parser.add_argument(
        '--candidates', required=True,
        help='candidate NPZ content-matched to source provenance')
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
    candidate_path = Path(args.candidates).expanduser().resolve(strict=True)
    controller_dir = resolve_controller_dir(args.controller_ckpt)
    if not source_path.is_file():
        raise ValueError('--source-checkpoint must be a file')
    if not candidate_path.is_file():
        raise ValueError('--candidates must be a file')
    output_path = _prepare_output_path(
        args.out,
        (source_path, candidate_path, controller_dir / 'agent.pt',
         controller_dir / 'config.yaml'))

    source_artifact = file_fingerprint(source_path)
    candidate_artifact = file_fingerprint(candidate_path)
    controller_artifact = controller_fingerprint(controller_dir)
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))

    checkpoint = torch.load(
        source_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('source checkpoint must contain a dictionary')
    dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    env = build_env_from_run(controller_dir, args.chunk_size, device)
    controller_agent = load_controller_agent(
        controller_dir, env, device).eval()
    controller_state_sha256 = state_dict_fingerprint(
        controller_agent.state_dict())
    controller_cfg = ppo_config_from_run(load_run_config(controller_dir))
    effective_controller_config = dataclasses.asdict(controller_cfg)
    seed_return, split_mode = _validate_source(
        checkpoint,
        candidate_artifact,
        controller_artifact,
        controller_state_sha256,
        effective_controller_config,
    )
    if not math.isfinite(controller_cfg.gamma) or not 0.0 <= controller_cfg.gamma <= 1.0:
        raise ValueError('controller gamma must be finite and in [0, 1]')

    # Reproduce the same derived action mask used by bidirectional training.
    # A dependency or collision-model change therefore fails closed below.
    dataset, valid_stats = validate_cached_dataset(
        dataset,
        env.kin,
        env.collision,
        chunk_size=args.chunk_size,
        cone_deg=env.cfg.cone_deg,
    )
    train_dataset = dataset.select_source_tasks(
        checkpoint['train_task_indices'].cpu())
    assert_same_valid_mask(
        train_dataset, checkpoint['train_valid_mask'], label='training')

    n_tasks = len(train_dataset)
    n_candidates = train_dataset.batch.n_candidates
    valid = train_dataset.batch.valid.numpy().copy()
    valid_pairs = torch.nonzero(
        train_dataset.batch.valid, as_tuple=False).to(dtype=torch.long)
    if valid_pairs.numel() == 0:
        raise ValueError('source training split contains no valid candidates')

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
    for start in range(0, n_pairs, args.chunk_size):
        end = min(start + args.chunk_size, n_pairs)
        pairs = valid_pairs[start:end]
        n_real = end - start
        if n_real < args.chunk_size:
            pairs = torch.cat([
                pairs,
                pairs[-1:].expand(args.chunk_size - n_real, -1),
            ])
        task_row = pairs[:, 0]
        candidate_index = pairs[:, 1].to(device=device)
        candidates = train_dataset.batch.index_select(task_row).to(
            device=device, dtype=env.kin.dtype)
        result = rollout_selected_seeds(
            env,
            candidates,
            candidate_index,
            frozen_controller,
            gamma=float(controller_cfg.gamma),
        )

        real_result = {
            'discounted_return': result.discounted_return[:n_real],
            'undiscounted_return': result.undiscounted_return[:n_real],
            'progress_m': result.progress_m[:n_real],
        }
        for name, value in real_result.items():
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f'controller produced non-finite {name} for a valid seed')
        row_np = pairs[:n_real, 0].numpy()
        slot_np = pairs[:n_real, 1].numpy()
        for name, value in real_result.items():
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
            f'[seed-return-cache] {end}/{n_pairs} valid candidate episodes',
            flush=True)

    train_indices = torch.as_tensor(
        checkpoint['train_indices'], device='cpu', dtype=torch.long)
    if train_indices.shape != (n_tasks,):
        raise ValueError(
            'source train_indices shape disagrees with train_task_indices')
    task_fingerprints = np.asarray(
        train_dataset.task_fingerprints, dtype='<U64')
    payload: dict[str, np.ndarray] = {
        'format': np.asarray('seed-return-cache-v1'),
        'format_version': np.asarray(1, dtype=np.int64),
        'valid': valid,
        **float_outputs,
        **integer_outputs,
        'q0': train_dataset.batch.q0.numpy().astype(np.float32, copy=True),
        'p0': train_dataset.batch.p0.numpy().astype(np.float32, copy=True),
        'line_dir': train_dataset.batch.line_dir.numpy().astype(
            np.float32, copy=True),
        'n_target': train_dataset.batch.n_target.numpy().astype(
            np.float32, copy=True),
        'task_indices': train_dataset.task_indices.numpy().copy(),
        'task_fingerprints': task_fingerprints,
        'train_row_indices': train_indices.numpy().copy(),
        'candidate_indices': np.arange(n_candidates, dtype=np.int64),
        'fallback_index': np.asarray(
            -1 if train_dataset.fallback_index is None
            else train_dataset.fallback_index,
            dtype=np.int64),
        'n_tasks': np.asarray(n_tasks, dtype=np.int64),
        'n_candidates': np.asarray(n_candidates, dtype=np.int64),
        'n_valid_rollouts': np.asarray(n_pairs, dtype=np.int64),
        'source_checkpoint_path': np.asarray(source_artifact['path']),
        'source_checkpoint_size': np.asarray(
            source_artifact['size'], dtype=np.int64),
        'source_checkpoint_sha256': np.asarray(
            source_artifact['sha256']),
        'candidate_cache_path': np.asarray(candidate_artifact['path']),
        'candidate_cache_size': np.asarray(
            candidate_artifact['size'], dtype=np.int64),
        'candidate_cache_sha256': np.asarray(
            candidate_artifact['sha256']),
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
        'split_mode': np.asarray(split_mode),
        'physical_validation': np.asarray(True, dtype=np.bool_),
    }
    for key, value in valid_stats.items():
        if key.startswith('frac_'):
            payload[f'physical_{key}'] = np.asarray(value, dtype=np.float64)
    payload['physical_n_tasks'] = np.asarray(
        valid_stats['n_tasks'], dtype=np.int64)
    payload['physical_n_tasks_retained'] = np.asarray(
        valid_stats['n_tasks_retained'], dtype=np.int64)
    payload['physical_n_tasks_rejected'] = np.asarray(
        valid_stats['n_tasks_rejected'], dtype=np.int64)

    # Re-fingerprint all immutable inputs immediately before publication so a
    # long sweep cannot silently combine more than one artifact generation.
    _assert_unchanged(
        'source checkpoint', source_artifact,
        file_fingerprint(source_path))
    _assert_unchanged(
        'candidate cache', candidate_artifact,
        file_fingerprint(candidate_path))
    _assert_unchanged(
        'controller', controller_artifact,
        controller_fingerprint(controller_dir))
    _atomic_savez_new(output_path, payload)
    print(f'[seed-return-cache] saved -> {output_path}')


if __name__ == '__main__':
    main()
