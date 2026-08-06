"""Evaluate one generated seed, at most one IK projection, and one rollout.

The script reports generator, projection, and controller wall time separately.
IK-pool returns are optional teacher-only diagnostics (fallback, pool oracle,
and capture); they are never queried to choose the deployed seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedCandidateBatch,
)
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_controller_agent,
    load_run_config,
    ppo_config_from_run,
    resolve_controller_dir,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    rollout_selected_seeds,
)
from Yuan.unified_rl.direct_seed_model import (
    direct_seed_task,
    load_deployment_generator,
)
from Yuan.unified_rl.direct_seed_projection import (
    DirectSeedProjectionConfig,
    ROUTE_NAMES,
    route_generated_seed,
    strict_seed_validity,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    """Return a canonical int64 manifest vector without lossy coercion."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(
            f'fallback filter manifest {name} must be one-dimensional')
    if not np.issubdtype(raw.dtype, np.integer) \
            or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(
            f'fallback filter manifest {name} must have an integer dtype')
    if np.issubdtype(raw.dtype, np.unsignedinteger) \
            and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError(
            f'fallback filter manifest {name} exceeds int64 range')
    return raw.astype(np.int64, copy=False)


def _integer_vector_sha256(value: np.ndarray) -> str:
    canonical = np.asarray(
        value, dtype=np.dtype('<i8'), order='C')
    return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def _paired_summary(
    new: np.ndarray,
    old: np.ndarray,
    *,
    seed: int = 20260728,
    bootstrap_samples: int = 5000,
) -> dict[str, float | list[float]]:
    new = np.asarray(new, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    if new.shape != old.shape or new.ndim != 1 or new.size < 1:
        raise ValueError('paired arrays must be non-empty and one-dimensional')
    if not np.isfinite(new).all() or not np.isfinite(old).all():
        raise ValueError('paired arrays must be finite')
    delta = new - old
    ordered = np.sort(delta)
    trim = int(0.05 * len(delta))
    trimmed = ordered[trim:-trim] if trim else ordered
    rng = np.random.default_rng(int(seed))
    boot = np.empty(bootstrap_samples, dtype=np.float64)
    chunk = min(256, bootstrap_samples)
    for start in range(0, bootstrap_samples, chunk):
        end = min(start + chunk, bootstrap_samples)
        sample = rng.integers(
            0, len(delta), size=(end - start, len(delta)))
        boot[start:end] = delta[sample].mean(axis=1)
    return {
        'delta_mm': float(delta.mean() * 1e3),
        'ci95_mm': [
            float(np.percentile(boot, 2.5) * 1e3),
            float(np.percentile(boot, 97.5) * 1e3),
        ],
        'trimmed_5pct_mm': float(trimmed.mean() * 1e3),
        'harm_gt_1mm_pct': float((delta < -1e-3).mean() * 100.0),
        'win_gt_1mm_pct': float((delta > 1e-3).mean() * 100.0),
    }


def summarize_direct_seed(
    progress_m: np.ndarray,
    route: np.ndarray,
    valid: np.ndarray,
    *,
    fallback_progress_m: np.ndarray | None = None,
    pool_oracle_progress_m: np.ndarray | None = None,
    reference_progress_m: np.ndarray | None = None,
) -> dict:
    """Build JSON-safe routing, progress, and paired result metrics."""
    progress = np.asarray(progress_m, dtype=np.float64)
    route = np.asarray(route)
    valid = np.asarray(valid)
    if progress.ndim != 1 or route.shape != progress.shape \
            or valid.shape != progress.shape:
        raise ValueError('progress, route, and valid must have shape (N,)')
    if not np.isfinite(progress).all():
        raise ValueError('progress must be finite')
    if valid.dtype != np.bool_ or not valid.all():
        raise ValueError('evaluation must not execute invalid seeds')
    if not np.isin(route, np.arange(len(ROUTE_NAMES))).all():
        raise ValueError('route contains an unknown code')
    count = {
        name: int((route == index).sum())
        for index, name in enumerate(ROUTE_NAMES)
    }
    report = {
        'n_tasks': int(len(progress)),
        'progress_mean_m': float(progress.mean()),
        'route_count': count,
        'route_pct': {
            name: float(value / len(progress) * 100.0)
            for name, value in count.items()
        },
        'mean_ik_attempts_per_task': float(
            1.0 - count['direct'] / len(progress)),
        'controller_rollouts_per_task': 1.0,
        'candidate_enumeration_per_task': 0,
    }

    def checked_optional(value, name):
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        if result.shape != progress.shape or not np.isfinite(result).all():
            raise ValueError(f'{name} must be finite with shape (N,)')
        return result

    fallback = checked_optional(
        fallback_progress_m, 'fallback_progress_m')
    oracle = checked_optional(
        pool_oracle_progress_m, 'pool_oracle_progress_m')
    reference = checked_optional(
        reference_progress_m, 'reference_progress_m')
    if fallback is not None:
        report['vs_fallback'] = _paired_summary(progress, fallback)
    if reference is not None:
        report['vs_reference'] = _paired_summary(progress, reference)
    if fallback is not None and oracle is not None:
        denominator = float((oracle - fallback).sum())
        report['pool_oracle_mean_m'] = float(oracle.mean())
        report['pool_capture_pct'] = (
            float((progress - fallback).sum() / denominator * 100.0)
            if abs(denominator) > 1e-12 else None)
    return report


def _load_teacher_returns(
    path: Path,
    dataset: CachedSeedCandidateDataset,
    rows: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset.fallback_index is None:
        raise ValueError('teacher comparison requires q0_pilot fallback')
    with np.load(path, allow_pickle=False) as archive:
        progress = np.asarray(archive['progress_m'], dtype=np.float32)
        valid = np.asarray(archive['valid'], dtype=np.bool_)
        task_indices = np.asarray(archive['task_indices'], dtype=np.int64)
    expected = (len(dataset), dataset.batch.n_candidates)
    if progress.shape != expected or valid.shape != expected:
        raise ValueError('teacher return cache shape does not match candidates')
    if not np.array_equal(task_indices, dataset.task_indices.numpy()):
        raise ValueError('teacher return task_indices do not match candidates')
    selected = rows.numpy()
    p = progress[selected]
    v = valid[selected] & np.isfinite(p)
    if not v.any(axis=1).all():
        raise ValueError('teacher return cache has a task with no valid action')
    oracle = np.where(v, p, -np.inf).max(axis=1)
    fallback = p[:, dataset.fallback_index]
    if not np.isfinite(fallback).all():
        raise ValueError('teacher fallback returns must be finite')
    return fallback, oracle


def _load_reference_progress(
    path: Path,
    task_indices: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if 'task_indices' not in archive:
            raise ValueError(
                'reference eval must contain task_indices')
        if 'policy_progress_m' in archive:
            progress_key = 'policy_progress_m'
        elif 'progress_m' in archive:
            # Native direct-seed evaluation artifact.  Supporting it makes
            # pre/post RL comparisons task-paired without rewriting either
            # result or silently changing the common task set.
            progress_key = 'progress_m'
        else:
            raise ValueError(
                'reference eval must contain policy_progress_m or progress_m')
        source_indices = np.asarray(archive['task_indices'], dtype=np.int64)
        source_progress = np.asarray(
            archive[progress_key], dtype=np.float32)
    if source_indices.ndim != 1 or source_progress.shape != source_indices.shape:
        raise ValueError(
            'reference task_indices and progress must be same-length vectors')
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError('reference eval contains duplicate task indices')
    if not np.isfinite(source_progress).all():
        raise ValueError('reference eval progress must be finite')
    lookup = {
        int(task): float(value)
        for task, value in zip(source_indices, source_progress)
    }
    missing = [int(task) for task in task_indices if int(task) not in lookup]
    if missing:
        raise ValueError(
            f'reference eval is missing task indices: {missing[:20]}')
    return np.asarray(
        [lookup[int(task)] for task in task_indices], dtype=np.float32)


def _load_fallback_filter_manifest(
    path: Path,
    dataset: CachedSeedCandidateDataset,
    n_source: int,
    *,
    candidates_sha256: str,
    allow_legacy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Reuse an audited common row set, requiring candidate-file identity."""
    resolved = path.expanduser().resolve(strict=True)
    manifest_sha256 = _file_sha256(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        required = {
            'source_row_index', 'excluded_source_row_index',
            'task_indices', 'excluded_task_indices',
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f'fallback filter manifest is missing {sorted(missing)}')
        rows_np = _integer_vector(
            archive['source_row_index'], 'source_row_index')
        excluded_np = _integer_vector(
            archive['excluded_source_row_index'],
            'excluded_source_row_index')
        kept_task_np = _integer_vector(
            archive['task_indices'], 'task_indices')
        excluded_task_np = _integer_vector(
            archive['excluded_task_indices'], 'excluded_task_indices')
    if kept_task_np.shape != rows_np.shape \
            or excluded_task_np.shape != excluded_np.shape:
        raise ValueError(
            'fallback filter manifest row/task arrays differ in length')
    partition = np.concatenate([rows_np, excluded_np])
    if (len(partition) != n_source
            or len(np.unique(partition)) != n_source
            or not np.array_equal(
                np.sort(partition), np.arange(n_source, dtype=np.int64))):
        raise ValueError(
            'fallback filter manifest rows do not exactly partition source')
    source_task_indices = dataset.task_indices[:n_source].numpy()
    if not np.array_equal(
            source_task_indices[rows_np], kept_task_np) \
            or not np.array_equal(
                source_task_indices[excluded_np], excluded_task_np):
        raise ValueError(
            'fallback filter manifest task indices differ from candidates')

    rows = torch.from_numpy(rows_np.copy())
    excluded_rows = torch.from_numpy(excluded_np.copy())
    task_fingerprints = dataset.task_fingerprints
    kept_fingerprints = [
        task_fingerprints[int(row)] for row in rows_np]
    excluded_fingerprints = [
        task_fingerprints[int(row)] for row in excluded_np]
    report = {
        'explicit_filter_enabled': True,
        'n_source_tasks': int(n_source),
        'n_kept_tasks': int(len(rows_np)),
        'n_excluded_tasks': int(len(excluded_np)),
        'kept_geometry_fingerprint_list_sha256': hashlib.sha256(
            '\n'.join(kept_fingerprints).encode('ascii')).hexdigest(),
        'excluded_geometry_fingerprint_list_sha256': hashlib.sha256(
            '\n'.join(excluded_fingerprints).encode('ascii')).hexdigest(),
        'kept_source_row_list_sha256': _integer_vector_sha256(rows_np),
        'excluded_source_row_list_sha256': _integer_vector_sha256(
            excluded_np),
        'kept_task_index_list_sha256': _integer_vector_sha256(kept_task_np),
        'excluded_task_index_list_sha256': _integer_vector_sha256(
            excluded_task_np),
        'reused_manifest': str(resolved),
        'reused_manifest_sha256': manifest_sha256,
    }
    json_path = resolved.with_suffix('.json')
    if not json_path.is_file():
        raise ValueError(
            'fallback filter manifest requires its sibling JSON audit')
    manifest_json_sha256 = _file_sha256(json_path)
    saved = json.loads(json_path.read_text(encoding='utf-8'))
    saved_filter = saved.get('fallback_strict_filter')
    if not isinstance(saved_filter, dict):
        raise ValueError(
            'fallback filter manifest JSON has no filter audit')
    if saved_filter.get('explicit_filter_enabled') is not True:
        raise ValueError(
            'fallback filter manifest JSON is not an explicit filter audit')
    for key in (
            'n_source_tasks', 'n_kept_tasks', 'n_excluded_tasks',
            'kept_geometry_fingerprint_list_sha256',
            'excluded_geometry_fingerprint_list_sha256'):
        if saved_filter.get(key) != report[key]:
            raise ValueError(
                f'fallback filter manifest audit differs for {key}')
    # Row/task-list hashes were added after the first audited manifests.
    # Validate them whenever present. Candidate-file identity below remains
    # mandatory for the default path, so absence does not permit a manifest
    # to be transplanted to another candidate cache.
    for key in (
            'kept_source_row_list_sha256',
            'excluded_source_row_list_sha256',
            'kept_task_index_list_sha256',
            'excluded_task_index_list_sha256'):
        if key in saved_filter and saved_filter[key] != report[key]:
            raise ValueError(
                f'fallback filter manifest audit differs for {key}')
    saved_artifacts = saved.get('artifacts')
    saved_candidates_sha256 = (
        saved_artifacts.get('candidates_sha256')
        if isinstance(saved_artifacts, dict) else None)
    if saved_candidates_sha256 is None:
        if not allow_legacy:
            raise ValueError(
                'fallback filter manifest has no candidates SHA256; '
                'default reuse fails closed. Use '
                '--allow-legacy-filter-manifest only for an explicitly '
                'labelled historical diagnostic')
        candidate_identity_verified = False
    else:
        if (not isinstance(saved_candidates_sha256, str)
                or len(saved_candidates_sha256) != 64
                or any(character not in '0123456789abcdef'
                       for character in saved_candidates_sha256)):
            raise ValueError(
                'fallback filter manifest candidates SHA256 is malformed')
        if saved_candidates_sha256 != candidates_sha256:
            raise ValueError(
                'fallback filter manifest candidates SHA256 differs from '
                'the current candidate file')
        candidate_identity_verified = True
    if saved.get('n_tasks') not in (None, int(len(rows_np))):
        raise ValueError(
            'fallback filter manifest JSON n_tasks differs from kept rows')
    upstream_legacy = (
        saved_filter.get('legacy_unverified') is True
        or ('reused_manifest' in saved_filter
            and saved_filter.get('candidate_identity_verified') is not True))
    report.update({
        'candidate_identity_verified': candidate_identity_verified,
        'legacy_unverified': not candidate_identity_verified,
        'upstream_filter_lineage_legacy_unverified': upstream_legacy,
        'current_candidates_sha256': candidates_sha256,
        'manifest_candidates_sha256': saved_candidates_sha256,
    })
    if _file_sha256(resolved) != manifest_sha256 \
            or _file_sha256(json_path) != manifest_json_sha256:
        raise RuntimeError(
            'fallback filter manifest changed while it was being audited')
    report['reused_manifest_json_sha256'] = manifest_json_sha256
    return rows, excluded_rows, report


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument(
        '--controller-dir',
        default='Yuan/unified_rl/runs/r2_grouped_best')
    parser.add_argument('--teacher-returns', default=None)
    parser.add_argument('--reference-eval', default=None)
    parser.add_argument(
        '--fallback-filter-manifest', default=None,
        help='reuse source/excluded rows from a prior audited direct-seed '
             'evaluation on the identical candidate file')
    parser.add_argument(
        '--allow-legacy-filter-manifest', action='store_true',
        help='allow a historical manifest without a candidates SHA256; '
             'the output is explicitly marked legacy_unverified')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument(
        '--filter-invalid-fallback', action='store_true',
        help=(
            'explicitly evaluate the fixed common subset whose supplied '
            'fallback passes the same strict gate; kept/excluded indices and '
            'geometry fingerprints are saved. Default is fail loud.'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be positive')
    if args.allow_legacy_filter_manifest \
            and args.fallback_filter_manifest is None:
        raise ValueError(
            '--allow-legacy-filter-manifest requires '
            '--fallback-filter-manifest')
    output = Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'refusing to overwrite output {output}')
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    controller_dir = resolve_controller_dir(args.controller_dir)
    controller_agent_path = controller_dir / 'agent.pt'
    controller_config_path = controller_dir / 'config.yaml'
    controller_agent_sha256 = _file_sha256(controller_agent_path)
    controller_config_sha256 = _file_sha256(controller_config_path)
    candidates_path = Path(
        args.candidates).expanduser().resolve(strict=True)
    candidates_sha256 = _file_sha256(candidates_path)
    checkpoint_path = Path(
        args.checkpoint).expanduser().resolve(strict=True)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    dataset = CachedSeedCandidateDataset.from_npz(candidates_path)
    if dataset.fallback_index is None:
        raise ValueError(
            'direct-seed evaluation requires q0_pilot fallback')
    n_source = len(dataset) if args.limit is None \
        else min(len(dataset), args.limit)
    if n_source < 1:
        raise ValueError('evaluation task count must be positive')

    env = build_env_from_run(
        controller_dir, args.batch_size, device)
    projection_config = DirectSeedProjectionConfig()

    # Freeze the common task subset before looking at model output.  FALLBACK
    # is not repaired or replaced: either it passes the identical strict gate,
    # or the task is excluded under an explicit manifest-bearing protocol.
    if args.fallback_filter_manifest is not None:
        if not args.filter_invalid_fallback:
            raise ValueError(
                '--fallback-filter-manifest requires '
                '--filter-invalid-fallback')
        rows, excluded_rows, fallback_filter_report = (
            _load_fallback_filter_manifest(
                Path(args.fallback_filter_manifest),
                dataset, n_source,
                candidates_sha256=candidates_sha256,
                allow_legacy=args.allow_legacy_filter_manifest))
    else:
        candidate_rows = torch.arange(n_source)
        fallback_valid_parts = []
        for start in range(0, n_source, args.batch_size):
            source = candidate_rows[start:start + args.batch_size]
            batch = dataset.batch.index_select(source).to(
                device, dtype=env.kin.dtype)
            fallback_valid_parts.append(strict_seed_validity(
                env.kin, env.collision,
                batch.q0[:, dataset.fallback_index],
                batch.p0, batch.line_dir, batch.n_target,
                projection_config).valid.cpu())
        fallback_valid = torch.cat(fallback_valid_parts)
        excluded_rows = candidate_rows[~fallback_valid]
        rows = candidate_rows[fallback_valid]
        if excluded_rows.numel() and not args.filter_invalid_fallback:
            raise RuntimeError(
                f'{excluded_rows.numel()}/{n_source} supplied fallbacks fail '
                'the strict deployment gate; default protocol refuses to '
                'evaluate. Use --filter-invalid-fallback only with an '
                'explicit common-set manifest. '
                f'First rows={excluded_rows[:20].tolist()}')
        if rows.numel() < 1:
            raise RuntimeError('strict fallback filter removed every task')
        task_fingerprints = dataset.task_fingerprints
        kept_fingerprints = [
            task_fingerprints[int(row)]
            for row in rows.tolist()]
        excluded_fingerprints = [
            task_fingerprints[int(row)]
            for row in excluded_rows.tolist()]
        fallback_filter_report = {
            'explicit_filter_enabled': bool(
                args.filter_invalid_fallback),
            'n_source_tasks': int(n_source),
            'n_kept_tasks': int(rows.numel()),
            'n_excluded_tasks': int(excluded_rows.numel()),
            'kept_geometry_fingerprint_list_sha256': hashlib.sha256(
                '\n'.join(kept_fingerprints).encode('ascii')).hexdigest(),
            'excluded_geometry_fingerprint_list_sha256': hashlib.sha256(
                '\n'.join(excluded_fingerprints).encode('ascii')).hexdigest(),
            'kept_source_row_list_sha256': _integer_vector_sha256(
                rows.numpy()),
            'excluded_source_row_list_sha256': _integer_vector_sha256(
                excluded_rows.numpy()),
            'kept_task_index_list_sha256': _integer_vector_sha256(
                dataset.task_indices[rows].numpy()),
            'excluded_task_index_list_sha256': _integer_vector_sha256(
                dataset.task_indices[excluded_rows].numpy()),
            'candidate_identity_verified': True,
            'legacy_unverified': False,
            'upstream_filter_lineage_legacy_unverified': False,
            'current_candidates_sha256': candidates_sha256,
            'manifest_candidates_sha256': candidates_sha256,
        }
    if rows.numel() < 1:
        raise RuntimeError('strict fallback filter removed every task')
    n_tasks = int(rows.numel())

    # Main path: contextual-RL actor deterministic mean.  The same adapter
    # accepts the explicitly labelled supervised bootstrap/ablation.
    model, model_payload = load_deployment_generator(
        checkpoint_path, device)
    agent = load_controller_agent(
        controller_dir, env, device).eval()
    controller = FrozenRLController(agent)
    gamma = float(ppo_config_from_run(
        load_run_config(controller_dir)).gamma)

    progress = np.empty(n_tasks, dtype=np.float32)
    episode_len = np.empty(n_tasks, dtype=np.int64)
    route = np.empty(n_tasks, dtype=np.int8)
    valid = np.empty(n_tasks, dtype=np.bool_)
    raw_position_error = np.empty(n_tasks, dtype=np.float32)
    generator_s = projection_s = controller_s = 0.0

    for start in range(0, n_tasks, args.batch_size):
        end = min(start + args.batch_size, n_tasks)
        real = end - start
        chunk_rows = rows[start:end]
        if real < args.batch_size:
            chunk_rows = torch.cat([
                chunk_rows,
                chunk_rows[-1:].expand(args.batch_size - real)])
        batch = dataset.batch.index_select(chunk_rows).to(
            device, dtype=env.kin.dtype)
        fallback_q = batch.q0[:, dataset.fallback_index]
        task = direct_seed_task(
            batch.p0, batch.line_dir, batch.n_target)

        _sync(device)
        before = time.perf_counter()
        with torch.no_grad():
            q_raw = model(task)
        _sync(device)
        generator_s += time.perf_counter() - before

        before = time.perf_counter()
        result = route_generated_seed(
            env.kin, env.collision, q_raw,
            batch.p0, batch.line_dir, batch.n_target,
            fallback_q, projection_config)
        _sync(device)
        projection_s += time.perf_counter() - before
        if not bool(result.valid[:real].all()):
            bad = torch.nonzero(
                ~result.valid[:real], as_tuple=False).flatten()
            raise RuntimeError(
                'fail-closed direct seed routing produced invalid tasks; '
                f'chunk rows={bad.cpu().tolist()}')

        one = SeedCandidateBatch(
            q0=result.q.unsqueeze(1),
            p0=batch.p0,
            line_dir=batch.line_dir,
            n_target=batch.n_target,
            valid=result.valid.unsqueeze(1))
        before = time.perf_counter()
        rollout = rollout_selected_seeds(
            env, one,
            torch.zeros(args.batch_size, device=device, dtype=torch.long),
            controller, gamma=gamma)
        _sync(device)
        controller_s += time.perf_counter() - before

        progress[start:end] = rollout.progress_m[:real].cpu().numpy()
        episode_len[start:end] = rollout.episode_len[:real].cpu().numpy()
        route[start:end] = result.route[:real].cpu().numpy()
        valid[start:end] = result.valid[:real].cpu().numpy()
        raw_position_error[start:end] = (
            result.raw.position_error_m[:real].cpu().numpy())
        print(
            f'[direct-seed-eval] {end}/{n_tasks} '
            f'direct={(route[:end] == 0).mean() * 100:.1f}%',
            flush=True)

    fallback_progress = oracle_progress = None
    if args.teacher_returns is not None:
        fallback_progress, oracle_progress = _load_teacher_returns(
            Path(args.teacher_returns), dataset, rows)
    reference_progress = None
    if args.reference_eval is not None:
        reference_progress = _load_reference_progress(
            Path(args.reference_eval),
            dataset.task_indices[rows].numpy())
    report = summarize_direct_seed(
        progress, route, valid,
        fallback_progress_m=fallback_progress,
        pool_oracle_progress_m=oracle_progress,
        reference_progress_m=reference_progress)
    report['raw_seed_position_error_m'] = {
        'mean': float(raw_position_error.mean()),
        'p50': float(np.percentile(raw_position_error, 50)),
        'p90': float(np.percentile(raw_position_error, 90)),
        'p95': float(np.percentile(raw_position_error, 95)),
    }
    unchanged_artifacts = {
        checkpoint_path: checkpoint_sha256,
        candidates_path: candidates_sha256,
        controller_agent_path: controller_agent_sha256,
        controller_config_path: controller_config_sha256,
    }
    for artifact_path, expected_sha256 in unchanged_artifacts.items():
        if _file_sha256(artifact_path) != expected_sha256:
            raise RuntimeError(
                f'evaluation artifact changed during execution: '
                f'{artifact_path}')
    report['fallback_strict_filter'] = fallback_filter_report
    report['artifacts'] = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_format': model_payload.get('format'),
        'candidates': str(candidates_path),
        'candidates_sha256': candidates_sha256,
        'controller_dir': str(controller_dir),
        'controller_agent_sha256': controller_agent_sha256,
        'controller_config_sha256': controller_config_sha256,
        'projection_config': dataclasses.asdict(projection_config),
    }
    report['timing'] = {
        'generator_total_s': generator_s,
        'projection_and_gate_total_s': projection_s,
        'controller_total_s': controller_s,
        'generator_ms_per_task': generator_s / n_tasks * 1e3,
        'projection_and_gate_ms_per_task': projection_s / n_tasks * 1e3,
        'controller_ms_per_task': controller_s / n_tasks * 1e3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        task_indices=dataset.task_indices[rows].numpy(),
        source_row_index=rows.numpy(),
        excluded_source_row_index=excluded_rows.numpy(),
        excluded_task_indices=dataset.task_indices[excluded_rows].numpy(),
        progress_m=progress,
        episode_len=episode_len,
        route=route,
        valid=valid,
        raw_position_error_m=raw_position_error)
    output.with_suffix('.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()


__all__ = [
    'summarize_direct_seed',
]
