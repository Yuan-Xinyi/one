"""Select and publish a static actor-Q seed deployment under controller C1.

The selector network is evaluated exactly once at deployment.  Its already
available ensemble actor and feasibility outputs are fused as

``actor_log_probability + weight * feasibility / 0.01``.

Every candidate weight and its abstention threshold are selected only on the
recorded source model split.  The disjoint calibration split is read once by
this materializer, after the complete rule has been frozen, and can only
approve it or roll the selector back to S0.  This does not recharacterize its
historical use by older selector calibration.  Neither validation nor
external/sealed tasks are read.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_run_config,
    ppo_config_from_run,
    resolve_controller_dir,
)
from Yuan.unified_rl.materialize_seed_blend import (
    _SPLIT_LOCAL_KEYS,
    _candidate_artifact,
    _fingerprint_sequence,
    _load_checkpoint,
    _members_from_states,
    _resolve_unified_checkpoint,
    _validate_selector_identity,
    _validate_shared_splits,
)
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.offline_seed_train import (
    _assert_artifact_unchanged,
    _calibrate_one_head,
    _validate_source_checkpoint,
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
from Yuan.unified_rl.seed_deployment import (
    SeedDeploymentConfig,
    deployment_config_from_checkpoint,
    select_seed_deployment,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    CandidateSeedPolicyEnsemble,
    SEED_ENSEMBLE_AGGREGATION,
    infer_seed_policy_config,
    seed_policy_ensemble_states,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


ACTOR_Q_WEIGHT_GRID = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.75, 1.0,
)
ACTOR_Q_SCALE_M = 0.01
CONFIDENCE_Z = 1.96
HARM_THRESHOLD_M = 0.001
MAX_GEOMETRY_HARM_RATE = 0.06


def _copy_ensemble_states(
    checkpoint: Mapping[str, Any], *, label: str,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any], dict[str, Any]]:
    data = seed_policy_ensemble_states(checkpoint)
    if data is None:
        raise ValueError(f'{label} must contain a seed-policy ensemble')
    source_states, metadata = data
    if metadata['aggregation'] != SEED_ENSEMBLE_AGGREGATION:
        raise ValueError(f'{label} uses an unsupported ensemble aggregation')
    states = []
    for member_index, state in enumerate(source_states):
        copied = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                raise ValueError(
                    f'{label} member {member_index} field {key!r} is not a tensor')
            copied[key] = value.detach().cpu().clone().contiguous()
        states.append(copied)
    architecture = infer_seed_policy_config(checkpoint).to_dict()
    return states, copy.deepcopy(dict(metadata)), architecture


def _validate_ensemble_pair(
    base: Mapping[str, Any], updated: Mapping[str, Any],
) -> tuple[
        list[dict[str, torch.Tensor]],
        list[dict[str, torch.Tensor]],
        dict[str, Any],
        dict[str, Any],
]:
    base_states, base_metadata, base_architecture = _copy_ensemble_states(
        base, label='base selector')
    updated_states, updated_metadata, updated_architecture = (
        _copy_ensemble_states(updated, label='updated selector'))
    if base_metadata != updated_metadata:
        raise ValueError('base and updated ensemble metadata differ')
    if base_architecture != updated_architecture:
        raise ValueError('base and updated selector architectures differ')
    if len(base_states) != len(updated_states):
        raise ValueError('base and updated ensemble sizes differ')
    for member_index, (base_state, updated_state) in enumerate(
            zip(base_states, updated_states)):
        if set(base_state) != set(updated_state):
            raise ValueError(
                f'base/updated member {member_index} state keys differ')
        for key in base_state:
            left = base_state[key]
            right = updated_state[key]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(
                    f'base/updated member {member_index} field {key!r} '
                    'shape or dtype differs')
            if (torch.is_floating_point(left)
                    and (not bool(torch.isfinite(left).all().item())
                         or not bool(torch.isfinite(right).all().item()))):
                raise ValueError(
                    f'base/updated member {member_index} field {key!r} '
                    'contains non-finite parameters')
    return base_states, updated_states, base_metadata, base_architecture


@torch.no_grad()
def _ensemble_outputs(
    members: Sequence[CandidateSeedActorCritic],
    features: torch.Tensor,
    valid: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensemble = CandidateSeedPolicyEnsemble(list(members)).to(device).eval()
    actor_parts = []
    feasibility_parts = []
    first_parts = []
    for start in range(0, indices.numel(), batch_size):
        local = indices[start:start + batch_size]
        batch_features = features[local].to(device)
        batch_valid = valid[local].to(device)
        distribution, _, feasibility = ensemble.distribution_and_values(
            batch_features, batch_valid)
        actor_parts.append(distribution.logits.cpu())
        feasibility_parts.append(feasibility.cpu())
        first_parts.append(batch_valid.to(torch.int64).argmax(dim=-1).cpu())
    actor = torch.cat(actor_parts).numpy()
    feasibility = torch.cat(feasibility_parts).numpy()
    first = torch.cat(first_parts).numpy()
    selected_valid = valid[indices].numpy()
    if (not np.isfinite(actor[selected_valid]).all()
            or not np.isfinite(feasibility[selected_valid]).all()):
        raise ValueError('selector emitted non-finite scores for valid candidates')
    return actor, feasibility, first


def _actor_q_proposal(
    actor_logits: np.ndarray,
    feasibility: np.ndarray,
    valid: np.ndarray,
    weight: float,
    *,
    scale_m: float = ACTOR_Q_SCALE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (actor_logits.ndim != 2 or feasibility.shape != actor_logits.shape
            or valid.shape != actor_logits.shape or valid.dtype != np.bool_):
        raise ValueError('actor-Q inputs must align in (N,K)')
    if not valid.any(axis=1).all():
        raise ValueError('every actor-Q row must have a valid candidate')
    if (not math.isfinite(weight) or weight < 0.0
            or not math.isfinite(scale_m) or scale_m <= 0.0):
        raise ValueError('actor-Q weight/scale are invalid')
    if (not np.isfinite(actor_logits[valid]).all()
            or not np.isfinite(feasibility[valid]).all()):
        raise ValueError('valid actor-Q scores must be finite')
    # Match the deployment tensor path: arithmetic remains float32 when the
    # policy outputs are float32.  Invalid candidates are masked afterwards.
    combined = actor_logits + weight * feasibility / scale_m
    proposal = np.where(valid, combined, -np.inf).argmax(axis=1)
    first = valid.argmax(axis=1)
    row = np.arange(len(first))
    margin = feasibility[row, proposal] - feasibility[row, first]
    if not np.isfinite(margin).all():
        raise ValueError('actor-Q proposal produced a non-finite Q margin')
    return proposal.astype(np.int64), margin, first.astype(np.int64)


def _geometry_values(
    values: np.ndarray, fingerprints: Sequence[str],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(fingerprints) or len(values) < 1:
        raise ValueError('geometry values/fingerprints are inconsistent')
    lookup: dict[str, int] = {}
    group_id = np.empty(len(values), dtype=np.int64)
    for row, fingerprint in enumerate(fingerprints):
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('geometry fingerprints must be non-empty strings')
        group_id[row] = lookup.setdefault(fingerprint, len(lookup))
    count = np.bincount(group_id, minlength=len(lookup)).astype(np.float64)
    return np.bincount(
        group_id, weights=values, minlength=len(lookup)) / count


def _mean_se_lcb(
    geometry_values: np.ndarray, confidence_z: float,
) -> tuple[float, float, float]:
    values = np.asarray(geometry_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError('geometry statistics require finite one-dimensional data')
    mean = float(values.mean())
    standard_error = float(
        values.std(ddof=1) / math.sqrt(values.size)
        if values.size > 1 else 0.0)
    return mean, standard_error, mean - confidence_z * standard_error


def _fixed_rule_report(
    proposal: np.ndarray,
    margin: np.ndarray,
    first: np.ndarray,
    threshold: float,
    progress_m: np.ndarray,
    valid: np.ndarray,
    baseline_selected: np.ndarray,
    fingerprints: Sequence[str],
    *,
    confidence_z: float = CONFIDENCE_Z,
    harm_threshold_m: float = HARM_THRESHOLD_M,
) -> dict[str, Any]:
    n_rows = len(proposal)
    arrays = (margin, first, baseline_selected)
    if (proposal.shape != (n_rows,)
            or any(value.shape != (n_rows,) for value in arrays)
            or progress_m.ndim != 2 or progress_m.shape != valid.shape
            or progress_m.shape[0] != n_rows or valid.dtype != np.bool_):
        raise ValueError('fixed-rule arrays have inconsistent shapes')
    if (not math.isfinite(threshold) or threshold < 0.0
            or not math.isfinite(confidence_z) or confidence_z <= 0.0
            or not math.isfinite(harm_threshold_m) or harm_threshold_m <= 0.0):
        raise ValueError('fixed-rule thresholds are invalid')
    row = np.arange(n_rows)
    if (not valid[row, proposal].all() or not valid[row, first].all()
            or not valid[row, baseline_selected].all()):
        raise ValueError('fixed rule selected an invalid candidate')
    selected_mask = margin.astype(np.float64) >= threshold
    selected = np.where(selected_mask, proposal, first)
    total_gain = progress_m[row, selected] - progress_m[row, first]
    paired_delta = (
        progress_m[row, selected] - progress_m[row, baseline_selected])
    total_group = _geometry_values(total_gain, fingerprints)
    paired_group = _geometry_values(paired_delta, fingerprints)
    total_mean, total_se, total_lcb = _mean_se_lcb(
        total_group, confidence_z)
    paired_mean, paired_se, paired_lcb = _mean_se_lcb(
        paired_group, confidence_z)
    geometry_harm = _geometry_values(
        (paired_delta < -harm_threshold_m).astype(np.float64), fingerprints)
    geometry_win = _geometry_values(
        (paired_delta > harm_threshold_m).astype(np.float64), fingerprints)
    geometry_changed = _geometry_values(
        (selected != baseline_selected).astype(np.float64), fingerprints)
    geometry_selected = _geometry_values(
        selected_mask.astype(np.float64), fingerprints)
    trim = int(math.floor(0.05 * paired_group.size))
    trimmed = np.sort(paired_group)
    if trim > 0 and 2 * trim < trimmed.size:
        trimmed = trimmed[trim:-trim]
    return {
        'n_rows': int(n_rows),
        'n_geometry_groups': int(paired_group.size),
        'threshold': float(threshold),
        'total_gain_mean_m': total_mean,
        'total_gain_standard_error_m': total_se,
        'total_gain_lower_bound_m': total_lcb,
        'paired_mean_delta_m': paired_mean,
        'paired_standard_error_m': paired_se,
        'paired_lower_bound_m': paired_lcb,
        'paired_trimmed_mean_delta_m': float(trimmed.mean()),
        'paired_clipped_50mm_mean_delta_m': float(
            np.clip(paired_group, -0.05, 0.05).mean()),
        'geometry_harm_rate_gt_1mm': float(geometry_harm.mean()),
        'geometry_win_rate_gt_1mm': float(geometry_win.mean()),
        'row_harm_rate_gt_1mm': float(
            np.mean(paired_delta < -harm_threshold_m)),
        'row_win_rate_gt_1mm': float(
            np.mean(paired_delta > harm_threshold_m)),
        'geometry_changed_rate': float(geometry_changed.mean()),
        'row_changed_rate': float(np.mean(selected != baseline_selected)),
        'geometry_selection_rate': float(geometry_selected.mean()),
        'row_selection_rate': float(selected_mask.mean()),
        'selected_index_sha256': state_dict_fingerprint({
            'selected': torch.from_numpy(selected.astype(np.int64))}),
    }


def _promotion_reasons(
    report: Mapping[str, Any],
    *,
    maximum_geometry_harm_rate: float = MAX_GEOMETRY_HARM_RATE,
) -> list[str]:
    reasons = []
    if float(report['total_gain_lower_bound_m']) < 0.0:
        reasons.append('negative-total-gain-lower-bound')
    if float(report['paired_mean_delta_m']) <= 0.0:
        reasons.append('nonpositive-paired-mean')
    if (float(report['geometry_harm_rate_gt_1mm'])
            > maximum_geometry_harm_rate):
        reasons.append('geometry-harm-rate-exceeds-limit')
    return reasons


def _choose_model_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [candidate for candidate in candidates
                if bool(candidate['eligible'])]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            float(candidate['model_report']['paired_mean_delta_m']),
            float(candidate['model_report']['paired_lower_bound_m']),
            -float(candidate['model_report'][
                'geometry_harm_rate_gt_1mm']),
            -float(candidate['weight']),
        ))


def _base_deployment_indices(
    actor_logits: np.ndarray,
    feasibility: np.ndarray,
    valid: np.ndarray,
    config: SeedDeploymentConfig,
) -> np.ndarray:
    decision = select_seed_deployment(
        torch.from_numpy(actor_logits), torch.from_numpy(feasibility),
        torch.from_numpy(valid), config)
    return decision.selected_index.numpy().astype(np.int64, copy=False)


def _publish(
    out_dir: Path,
    *,
    selector_source: Mapping[str, Any],
    controller_source: Mapping[str, Any],
    controller_dir: Path,
    states: Sequence[Mapping[str, torch.Tensor]],
    ensemble_metadata: Mapping[str, Any],
    architecture: Mapping[str, Any],
    deployment: Mapping[str, Any],
    deployment_calibration: Mapping[str, Any],
    materialization: Mapping[str, Any],
    device: torch.device,
) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(mode=0o755)
    except FileExistsError as error:
        raise FileExistsError(
            f'refusing to overwrite output directory: {out_dir}') from error
    copied_states = [
        {key: value.detach().cpu().clone().contiguous()
         for key, value in state.items()}
        for state in states
    ]
    with open(controller_dir / 'agent.pt', 'rb') as source_stream:
        with open(out_dir / 'agent.pt', 'xb') as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())

    output_config = copy.deepcopy(load_run_config(controller_dir))
    output_config.setdefault('unified', {}).update({
        'seed_architecture': copy.deepcopy(dict(architecture)),
        'seed_feature_schema': (
            'initial-observation-ray-logmanip-directional-45d-v1'),
        'seed_ensemble': copy.deepcopy(dict(ensemble_metadata)),
        'seed_deployment': copy.deepcopy(dict(deployment)),
        'offline_seed_selector': (
            'static-actor-q-fusion-v1'
            if bool(materialization['promoted'])
            else 'static-actor-q-rollback-s0-v1'),
        'seed_actor_q_materialization': {
            'format': str(materialization['format']),
            'promoted': bool(materialization['promoted']),
            'selected_weight': materialization['selected_weight'],
            'scale_m': float(materialization['scale_m']),
            'inference': 'one-static-seed-one-controller-rollout-v1',
            'inference_controller_probes': 0,
            'inference_model_rollouts': 0,
        },
    })
    with open(out_dir / 'config.yaml', 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    config_artifact = file_fingerprint(out_dir / 'config.yaml')

    result = copy.deepcopy(dict(selector_source))
    result.update({
        'phase': 'round_complete',
        'controller': copy.deepcopy(controller_source['controller']),
        'controller_optimizer': copy.deepcopy(
            controller_source['controller_optimizer']),
        'controller_config': copy.deepcopy(
            controller_source['controller_config']),
        'controller_kind': controller_source.get('controller_kind', 'pure'),
        'controller_state_sha256': controller_source[
            'controller_state_sha256'],
        'controller_run_config_sha256': config_artifact['sha256'],
        'seed_policy': copy.deepcopy(copied_states[0]),
        'seed_policy_ensemble': copy.deepcopy(copied_states),
        'seed_ensemble': copy.deepcopy(dict(ensemble_metadata)),
        'seed_ensemble_inference_only': True,
        'seed_architecture': copy.deepcopy(dict(architecture)),
        'feature_dim': int(architecture['feature_dim']),
        'hidden_dim': int(architecture['hidden_dim']),
        'seed_policy_feature_schema': (
            'initial-observation-ray-logmanip-directional-45d-v1'),
        'seed_include_directional_dynamics': True,
        'seed_deployment': copy.deepcopy(dict(deployment)),
        'seed_deployment_calibration': copy.deepcopy(
            dict(deployment_calibration)),
        'seed_actor_q_materialization': copy.deepcopy(dict(materialization)),
    })
    provenance = copy.deepcopy(result.get('provenance', {}))
    if not isinstance(provenance, dict):
        raise ValueError('selector provenance must be a mapping')
    provenance['actor_q_materialization'] = copy.deepcopy(
        dict(materialization))
    result['provenance'] = provenance
    result.update(global_rng_state(device))

    copied_agent = torch.load(
        out_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if (not isinstance(copied_agent, dict)
            or state_dict_fingerprint(copied_agent)
            != result['controller_state_sha256']):
        raise RuntimeError('published agent.pt differs from controller source')
    if state_dict_fingerprint(result['seed_policy_ensemble'][0]) != (
            state_dict_fingerprint(copied_states[0])):
        raise RuntimeError('published selector state changed before saving')
    with open(out_dir / 'unified.pt', 'xb') as stream:
        torch.save(result, stream)
        stream.flush()
        os.fsync(stream.fileno())
    published = torch.load(
        out_dir / 'unified.pt', map_location='cpu', weights_only=False)
    if (not isinstance(published, dict)
            or published.get('seed_deployment') != dict(deployment)
            or published.get('controller_state_sha256')
            != controller_source['controller_state_sha256']):
        raise RuntimeError('published actor-Q checkpoint metadata changed')
    published_states = published.get('seed_policy_ensemble')
    if (not isinstance(published_states, (list, tuple))
            or len(published_states) != len(copied_states)
            or any(
                state_dict_fingerprint(actual)
                != state_dict_fingerprint(expected)
                for actual, expected in zip(
                    published_states, copied_states))):
        raise RuntimeError('published actor-Q selector states changed')
    directory_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Select a static actor-Q proposal weight on the source model '
            'split, certify it once on calibration, and bind it to C1.'))
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--updated-checkpoint', required=True)
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--return-cache', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--feature-chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.feature_chunk_size < 1:
        raise ValueError('batch and feature chunk sizes must be positive')

    base_path = _resolve_unified_checkpoint(
        args.base_checkpoint, label='--base-checkpoint')
    updated_path = _resolve_unified_checkpoint(
        args.updated_checkpoint, label='--updated-checkpoint')
    controller_dir = resolve_controller_dir(args.controller_ckpt)
    controller_source_path = (controller_dir / 'unified.pt').resolve(
        strict=True)
    return_path = Path(args.return_cache).expanduser().resolve(strict=True)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if return_path.suffix.lower() != '.npz':
        raise ValueError('--return-cache must name an NPZ file')
    if os.path.lexists(out_dir):
        raise FileExistsError(f'refusing to overwrite output: {out_dir}')
    if len({base_path, updated_path, controller_source_path}) != 3:
        raise ValueError('base, updated, and controller source must be distinct')

    artifacts = {
        'base_checkpoint': file_fingerprint(base_path),
        'updated_checkpoint': file_fingerprint(updated_path),
        'controller_source_checkpoint': file_fingerprint(
            controller_source_path),
        'return_cache': file_fingerprint(return_path),
    }
    base = _load_checkpoint(base_path, label='base selector')
    updated = _load_checkpoint(updated_path, label='updated selector')
    controller_source = _load_checkpoint(
        controller_source_path, label='controller source')
    base_controller_artifact = _validate_selector_identity(
        base, base_path, label='base selector')
    updated_controller_artifact = _validate_selector_identity(
        updated, updated_path, label='updated selector')
    controller_artifact = controller_fingerprint(controller_dir)
    candidate_path, candidate_artifact = _candidate_artifact(
        controller_source, label='controller source')
    for checkpoint, label in (
            (base, 'base selector'), (updated, 'updated selector')):
        _, selector_candidate = _candidate_artifact(checkpoint, label=label)
        if (selector_candidate['size'] != candidate_artifact['size']
                or selector_candidate['sha256']
                != candidate_artifact['sha256']):
            raise ValueError(f'{label} uses a different candidate cache')

    controller_agent = torch.load(
        controller_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if not isinstance(controller_agent, dict):
        raise ValueError('controller agent.pt must contain a state dictionary')
    controller_state_sha256 = state_dict_fingerprint(controller_agent)
    effective_config = dataclasses.asdict(
        ppo_config_from_run(load_run_config(controller_dir)))
    objective, gamma = _validate_source_checkpoint(
        controller_source, artifacts['controller_source_checkpoint'],
        candidate_artifact, controller_artifact, controller_state_sha256,
        effective_config)
    # ``objective`` binds the cached controller-return semantics (for these
    # runs it is typically undiscounted return).  Actor-Q promotion itself is
    # deliberately evaluated in progress metres, matching the selector label.
    if controller_source.get('seed_selector_objective') != 'progress_m':
        raise ValueError(
            'actor-Q materialization requires progress_m selector objective')
    if (updated['controller_state_sha256'] != controller_state_sha256
            or state_dict_fingerprint(updated['controller'])
            != controller_state_sha256):
        raise ValueError('updated selector was not relabeled against C1')
    if updated['controller_config'] != effective_config:
        raise ValueError('updated selector PPO semantics differ from C1')

    base_states, updated_states, ensemble_metadata, architecture = (
        _validate_ensemble_pair(base, updated))
    if int(architecture['feature_dim']) != 45:
        raise ValueError('actor-Q materialization requires the 45-D schema')

    seed_global_rng(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    env = build_env_from_run(controller_dir, 1, device)
    source_dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    source_dataset, physical_stats = validate_cached_dataset(
        source_dataset, env.kin, env.collision,
        chunk_size=args.feature_chunk_size, cone_deg=env.cfg.cone_deg)
    train_dataset = source_dataset.select_source_tasks(
        torch.as_tensor(controller_source['train_task_indices']).cpu())
    validation_dataset = source_dataset.select_source_tasks(
        torch.as_tensor(controller_source['validation_task_indices']).cpu())
    assert_same_valid_mask(
        train_dataset, controller_source['train_valid_mask'], label='training')
    assert_same_valid_mask(
        validation_dataset, controller_source['validation_valid_mask'],
        label='validation')
    if set(train_dataset.task_fingerprints) & set(
            validation_dataset.task_fingerprints):
        raise ValueError('source train/validation task geometries overlap')
    cached = load_return_cache(
        return_path, source=controller_source,
        source_artifact=artifacts['controller_source_checkpoint'],
        candidate_artifact=candidate_artifact,
        controller_artifact=controller_artifact,
        controller_state_sha256=controller_state_sha256,
        objective=objective, gamma=gamma, train_dataset=train_dataset)
    if cached.artifact != artifacts['return_cache']:
        raise RuntimeError('return cache changed while loading')
    fit_indices, model_indices, calibration_indices = _validate_shared_splits(
        base, updated, controller_source, cached.task_fingerprints)

    print(
        '[actor-q] building directional features; weight/threshold selection '
        'uses source model split only', flush=True)
    features = _build_features(
        env.kin, train_dataset, args.feature_chunk_size)
    base_members = _members_from_states(base_states, architecture, device)
    updated_members = _members_from_states(
        updated_states, architecture, device)
    selection_indices = torch.cat([model_indices, calibration_indices])
    base_actor, base_q, _ = _ensemble_outputs(
        base_members, features, cached.valid, selection_indices,
        batch_size=args.batch_size, device=device)
    updated_actor, updated_q, _ = _ensemble_outputs(
        updated_members, features, cached.valid, selection_indices,
        batch_size=args.batch_size, device=device)
    n_model = model_indices.numel()
    base_config = deployment_config_from_checkpoint(base)

    def partition(values: np.ndarray, calibration: bool) -> np.ndarray:
        return values[n_model:] if calibration else values[:n_model]

    model_valid = cached.valid[model_indices].numpy()
    model_progress = cached.progress_m[model_indices].numpy()
    model_fingerprints = [cached.task_fingerprints[int(index)]
                          for index in model_indices.tolist()]
    base_model_selected = _base_deployment_indices(
        partition(base_actor, False), partition(base_q, False), model_valid,
        base_config)

    candidates = []
    for weight in ACTOR_Q_WEIGHT_GRID:
        proposal, margin, first = _actor_q_proposal(
            partition(updated_actor, False), partition(updated_q, False),
            model_valid, weight)
        threshold_selection = _calibrate_one_head(
            'actor-q', proposal, margin.astype(np.float64), first,
            model_progress, model_valid, model_fingerprints, CONFIDENCE_Z)
        model_report = _fixed_rule_report(
            proposal, margin, first,
            float(threshold_selection['threshold']), model_progress,
            model_valid, base_model_selected, model_fingerprints)
        if not math.isclose(
                model_report['total_gain_lower_bound_m'],
                float(threshold_selection['lower_bound']),
                rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError('actor-Q threshold report is inconsistent')
        reasons = _promotion_reasons(model_report)
        candidate = {
            'weight': float(weight),
            'scale_m': ACTOR_Q_SCALE_M,
            'threshold_selection': copy.deepcopy(threshold_selection),
            'model_report': model_report,
            'eligible': not reasons,
            'ineligible_reasons': reasons,
        }
        candidates.append(candidate)
        print(
            f'[actor-q] w={weight:>4.2f}  '
            f'paired={model_report["paired_mean_delta_m"] * 1e3:+.3f} mm  '
            f'total-LCB={model_report["total_gain_lower_bound_m"]:+.6f} m  '
            f'harm={100 * model_report["geometry_harm_rate_gt_1mm"]:.2f}%  '
            f'eligible={not reasons}', flush=True)

    chosen = _choose_model_candidate(candidates)
    calibration_report = None
    calibration_reasons: list[str] = []
    promoted = False
    if chosen is not None:
        calibration_valid = cached.valid[calibration_indices].numpy()
        calibration_progress = cached.progress_m[calibration_indices].numpy()
        calibration_fingerprints = [
            cached.task_fingerprints[int(index)]
            for index in calibration_indices.tolist()
        ]
        base_calibration_selected = _base_deployment_indices(
            partition(base_actor, True), partition(base_q, True),
            calibration_valid, base_config)
        calibration_proposal, calibration_margin, calibration_first = (
            _actor_q_proposal(
                partition(updated_actor, True), partition(updated_q, True),
                calibration_valid, float(chosen['weight'])))
        # This is the sole calibration access: the model-selected weight and
        # threshold are immutable before these metrics are computed.
        calibration_report = _fixed_rule_report(
            calibration_proposal, calibration_margin, calibration_first,
            float(chosen['threshold_selection']['threshold']),
            calibration_progress, calibration_valid,
            base_calibration_selected, calibration_fingerprints)
        calibration_reasons = _promotion_reasons(calibration_report)
        promoted = not calibration_reasons

    if promoted:
        assert chosen is not None and calibration_report is not None
        deployment_config = SeedDeploymentConfig(
            mode='conservative', proposal_head='actor-q',
            threshold=float(chosen['threshold_selection']['threshold']),
            comparison='ge', proposal_q_weight=float(chosen['weight']),
            proposal_q_scale_m=ACTOR_Q_SCALE_M)
        selector_source = updated
        selected_states = updated_states
        actual_calibration = {
            'format': 'static-actor-q-fixed-calibration-v1',
            'proposal': (
                'mean-member-log-probability-plus-weighted-mean-feasibility'),
            'gate_score': 'mean-feasibility-proposal-minus-first-valid',
            'fallback': 'first_valid',
            'confidence_z': CONFIDENCE_Z,
            'weight': float(chosen['weight']),
            'scale_m': ACTOR_Q_SCALE_M,
            'threshold': float(chosen['threshold_selection']['threshold']),
            'selected': copy.deepcopy(calibration_report),
            'fixed_rule_certified': True,
            'actor_q_fixed_rule_audit_count': 1,
            'used_for_actor_q_weight_selection': False,
            'historical_selector_calibration_usage_not_recharacterized': True,
        }
    else:
        deployment_config = base_config
        selector_source = base
        selected_states = base_states
        actual_calibration = copy.deepcopy(
            base.get('seed_deployment_calibration', {}))

    materialization = {
        'format': 'static-actor-q-selector-materialization-v1',
        'base_checkpoint': copy.deepcopy(artifacts['base_checkpoint']),
        'updated_checkpoint': copy.deepcopy(artifacts['updated_checkpoint']),
        'controller_source_checkpoint': copy.deepcopy(
            artifacts['controller_source_checkpoint']),
        'return_cache': copy.deepcopy(artifacts['return_cache']),
        'candidate_cache': copy.deepcopy(candidate_artifact),
        'controller': copy.deepcopy(controller_artifact),
        'controller_state_sha256': controller_state_sha256,
        'base_controller_state_sha256': base['controller_state_sha256'],
        'updated_controller_state_sha256': updated[
            'controller_state_sha256'],
        'fixed_weight_grid': list(ACTOR_Q_WEIGHT_GRID),
        'scale_m': ACTOR_Q_SCALE_M,
        'confidence_z': CONFIDENCE_Z,
        'promotion_constraints': {
            'total_gain_lower_bound_m': {'comparison': 'ge', 'value': 0.0},
            'paired_mean_delta_m': {'comparison': 'gt', 'value': 0.0},
            'geometry_harm_rate_gt_1mm': {
                'comparison': 'le', 'value': MAX_GEOMETRY_HARM_RATE},
            'harm_threshold_m': HARM_THRESHOLD_M,
        },
        'model_selection_candidates': candidates,
        'model_selected_weight': (
            float(chosen['weight']) if chosen is not None else None),
        'model_selected_threshold': (
            float(chosen['threshold_selection']['threshold'])
            if chosen is not None else None),
        'calibration_report': copy.deepcopy(calibration_report),
        'calibration_ineligible_reasons': list(calibration_reasons),
        'calibration_used_candidate_count': 1 if chosen is not None else 0,
        'actor_q_fixed_rule_audit_count': 1 if chosen is not None else 0,
        'calibration_used_for_actor_q_weight_selection': False,
        'historical_selector_calibration_usage_not_recharacterized': True,
        'promoted': bool(promoted),
        'selected_weight': (
            float(chosen['weight']) if promoted and chosen is not None else None),
        'rollback_to_exact_s0': not promoted,
        'rollback_reason': (
            None if promoted else (
                'no-model-eligible-candidate'
                if chosen is None else 'fixed-calibration-failed')),
        'architecture': copy.deepcopy(architecture),
        'ensemble_size': len(selected_states),
        'selected_ensemble_member_state_sha256': [
            state_dict_fingerprint(state) for state in selected_states],
        'split_local_indices_sha256': {
            key: state_dict_fingerprint({'indices': value})
            for key, value in zip(_SPLIT_LOCAL_KEYS, (
                fit_indices, model_indices, calibration_indices))
        },
        'train_task_fingerprints': _fingerprint_sequence(
            cached.task_fingerprints),
        'validation_task_fingerprints': _fingerprint_sequence(
            validation_dataset.task_fingerprints),
        'physical_validation': copy.deepcopy(physical_stats),
        'model_selection_data': 'source-model-split-only-v1',
        'calibration_data': 'source-calibration-fixed-rule-once-v1',
        'validation_data_used_for_selection': False,
        'external_data_used_for_selection': False,
        'sealed_data_used_for_selection': False,
        'seed': int(args.seed),
        'device': device_identity(device),
        'deployment': 'one-static-seed-one-controller-rollout-v1',
        'inference_selector_forwards': 1,
        'inference_controller_probes': 0,
        'inference_model_rollouts': 0,
    }

    # Recheck every content-addressed input immediately before publication.
    for label, path in (
            ('base checkpoint', base_path),
            ('updated checkpoint', updated_path),
            ('controller source checkpoint', controller_source_path),
            ('return cache', return_path)):
        _assert_artifact_unchanged(
            label, artifacts[label.replace(' ', '_')], file_fingerprint(path))
    _assert_artifact_unchanged(
        'candidate cache', candidate_artifact, file_fingerprint(candidate_path))
    current_controller = controller_fingerprint(controller_dir)
    for kind in ('agent', 'config'):
        _assert_artifact_unchanged(
            f'controller {kind}', controller_artifact[kind],
            current_controller[kind])
    for label, before, path in (
            ('base selector agent', base_controller_artifact['agent'],
             base_path.parent / 'agent.pt'),
            ('base selector config', base_controller_artifact['config'],
             base_path.parent / 'config.yaml'),
            ('updated selector agent', updated_controller_artifact['agent'],
             updated_path.parent / 'agent.pt'),
            ('updated selector config', updated_controller_artifact['config'],
             updated_path.parent / 'config.yaml')):
        _assert_artifact_unchanged(label, before, file_fingerprint(path))

    _publish(
        out_dir, selector_source=selector_source,
        controller_source=controller_source, controller_dir=controller_dir,
        states=selected_states, ensemble_metadata=ensemble_metadata,
        architecture=architecture, deployment=deployment_config.to_dict(),
        deployment_calibration=actual_calibration,
        materialization=materialization, device=device)
    if promoted:
        print(
            '[actor-q] promoted '
            f'w={float(chosen["weight"]):.2f}, '
            f'threshold={float(chosen["threshold_selection"]["threshold"]):.9g}; '
            f'calibration paired='
            f'{float(calibration_report["paired_mean_delta_m"]) * 1e3:+.3f} mm '
            f'-> {out_dir}', flush=True)
    else:
        print(
            '[actor-q] fixed calibration rejected the update; published '
            f'exact S0 selector with C1 controller -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
