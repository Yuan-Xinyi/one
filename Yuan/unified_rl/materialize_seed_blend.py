"""Publish a calibrated static selector obtained by weight interpolation.

This is a training-time model-selection utility for the backward half of a
bidirectional update.  It linearly interpolates two compatible seed-policy
ensembles, recalibrates their conservative first-valid gate against an
exhaustive return cache collected with the requested controller, and emits an
ordinary ``round_complete`` checkpoint.  Deployment remains exactly one
static seed decision followed by one controller rollout; no controller probe
or model rollout is embedded in the published artifact.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import torch
import yaml

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    resolve_controller_dir,
)
from Yuan.unified_rl.offline_seed_ensemble_train import (
    _build_features,
    _calibrate_final,
    _report_split,
)
from Yuan.unified_rl.offline_seed_train import (
    _assert_artifact_unchanged,
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
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    SEED_ENSEMBLE_AGGREGATION,
    infer_seed_policy_config,
    seed_policy_ensemble_states,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


_SPLIT_LOCAL_KEYS = (
    'offline_ensemble_fit_local_indices',
    'offline_ensemble_model_select_local_indices',
    'offline_ensemble_calibration_local_indices',
)
_SPLIT_TASK_KEYS = (
    'offline_ensemble_fit_task_indices',
    'offline_ensemble_model_select_task_indices',
    'offline_ensemble_calibration_task_indices',
)
_SOURCE_TASK_KEYS = (
    'train_indices',
    'validation_indices',
    'train_task_indices',
    'validation_task_indices',
    'train_valid_mask',
    'validation_valid_mask',
)


def _resolve_unified_checkpoint(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_dir():
        resolved = (resolved / 'unified.pt').resolve(strict=True)
    if not resolved.is_file() or resolved.name != 'unified.pt':
        raise ValueError(f'{label} must name a run directory or unified.pt')
    return resolved


def _load_checkpoint(path: Path, *, label: str) -> dict[str, Any]:
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f'{label} must contain a checkpoint dictionary')
    require_checkpoint_format_version(value, 4, kind=label)
    return value


def _blend_state_dicts(
    base: Mapping[str, Any],
    updated: Mapping[str, Any],
    alpha: float,
    *,
    label: str = 'seed-policy member',
) -> dict[str, torch.Tensor]:
    """Strictly interpolate floating tensors and preserve equal buffers."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError('blend alpha must be finite and in [0,1]')
    if set(base) != set(updated):
        missing = sorted(set(base) - set(updated))
        extra = sorted(set(updated) - set(base))
        raise ValueError(
            f'{label} state keys differ: missing={missing}, extra={extra}')
    result: dict[str, torch.Tensor] = {}
    for key in base:
        left = base[key]
        right = updated[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f'{label} state {key!r} must contain tensors')
        left = left.detach().cpu()
        right = right.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(
                f'{label} state {key!r} shape/dtype differs: '
                f'{tuple(left.shape)}/{left.dtype} versus '
                f'{tuple(right.shape)}/{right.dtype}')
        if torch.is_floating_point(left):
            if (not bool(torch.isfinite(left).all().item())
                    or not bool(torch.isfinite(right).all().item())):
                raise ValueError(
                    f'{label} floating state {key!r} must be finite')
            # Compute in float64 to make the interpolation independent of the
            # source tensor device, then cast once to the recorded dtype.
            mixed = torch.lerp(left.double(), right.double(), alpha).to(
                dtype=left.dtype)
            if not bool(torch.isfinite(mixed).all().item()):
                raise ValueError(
                    f'{label} blended state {key!r} is not finite')
            result[key] = mixed.contiguous()
        else:
            if not torch.equal(left, right):
                raise ValueError(
                    f'{label} non-floating state {key!r} differs')
            result[key] = left.clone().contiguous()
    return result


def _blend_ensemble_states(
    base_checkpoint: Mapping[str, Any],
    updated_checkpoint: Mapping[str, Any],
    alpha: float,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Validate ensemble metadata/architecture and blend member-by-member."""
    base_data = seed_policy_ensemble_states(base_checkpoint)
    updated_data = seed_policy_ensemble_states(updated_checkpoint)
    if base_data is None or updated_data is None:
        raise ValueError('both selector checkpoints must contain an ensemble')
    base_states, base_metadata = base_data
    updated_states, updated_metadata = updated_data
    base_config = infer_seed_policy_config(base_checkpoint)
    updated_config = infer_seed_policy_config(updated_checkpoint)
    if base_config != updated_config:
        raise ValueError('base and updated selector architectures differ')
    if dict(base_metadata) != dict(updated_metadata):
        raise ValueError('base and updated ensemble metadata differ')
    if base_metadata['aggregation'] != SEED_ENSEMBLE_AGGREGATION:
        raise ValueError('selector ensemble aggregation is unsupported')
    if len(base_states) != len(updated_states):
        raise ValueError('base and updated ensemble member counts differ')
    states = [
        _blend_state_dicts(
            base_state, updated_state, alpha,
            label=f'seed-policy member {index}')
        for index, (base_state, updated_state) in enumerate(
            zip(base_states, updated_states))
    ]
    return states, base_config.to_dict()


def _members_from_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    architecture: Mapping[str, Any],
    device: torch.device,
) -> list[CandidateSeedActorCritic]:
    members = []
    for index, state in enumerate(states):
        member = CandidateSeedActorCritic(**dict(architecture)).to(device)
        try:
            member.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                f'invalid blended seed-policy member {index}: {error}') from error
        member.eval()
        members.append(member)
    return members


def _require_equal_tensor(
    left: Mapping[str, Any], right: Mapping[str, Any], key: str, *, label: str,
) -> None:
    if key not in left or key not in right:
        raise ValueError(f'{label} is missing required split field {key!r}')
    left_value = torch.as_tensor(left[key]).cpu()
    right_value = torch.as_tensor(right[key]).cpu()
    if left_value.dtype != right_value.dtype or not torch.equal(
            left_value, right_value):
        raise ValueError(f'{label} differs in immutable field {key!r}')


def _candidate_artifact(
    checkpoint: Mapping[str, Any], *, label: str,
) -> tuple[Path, dict[str, str | int]]:
    provenance = checkpoint.get('provenance')
    if not isinstance(provenance, Mapping):
        raise ValueError(f'{label} provenance must be a mapping')
    record = provenance.get('candidate_cache')
    if not isinstance(record, Mapping) or not all(
            key in record for key in ('path', 'size', 'sha256')):
        raise ValueError(f'{label} lacks complete candidate-cache provenance')
    path = Path(str(record['path'])).expanduser().resolve(strict=True)
    artifact = file_fingerprint(path)
    if (artifact['size'] != int(record['size'])
            or artifact['sha256'] != str(record['sha256'])):
        raise ValueError(f'{label} candidate-cache content has changed')
    return path, artifact


def _validate_selector_identity(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Bind a selector's embedded controller to its sibling run files."""
    for key in ('controller', 'controller_state_sha256',
                'controller_run_config_sha256', 'controller_config'):
        if key not in checkpoint:
            raise ValueError(f'{label} is missing {key!r}')
    embedded = checkpoint['controller']
    if not isinstance(embedded, dict):
        raise ValueError(f'{label} controller must be a state dictionary')
    embedded_hash = state_dict_fingerprint(embedded)
    if embedded_hash != checkpoint['controller_state_sha256']:
        raise ValueError(f'{label} embedded controller hash is inconsistent')
    run_dir = checkpoint_path.parent
    artifact = controller_fingerprint(run_dir)
    agent = torch.load(
        run_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if not isinstance(agent, dict) or state_dict_fingerprint(agent) != embedded_hash:
        raise ValueError(f'{label} agent.pt differs from embedded controller')
    if (artifact['config']['sha256']
            != checkpoint['controller_run_config_sha256']):
        raise ValueError(f'{label} config.yaml hash is inconsistent')
    effective = dataclasses.asdict(
        ppo_config_from_run(load_run_config(run_dir)))
    if checkpoint['controller_config'] != effective:
        raise ValueError(f'{label} PPO semantics differ from config.yaml')
    return artifact


def _validate_shared_splits(
    base: Mapping[str, Any],
    updated: Mapping[str, Any],
    source: Mapping[str, Any],
    task_fingerprints: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Require identical source tasks and a geometry-disjoint 3-way split."""
    for key in (*_SOURCE_TASK_KEYS, *_SPLIT_LOCAL_KEYS, *_SPLIT_TASK_KEYS):
        _require_equal_tensor(base, updated, key, label='base/updated selectors')
        _require_equal_tensor(base, source, key, label='selector/controller source')
    for key in ('split_mode', 'seed_policy_feature_schema',
                'seed_include_directional_dynamics'):
        if key not in base or key not in updated or key not in source:
            raise ValueError(f'checkpoints are missing selector field {key!r}')
        if not (base[key] == updated[key] == source[key]):
            raise ValueError(f'checkpoints differ in selector field {key!r}')
    if source['split_mode'] != 'task-geometry-grouped-v1':
        raise ValueError('seed blending requires task-geometry-grouped-v1')
    if (source['seed_policy_feature_schema']
            != 'initial-observation-ray-logmanip-directional-45d-v1'
            or not bool(source['seed_include_directional_dynamics'])):
        raise ValueError('seed blending requires the directional 45-D schema')

    indices = tuple(
        torch.as_tensor(source[key], dtype=torch.long).cpu()
        for key in _SPLIT_LOCAL_KEYS)
    n_train = int(torch.as_tensor(source['train_task_indices']).numel())
    combined = torch.cat(indices)
    if (any(index.ndim != 1 or index.numel() < 1 for index in indices)
            or combined.numel() != n_train
            or not torch.equal(
                combined.sort().values, torch.arange(n_train))):
        raise ValueError('ensemble split indices do not partition train rows')
    if len(task_fingerprints) != n_train:
        raise ValueError('return-cache fingerprint count differs from train rows')
    groups = [
        {task_fingerprints[int(row)] for row in index.tolist()}
        for index in indices
    ]
    if any(groups[i] & groups[j]
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('ensemble fit/model/calibration geometries overlap')

    train_tasks = torch.as_tensor(source['train_task_indices']).cpu()
    for local, task_key in zip(indices, _SPLIT_TASK_KEYS):
        recorded = torch.as_tensor(source[task_key]).cpu()
        if recorded.dtype != train_tasks.dtype or not torch.equal(
                recorded, train_tasks[local]):
            raise ValueError(f'source field {task_key!r} is inconsistent')
    return indices


def _fingerprint_sequence(values: Sequence[str]) -> dict[str, str | int]:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode('ascii')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return {'count': len(values), 'sha256': digest.hexdigest()}


def _optimizer_hyperparameters(checkpoint: Mapping[str, Any]
                               ) -> tuple[float, float]:
    settings = checkpoint.get('offline_seed_ensemble_provenance', {}).get(
        'settings', {})
    learning_rate = float(settings.get('learning_rate', 1e-4))
    weight_decay = float(settings.get('weight_decay', 0.0))
    if (not math.isfinite(learning_rate) or learning_rate <= 0.0
            or not math.isfinite(weight_decay) or weight_decay < 0.0):
        raise ValueError('updated selector optimizer hyperparameters are invalid')
    return learning_rate, weight_decay


def _publish(
    out_dir: Path,
    *,
    updated: Mapping[str, Any],
    controller_source: Mapping[str, Any],
    controller_dir: Path,
    members: Sequence[CandidateSeedActorCritic],
    deployment: Mapping[str, Any],
    calibration: Mapping[str, Any],
    model_report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    device: torch.device,
) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(mode=0o755)
    except FileExistsError as error:
        raise FileExistsError(
            f'refusing to overwrite output directory: {out_dir}') from error

    states = [
        {key: value.detach().cpu().clone()
         for key, value in member.state_dict().items()}
        for member in members
    ]
    architecture = members[0].architecture
    ensemble_metadata = copy.deepcopy(updated['seed_ensemble'])
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
        'seed_deployment': copy.deepcopy(dict(deployment)),
        'offline_seed_selector': 'seed-policy-linear-blend-v1',
        'seed_policy_blend': {
            'format': 'static-seed-policy-linear-blend-v1',
            'alpha': float(provenance['alpha']),
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

    learning_rate, weight_decay = _optimizer_hyperparameters(updated)
    fresh_optimizers = [
        torch.optim.AdamW(
            member.parameters(), lr=learning_rate,
            weight_decay=weight_decay).state_dict()
        for member in members
    ]
    result = copy.deepcopy(dict(updated))
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
        'seed_policy': copy.deepcopy(states[0]),
        'seed_policy_ensemble': copy.deepcopy(states),
        'seed_ensemble': copy.deepcopy(ensemble_metadata),
        'seed_ensemble_inference_only': True,
        'seed_architecture': copy.deepcopy(architecture),
        'feature_dim': int(architecture['feature_dim']),
        'hidden_dim': int(architecture['hidden_dim']),
        'seed_policy_feature_schema': (
            'initial-observation-ray-logmanip-directional-45d-v1'),
        'seed_include_directional_dynamics': True,
        'seed_deployment': copy.deepcopy(dict(deployment)),
        'seed_deployment_calibration': copy.deepcopy(dict(calibration)),
        'seed_optimizer': copy.deepcopy(fresh_optimizers[0]),
        'offline_seed_ensemble_optimizers': copy.deepcopy(fresh_optimizers),
        'seed_policy_blend': copy.deepcopy(dict(provenance)),
        'seed_policy_blend_model_report': copy.deepcopy(dict(model_report)),
    })
    combined_provenance = copy.deepcopy(result['provenance'])
    combined_provenance['seed_policy_blend'] = copy.deepcopy(dict(provenance))
    result['provenance'] = combined_provenance
    result.update(global_rng_state(device))

    copied_agent = torch.load(
        out_dir / 'agent.pt', map_location='cpu', weights_only=True)
    if (not isinstance(copied_agent, dict)
            or state_dict_fingerprint(copied_agent)
            != result['controller_state_sha256']):
        raise RuntimeError('published agent.pt differs from controller source')
    with open(out_dir / 'unified.pt', 'xb') as stream:
        torch.save(result, stream)
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
            'Blend S0/S1 selector weights and recalibrate a static one-seed '
            'deployment against exhaustive returns from controller C1.'))
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--updated-checkpoint', required=True)
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--return-cache', required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--feature-chunk-size', type=int, default=4096)
    parser.add_argument('--calibration-z', type=float, default=1.96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
        raise ValueError('--alpha must be finite and in [0,1]')
    if args.batch_size < 1 or args.feature_chunk_size < 1:
        raise ValueError('batch and feature chunk sizes must be positive')
    if not math.isfinite(args.calibration_z) or args.calibration_z <= 0.0:
        raise ValueError('--calibration-z must be positive')

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
        _, selector_candidate_artifact = _candidate_artifact(
            checkpoint, label=label)
        if (selector_candidate_artifact['size'] != candidate_artifact['size']
                or selector_candidate_artifact['sha256']
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
    if (updated['controller_state_sha256'] != controller_state_sha256
            or state_dict_fingerprint(updated['controller'])
            != controller_state_sha256):
        raise ValueError(
            'updated selector was not trained against the requested C1')
    if updated['controller_config'] != effective_config:
        raise ValueError('updated selector PPO semantics differ from C1')

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

    blended_states, architecture = _blend_ensemble_states(
        base, updated, args.alpha)
    if int(architecture['feature_dim']) != 45:
        raise ValueError('seed blend currently requires 45-D selector features')
    members = _members_from_states(blended_states, architecture, device)
    print(
        '[seed-blend] building directional features and recalibrating the '
        f'static gate: alpha={args.alpha:g}', flush=True)
    features = _build_features(
        env.kin, train_dataset, args.feature_chunk_size)
    model_report = _report_split(
        members, features, cached.valid, cached.progress_m, model_indices,
        cached.task_fingerprints, batch_size=args.batch_size, device=device)
    deployment, calibration = _calibrate_final(
        members, features, cached.valid, cached.progress_m, model_indices,
        calibration_indices, cached.task_fingerprints,
        batch_size=args.batch_size, device=device,
        confidence_z=args.calibration_z)

    # Every input is content-addressed before the expensive feature pass and
    # checked again immediately before publication.
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
    # Selector sibling identities matter too: they certify that each embedded
    # controller came from the recorded run rather than a detached state dict.
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

    provenance = {
        'format': 'static-seed-policy-linear-blend-v1',
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
        'alpha': float(args.alpha),
        'architecture': copy.deepcopy(architecture),
        'ensemble_size': len(members),
        'ensemble_member_state_sha256': [
            state_dict_fingerprint(state) for state in blended_states],
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
        'calibration_z': float(args.calibration_z),
        'seed': int(args.seed),
        'device': device_identity(device),
        'optimizer_reset': (
            'fresh-empty-adamw-after-parameter-interpolation-v1'),
        'deployment': 'one-static-seed-one-controller-rollout-v1',
        'inference_controller_probes': 0,
        'inference_model_rollouts': 0,
    }
    _publish(
        out_dir, updated=updated, controller_source=controller_source,
        controller_dir=controller_dir, members=members,
        deployment=deployment, calibration=calibration,
        model_report=model_report, provenance=provenance, device=device)
    selected = calibration['selected']
    print(
        '[seed-blend] model gain='
        f'{float(model_report["mean_gain_m"]):+.6f} m; '
        f'gate={float(deployment["threshold"]):.9g}; '
        f'calibration gain={float(selected["mean_gain"]):+.6f} m; '
        f'LCB={float(selected["lower_bound"]):+.6f} m; saved -> {out_dir}',
        flush=True)


if __name__ == '__main__':
    main()
