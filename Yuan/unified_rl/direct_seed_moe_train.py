"""Fit a hard-gated single-seed actor from paired real-return targets.

This is a training-only branch-specialisation stage.  The target archive is
built from one deterministic baseline rollout per task and a strictly better
online explorer projection when available.  Deployment remains one network
forward, one hard-selected joint head, at most one IK refinement, and one
controller rollout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from Yuan.unified_rl.checkpoint import atomic_torch_save
from Yuan.unified_rl.direct_seed_rl import (
    DirectSeedEliteMemory,
    DirectSeedPairedArchive,
    DirectSeedRLBatch,
    direct_seed_moe_checkpoint,
    direct_seed_moe_from_actor,
    load_direct_seed_moe_checkpoint,
    load_direct_seed_rl_checkpoint,
    update_direct_seed_moe_advantage,
    update_direct_seed_moe_projection,
)
from Yuan.unified_rl.direct_seed_projection import ROUTE_REFINED
from Yuan.unified_rl.provenance import state_dict_fingerprint
from Yuan.unified_rl.reproducibility import (
    global_rng_state,
    restore_global_rng,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


_RUNNER_FORMAT = 'direct-seed-bidirectional-v1'
_TRAINER_STATE_FORMAT = 'direct-seed-moe-trainer-v1'
_FROZEN_CONTRACT_FORMAT = 'direct-seed-moe-frozen-contract-v1'


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _semantic_sha256(value: Any) -> str:
    """Hash nested checkpoint values independently of torch container bytes."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b'tensor:')
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b'mapping{')
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                digest.update(b'=')
                update(item[key])
            digest.update(b'}')
        elif isinstance(item, (list, tuple)):
            digest.update(b'sequence[')
            for child in item:
                update(child)
            digest.update(b']')
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(
                json.dumps(
                    item, sort_keys=True, allow_nan=False
                ).encode())
        else:
            raise TypeError(
                f'unsupported checkpoint provenance value {type(item)!r}')

    update(value)
    return digest.hexdigest()


def _load_mapping(path: Path, kind: str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f'{kind} checkpoint must be a mapping')
    return payload


def _extract_paired_archive_state(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Accept a bare archive or a complete bidirectional runner checkpoint."""
    if payload.get('format') == _RUNNER_FORMAT:
        state = payload.get('paired_archive')
        if not isinstance(state, Mapping):
            raise ValueError(
                'paired runner checkpoint has no paired_archive state')
        return state, payload
    if payload.get('format') == 'direct-seed-paired-archive-v1':
        return payload, None
    raise ValueError(
        'paired archive must be a direct-seed paired archive or complete '
        'bidirectional runner checkpoint')


def _extract_explorer_elite_state(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Accept a bare elite memory or its complete runner checkpoint."""
    if payload.get('format') == _RUNNER_FORMAT:
        state = payload.get('per_task_elite_memory')
        if not isinstance(state, Mapping):
            raise ValueError(
                'explorer runner checkpoint has no per_task_elite_memory')
        return state, payload
    if payload.get('format') == 'direct-seed-elite-memory-v1':
        return payload, None
    raise ValueError(
        'explorer checkpoint must be a direct-seed elite memory or complete '
        'bidirectional runner checkpoint')


def _load_archive_objects(
    paired_path: Path,
    explorer_path: Path,
) -> tuple[
    DirectSeedPairedArchive,
    DirectSeedEliteMemory,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    paired_payload = _load_mapping(paired_path, 'paired archive')
    paired_state, paired_runner = _extract_paired_archive_state(
        paired_payload)
    task_ids = paired_state.get('task_ids')
    if not torch.is_tensor(task_ids):
        raise ValueError('paired archive checkpoint has no task_ids')
    archive = DirectSeedPairedArchive(task_ids)
    archive.load_state_dict(paired_state)

    explorer_payload = _load_mapping(
        explorer_path, 'explorer')
    elite_state, explorer_runner = _extract_explorer_elite_state(
        explorer_payload)
    elite_task_ids = elite_state.get('task_ids')
    if not torch.is_tensor(elite_task_ids):
        raise ValueError(
            'explorer checkpoint has no per_task_elite_memory task_ids')
    elite = DirectSeedEliteMemory(elite_task_ids)
    elite.load_state_dict(elite_state)
    return (
        archive, elite, paired_state, elite_state,
        paired_runner, explorer_runner,
    )


def _runner_task_identity(
    runner: Mapping[str, Any],
    kind: str,
) -> tuple[list[int], str]:
    task_ids = runner.get('kept_task_indices')
    fingerprint = runner.get('safe_task_fingerprint_list_sha256')
    if (not isinstance(task_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in task_ids)):
        raise ValueError(f'{kind} runner has invalid kept_task_indices')
    if not _is_sha256(fingerprint):
        raise ValueError(
            f'{kind} runner has no safe-task fingerprint')
    return task_ids, fingerprint


def _runner_controller_identity(
    runner: Mapping[str, Any],
    version_field: str,
    kind: str,
) -> tuple[str, int]:
    controller = runner.get('controller')
    if (not isinstance(controller, Mapping)
            or not controller
            or any(not isinstance(name, str) or not torch.is_tensor(value)
                   for name, value in controller.items())):
        raise ValueError(f'{kind} runner has no valid controller state')
    controller_version = runner.get('controller_update_count')
    data_version = runner.get(version_field)
    for name, value in {
            'controller_update_count': controller_version,
            version_field: data_version,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f'{kind} runner {name} must be a non-negative integer')
    if data_version != controller_version:
        raise ValueError(
            f'{kind} runner data/controller versions do not match')
    return state_dict_fingerprint(dict(controller)), controller_version


def _verify_training_provenance(
    *,
    base_payload: Mapping[str, Any],
    base_actor_state_sha256: str,
    archive: DirectSeedPairedArchive,
    paired_state: Mapping[str, Any],
    paired_runner: Mapping[str, Any] | None,
    elite: DirectSeedEliteMemory,
    explorer_runner: Mapping[str, Any] | None,
    baseline_runner: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every semantic dependency behind real-return advantage labels."""
    if baseline_runner.get('format') != _RUNNER_FORMAT:
        raise ValueError(
            'baseline run checkpoint is not a bidirectional runner checkpoint')
    if paired_runner is not None \
            and paired_runner.get('format') != _RUNNER_FORMAT:
        raise ValueError(
            'paired container is not a bidirectional runner checkpoint')
    if explorer_runner is None:
        raise ValueError(
            'verified provenance requires a complete explorer runner '
            'checkpoint, not a bare elite memory')

    baseline_archive = baseline_runner.get('paired_archive')
    if not isinstance(baseline_archive, Mapping):
        raise ValueError(
            'baseline runner has no paired_archive; use '
            '--allow-unverified-provenance only for a legacy ablation')
    paired_semantic_sha256 = _semantic_sha256(paired_state)
    if _semantic_sha256(baseline_archive) != paired_semantic_sha256:
        raise ValueError(
            'paired archive does not equal the baseline runner archive')
    if len(archive) != int(archive.task_ids.numel()):
        raise ValueError(
            'verified paired archive must have full task coverage')

    baseline_task_ids, baseline_task_fingerprint = _runner_task_identity(
        baseline_runner, 'baseline')
    explorer_task_ids, explorer_task_fingerprint = _runner_task_identity(
        explorer_runner, 'explorer')
    archive_task_ids = archive.task_ids.tolist()
    if (baseline_task_ids != archive_task_ids
            or explorer_task_ids != archive_task_ids
            or not torch.equal(elite.task_ids, archive.task_ids)):
        raise ValueError(
            'baseline, explorer, and paired archive task order differs')
    if baseline_task_fingerprint != explorer_task_fingerprint:
        raise ValueError(
            'baseline and explorer safe-task fingerprints differ')

    base_metadata = base_payload.get('metadata')
    base_task_fingerprint = (
        base_metadata.get('safe_task_fingerprint_list_sha256')
        if isinstance(base_metadata, Mapping) else None)
    if base_task_fingerprint != baseline_task_fingerprint:
        raise ValueError(
            'base actor safe-task fingerprint differs from the paired runs')

    baseline_projection = baseline_runner.get('projection_config')
    explorer_projection = explorer_runner.get('projection_config')
    if (not isinstance(baseline_projection, Mapping)
            or not isinstance(explorer_projection, Mapping)
            or dict(baseline_projection) != dict(explorer_projection)):
        raise ValueError(
            'baseline and explorer projection configs differ')

    baseline_controller_sha256, controller_version = (
        _runner_controller_identity(
            baseline_runner,
            'paired_archive_controller_update_count',
            'baseline'))
    explorer_controller_sha256, explorer_controller_version = (
        _runner_controller_identity(
            explorer_runner,
            'per_task_elite_controller_update_count',
            'explorer'))
    if (baseline_controller_sha256 != explorer_controller_sha256
            or controller_version != explorer_controller_version):
        raise ValueError(
            'baseline and explorer controller states/versions differ')

    collection_contract = baseline_runner.get(
        'paired_collection_contract')
    baseline_actor_sha256 = baseline_runner.get(
        'paired_baseline_actor_state_sha256')
    direct_seed = baseline_runner.get('direct_seed')
    direct_metadata = (
        direct_seed.get('metadata')
        if isinstance(direct_seed, Mapping) else None)
    if (not isinstance(collection_contract, Mapping)
            or collection_contract.get('format')
            != 'direct-seed-paired-collection-contract-v1'
            or collection_contract.get('actor_action')
            != 'deterministic-mean'
            or collection_contract.get('deterministic_backward') is not True
            or collection_contract.get(
                'actor_frozen_during_collection') is not True
            or collection_contract.get('task_sampling') != 'cycle'):
        raise ValueError(
            'baseline runner has no verified deterministic/frozen/cycle '
            'collection contract')
    if (not _is_sha256(baseline_actor_sha256)
            or base_actor_state_sha256 != baseline_actor_sha256):
        raise ValueError(
            'base actor state differs from the baseline collection actor')
    if (not isinstance(direct_metadata, Mapping)
            or direct_metadata.get('paired_collection_contract')
            != collection_contract
            or direct_metadata.get(
                'paired_baseline_actor_state_sha256')
            != baseline_actor_sha256):
        raise ValueError(
            'baseline runner paired provenance is internally inconsistent')

    if paired_runner is not None:
        paired_runner_archive = paired_runner.get('paired_archive')
        if (not isinstance(paired_runner_archive, Mapping)
                or _semantic_sha256(paired_runner_archive)
                != paired_semantic_sha256):
            raise ValueError(
                'paired runner container archive differs from the requested '
                'paired archive')
        paired_ids, paired_fingerprint = _runner_task_identity(
            paired_runner, 'paired container')
        if (paired_ids != baseline_task_ids
                or paired_fingerprint != baseline_task_fingerprint
                or paired_runner.get('projection_config')
                != baseline_projection):
            raise ValueError(
                'paired runner container provenance differs from the baseline '
                'runner')
        paired_controller_sha256, paired_controller_version = (
            _runner_controller_identity(
                paired_runner,
                'paired_archive_controller_update_count',
                'paired container'))
        paired_direct_seed = paired_runner.get('direct_seed')
        paired_direct_metadata = (
            paired_direct_seed.get('metadata')
            if isinstance(paired_direct_seed, Mapping) else None)
        if (paired_controller_sha256 != baseline_controller_sha256
                or paired_controller_version != controller_version
                or paired_runner.get('paired_collection_contract')
                != collection_contract
                or paired_runner.get(
                    'paired_baseline_actor_state_sha256')
                != baseline_actor_sha256
                or not isinstance(paired_direct_metadata, Mapping)
                or paired_direct_metadata.get(
                    'paired_collection_contract') != collection_contract
                or paired_direct_metadata.get(
                    'paired_baseline_actor_state_sha256')
                != baseline_actor_sha256):
            raise ValueError(
                'paired runner controller/collection provenance differs from '
                'the baseline runner')

    return {
        'verified': True,
        'mode': 'strict-runner-provenance',
        'paired_archive_state_sha256': paired_semantic_sha256,
        'base_actor_state_sha256': base_actor_state_sha256,
        'baseline_actor_state_sha256': baseline_actor_sha256,
        'controller_state_sha256': baseline_controller_sha256,
        'baseline_controller_update_count': controller_version,
        'explorer_controller_update_count': explorer_controller_version,
        'projection_config': copy.deepcopy(dict(baseline_projection)),
        'safe_task_fingerprint_list_sha256': baseline_task_fingerprint,
        'kept_task_indices': list(baseline_task_ids),
        'paired_collection_contract': copy.deepcopy(
            dict(collection_contract)),
    }


def _parse_save_steps(text: str, updates: int) -> tuple[int, ...]:
    try:
        values = {
            int(value.strip())
            for value in text.split(',')
            if value.strip()
        }
    except ValueError as error:
        raise ValueError('--save-steps must be comma-separated integers') \
            from error
    if any(value < 1 or value > updates for value in values):
        raise ValueError(
            '--save-steps entries must be in [1, --updates]')
    values.add(updates)
    return tuple(sorted(values))


def _build_training_data(
    archive: DirectSeedPairedArchive,
    elite: DirectSeedEliteMemory,
    advantage_margin_m: float,
    objective: str,
) -> tuple[
    DirectSeedRLBatch,
    torch.Tensor | None,
    dict[str, int | float],
]:
    stats = archive.target_stats(elite, advantage_margin_m)
    if objective == 'paired-wta':
        targets = archive.build_targets(elite, advantage_margin_m)
        explorer_selected = None
    elif objective == 'advantage-safe':
        positive = (
            archive.valid
            & elite.valid
            & (
                elite.progress_m
                > archive.progress_m + float(advantage_margin_m))
        )
        index = torch.nonzero(
            archive.valid, as_tuple=False).flatten()
        selected_positive = positive.index_select(0, index)
        baseline_q = archive.q_projected.index_select(0, index)
        explorer_q = elite.q_projected.index_select(0, index)
        projected = torch.where(
            selected_positive.unsqueeze(-1), explorer_q, baseline_q)
        baseline_progress = archive.progress_m.index_select(0, index)
        explorer_progress = elite.progress_m.index_select(0, index)
        progress = torch.where(
            selected_positive, explorer_progress, baseline_progress)
        targets = DirectSeedRLBatch(
            task=archive.task.index_select(0, index),
            q_raw=projected.clone(),
            q_projected=projected,
            fallback_q=torch.zeros_like(projected),
            progress_m=progress,
            route=torch.full(
                (int(index.numel()),), ROUTE_REFINED,
                dtype=torch.int64),
        )
        explorer_selected = selected_positive
        stats = {
            **stats,
            'gate_training_count': int(index.numel()),
            'gate_positive_count': int(torch.count_nonzero(
                selected_positive)),
            'gate_positive_fraction': float(
                selected_positive.float().mean()),
        }
    else:
        raise ValueError(f'unsupported objective {objective!r}')
    if targets.batch_size < 1:
        raise RuntimeError('paired archive produced no legal targets')
    return targets, explorer_selected, stats


def _load_training_data(
    paired_path: Path,
    explorer_path: Path,
    advantage_margin_m: float,
    objective: str,
) -> tuple[
    DirectSeedRLBatch,
    torch.Tensor | None,
    dict[str, int | float],
]:
    """Compatibility wrapper accepting bare or complete runner checkpoints."""
    archive, elite, _, _, _, _ = _load_archive_objects(
        paired_path, explorer_path)
    return _build_training_data(
        archive, elite, advantage_margin_m, objective)


def _select_batch(
    source: DirectSeedRLBatch,
    index: torch.Tensor,
    device: torch.device,
) -> DirectSeedRLBatch:
    if index.device.type != 'cpu' or index.dtype != torch.int64:
        raise ValueError('training indices must be CPU int64')
    return DirectSeedRLBatch(
        task=source.task.index_select(0, index),
        q_raw=source.q_raw.index_select(0, index),
        q_projected=source.q_projected.index_select(0, index),
        fallback_q=source.fallback_q.index_select(0, index),
        progress_m=source.progress_m.index_select(0, index),
        route=source.route.index_select(0, index),
    ).to(device=device, dtype=torch.float32)


@torch.no_grad()
def _diagnostics(
    actor,
    targets: DirectSeedRLBatch,
    explorer_selected: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | list[float]]:
    actor.eval()
    winner_count = torch.zeros(
        actor.config.n_experts, dtype=torch.int64)
    gate_count = torch.zeros_like(winner_count)
    winner_correct = 0
    rows = 0
    oracle_loss = 0.0
    deployed_loss = 0.0
    classifier_true_positive = 0
    classifier_false_positive = 0
    classifier_false_negative = 0
    positive_count = 0
    specialist_oracle_loss = 0.0
    specialist_deployed_loss = 0.0
    for start in range(0, targets.batch_size, batch_size):
        end = min(start + batch_size, targets.batch_size)
        index = torch.arange(start, end, dtype=torch.int64)
        batch = _select_batch(targets, index, device)
        expert_q, logits = actor.expert_q_and_gate(batch.task)
        distance = (
            (expert_q - batch.q_projected.unsqueeze(1))
            / actor.q_half.view(1, 1, -1)
        ).square().mean(dim=-1)
        winner = distance.argmin(dim=-1)
        gate = logits.argmax(dim=-1)
        deployed = distance.gather(1, gate.unsqueeze(-1)).squeeze(-1)
        winner_count += torch.bincount(
            winner.cpu(), minlength=actor.config.n_experts)
        gate_count += torch.bincount(
            gate.cpu(), minlength=actor.config.n_experts)
        winner_correct += int(torch.count_nonzero(winner == gate))
        rows += end - start
        oracle_loss += float(distance.amin(dim=-1).sum())
        deployed_loss += float(deployed.sum())
        if explorer_selected is not None:
            positive = explorer_selected[start:end].to(device)
            predicted_positive = gate != 0
            classifier_true_positive += int(torch.count_nonzero(
                predicted_positive & positive))
            classifier_false_positive += int(torch.count_nonzero(
                predicted_positive & ~positive))
            classifier_false_negative += int(torch.count_nonzero(
                ~predicted_positive & positive))
            positive_count += int(torch.count_nonzero(positive))
            if bool(positive.any()):
                specialist_distance = distance[positive, 1:]
                specialist_oracle_loss += float(
                    specialist_distance.amin(dim=-1).sum())
                specialist_deployed_loss += float(
                    deployed[positive].sum())
    result = {
        'target_count': rows,
        'winner_fraction': [
            float(value / rows) for value in winner_count.tolist()
        ],
        'gate_fraction': [
            float(value / rows) for value in gate_count.tolist()
        ],
        'gate_winner_accuracy': winner_correct / rows,
        'oracle_normalized_joint_mse': oracle_loss / rows,
        'deployed_normalized_joint_mse': deployed_loss / rows,
    }
    if explorer_selected is not None:
        predicted_count = (
            classifier_true_positive + classifier_false_positive)
        result.update({
            'gate_positive_fraction': predicted_count / rows,
            'gate_positive_precision': (
                classifier_true_positive / predicted_count
                if predicted_count else 0.0),
            'gate_positive_recall': (
                classifier_true_positive / positive_count
                if positive_count else 0.0),
            'gate_false_positive_fraction': (
                classifier_false_positive / rows),
            'gate_false_negative_fraction': (
                classifier_false_negative / rows),
            'positive_specialist_oracle_normalized_joint_mse': (
                specialist_oracle_loss / positive_count
                if positive_count else 0.0),
            'positive_deployed_normalized_joint_mse': (
                specialist_deployed_loss / positive_count
                if positive_count else 0.0),
        })
    return result


def _method_for_objective(objective: str) -> str:
    if objective == 'advantage-safe':
        return 'paired-real-return-hard-moe-advantage-safe'
    if objective == 'paired-wta':
        return 'paired-real-return-hard-moe-wta'
    raise ValueError(f'unsupported objective {objective!r}')


def _configure_frozen_parameters(
    actor,
    objective: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply and fingerprint the immutable part of the training objective."""
    if objective == 'advantage-safe' and actor.config.n_experts < 2:
        raise ValueError(
            'advantage-safe objective requires at least two experts')
    if objective not in {'advantage-safe', 'paired-wta'}:
        raise ValueError(f'unsupported objective {objective!r}')

    frozen_names: set[str] = set()
    if objective == 'advantage-safe':
        frozen_names.update(
            f'trunk.{name}' for name, _ in actor.trunk.named_parameters())
        frozen_names.update(
            f'experts.0.{name}'
            for name, _ in actor.experts[0].named_parameters())

    frozen_values: dict[str, torch.Tensor] = {}
    trainable_names: list[str] = []
    for name, parameter in actor.named_parameters():
        frozen = name in frozen_names
        parameter.requires_grad_(not frozen)
        if frozen:
            frozen_values[name] = parameter.detach().cpu().clone()
        else:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError('objective leaves no trainable MoE parameters')
    if frozen_names != set(frozen_values):
        missing = sorted(frozen_names - set(frozen_values))
        raise ValueError(f'frozen parameter names do not exist: {missing}')
    contract = {
        'format': _FROZEN_CONTRACT_FORMAT,
        'objective': objective,
        'frozen_parameter_names': sorted(frozen_values),
        'trainable_parameter_names': sorted(trainable_names),
        'frozen_parameter_state_sha256': state_dict_fingerprint(
            frozen_values),
    }
    return frozen_values, contract


def _training_contract(
    args: argparse.Namespace,
    device: torch.device,
    provenance_mode: str,
) -> dict[str, Any]:
    """Settings that must stay identical for bitwise-faithful continuation."""
    return {
        'format': _TRAINER_STATE_FORMAT,
        'objective': args.objective,
        'method': _method_for_objective(args.objective),
        'n_experts': int(args.n_experts),
        'expert_perturb_std': float(args.expert_perturb_std),
        'batch_size': int(args.batch_size),
        'learning_rate': float(args.lr),
        'advantage_margin_m': float(args.advantage_margin_m),
        'gate_ce_weight': float(args.gate_ce_weight),
        'positive_gate_weight': float(args.positive_gate_weight),
        'load_balance_weight': float(args.load_balance_weight),
        'gradient_clip_norm': float(args.gradient_clip_norm),
        'seed': int(args.seed),
        'resolved_device': str(device),
        'provenance_mode': provenance_mode,
    }


def _require_equal(
    saved: Any,
    current: Any,
    label: str,
) -> None:
    if saved != current:
        raise ValueError(
            f'resume {label} mismatch: checkpoint={saved!r}, '
            f'current={current!r}')


def _validate_resume_state(
    payload: Mapping[str, Any],
    *,
    actor_state_sha256: str,
    source_hashes: Mapping[str, str | None],
    training_contract: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    frozen_contract: Mapping[str, Any],
    target_count: int,
) -> Mapping[str, Any]:
    state = payload.get('trainer_state')
    if not isinstance(state, Mapping):
        raise ValueError(
            'checkpoint has no resumable trainer_state; historical P9/P10 '
            'MoE files are deployment-only and cannot be resumed')
    if state.get('format') != _TRAINER_STATE_FORMAT:
        raise ValueError(
            f"unsupported trainer_state format {state.get('format')!r}")
    _require_equal(
        state.get('actor_state_sha256'), actor_state_sha256,
        'actor state identity')
    _require_equal(
        state.get('source_hashes'), dict(source_hashes), 'source hashes')
    _require_equal(
        state.get('training_contract'), dict(training_contract),
        'training contract')
    _require_equal(
        state.get('data_provenance'), dict(data_provenance),
        'data provenance')
    _require_equal(
        state.get('frozen_contract'), dict(frozen_contract),
        'frozen contract')
    if state.get('target_count') != target_count:
        raise ValueError(
            'resume target count differs from the current archives')
    return state


def _restore_sampler_state(
    state: Mapping[str, Any],
    sampler: torch.Generator,
    target_count: int,
) -> tuple[torch.Tensor, int]:
    sampler_state = state.get('sampler_state')
    permutation = state.get('permutation')
    cursor = state.get('cursor')
    if not torch.is_tensor(sampler_state):
        raise ValueError('resume trainer_state has no sampler_state')
    if (not torch.is_tensor(permutation)
            or permutation.device.type != 'cpu'
            or permutation.dtype != torch.int64
            or tuple(permutation.shape) != (target_count,)
            or not torch.equal(
                torch.sort(permutation).values,
                torch.arange(target_count, dtype=torch.int64))):
        raise ValueError('resume permutation is not a valid target ordering')
    if (isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor < target_count):
        raise ValueError('resume cursor is outside the target permutation')
    try:
        sampler.set_state(sampler_state.cpu())
    except RuntimeError as error:
        raise ValueError(
            'resume sampler_state is invalid') from error
    return permutation.clone(), cursor


def _validate_rng_state(
    state: Any,
    device: torch.device,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError('resume trainer_state has no global RNG state')
    torch_state = state.get('torch_rng_state')
    if (not torch.is_tensor(torch_state)
            or torch_state.dtype != torch.uint8
            or torch_state.ndim != 1):
        raise ValueError('resume torch RNG state is invalid')
    if state.get('numpy_rng_state') is None:
        raise ValueError('resume NumPy RNG state is missing')
    cuda_state = state.get('cuda_rng_state')
    if (device.type == 'cuda'
            and (not torch.is_tensor(cuda_state)
                 or cuda_state.dtype != torch.uint8
                 or cuda_state.ndim != 1)):
        raise ValueError('resume CUDA RNG state is invalid')
    return state


def _optimizer_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--paired-archive', required=True)
    parser.add_argument('--explorer-checkpoint', required=True)
    parser.add_argument(
        '--baseline-run-checkpoint',
        help=(
            'complete deterministic/frozen paired collection runner; '
            'required unless explicitly running a legacy unverified ablation'))
    parser.add_argument(
        '--allow-unverified-provenance',
        action='store_true',
        help=(
            'explicitly allow bare legacy archives without runner provenance'))
    parser.add_argument(
        '--resume',
        help=(
            'resumable MoE trainer checkpoint; deployment-only historical '
            'checkpoints are rejected'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--objective',
        choices=('advantage-safe', 'paired-wta'),
        default='advantage-safe')
    parser.add_argument('--n-experts', type=int, default=4)
    parser.add_argument('--expert-perturb-std', type=float, default=0.01)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--advantage-margin-m', type=float, default=0.01)
    parser.add_argument('--gate-ce-weight', type=float, default=0.1)
    parser.add_argument('--positive-gate-weight', type=float, default=1.0)
    parser.add_argument('--load-balance-weight', type=float, default=0.01)
    parser.add_argument('--gradient-clip-norm', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=20260728)
    parser.add_argument(
        '--save-steps', default='100,500,1000,2000')
    return parser.parse_args()


def main() -> None:
    session_start = time.perf_counter()
    args = parse_args()
    numeric_positive = {
        'n_experts': args.n_experts,
        'updates': args.updates,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'gradient_clip_norm': args.gradient_clip_norm,
    }
    if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in numeric_positive.values()):
        raise ValueError(
            'experts, updates, batch size, lr, and clip norm must be positive')
    for name in (
            'expert_perturb_std', 'advantage_margin_m',
            'gate_ce_weight', 'positive_gate_weight',
            'load_balance_weight'):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'--{name.replace("_", "-")} must be non-negative')
    save_steps = _parse_save_steps(args.save_steps, args.updates)
    if (not args.allow_unverified_provenance
            and args.baseline_run_checkpoint is None):
        raise ValueError(
            '--baseline-run-checkpoint is required by default; only pass '
            '--allow-unverified-provenance for an explicit legacy ablation')

    base_path = Path(args.base_checkpoint).expanduser().resolve(strict=True)
    paired_path = Path(args.paired_archive).expanduser().resolve(strict=True)
    explorer_path = Path(
        args.explorer_checkpoint).expanduser().resolve(strict=True)
    baseline_path = (
        Path(args.baseline_run_checkpoint).expanduser().resolve(strict=True)
        if args.baseline_run_checkpoint is not None else None)
    resume_path = (
        Path(args.resume).expanduser().resolve(strict=True)
        if args.resume is not None else None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'output directory already exists: {output_dir}')
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    base_actor, _, _, _, base_payload = (
        load_direct_seed_rl_checkpoint(base_path, device))
    actor_state = base_payload.get('actor')
    if (not isinstance(actor_state, Mapping)
            or any(not isinstance(name, str) or not torch.is_tensor(value)
                   for name, value in actor_state.items())):
        raise ValueError('base checkpoint has no valid actor state')
    base_actor_state_sha256 = state_dict_fingerprint(dict(actor_state))

    (
        archive, elite, paired_state, elite_state,
        paired_runner, explorer_runner,
    ) = _load_archive_objects(paired_path, explorer_path)
    if args.allow_unverified_provenance:
        data_provenance: dict[str, Any] = {
            'verified': False,
            'mode': 'legacy-unverified-explicit',
            'warning': (
                'runner provenance checks were explicitly disabled'),
            'paired_archive_state_sha256': _semantic_sha256(paired_state),
            'explorer_elite_state_sha256': _semantic_sha256(elite_state),
            'base_actor_state_sha256': base_actor_state_sha256,
            'paired_input_is_complete_runner': paired_runner is not None,
            'explorer_input_is_complete_runner': explorer_runner is not None,
        }
    else:
        if paired_runner is None:
            raise ValueError(
                'a bare legacy paired archive is accepted only with '
                '--allow-unverified-provenance; pass its complete runner '
                'checkpoint to --paired-archive')
        assert baseline_path is not None
        baseline_runner = _load_mapping(
            baseline_path, 'baseline run')
        data_provenance = _verify_training_provenance(
            base_payload=base_payload,
            base_actor_state_sha256=base_actor_state_sha256,
            archive=archive,
            paired_state=paired_state,
            paired_runner=paired_runner,
            elite=elite,
            explorer_runner=explorer_runner,
            baseline_runner=baseline_runner,
        )
        data_provenance['explorer_elite_state_sha256'] = (
            _semantic_sha256(elite_state))

    targets, explorer_selected, target_stats = _build_training_data(
        archive, elite, args.advantage_margin_m, args.objective)

    source_hashes = {
        'base_checkpoint_sha256': _sha256(base_path),
        'paired_archive_sha256': _sha256(paired_path),
        'explorer_checkpoint_sha256': _sha256(explorer_path),
        'baseline_run_checkpoint_sha256': (
            _sha256(baseline_path)
            if baseline_path is not None else None),
    }
    contract = _training_contract(
        args, device, str(data_provenance['mode']))

    resume_payload: Mapping[str, Any] | None = None
    resume_optimizer_state: Mapping[str, Any] | None = None
    if resume_path is None:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        actor = direct_seed_moe_from_actor(
            base_actor,
            n_experts=args.n_experts,
            expert_perturb_std=args.expert_perturb_std,
            seed=args.seed + 1).train()
        start_step = 1
    else:
        actor, resume_optimizer_state, resume_payload = (
            load_direct_seed_moe_checkpoint(resume_path, device))
        saved_step = resume_payload['update_step']
        if saved_step >= args.updates:
            raise ValueError(
                f'--updates must exceed resumed update_step {saved_step}')
        start_step = saved_step + 1
        actor.train()

    frozen_parameters, frozen_contract = _configure_frozen_parameters(
        actor, args.objective)
    optimizer = torch.optim.Adam(
        [
            parameter for parameter in actor.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr)
    sampler = torch.Generator(device='cpu')
    log: list[dict[str, Any]]
    if resume_payload is None:
        sampler.manual_seed(args.seed + 2)
        permutation = torch.randperm(
            targets.batch_size, generator=sampler)
        cursor = 0
        log = []
    else:
        trainer_state = _validate_resume_state(
            resume_payload,
            actor_state_sha256=state_dict_fingerprint(
                dict(actor.state_dict())),
            source_hashes=source_hashes,
            training_contract=contract,
            data_provenance=data_provenance,
            frozen_contract=frozen_contract,
            target_count=targets.batch_size,
        )
        if trainer_state.get('update_step') != resume_payload['update_step']:
            raise ValueError(
                'resume trainer_state/update_step is internally inconsistent')
        if not isinstance(resume_optimizer_state, Mapping):
            raise ValueError(
                'resumable trainer checkpoint has no actor optimizer state')
        optimizer.load_state_dict(resume_optimizer_state)
        _optimizer_to(optimizer, device)
        permutation, cursor = _restore_sampler_state(
            trainer_state, sampler, targets.batch_size)
        saved_log = trainer_state.get('log')
        if not isinstance(saved_log, list):
            raise ValueError('resume trainer_state log must be a list')
        log = copy.deepcopy(saved_log)
        rng_state = _validate_rng_state(
            trainer_state.get('rng_state'), device)
        restore_global_rng(rng_state, device)

    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        'base_checkpoint': str(base_path),
        'paired_archive': str(paired_path),
        'explorer_checkpoint': str(explorer_path),
        'baseline_run_checkpoint': (
            str(baseline_path) if baseline_path is not None else None),
        'resume': str(resume_path) if resume_path is not None else None,
        'output_dir': str(output_dir),
        'resolved_device': str(device),
        'save_steps': list(save_steps),
        'target_stats': target_stats,
        'data_provenance': data_provenance,
        'training_contract': contract,
        'frozen_contract': frozen_contract,
        'source_hashes': source_hashes,
        'base_format': base_payload.get('format'),
    }
    setup_io_start = time.perf_counter()
    setup_compute_s = setup_io_start - session_start
    (output_dir / 'config.json').write_text(
        json.dumps(config, indent=2), encoding='utf-8')
    timing = {
        'batch_preparation_s': 0.0,
        'pure_training_s': 0.0,
        'diagnostics_s': 0.0,
        'checkpoint_io_s': 0.0,
        'metrics_io_s': 0.0,
        'stdout_io_s': 0.0,
        'setup_compute_s': setup_compute_s,
        'setup_io_s': time.perf_counter() - setup_io_start,
    }
    method = _method_for_objective(args.objective)
    for step in range(start_step, args.updates + 1):
        _synchronize(device)
        preparation_start = time.perf_counter()
        parts = []
        remaining = args.batch_size
        while remaining:
            available = targets.batch_size - cursor
            take = min(remaining, available)
            parts.append(permutation[cursor:cursor + take])
            cursor += take
            remaining -= take
            if cursor == targets.batch_size:
                permutation = torch.randperm(
                    targets.batch_size, generator=sampler)
                cursor = 0
        index = torch.cat(parts)
        batch = _select_batch(targets, index, device)
        selected_labels = (
            explorer_selected.index_select(0, index).to(device)
            if explorer_selected is not None else None)
        _synchronize(device)
        timing['batch_preparation_s'] += (
            time.perf_counter() - preparation_start)

        _synchronize(device)
        training_start = time.perf_counter()
        if args.objective == 'advantage-safe':
            if selected_labels is None:
                raise RuntimeError(
                    'advantage-safe training has no explorer labels')
            metrics = update_direct_seed_moe_advantage(
                actor, optimizer, batch,
                selected_labels,
                gate_ce_weight=args.gate_ce_weight,
                positive_gate_weight=args.positive_gate_weight,
                specialist_load_balance_weight=(
                    args.load_balance_weight),
                gradient_clip_norm=args.gradient_clip_norm)
        else:
            metrics = update_direct_seed_moe_projection(
                actor, optimizer, batch,
                gate_ce_weight=args.gate_ce_weight,
                load_balance_weight=args.load_balance_weight,
                gradient_clip_norm=args.gradient_clip_norm)
        _synchronize(device)
        timing['pure_training_s'] += (
            time.perf_counter() - training_start)

        if step in save_steps:
            _synchronize(device)
            diagnostics_start = time.perf_counter()
            diagnostics = _diagnostics(
                actor, targets, explorer_selected,
                device, args.batch_size)
            record = {
                'step': step,
                **metrics,
                **diagnostics,
            }
            if frozen_parameters:
                current = dict(actor.named_parameters())
                record['frozen_baseline_max_abs_delta'] = max(
                    float((
                        current[name].detach().cpu() - expected
                    ).abs().max())
                    for name, expected in frozen_parameters.items())
            _synchronize(device)
            timing['diagnostics_s'] += (
                time.perf_counter() - diagnostics_start)
            record['timing_s'] = {
                key: value for key, value in timing.items()
            }
            log.append(record)
            _synchronize(device)
            checkpoint_start = time.perf_counter()
            metadata = {
                'method': method,
                'training_config': config,
                'target_stats': target_stats,
                'diagnostics': record,
                'data_provenance': data_provenance,
                'frozen_contract': frozen_contract,
            }
            trainer_state = {
                'format': _TRAINER_STATE_FORMAT,
                'update_step': step,
                'actor_state_sha256': state_dict_fingerprint(
                    dict(actor.state_dict())),
                'source_hashes': copy.deepcopy(source_hashes),
                'training_contract': copy.deepcopy(contract),
                'data_provenance': copy.deepcopy(data_provenance),
                'target_count': targets.batch_size,
                'sampler_state': sampler.get_state().clone(),
                'permutation': permutation.clone(),
                'cursor': cursor,
                'rng_state': global_rng_state(device),
                'frozen_contract': copy.deepcopy(frozen_contract),
                'log': copy.deepcopy(log),
                'timing_before_checkpoint_io_s': copy.deepcopy(timing),
            }
            checkpoint = direct_seed_moe_checkpoint(
                actor, update_step=step,
                actor_optimizer=optimizer,
                metadata=metadata)
            checkpoint['trainer_state'] = trainer_state
            step_path = output_dir / f'direct_seed_moe_step{step:06d}.pt'
            atomic_torch_save(checkpoint, step_path)
            if step == args.updates:
                atomic_torch_save(
                    checkpoint, output_dir / 'direct_seed.pt')
            _synchronize(device)
            checkpoint_io = time.perf_counter() - checkpoint_start
            timing['checkpoint_io_s'] += checkpoint_io
            record['timing_s'].update({
                'last_checkpoint_io_s': checkpoint_io,
                'checkpoint_io_s': timing['checkpoint_io_s'],
            })
            metrics_io_start = time.perf_counter()
            (output_dir / 'metrics.json').write_text(
                json.dumps(log, indent=2), encoding='utf-8')
            timing['metrics_io_s'] += (
                time.perf_counter() - metrics_io_start)
            record['timing_s']['metrics_io_s'] = timing['metrics_io_s']
            stdout_io_start = time.perf_counter()
            print(json.dumps(record), flush=True)
            timing['stdout_io_s'] += (
                time.perf_counter() - stdout_io_start)
            actor.train()

    timing_report = {
        'format': 'direct-seed-moe-timing-v1',
        'resumed_from': (
            str(resume_path) if resume_path is not None else None),
        'start_update_step': start_step,
        'end_update_step': args.updates,
        **timing,
        'session_end_to_end_before_timing_report_write_s': (
            time.perf_counter() - session_start),
        'definitions': {
            'pure_training_s': (
                'optimizer update only, device-synchronized'),
            'diagnostics_s': (
                'full-dataset diagnostics and frozen-parameter audit only'),
            'checkpoint_io_s': (
                'checkpoint materialization plus atomic torch serialization'),
            'metrics_io_s': 'metrics JSON writes only',
            'stdout_io_s': 'progress JSON writes to stdout only',
            'batch_preparation_s': (
                'sampling, host indexing, and device transfer'),
            'setup_compute_s': (
                'argument validation, source loading/hashing/provenance, '
                'target construction, and model/optimizer setup'),
            'setup_io_s': 'config JSON write only',
        },
    }
    (output_dir / 'timing.json').write_text(
        json.dumps(timing_report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
