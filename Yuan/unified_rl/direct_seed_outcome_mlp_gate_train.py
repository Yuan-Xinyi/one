"""Fit an auditable nonlinear expected-advantage gate on frozen seed branches.

The input is the same paired P12 training evidence consumed by
``direct_seed_outcome_gate_train``: one frozen contextual-RL baseline and
three frozen specialist branches, each evaluated exactly once per training
task under the same controller and projection contract.  Only the gate is
replaced.

Five-fold OOF predictions use exact float32 task bytes as GroupKFold groups.
Every fold fits its StandardScaler on its training rows only.  The final
scaler is folded analytically into the first gate layer, so deployment is one
task-only hard gate, one selected seed head, at most one IK refinement, and
one controller rollout.  It performs no candidate enumeration, return-model
query, controller probe, or validation-set read.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Yuan.unified_rl.direct_seed_outcome_gate_train import (
    _exclusive_save,
    _fingerprint_list_sha256,
    _frozen_features,
    _gated_metrics,
    _geometry_fingerprints,
    _geometry_group_keys,
    _load_outcomes,
    _quota_metrics,
    _require_mapping,
    _resolved_file,
    _validate_forced_specialist,
    _validate_moe_baseline_branch,
    _validate_outcome_manifest,
    _validate_runner,
)
from Yuan.unified_rl.direct_seed_rl import (
    DirectSeedMoEActor,
    DirectSeedPairedArchive,
    direct_seed_moe_checkpoint,
    load_direct_seed_moe_checkpoint,
    load_direct_seed_rl_checkpoint,
)
from Yuan.unified_rl.provenance import file_fingerprint


_DEFAULT_SELECTION_FRACTIONS = (0.10, 0.15)
_DEFAULT_QUOTAS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)


class _ExpectedAdvantageMLP(nn.Module):
    """Small training-only regressor for specialist advantages in metres."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        specialist_count: int,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, specialist_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _array_sha256(value: np.ndarray) -> str:
    """Hash canonical little-endian contiguous array bytes and shape."""
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder('<')
    canonical = np.asarray(array, dtype=dtype, order='C')
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode('ascii'))
    digest.update(b'\0')
    digest.update(canonical.dtype.str.encode('ascii'))
    digest.update(b'\0')
    digest.update(canonical.tobytes(order='C'))
    return digest.hexdigest()


def _validated_training_arrays(
    features: np.ndarray,
    advantage_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32, order='C')
    advantage_m = np.asarray(advantage_m, dtype=np.float32, order='C')
    if (features.ndim != 2 or advantage_m.ndim != 2
            or features.shape[0] != advantage_m.shape[0]
            or features.shape[0] < 1 or features.shape[1] < 1
            or advantage_m.shape[1] < 1
            or not np.isfinite(features).all()
            or not np.isfinite(advantage_m).all()):
        raise ValueError(
            'features and advantages must be finite non-empty matrices '
            'with matching rows')
    return features, advantage_m


def _new_mlp(
    feature_dim: int,
    hidden_dim: int,
    specialist_count: int,
    *,
    seed: int,
) -> _ExpectedAdvantageMLP:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError('MLP seed must be an integer')
    # Avoid perturbing unrelated process-global torch RNG state while
    # retaining the diagnostic's exact CPU initialization sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _ExpectedAdvantageMLP(
            feature_dim, hidden_dim, specialist_count)
    return model


def _fit_advantage_mlp(
    features: np.ndarray,
    advantage_m: np.ndarray,
    *,
    hidden_dim: int = 64,
    epochs: int = 120,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    model_seed: int = 20260728,
    shuffle_seed: int = 10000,
) -> tuple[_ExpectedAdvantageMLP, Any, dict[str, Any]]:
    """Fit StandardScaler -> MLP with deterministic CPU AdamW/MSE."""
    from sklearn.preprocessing import StandardScaler

    features, advantage_m = _validated_training_arrays(
        features, advantage_m)
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) \
            or hidden_dim < 1:
        raise ValueError('hidden_dim must be a positive integer')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
            or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    for name, value in (
            ('learning_rate', learning_rate),
            ('weight_decay', weight_decay)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).astype(
        np.float32, copy=False)
    if (not np.isfinite(scaled).all()
            or not np.isfinite(scaler.mean_).all()
            or not np.isfinite(scaler.scale_).all()
            or np.any(scaler.scale_ <= 0.0)):
        raise RuntimeError('fitted feature scaler is invalid')
    x = torch.from_numpy(np.ascontiguousarray(scaled))
    y = torch.from_numpy(np.ascontiguousarray(advantage_m))
    model = _new_mlp(
        features.shape[1], hidden_dim, advantage_m.shape[1],
        seed=model_seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(shuffle_seed)

    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            rows = order[start:start + batch_size]
            prediction = model(x.index_select(0, rows))
            loss = F.mse_loss(prediction, y.index_select(0, rows))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    elapsed_s = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        fitted = model(x)
        residual = fitted - y
        mse = float(torch.mean(residual.square()))
        mae = float(torch.mean(residual.abs()))
    audit = {
        'implementation': (
            'torch CPU Linear-SiLU-Linear expected-advantage regressor'),
        'target': 'raw specialist-minus-baseline progress in metres',
        'optimizer': 'AdamW',
        'loss': 'mean squared error over all specialist advantages',
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': float(learning_rate),
        'weight_decay': float(weight_decay),
        'model_seed': int(model_seed),
        'shuffle_seed': int(shuffle_seed),
        'torch_version': torch.__version__,
        'torch_num_threads': torch.get_num_threads(),
        'deterministic_algorithms_enabled': (
            torch.are_deterministic_algorithms_enabled()),
        'elapsed_s': float(elapsed_s),
        'final_full_train_mse_m2': mse,
        'final_full_train_mae_m': mae,
        'scaler_fit_row_count': int(len(features)),
        'scaler_mean_sha256': _array_sha256(
            np.asarray(scaler.mean_, dtype=np.float64)),
        'scaler_scale_sha256': _array_sha256(
            np.asarray(scaler.scale_, dtype=np.float64)),
    }
    return model, scaler, audit


@torch.no_grad()
def _predict_advantage(
    model: _ExpectedAdvantageMLP,
    scaler: Any,
    features: np.ndarray,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32, order='C')
    scaled = scaler.transform(features).astype(np.float32, copy=False)
    prediction = model(torch.from_numpy(
        np.ascontiguousarray(scaled))).cpu().numpy()
    if prediction.shape[0] != len(features) \
            or not np.isfinite(prediction).all():
        raise RuntimeError('MLP advantage predictions are invalid')
    return prediction.astype(np.float64, copy=False)


def _fold_scaler_into_first_layer(
    model: _ExpectedAdvantageMLP,
    scaler: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a raw-feature first layer exactly equivalent up to dtype."""
    first = model.network[0]
    if not isinstance(first, nn.Linear):
        raise TypeError('expected-advantage model first layer is not Linear')
    weight = first.weight.detach().cpu().numpy().astype(
        np.float64, copy=False)
    bias = first.bias.detach().cpu().numpy().astype(
        np.float64, copy=False)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if (mean.shape != (first.in_features,)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)):
        raise ValueError('feature scaler cannot be folded into first layer')
    raw_weight = weight / scale[None, :]
    raw_bias = bias - raw_weight @ mean
    return (
        torch.from_numpy(raw_weight).to(dtype=first.weight.dtype),
        torch.from_numpy(raw_bias).to(dtype=first.bias.dtype),
    )


@torch.no_grad()
def _nonlinear_gate_actor(
    source: DirectSeedMoEActor,
    model: _ExpectedAdvantageMLP,
    scaler: Any,
    *,
    threshold_m: float,
    gate_hidden_dim: int,
) -> DirectSeedMoEActor:
    """Clone frozen branches and install a calibrated nonlinear hard gate."""
    if not isinstance(source, DirectSeedMoEActor):
        raise TypeError('source must be a DirectSeedMoEActor')
    if source.config.gate_hidden_dim != 0:
        raise ValueError('source MoE must use the legacy linear gate')
    if not math.isfinite(threshold_m):
        raise ValueError('threshold_m must be finite')
    if model.network[0].out_features != gate_hidden_dim:
        raise ValueError('MLP and requested gate hidden dimensions differ')
    if model.network[-1].out_features != source.config.n_experts - 1:
        raise ValueError('MLP output count differs from specialist count')

    config = dataclasses.replace(
        source.config, gate_hidden_dim=gate_hidden_dim)
    actor = DirectSeedMoEActor(
        source.q_lower.detach().cpu(),
        source.q_upper.detach().cpu(),
        config,
        task_mean=source.task_mean.detach().cpu(),
        task_std=source.task_std.detach().cpu(),
    ).to(device='cpu', dtype=source.q_mid.dtype)
    actor.trunk.load_state_dict(source.trunk.state_dict(), strict=True)
    actor.experts.load_state_dict(source.experts.state_dict(), strict=True)
    if not isinstance(actor.gate, nn.Sequential) \
            or len(actor.gate) != 3 \
            or not isinstance(actor.gate[0], nn.Linear) \
            or not isinstance(actor.gate[1], nn.SiLU) \
            or not isinstance(actor.gate[2], nn.Linear):
        raise RuntimeError('nonlinear deployment gate layout is invalid')

    raw_weight, raw_bias = _fold_scaler_into_first_layer(model, scaler)
    actor.gate[0].weight.copy_(raw_weight)
    actor.gate[0].bias.copy_(raw_bias)
    actor.gate[2].weight.zero_()
    actor.gate[2].bias.zero_()
    actor.gate[2].weight[1:].copy_(model.network[2].weight)
    actor.gate[2].bias[1:].copy_(
        model.network[2].bias - float(threshold_m))

    source_non_gate = {
        name: value.detach().cpu()
        for name, value in source.state_dict().items()
        if not name.startswith('gate.')
    }
    actor_non_gate = {
        name: value.detach().cpu()
        for name, value in actor.state_dict().items()
        if not name.startswith('gate.')
    }
    if set(source_non_gate) != set(actor_non_gate):
        raise RuntimeError('non-gate state keys changed during gate fitting')
    for name, expected in source_non_gate.items():
        if not torch.equal(actor_non_gate[name], expected):
            raise RuntimeError(
                f'frozen branch tensor {name!r} changed during gate fitting')
    if (not bool((actor.gate[2].weight[0] == 0).all())
            or not bool((actor.gate[2].bias[0] == 0).all())):
        raise RuntimeError('baseline gate logit is not identically zero')
    return actor.eval()


def _grouped_oof_advantage_mlp(
    features: np.ndarray,
    advantage_m: np.ndarray,
    branch_progress_m: np.ndarray,
    fingerprints: Sequence[str],
    group_keys: Sequence[str],
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    shuffle_seed: int,
    quotas: Sequence[float],
    positive_margin_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate leakage-proof five-fold expected-advantage predictions."""
    from sklearn import __version__ as sklearn_version
    from sklearn.model_selection import GroupKFold

    features, advantage_m = _validated_training_arrays(
        features, advantage_m)
    branch_progress_m = np.asarray(
        branch_progress_m, dtype=np.float64)
    if branch_progress_m.shape != (
            len(features), advantage_m.shape[1] + 1):
        raise ValueError('branch progress shape differs from advantages')
    groups = np.asarray(tuple(group_keys))
    if len(groups) != len(features) \
            or len(fingerprints) != len(features):
        raise ValueError('task group metadata differs from feature rows')
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError('five-fold OOF requires at least five task groups')

    oof_prediction = np.full(
        advantage_m.shape, np.nan, dtype=np.float64)
    fold_reports = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, heldout) in enumerate(
            splitter.split(features, groups=groups)):
        if set(groups[train]).intersection(groups[heldout]):
            raise RuntimeError('exact task-byte group leaked across folds')
        model, scaler, fit_audit = _fit_advantage_mlp(
            features[train], advantage_m[train],
            hidden_dim=hidden_dim, epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_seed=seed + fold,
            shuffle_seed=shuffle_seed + fold)
        prediction = _predict_advantage(
            model, scaler, features[heldout])
        oof_prediction[heldout] = prediction
        heldout_error = prediction - advantage_m[heldout]
        fold_reports.append({
            'fold': fold + 1,
            'train_count': int(len(train)),
            'heldout_count': int(len(heldout)),
            'train_unique_geometry_groups': int(
                len(np.unique(groups[train]))),
            'heldout_unique_geometry_groups': int(
                len(np.unique(groups[heldout]))),
            'heldout_group_list_sha256': _fingerprint_list_sha256(
                sorted(set(groups[heldout].tolist()))),
            'heldout_prediction_mse_m2': float(
                np.mean(np.square(heldout_error), dtype=np.float64)),
            'heldout_prediction_mae_m': float(
                np.mean(np.abs(heldout_error), dtype=np.float64)),
            'fit': fit_audit,
        })
    if not np.isfinite(oof_prediction).all():
        raise RuntimeError('OOF predictions do not cover every task')

    score = oof_prediction.max(axis=1)
    specialist_choice = oof_prediction.argmax(axis=1) + 1
    target_branch = branch_progress_m.argmax(axis=1)
    error = oof_prediction - advantage_m.astype(
        np.float64, copy=False)
    report = {
        'protocol': 'exact-float32-task-bytes-grouped-5-fold',
        'shuffle': False,
        'validation_artifacts_read': False,
        'unique_geometry_group_count': int(len(unique_groups)),
        'geometry_fingerprint_list_sha256': (
            _fingerprint_list_sha256(fingerprints)),
        'sklearn_version': sklearn_version,
        'oof_prediction_mse_m2': float(
            np.mean(np.square(error), dtype=np.float64)),
        'oof_prediction_mae_m': float(
            np.mean(np.abs(error), dtype=np.float64)),
        'folds': fold_reports,
        'quota_grid': _quota_metrics(
            score, specialist_choice, branch_progress_m,
            target_branch, quotas,
            positive_margin_m=positive_margin_m),
    }
    return oof_prediction, report


def _parse_fractions(text: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(part.strip()) for part in text.split(',')
            if part.strip())
    except ValueError as error:
        raise ValueError(
            f'{label} must contain comma-separated floats') from error
    if not values or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in values):
        raise ValueError(f'{label} values must be in (0, 1]')
    if len(set(values)) != len(values):
        raise ValueError(f'{label} values must be unique')
    return values


def _fraction_tag(fraction: float) -> str:
    percent = fraction * 100.0
    if math.isclose(percent, round(percent), abs_tol=1e-10):
        return f'q{int(round(percent)):02d}'
    per_mille = fraction * 1000.0
    if math.isclose(per_mille, round(per_mille), abs_tol=1e-10):
        return f'q{int(round(per_mille)):03d}'
    raise ValueError(
        'selection fractions must be representable in whole per-mille units')


@torch.inference_mode()
def _generator_timing(
    actor: DirectSeedMoEActor,
    task: torch.Tensor,
    *,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, Any]:
    """Small CPU generator-only benchmark; excludes IK/controller work."""
    actor = actor.to('cpu').eval()
    task = task.detach().to(device='cpu', dtype=actor.q_mid.dtype)
    if len(task) > 1024:
        task = task[:1024]
    for _ in range(warmup):
        actor.mean_q(task)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        actor.mean_q(task)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(samples, dtype=np.float64)
    return {
        'device': 'cpu',
        'scope': (
            'task trunk + task-only hard gate + one selected seed head; '
            'excludes projection and controller'),
        'batch_size': int(len(task)),
        'warmup': warmup,
        'repeats': repeats,
        'median_batch_ms': float(np.median(values)),
        'mean_batch_ms': float(np.mean(values)),
        'p95_batch_ms': float(np.quantile(values, 0.95)),
        'median_us_per_task': float(
            np.median(values) * 1e3 / len(task)),
        'gate_parameter_count': int(sum(
            parameter.numel() for parameter in actor.gate.parameters())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Fit a frozen P12 multi-branch expected-advantage MLP gate.'))
    parser.add_argument('--moe-checkpoint', required=True)
    parser.add_argument(
        '--forced-specialist-checkpoint',
        required=True, action='append',
        help='repeat in expert-index order for experts 1..K-1')
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--baseline-archive', required=True)
    parser.add_argument('--baseline-runner-checkpoint', required=True)
    parser.add_argument(
        '--specialist-outcomes',
        required=True, action='append',
        help='repeat in expert-index order for experts 1..K-1')
    parser.add_argument(
        '--output-prefix', required=True,
        help='writes <prefix>_q10.pt/json and <prefix>_q15.pt/json')
    parser.add_argument(
        '--selection-fractions',
        default=','.join(
            f'{value:g}' for value in _DEFAULT_SELECTION_FRACTIONS))
    parser.add_argument(
        '--quota-grid',
        default=','.join(f'{value:g}' for value in _DEFAULT_QUOTAS))
    parser.add_argument('--positive-margin-m', type=float, default=0.01)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--feature-batch-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20260728)
    parser.add_argument('--shuffle-seed', type=int, default=10000)
    parser.add_argument(
        '--allow-legacy-outcome-manifest', action='store_true')
    parser.add_argument(
        '--allow-legacy-runner-provenance', action='store_true')
    return parser


def main() -> None:
    args = _parser().parse_args()
    selection_fractions = _parse_fractions(
        args.selection_fractions, '--selection-fractions')
    quotas = _parse_fractions(args.quota_grid, '--quota-grid')
    if not math.isfinite(args.positive_margin_m) \
            or args.positive_margin_m < 0.0:
        raise ValueError('--positive-margin-m must be non-negative')
    if args.feature_batch_size < 1:
        raise ValueError('--feature-batch-size must be positive')
    if args.seed < 0 or args.shuffle_seed < 0:
        raise ValueError('training seeds must be non-negative')
    torch.use_deterministic_algorithms(True)

    paths = {
        'moe_checkpoint': _resolved_file(args.moe_checkpoint),
        'base_checkpoint': _resolved_file(args.base_checkpoint),
        'baseline_archive': _resolved_file(args.baseline_archive),
        'baseline_runner_checkpoint': _resolved_file(
            args.baseline_runner_checkpoint),
    }
    forced_paths = tuple(
        _resolved_file(value)
        for value in args.forced_specialist_checkpoint)
    outcome_paths = tuple(
        _resolved_file(value) for value in args.specialist_outcomes)
    prefix = Path(args.output_prefix).expanduser().resolve()
    if prefix.suffix in ('.pt', '.json'):
        raise ValueError('--output-prefix must not have a .pt/.json suffix')
    tags = tuple(_fraction_tag(value) for value in selection_fractions)
    if len(set(tags)) != len(tags):
        raise ValueError('selection fractions map to duplicate output tags')
    output_paths = {
        tag: (Path(f'{prefix}_{tag}.pt'), Path(f'{prefix}_{tag}.json'))
        for tag in tags
    }
    collisions = [
        str(path)
        for pair in output_paths.values()
        for path in pair if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            f'refusing to overwrite existing outputs: {collisions}')

    source_files: dict[str, Any] = {
        name: file_fingerprint(path) for name, path in paths.items()
    }
    source_files['forced_specialist_checkpoints'] = [
        file_fingerprint(path) for path in forced_paths]
    source_files['specialist_outcomes'] = [
        file_fingerprint(path) for path in outcome_paths]

    moe, _, moe_payload = load_direct_seed_moe_checkpoint(
        paths['moe_checkpoint'], device='cpu')
    if moe.config.gate_hidden_dim != 0:
        raise ValueError('P12 source MoE must use its original linear gate')
    base, _, _, _, base_payload = load_direct_seed_rl_checkpoint(
        paths['base_checkpoint'], device='cpu')
    branch_audit = _validate_moe_baseline_branch(moe, base)
    n_experts = moe.config.n_experts
    if n_experts != 4:
        raise ValueError(
            'formal P12 expected-advantage gate requires exactly K=4')
    if (len(forced_paths) != n_experts - 1
            or len(outcome_paths) != n_experts - 1):
        raise ValueError(
            'K=4 requires three ordered forced checkpoints and outcomes')
    forced_audit = []
    for expert_index, forced_path in enumerate(forced_paths, start=1):
        forced, _, forced_payload = load_direct_seed_moe_checkpoint(
            forced_path, device='cpu')
        if forced.config != moe.config:
            raise ValueError(
                f'forced expert {expert_index} MoE config differs')
        forced_audit.append(_validate_forced_specialist(
            moe_payload, forced_payload,
            moe_sha256=source_files['moe_checkpoint']['sha256'],
            expert_index=expert_index))

    archive_state = torch.load(
        paths['baseline_archive'],
        map_location='cpu', weights_only=False)
    archive_state = _require_mapping(
        archive_state, 'baseline paired archive')
    task_ids = archive_state.get('task_ids')
    if not torch.is_tensor(task_ids):
        raise ValueError('baseline paired archive has no task_ids')
    archive = DirectSeedPairedArchive(task_ids)
    archive.load_state_dict(archive_state)
    if len(archive) != int(archive.task_ids.numel()):
        raise ValueError('baseline paired archive is not full coverage')

    runner = torch.load(
        paths['baseline_runner_checkpoint'],
        map_location='cpu', weights_only=False)
    runner = _require_mapping(runner, 'baseline full runner')
    runner_audit = _validate_runner(
        runner, archive, archive_state, base_payload,
        allow_legacy=args.allow_legacy_runner_provenance)
    outcomes = [
        _load_outcomes(path, archive) for path in outcome_paths]
    outcome_manifest = [
        _validate_outcome_manifest(
            outcome_path, outcome,
            forced_path=forced_path,
            runner=runner, runner_audit=runner_audit,
            allow_legacy=args.allow_legacy_outcome_manifest)
        for outcome_path, forced_path, outcome in zip(
            outcome_paths, forced_paths, outcomes)
    ]

    baseline_progress = archive.progress_m.numpy().astype(
        np.float64, copy=False)
    branch_progress = np.column_stack([
        baseline_progress,
        *[
            outcome['progress_m'].astype(np.float64, copy=False)
            for outcome in outcomes
        ],
    ])
    advantage_m = branch_progress[:, 1:] - baseline_progress[:, None]
    fingerprints = _geometry_fingerprints(archive.task)
    group_keys = _geometry_group_keys(archive.task)
    feature_started = time.perf_counter()
    features = _frozen_features(
        moe, archive.task, args.feature_batch_size)
    feature_elapsed_s = time.perf_counter() - feature_started

    oof_prediction, oof_report = _grouped_oof_advantage_mlp(
        features, advantage_m, branch_progress,
        fingerprints, group_keys,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        quotas=quotas,
        positive_margin_m=args.positive_margin_m)
    oof_score = oof_prediction.max(axis=1)
    oof_choice = oof_prediction.argmax(axis=1) + 1

    model, scaler, full_fit_audit = _fit_advantage_mlp(
        features, advantage_m,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_seed=args.seed,
        shuffle_seed=args.shuffle_seed)
    full_prediction = _predict_advantage(model, scaler, features)

    actors: dict[str, DirectSeedMoEActor] = {}
    fraction_audits: dict[str, dict[str, Any]] = {}
    for fraction, tag in zip(selection_fractions, tags):
        if fraction == 1.0:
            threshold = float(np.nextafter(oof_score.min(), -np.inf))
        else:
            threshold = float(np.quantile(
                oof_score, 1.0 - fraction, method='linear'))
        oof_deployed = np.where(
            oof_score > threshold, oof_choice, 0)
        actor = _nonlinear_gate_actor(
            moe, model, scaler,
            threshold_m=threshold,
            gate_hidden_dim=args.hidden_dim)
        _validate_moe_baseline_branch(actor, base)
        deployed = actor.expert_index(archive.task).numpy()
        raw_folded_logits = actor.gate_logits(archive.task).detach().numpy()
        if not bool((raw_folded_logits[:, 0] == 0.0).all()):
            raise RuntimeError(
                f'{tag} deployed baseline logit is not exactly zero')
        expected = np.where(
            full_prediction.max(axis=1) > threshold,
            full_prediction.argmax(axis=1) + 1, 0)
        if not np.array_equal(deployed, expected):
            mismatch = int(np.count_nonzero(deployed != expected))
            raise RuntimeError(
                f'{tag} folded hard route differs on {mismatch} tasks')
        rows = np.arange(len(branch_progress))
        gated_progress = branch_progress[rows, deployed]
        fraction_audits[tag] = {
            'selection_fraction': float(fraction),
            'role': (
                'primary-conservative' if math.isclose(fraction, 0.10)
                else 'higher-selection-ablation'),
            'threshold_source': (
                'quantile of grouped training OOF maximum predicted '
                'specialist advantage only'),
            'validation_artifacts_read': False,
            'oof_quantile': float(1.0 - fraction),
            'predicted_advantage_threshold_m': threshold,
            'strict_comparison': (
                'best specialist predicted advantage > OOF threshold; '
                'otherwise baseline'),
            'oof_realization': _gated_metrics(
                oof_deployed, branch_progress,
                positive_margin_m=args.positive_margin_m),
            'full_fit_train_realization': {
                'gated_progress_mean_m': float(gated_progress.mean()),
                **_gated_metrics(
                    deployed, branch_progress,
                    positive_margin_m=args.positive_margin_m),
            },
            'baseline_logit_parameter_row_exact_zero': True,
            'serialized_route_matches_folded_full_model': True,
        }
        actors[tag] = actor

    deployment_timing = {
        'measurement_note': (
            'single-process CPU microbenchmark; informative, not a latency '
            'guarantee; projection/controller dominate end-to-end runtime'),
        'source_linear_gate': _generator_timing(moe, archive.task),
        'nonlinear_gate': {
            tag: _generator_timing(actor, archive.task)
            for tag, actor in actors.items()
        },
    }
    common_audit = {
        'method': 'outcome-matched-frozen-multi-branch-mlp-gate-v1',
        'seed': int(args.seed),
        'shuffle_seed': int(args.shuffle_seed),
        'source_files': source_files,
        'branch_audit': branch_audit,
        'forced_specialist_audit': forced_audit,
        'runner_audit': runner_audit,
        'outcome_manifest': outcome_manifest,
        'training_data': {
            'task_count': int(len(branch_progress)),
            'n_experts': n_experts,
            'feature_dim': int(features.shape[1]),
            'specialist_advantage_target_count': n_experts - 1,
            'baseline_progress_mean_m': float(baseline_progress.mean()),
            'branch_progress_mean_m': {
                str(index): float(branch_progress[:, index].mean())
                for index in range(n_experts)
            },
            'branch_oracle_progress_mean_m': float(
                branch_progress.max(axis=1).mean()),
            'branch_oracle_gain_mm': float(
                (branch_progress.max(axis=1)
                 - baseline_progress).mean() * 1e3),
            'feature_matrix_sha256': _array_sha256(features),
            'advantage_target_sha256': _array_sha256(advantage_m),
            'feature_extraction_elapsed_s': float(feature_elapsed_s),
        },
        'training_config': {
            'architecture': 'Linear(256,64)-SiLU-Linear(64,3)',
            'target': 'raw metre expected advantage; no target scaling',
            'hidden_dim': int(args.hidden_dim),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'learning_rate': float(args.learning_rate),
            'weight_decay': float(args.weight_decay),
            'device': 'cpu',
            'deterministic_algorithms': True,
            'fold_model_seed': 'seed + zero-based fold index',
            'fold_shuffle_seed': (
                'shuffle_seed + zero-based fold index'),
            'full_model_seed': int(args.seed),
            'full_shuffle_seed': int(args.shuffle_seed),
        },
        'oof': oof_report,
        'full_fit': full_fit_audit,
        'timing': deployment_timing,
        'deployment_protocol': {
            'one_deterministic_seed': True,
            'task_only_hard_argmax_gate': True,
            'hard_selected_expert_heads_per_task': 1,
            'candidate_enumeration': 0,
            'return_model_queries': 0,
            'controller_probes': 0,
            'max_ik_refinements': 1,
            'controller_rollouts': 1,
        },
        'validation_artifacts_read': False,
    }

    reports = {}
    for tag in tags:
        output, report_path = output_paths[tag]
        metadata = {
            **common_audit,
            'selection': fraction_audits[tag],
            'source_moe_update_step': int(moe_payload['update_step']),
            'output_contract': (
                'self-contained direct-seed-hard-moe-v1; nonlinear '
                'task-only gate, all seed branches frozen'),
        }
        checkpoint = direct_seed_moe_checkpoint(
            actors[tag],
            update_step=int(moe_payload['update_step']),
            metadata=metadata)
        _exclusive_save(checkpoint, output, json_value=False)
        report = {
            **common_audit,
            'selection': fraction_audits[tag],
            'output': file_fingerprint(output),
            'output_checkpoint_format': checkpoint['format'],
            'output_update_step': int(checkpoint['update_step']),
        }
        _exclusive_save(report, report_path, json_value=True)
        reports[tag] = {
            'output': str(output),
            'output_sha256': report['output']['sha256'],
            'oof': fraction_audits[tag]['oof_realization'],
            'full_fit_train': fraction_audits[
                tag]['full_fit_train_realization'],
        }
    print(json.dumps({
        'outputs': reports,
        'validation_artifacts_read': False,
        'oof_quota_grid': oof_report['quota_grid'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()


__all__ = [
    '_fit_advantage_mlp',
    '_fold_scaler_into_first_layer',
    '_grouped_oof_advantage_mlp',
    '_nonlinear_gate_actor',
]
