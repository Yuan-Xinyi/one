"""Optional IK-pool bootstrap/ablation for the one-shot seed generator.

This module is deliberately **not** the main direct-seed learning algorithm;
that is contextual macro-RL in :mod:`Yuan.unified_rl.direct_seed_rl`, trained
from real downstream controller progress.  Here the IK pool is an optional
training-time teacher used to initialise that actor or to report a supervised
ablation.  For each task, a return-weighted soft nearest-support objective
lets a deterministic generator commit to one good IK mode instead of averaging
several incompatible joint-space branches.

Task geometries, rather than rows, are split into fit/model/calibration
partitions.  This prevents duplicated task geometry from leaking across model
selection boundaries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import math
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.direct_seed_model import (
    DirectSeedConfig,
    DirectSeedGenerator,
    direct_seed_checkpoint,
    direct_seed_task,
)


@dataclass(frozen=True)
class DirectSeedLossConfig:
    """Temperatures and differentiable kinematic regularisation."""

    return_temperature_m: float = 0.02
    support_temperature: float = 0.05
    support_weight: float = 1.0
    kinematic_weight: float = 0.01
    position_scale_m: float = 0.01
    cone_deg: float = 30.0
    collision_margin_m: float = 0.0
    collision_scale_m: float = 0.01

    def __post_init__(self) -> None:
        positive = {
            'return_temperature_m': self.return_temperature_m,
            'support_temperature': self.support_temperature,
            'support_weight': self.support_weight,
            'position_scale_m': self.position_scale_m,
            'collision_scale_m': self.collision_scale_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if (not math.isfinite(self.kinematic_weight)
                or self.kinematic_weight < 0.0):
            raise ValueError(
                'kinematic_weight must be finite and non-negative')
        if (not math.isfinite(self.cone_deg)
                or not 0.0 < self.cone_deg <= 30.0):
            raise ValueError('cone_deg must be finite and in (0, 30]')
        if (not math.isfinite(self.collision_margin_m)
                or self.collision_margin_m < 0.0):
            raise ValueError(
                'collision_margin_m must be finite and non-negative')


@dataclass(frozen=True)
class DirectSeedTrainingData:
    task: torch.Tensor
    candidates: torch.Tensor
    returns_m: torch.Tensor
    valid: torch.Tensor
    fallback_q: torch.Tensor
    task_indices: torch.Tensor
    dataset: CachedSeedCandidateDataset


def return_weighted_soft_nearest_support_loss(
    predicted_q: torch.Tensor,
    candidate_q: torch.Tensor,
    returns_m: torch.Tensor,
    valid: torch.Tensor,
    q_half: torch.Tensor,
    *,
    return_temperature_m: float = 0.02,
    support_temperature: float = 0.05,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Return-weighted soft distance to the closest supported IK branch.

    For task ``b`` and candidate ``i``:

    ``log w_bi = log_softmax(return_bi / tau_return)``

    ``loss_b = -tau_support * logsumexp(log w_bi - d(q, q_bi)/tau_support)``

    Invalid slots contribute exactly zero probability.  Unlike MSE to an
    averaged target, this objective has a mode-seeking soft-min gradient.
    """
    if predicted_q.ndim != 2 or predicted_q.shape[-1] != 7:
        raise ValueError('predicted_q must have shape (B, 7)')
    if candidate_q.ndim != 3 or candidate_q.shape[0] != predicted_q.shape[0] \
            or candidate_q.shape[-1] != 7:
        raise ValueError('candidate_q must have shape (B, K, 7)')
    expected = candidate_q.shape[:2]
    if returns_m.shape != expected or valid.shape != expected:
        raise ValueError('returns_m and valid must have shape (B, K)')
    if valid.dtype != torch.bool:
        raise TypeError('valid must have boolean dtype')
    if not all(value.device == predicted_q.device
               for value in (candidate_q, returns_m, valid, q_half)):
        raise ValueError('all loss tensors must be on the same device')
    if not all(value.dtype == predicted_q.dtype
               for value in (candidate_q, returns_m, q_half)):
        raise ValueError(
            'candidate_q, returns_m, and q_half must match predicted dtype')
    if q_half.shape != (7,) or not bool(torch.isfinite(q_half).all()) \
            or not bool((q_half > 0.0).all()):
        raise ValueError('q_half must be finite, positive, and shape (7,)')
    if (not math.isfinite(return_temperature_m)
            or return_temperature_m <= 0.0):
        raise ValueError(
            'return_temperature_m must be finite and positive')
    if (not math.isfinite(support_temperature)
            or support_temperature <= 0.0):
        raise ValueError(
            'support_temperature must be finite and positive')
    if not bool(valid.any(dim=-1).all()):
        raise ValueError('every task must have at least one valid IK support')
    if not bool(torch.isfinite(predicted_q).all()):
        raise ValueError('predicted_q must be finite')
    if not bool(torch.isfinite(candidate_q[valid]).all()):
        raise ValueError('valid candidate_q entries must be finite')
    if not bool(torch.isfinite(returns_m[valid]).all()):
        raise ValueError('valid returns_m entries must be finite')

    safe_candidates = torch.where(
        valid.unsqueeze(-1), candidate_q, predicted_q.unsqueeze(1))
    distance = (
        (predicted_q.unsqueeze(1) - safe_candidates)
        / q_half.view(1, 1, 7)
    ).square().mean(dim=-1)
    minus_inf = torch.full_like(returns_m, float('-inf'))
    masked_returns = torch.where(valid, returns_m, minus_inf)
    log_weight = torch.log_softmax(
        masked_returns / float(return_temperature_m), dim=-1)
    log_kernel = -distance / float(support_temperature)
    per_task = -float(support_temperature) * torch.logsumexp(
        log_weight + log_kernel, dim=-1)

    if reduction == 'none':
        return per_task
    if reduction == 'mean':
        return per_task.mean()
    if reduction == 'sum':
        return per_task.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def direct_seed_kinematic_loss(
    kin,
    collision,
    predicted_q: torch.Tensor,
    p0: torch.Tensor,
    n_target: torch.Tensor,
    config: DirectSeedLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable FK/cone/collision regulariser for approximate IK."""
    if predicted_q.ndim != 2 or predicted_q.shape[-1] != 7:
        raise ValueError('predicted_q must have shape (B, 7)')
    batch_size = predicted_q.shape[0]
    if p0.shape != (batch_size, 3) or n_target.shape != (batch_size, 3):
        raise ValueError('p0 and n_target must have shape (B, 3)')
    p_tcp, rotation, _, _ = kin.tcp_fk_jac(predicted_q)
    position_error = (p_tcp - p0).norm(dim=-1)
    cone_cosine = (rotation[:, :, 2] * n_target).sum(dim=-1)
    position_loss = (
        position_error / float(config.position_scale_m)).square().mean()
    cone_violation = torch.relu(
        math.cos(math.radians(config.cone_deg)) - cone_cosine)
    cone_loss = cone_violation.square().mean()

    if collision is None or not hasattr(collision, 'min_margin'):
        raise ValueError(
            'differentiable training requires collision.min_margin')
    collision_margin = collision.min_margin(
        kin.link_transforms(predicted_q))
    if collision_margin.shape != (batch_size,):
        raise ValueError('collision.min_margin must return shape (B,)')
    collision_violation = torch.relu(
        float(config.collision_margin_m) - collision_margin)
    collision_loss = (
        collision_violation / float(config.collision_scale_m)
    ).square().mean()
    total = position_loss + cone_loss + collision_loss
    return total, {
        'position': position_loss,
        'cone': cone_loss,
        'collision': collision_loss,
        'position_error_m': position_error.mean(),
        'cone_cosine': cone_cosine.mean(),
        'collision_margin_m': collision_margin.mean(),
    }


def geometry_grouped_three_way_split(
    dataset: CachedSeedCandidateDataset,
    *,
    model_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    seed: int = 20260728,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic, geometry-disjoint fit/model/calibration rows."""
    if not 0.0 < model_fraction < 1.0:
        raise ValueError('model_fraction must be in (0, 1)')
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError('calibration_fraction must be in (0, 1)')
    if model_fraction + calibration_fraction >= 1.0:
        raise ValueError('model and calibration fractions leave no fit rows')
    remaining, _, remaining_index, calibration_index = (
        dataset.train_validation_split(calibration_fraction, seed + 1))
    adjusted_model = model_fraction / (1.0 - calibration_fraction)
    _, _, fit_local, model_local = remaining.train_validation_split(
        adjusted_model, seed + 2)
    fit_index = remaining_index[fit_local]
    model_index = remaining_index[model_local]

    fingerprints = dataset.task_fingerprints
    partitions = [
        {fingerprints[int(row)] for row in index.tolist()}
        for index in (fit_index, model_index, calibration_index)
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            overlap = partitions[left].intersection(partitions[right])
            if overlap:
                raise RuntimeError(
                    'geometry-grouped direct-seed split overlaps: '
                    f'partitions {left}/{right}, first={sorted(overlap)[0]}')
    joined = torch.cat([fit_index, model_index, calibration_index])
    if (joined.numel() != len(dataset)
            or joined.unique().numel() != len(dataset)):
        raise RuntimeError(
            'geometry-grouped direct-seed split lost or duplicated rows')
    return fit_index, model_index, calibration_index


def load_direct_seed_training_data(
    candidate_path: str | Path,
    return_path: str | Path,
) -> DirectSeedTrainingData:
    """Load aligned IK solutions and C0/C1 complete-candidate returns."""
    dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    if dataset.fallback_index is None:
        raise ValueError(
            'direct-seed training requires an explicit q0_pilot fallback')
    with np.load(Path(return_path), allow_pickle=False) as archive:
        required = {'progress_m', 'valid', 'task_indices'}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f'return cache is missing keys: {sorted(missing)}')
        progress = torch.from_numpy(
            np.asarray(archive['progress_m'], dtype=np.float32).copy())
        return_valid = torch.from_numpy(
            np.asarray(archive['valid'], dtype=np.bool_).copy())
        return_indices = torch.from_numpy(
            np.asarray(archive['task_indices'], dtype=np.int64).copy())
    expected = (
        dataset.batch.n_tasks, dataset.batch.n_candidates)
    if progress.shape != expected or return_valid.shape != expected:
        raise ValueError(
            f'return cache must have shape {expected}, got '
            f'{tuple(progress.shape)} and {tuple(return_valid.shape)}')
    if not torch.equal(return_indices, dataset.task_indices):
        raise ValueError(
            'candidate and return task_indices are not exactly aligned')

    # The final slot is the classical fallback.  It is available to deployment
    # but must not become a hidden candidate for the learned direct generator.
    support_end = dataset.fallback_index
    candidates = dataset.batch.q0[:, :support_end].clone()
    valid = (
        dataset.batch.valid[:, :support_end]
        & return_valid[:, :support_end]
        & torch.isfinite(progress[:, :support_end])
    )
    if not bool(valid.any(dim=-1).all()):
        bad = torch.nonzero(~valid.any(dim=-1), as_tuple=False).flatten()
        raise ValueError(
            'some tasks have no valid exact IK support: '
            f'{bad[:20].tolist()}')
    task = direct_seed_task(
        dataset.batch.p0, dataset.batch.line_dir, dataset.batch.n_target)
    return DirectSeedTrainingData(
        task=task,
        candidates=candidates,
        returns_m=progress[:, :support_end].clone(),
        valid=valid,
        fallback_q=dataset.batch.q0[:, dataset.fallback_index].clone(),
        task_indices=dataset.task_indices.clone(),
        dataset=dataset,
    )


def _partition_loss(
    model: DirectSeedGenerator,
    data: DirectSeedTrainingData,
    index: torch.Tensor,
    device: torch.device,
    loss_config: DirectSeedLossConfig,
    batch_size: int,
) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, index.numel(), batch_size):
            rows = index[start:start + batch_size]
            task = data.task[rows].to(device)
            predicted = model(task)
            loss = return_weighted_soft_nearest_support_loss(
                predicted,
                data.candidates[rows].to(device),
                data.returns_m[rows].to(device),
                data.valid[rows].to(device),
                model.q_half.to(device),
                return_temperature_m=loss_config.return_temperature_m,
                support_temperature=loss_config.support_temperature,
                reduction='none')
            values.append(loss.cpu())
    return float(torch.cat(values).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--candidates',
        default='Yuan/unified_rl/runs/ikpool_full_v1/ikpool_candidates.npz')
    parser.add_argument(
        '--returns',
        default='Yuan/unified_rl/runs/ikpool_full_v1/ikpool_returns.npz')
    parser.add_argument(
        '--controller-dir',
        default='Yuan/unified_rl/runs/r2_grouped_best')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=20260728)
    parser.add_argument('--model-fraction', type=float, default=0.15)
    parser.add_argument('--calibration-fraction', type=float, default=0.15)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument(
        '--smoke', action='store_true',
        help='one epoch on at most 512 tasks; never a publishable run')
    parser.add_argument(
        '--bootstrap-ablation', action='store_true',
        help=(
            'required acknowledgement: this produces only an optional '
            'IK-pool bootstrap/ablation, not the contextual-RL main method'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bootstrap_ablation:
        raise SystemExit(
            'direct_seed_train.py is optional IK-pool supervision only; '
            'pass --bootstrap-ablation explicitly, or use direct_seed_rl.py '
            'for the main contextual-RL method')
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit('--epochs and --batch-size must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_direct_seed_training_data(args.candidates, args.returns)
    limit = 512 if args.smoke else args.max_tasks
    if limit is not None and limit < len(data.dataset):
        if limit < 8:
            raise ValueError('--max-tasks must be at least 8')
        subset = torch.randperm(
            len(data.dataset),
            generator=torch.Generator().manual_seed(args.seed))[:limit]
        selected = data.dataset.index_select(subset)
        data = DirectSeedTrainingData(
            task=data.task[subset],
            candidates=data.candidates[subset],
            returns_m=data.returns_m[subset],
            valid=data.valid[subset],
            fallback_q=data.fallback_q[subset],
            task_indices=data.task_indices[subset],
            dataset=selected)

    fit, model_rows, calibration = geometry_grouped_three_way_split(
        data.dataset,
        model_fraction=args.model_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed)
    task_mean = data.task[fit].mean(dim=0)
    task_std = data.task[fit].std(dim=0).clamp_min(1e-6)

    from Yuan.unified_rl.checkpoint import (
        build_env_from_run, resolve_controller_dir)
    env = build_env_from_run(
        resolve_controller_dir(args.controller_dir), 1, device)
    generator = DirectSeedGenerator(
        env.kin.lmt_lo.cpu(), env.kin.lmt_up.cpu(),
        DirectSeedConfig(),
        task_mean=task_mean, task_std=task_std).to(device)
    loss_config = DirectSeedLossConfig()
    optimiser = torch.optim.AdamW(
        generator.parameters(), lr=args.lr, weight_decay=1e-4)
    n_epochs = 1 if args.smoke else args.epochs

    for epoch in range(n_epochs):
        generator.train()
        order = fit[torch.randperm(fit.numel())]
        sum_loss = 0.0
        n_seen = 0
        for start in range(0, order.numel(), args.batch_size):
            rows = order[start:start + args.batch_size]
            task = data.task[rows].to(device)
            candidate_q = data.candidates[rows].to(device)
            returns_m = data.returns_m[rows].to(device)
            valid = data.valid[rows].to(device)
            predicted = generator(task)
            support = return_weighted_soft_nearest_support_loss(
                predicted, candidate_q, returns_m, valid,
                generator.q_half,
                return_temperature_m=loss_config.return_temperature_m,
                support_temperature=loss_config.support_temperature)
            kinematic, _ = direct_seed_kinematic_loss(
                env.kin, env.collision, predicted,
                task[:, :3], task[:, 6:9], loss_config)
            loss = (
                loss_config.support_weight * support
                + loss_config.kinematic_weight * kinematic)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimiser.step()
            n_batch = rows.numel()
            sum_loss += float(loss.detach()) * n_batch
            n_seen += n_batch
        print(
            f'[direct-seed] epoch={epoch + 1}/{n_epochs} '
            f'fit_loss={sum_loss / max(n_seen, 1):.6f}',
            flush=True)

    metrics = {
        'model_support_loss': _partition_loss(
            generator, data, model_rows, device, loss_config,
            args.batch_size),
        'calibration_support_loss': _partition_loss(
            generator, data, calibration, device, loss_config,
            args.batch_size),
    }
    payload = direct_seed_checkpoint(generator)
    payload.update({
        'training_format': 'direct-seed-ikpool-bootstrap-ablation-v1',
        'method_role': 'optional-bootstrap-or-supervised-ablation',
        'loss_config': asdict(loss_config),
        'candidate_path': str(Path(args.candidates)),
        'return_path': str(Path(args.returns)),
        'controller_dir': str(Path(args.controller_dir)),
        'task_indices': data.task_indices,
        'fit_rows': fit,
        'model_rows': model_rows,
        'calibration_rows': calibration,
        'metrics': metrics,
        'seed': int(args.seed),
        'epochs': int(n_epochs),
        'smoke': bool(args.smoke),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f'[direct-seed] saved {output}  metrics={metrics}', flush=True)


if __name__ == '__main__':
    main()


__all__ = [
    'DirectSeedLossConfig',
    'DirectSeedTrainingData',
    'direct_seed_kinematic_loss',
    'geometry_grouped_three_way_split',
    'load_direct_seed_training_data',
    'return_weighted_soft_nearest_support_loss',
]
