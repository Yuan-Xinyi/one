"""Stable controller phase starting from a strong static seed ensemble.

This is the forward half of the production bidirectional loop.  The seed
ensemble is frozen and deployment remains search-free: one batched static
selector call chooses one seed, then exactly one controller episode runs.

Compared with the historical joint loop, this phase:

* reproduces the calibrated conservative seed deployment rule in reset lanes;
* uses a low actor learning rate and a phase-start deterministic-action anchor;
* evaluates immutable controller blocks with the same frozen seed choices;
* promotes only a validation-improving block while retaining first-valid
  coverage.  The next backward phase can exhaustively relabel candidates under
  the promoted controller and refit the static ensemble.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from Yuan.RL_controller.algorithms.ppo import RewardScaler, train as ppo_train
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    atomic_torch_save,
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
from Yuan.unified_rl.evaluate import load_seed_policy
from Yuan.unified_rl.evaluate_residual import geometry_grouped_bootstrap_ci
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.provenance import (
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.reproducibility import device_identity, seed_global_rng
from Yuan.unified_rl.seed_deployment import (
    deployment_config_from_checkpoint,
    select_seed_deployment,
)
from Yuan.unified_rl.seed_distribution import SeedPolicyLineDistribution
from Yuan.unified_rl.seed_policy import seed_policy_ensemble_states
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _same_artifact(saved: dict, current: dict, *, label: str) -> None:
    if (saved.get('size') != current.get('size')
            or saved.get('sha256') != current.get('sha256')):
        raise ValueError(
            f'{label} differs from source provenance: '
            f'expected {saved.get("sha256")}, got {current.get("sha256")}')


def _pad_index(start: int, end: int, total: int,
               chunk_size: int) -> tuple[torch.Tensor, int]:
    n_real = end - start
    index = torch.arange(start, end, dtype=torch.long)
    if n_real < chunk_size:
        if total < 1:
            raise ValueError('cannot pad an empty evaluation dataset')
        index = torch.cat([
            index,
            torch.full((chunk_size - n_real,), end - 1, dtype=torch.long),
        ])
    return index, n_real


@torch.no_grad()
def _static_seed_indices(
    policy,
    checkpoint: dict,
    dataset: CachedSeedCandidateDataset,
    kin,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one deployment seed and one first-valid seed per task."""
    deployment = deployment_config_from_checkpoint(checkpoint)
    include_log_manip = bool(checkpoint.get('seed_include_log_manip', True))
    include_ray_error = bool(checkpoint.get('seed_include_ray_error', True))
    include_directional = bool(
        checkpoint.get('seed_include_directional_dynamics', False))
    selected_parts = []
    first_parts = []
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        index = torch.arange(start, end)
        candidates = dataset.batch.index_select(index).to(
            kin.device, dtype=kin.dtype)
        features = initial_observation_features(
            kin, candidates,
            include_log_manip=include_log_manip,
            include_ray_error=include_ray_error,
            include_directional_dynamics=include_directional)
        dist, _, feasibility = policy.distribution_and_values(
            features, candidates.valid)
        decision = select_seed_deployment(
            dist.logits, feasibility, candidates.valid, deployment)
        selected_parts.append(decision.selected_index.cpu())
        first_parts.append(
            candidates.valid.float().argmax(dim=-1).cpu())
    return torch.cat(selected_parts), torch.cat(first_parts)


@torch.no_grad()
def _evaluate_fixed_seeds(
    agent,
    env,
    dataset: CachedSeedCandidateDataset,
    selected_index: torch.Tensor,
    first_index: torch.Tensor,
    *,
    chunk_size: int,
    gamma: float,
) -> dict[str, np.ndarray]:
    n = len(dataset)
    outputs = {
        'policy_progress_m': np.empty(n, dtype=np.float32),
        'policy_episode_len': np.empty(n, dtype=np.int64),
        'policy_term_reason': np.empty(n, dtype=np.int32),
        'first_valid_progress_m': np.empty(n, dtype=np.float32),
        'first_valid_episode_len': np.empty(n, dtype=np.int64),
        'first_valid_term_reason': np.empty(n, dtype=np.int32),
    }
    agent.eval()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        index, n_real = _pad_index(start, end, n, chunk_size)
        candidates = dataset.batch.index_select(index).to(
            env.device, dtype=env.kin.dtype)
        for prefix, source_index in (
                ('policy', selected_index),
                ('first_valid', first_index)):
            action = source_index[index].to(env.device)
            result = rollout_selected_seeds(
                env, candidates, action, FrozenRLController(agent),
                gamma=gamma)
            target = slice(start, end)
            outputs[f'{prefix}_progress_m'][target] = (
                result.progress_m[:n_real].cpu().numpy())
            outputs[f'{prefix}_episode_len'][target] = (
                result.episode_len[:n_real].cpu().numpy())
            outputs[f'{prefix}_term_reason'][target] = (
                result.term_reason[:n_real].cpu().numpy())
    return outputs


def select_promoted_block(
    evaluations: Sequence[dict[str, Any]],
    *,
    first_valid_tolerance_m: float,
    require_positive_ci: bool = False,
    harm_rate_tolerance: float = math.inf,
) -> int:
    """Return the highest-progress eligible block, including block zero.

    The mean-only mode is retained for small unit-level callers.  Production
    controller refinement additionally requires a positive grouped-bootstrap
    lower bound and no material increase in the rate of seeds that lose more
    than one millimetre relative to first-valid.
    """
    if not evaluations:
        raise ValueError('at least one evaluation is required')
    if (not math.isfinite(first_valid_tolerance_m)
            or first_valid_tolerance_m < 0.0):
        raise ValueError('first_valid_tolerance_m must be finite and non-negative')
    if (math.isnan(harm_rate_tolerance)
            or harm_rate_tolerance < 0.0):
        raise ValueError('harm_rate_tolerance must be non-negative')
    baseline_first = float(evaluations[0]['first_valid_progress_mean_m'])
    baseline_harm = float(evaluations[0].get(
        'policy_harm_gt_1mm_rate', 0.0))
    best_index = 0
    best_progress = float(evaluations[0]['policy_progress_mean_m'])
    for index, result in enumerate(evaluations[1:], start=1):
        first = float(result['first_valid_progress_mean_m'])
        progress = float(result['policy_progress_mean_m'])
        if first < baseline_first - first_valid_tolerance_m:
            continue
        if (require_positive_ci
                and float(result.get(
                    'gain_vs_baseline_ci95_low_m', -math.inf)) <= 0.0):
            continue
        harm = float(result.get('policy_harm_gt_1mm_rate', math.inf))
        if harm > baseline_harm + harm_rate_tolerance:
            continue
        if progress > best_progress:
            best_index = index
            best_progress = progress
    return best_index


def _summary(
    block: int,
    outputs: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    fingerprints: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    policy = outputs['policy_progress_m']
    first = outputs['first_valid_progress_m']
    delta = policy - baseline['policy_progress_m']
    estimate, low, high, groups = geometry_grouped_bootstrap_ci(
        delta, fingerprints, seed=bootstrap_seed,
        samples=bootstrap_samples)
    return {
        'block': int(block),
        'policy_progress_mean_m': float(policy.mean()),
        'first_valid_progress_mean_m': float(first.mean()),
        'policy_episode_len_mean': float(
            outputs['policy_episode_len'].mean()),
        'policy_gain_over_first_mean_m': float((policy - first).mean()),
        'policy_harm_gt_1mm_rate': float(
            (policy < first - 0.001).mean()),
        'gain_vs_baseline_geometry_macro_m': estimate,
        'gain_vs_baseline_ci95_low_m': low,
        'gain_vs_baseline_ci95_high_m': high,
        'geometry_groups': groups,
    }


def _save_evaluation(
    path: Path,
    outputs: dict[str, np.ndarray],
    summary: dict[str, Any],
    selected_index: torch.Tensor,
    first_index: torch.Tensor,
    fingerprints: Sequence[str],
) -> None:
    payload = dict(outputs)
    payload.update({
        key: np.asarray(value)
        for key, value in summary.items()
    })
    payload.update({
        'policy_candidate_index': selected_index.numpy(),
        'first_valid_candidate_index': first_index.numpy(),
        'task_geometry_sha256': np.asarray(fingerprints),
    })
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Refine a controller on the exact single-seed deployment '
            'distribution of a frozen static ensemble.'))
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--controller-ckpt', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--block-steps', type=int, default=65_536)
    parser.add_argument('--controller-n-envs', type=int, default=128)
    parser.add_argument('--controller-n-steps', type=int, default=64)
    parser.add_argument('--first-block-actor-warmup', type=int, default=0)
    parser.add_argument('--actor-lr', type=float, default=3e-6)
    parser.add_argument('--critic-lr', type=float, default=1e-5)
    parser.add_argument('--anchor-coef', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--train-log-std', type=float, default=-1.5)
    parser.add_argument('--policy-reset-prob', type=float, default=0.85)
    parser.add_argument('--uniform-reset-prob', type=float, default=0.10)
    parser.add_argument('--fallback-reset-prob', type=float, default=0.05)
    parser.add_argument('--eval-chunk-size', type=int, default=1024)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--first-valid-tolerance-m', type=float, default=0.001)
    parser.add_argument('--harm-rate-tolerance', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=41000)
    args = parser.parse_args()

    for name in ('blocks', 'block_steps', 'controller_n_envs',
                 'controller_n_steps', 'eval_chunk_size',
                 'bootstrap_samples'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.first_block_actor_warmup < 0:
        raise ValueError('--first-block-actor-warmup must be non-negative')
    for name in ('actor_lr', 'critic_lr', 'target_kl'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    for name in ('anchor_coef', 'ent_coef', 'first_valid_tolerance_m',
                 'harm_rate_tolerance'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{name.replace("_", "-")} must be non-negative')
    if not math.isfinite(args.train_log_std):
        raise ValueError('--train-log-std must be finite')
    reset_probs = (
        args.policy_reset_prob, args.uniform_reset_prob,
        args.fallback_reset_prob)
    if (not all(math.isfinite(value) and value >= 0.0
                for value in reset_probs)
            or not math.isclose(sum(reset_probs), 1.0, abs_tol=1e-6)):
        raise ValueError('reset probabilities must be non-negative and sum to 1')

    source_path = Path(args.source_checkpoint).expanduser().resolve(strict=True)
    candidate_path = Path(args.candidates).expanduser().resolve(strict=True)
    source_dir = source_path.parent
    controller_dir = resolve_controller_dir(
        source_dir if args.controller_ckpt is None else args.controller_ckpt)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if os.path.lexists(out_dir):
        raise FileExistsError(f'refusing to overwrite output directory: {out_dir}')
    out_dir.mkdir(parents=True)
    snapshot_dir = out_dir / 'snapshots'
    snapshot_dir.mkdir()

    seed_global_rng(args.seed)
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if not isinstance(source, dict):
        raise ValueError('source checkpoint must contain a dictionary')
    if seed_policy_ensemble_states(source) is None:
        raise ValueError('source checkpoint must contain a seed ensemble')
    if source.get('seed_selector_objective') != 'progress_m':
        raise ValueError('source selector must optimize progress_m')
    provenance = source.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('source checkpoint has no provenance dictionary')
    _same_artifact(
        provenance['candidate_cache'], file_fingerprint(candidate_path),
        label='candidate cache')

    policy, loaded = load_seed_policy(source_path, device)
    if loaded.get('seed_ensemble') != source.get('seed_ensemble'):
        raise RuntimeError('seed ensemble changed while loading')
    policy.eval()

    train_env = build_env_from_run(
        controller_dir, args.controller_n_envs, device,
        env_overrides={'observe_ray_error': True})
    eval_env = build_env_from_run(
        controller_dir, args.eval_chunk_size, device,
        env_overrides={'observe_ray_error': True})
    dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    dataset, validity = validate_cached_dataset(
        dataset, train_env.kin, train_env.collision,
        cone_deg=train_env.cfg.cone_deg)
    train_dataset = dataset.select_source_tasks(
        torch.as_tensor(source['train_task_indices']).cpu())
    validation_dataset = dataset.select_source_tasks(
        torch.as_tensor(source['validation_task_indices']).cpu())
    assert_same_valid_mask(
        train_dataset, source['train_valid_mask'], label='training')
    assert_same_valid_mask(
        validation_dataset, source['validation_valid_mask'],
        label='validation')
    if set(train_dataset.task_fingerprints) & set(
            validation_dataset.task_fingerprints):
        raise ValueError('training and validation geometries overlap')

    include_log_manip = bool(source.get('seed_include_log_manip', True))
    include_ray_error = bool(source.get('seed_include_ray_error', True))
    include_directional = bool(
        source.get('seed_include_directional_dynamics', False))
    train_env.line_dist = SeedPolicyLineDistribution(
        train_dataset, policy, train_env.kin,
        policy_prob=args.policy_reset_prob,
        uniform_prob=args.uniform_reset_prob,
        fallback_prob=args.fallback_reset_prob,
        deterministic_policy=True,
        include_log_manip=include_log_manip,
        include_ray_error=include_ray_error,
        include_directional_dynamics=include_directional,
        independent_rng_streams=True,
        seed_deployment=source['seed_deployment'],
        seed=args.seed + 1)

    controller = load_controller_agent(controller_dir, train_env, device)
    embedded_hash = state_dict_fingerprint(source['controller'])
    if embedded_hash != source.get('controller_state_sha256'):
        raise ValueError('source controller hash metadata is inconsistent')
    if state_dict_fingerprint(controller.state_dict()) != embedded_hash:
        raise ValueError('controller checkpoint differs from source controller')
    anchor = copy.deepcopy(controller).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    baseline_controller_state = _cpu_tree(controller.state_dict())
    controller.log_std.data.fill_(args.train_log_std)

    actor_parameters = [
        *controller._actor_trunk.parameters(),
        *controller._mean_head.parameters(),
    ]
    optimizer = torch.optim.Adam([
        {'params': actor_parameters, 'lr': args.actor_lr},
        {'params': controller.critic.parameters(), 'lr': args.critic_lr},
        {'params': [controller.log_std], 'lr': 0.0},
    ], eps=1e-5)

    source_config = load_run_config(controller_dir)
    base_cfg = ppo_config_from_run(source_config)
    gamma = base_cfg.gamma if args.gamma is None else args.gamma
    gae_lambda = (
        base_cfg.gae_lambda if args.gae_lambda is None else args.gae_lambda)
    if (not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0
            or not math.isfinite(gae_lambda)
            or not 0.0 <= gae_lambda <= 1.0):
        raise ValueError('gamma and gae-lambda must be finite in [0, 1]')
    block_cfg = dataclasses.replace(
        base_cfg,
        total_timesteps=args.block_steps,
        learning_rate=args.actor_lr,
        n_steps=args.controller_n_steps,
        anneal_lr=False,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        actor_warmup_updates=0,
    )
    updates_per_block = (
        block_cfg.total_timesteps
        // (args.controller_n_envs * block_cfg.n_steps))
    if updates_per_block < 1:
        raise ValueError('block-steps is smaller than one PPO update')
    if args.first_block_actor_warmup >= updates_per_block:
        raise ValueError(
            '--first-block-actor-warmup must leave at least one actor update')
    scaler = RewardScaler(args.controller_n_envs, gamma, device)
    if (gamma == base_cfg.gamma
            and source.get('controller_scaler') is not None):
        try:
            scaler.load_state_dict(source['controller_scaler'])
        except ValueError:
            # Old checkpoints can use a different n_env count.  Starting new
            # running statistics is safe; silently reshaping return_acc is not.
            pass

    selected_index, first_index = _static_seed_indices(
        policy, source, validation_dataset, eval_env.kin,
        chunk_size=args.eval_chunk_size)
    fingerprints = validation_dataset.task_fingerprints
    baseline_outputs = _evaluate_fixed_seeds(
        controller, eval_env, validation_dataset,
        selected_index, first_index,
        chunk_size=args.eval_chunk_size, gamma=gamma)
    baseline_summary = _summary(
        0, baseline_outputs, baseline_outputs, fingerprints,
        bootstrap_seed=args.seed + 100,
        bootstrap_samples=args.bootstrap_samples)
    _save_evaluation(
        snapshot_dir / 'eval_block_000.npz', baseline_outputs,
        baseline_summary, selected_index, first_index, fingerprints)
    # Block zero must be the exact source controller.  The lower exploration
    # variance is a training intervention and is not published when no block
    # passes the promotion gate.
    block_states = [baseline_controller_state]
    evaluations = [baseline_summary]
    print(
        '[joint-controller] block 0 baseline: '
        f'policy={baseline_summary["policy_progress_mean_m"]:.6f} m, '
        f'first={baseline_summary["first_valid_progress_mean_m"]:.6f} m',
        flush=True)

    train_log: list[dict[str, Any]] = []
    for block in range(1, args.blocks + 1):
        def log_fn(stats: dict) -> None:
            record = {'block': block, **stats}
            train_log.append(record)
            if stats.get('update') in (1, updates_per_block):
                print(
                    f'[joint-controller] block {block} update '
                    f'{stats["update"]}/{updates_per_block}: '
                    f'kl={stats.get("train/approx_kl", 0.0):.5f}, '
                    f'anchor={stats.get("train/actor_anchor_loss", 0.0):.6f}',
                    flush=True)

        controller.train()
        active_block_cfg = dataclasses.replace(
            block_cfg,
            actor_warmup_updates=(
                args.first_block_actor_warmup if block == 1 else 0))
        ppo_train(
            active_block_cfg, train_env, device,
            log_fn=log_fn,
            agent=controller,
            optimizer=optimizer,
            reward_scaler=scaler,
            anchor_agent=anchor,
            actor_anchor_coef=args.anchor_coef,
            actor_anchor_anneal=False)
        outputs = _evaluate_fixed_seeds(
            controller, eval_env, validation_dataset,
            selected_index, first_index,
            chunk_size=args.eval_chunk_size, gamma=gamma)
        summary = _summary(
            block, outputs, baseline_outputs, fingerprints,
            bootstrap_seed=args.seed + 100 + block,
            bootstrap_samples=args.bootstrap_samples)
        evaluations.append(summary)
        block_states.append(_cpu_tree(controller.state_dict()))
        _save_evaluation(
            snapshot_dir / f'eval_block_{block:03d}.npz', outputs,
            summary, selected_index, first_index, fingerprints)
        atomic_torch_save(
            block_states[-1], snapshot_dir / f'agent_block_{block:03d}.pt')
        print(
            f'[joint-controller] block {block}: '
            f'policy={summary["policy_progress_mean_m"]:.6f} m, '
            f'first={summary["first_valid_progress_mean_m"]:.6f} m, '
            f'delta={summary["gain_vs_baseline_geometry_macro_m"]:+.6f} m '
            f'CI=[{summary["gain_vs_baseline_ci95_low_m"]:+.6f}, '
            f'{summary["gain_vs_baseline_ci95_high_m"]:+.6f}]',
            flush=True)

    best_block = select_promoted_block(
        evaluations,
        first_valid_tolerance_m=args.first_valid_tolerance_m,
        require_positive_ci=True,
        harm_rate_tolerance=args.harm_rate_tolerance)
    controller.load_state_dict(block_states[best_block])
    best_summary = evaluations[best_block]
    promoted = best_block > 0

    output_config = copy.deepcopy(source_config)
    output_config.setdefault('env', {})['n_envs'] = args.controller_n_envs
    output_config['env']['observe_ray_error'] = True
    output_config['ppo'] = dataclasses.asdict(block_cfg)
    output_config.setdefault('unified', {})['joint_controller_refinement'] = {
        'format': 'static-single-seed-controller-refinement-v1',
        'source_checkpoint': str(source_path),
        'blocks': args.blocks,
        'block_steps': args.block_steps,
        'updates_per_block': updates_per_block,
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
        'anchor_coef': args.anchor_coef,
        'train_log_std': args.train_log_std,
        'first_block_actor_warmup': args.first_block_actor_warmup,
        'promotion_requires_positive_ci': True,
        'harm_rate_tolerance': args.harm_rate_tolerance,
        'reset_probabilities': {
            'deployment': args.policy_reset_prob,
            'uniform_valid': args.uniform_reset_prob,
            'first_valid': args.fallback_reset_prob,
        },
        'best_block': best_block,
        'promoted': promoted,
        'inference': 'one-static-seed-one-controller-rollout-v1',
    }
    with open(out_dir / 'config.yaml', 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
    output_config_hash = file_fingerprint(out_dir / 'config.yaml')['sha256']
    controller_state = _cpu_tree(controller.state_dict())
    controller_hash = state_dict_fingerprint(controller_state)

    result = copy.deepcopy(source)
    result['outer_round'] = int(source.get('outer_round', 0)) + 1
    result['phase'] = 'round_complete'
    result['controller'] = controller_state
    result['controller_state_sha256'] = controller_hash
    result['controller_config'] = dataclasses.asdict(block_cfg)
    result['controller_run_config_sha256'] = output_config_hash
    result['controller_optimizer'] = _cpu_tree(optimizer.state_dict())
    result['controller_scaler'] = _cpu_tree(scaler.state_dict())
    result['joint_controller_refinement'] = {
        'format': 'static-single-seed-controller-refinement-v1',
        'source_checkpoint': file_fingerprint(source_path),
        'candidate_cache': file_fingerprint(candidate_path),
        'device': device_identity(device),
        'physical_validation': validity,
        'settings': copy.deepcopy(
            output_config['unified']['joint_controller_refinement']),
        'evaluations': copy.deepcopy(evaluations),
        'best_block': best_block,
        'promoted': promoted,
        'baseline_controller_state_sha256': embedded_hash,
        'promoted_controller_state_sha256': controller_hash,
        'validation_used_for_controller_promotion': True,
        'external_holdout_used': False,
    }
    result_provenance = copy.deepcopy(result['provenance'])
    result_provenance['joint_controller_refinement'] = copy.deepcopy(
        result['joint_controller_refinement'])
    result['provenance'] = result_provenance
    atomic_torch_save(controller_state, out_dir / 'agent.pt')
    atomic_torch_save(result, out_dir / 'unified.pt')
    np.savez_compressed(
        out_dir / 'training_log.npz',
        records=np.asarray(train_log, dtype=object))
    print(
        f'[joint-controller] selected block {best_block}/{args.blocks}; '
        f'promoted={promoted}; policy='
        f'{best_summary["policy_progress_mean_m"]:.6f} m; '
        f'controller_sha256={controller_hash}', flush=True)


if __name__ == '__main__':
    main()
