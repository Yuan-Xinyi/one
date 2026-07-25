"""Train a shielded continuous residual around a frozen unified seed policy."""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import torch

from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedSelection,
)
from Yuan.unified_rl.checkpoint import (
    atomic_torch_save,
    build_env_from_run,
    load_controller_agent,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    require_checkpoint_keys,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    rollout_seed_selection,
)
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.provenance import (
    assert_same_provenance,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.reproducibility import (
    device_identity,
    global_rng_state,
    restore_global_rng,
    seed_global_rng,
)
from Yuan.unified_rl.residual_bandit import (
    ResidualBanditConfig,
    geometry_groups,
    residual_bandit_loss,
    sample_group_balanced_indices,
)
from Yuan.unified_rl.residual_policy import (
    ResidualSeedHead,
    antithetic_gaussian_actions_and_log_prob,
)
from Yuan.unified_rl.residual_seed import (
    ResidualSeedConfig,
    apply_residual_seed,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    infer_seed_policy_config,
)


FORMAT = 'unified-residual-seed-v2'
FORMAT_VERSION = 2
GATE_THRESHOLD = 0.5


def _optimizer_to(optimizer: torch.optim.Optimizer,
                  device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f'{label} mismatch: {actual!r} != {expected!r}')


def _initial_shield_stats(config: ResidualSeedConfig) -> dict:
    return {
        'attempts': 0,
        'valid': 0,
        'alpha_counts': {str(value): 0 for value in config.alphas},
        'basis_fallback_rows': 0,
        'max_position_error': 0.0,
        'max_branch_distance': 0.0,
        'min_cone_cosine': 1.0,
        'min_collision_margin': float('inf'),
        'min_joint_margin': float('inf'),
    }


def _accumulate_shield_stats(stats: dict, result,
                             config: ResidualSeedConfig) -> None:
    alpha = result.accepted_alpha.detach()
    stats['attempts'] += alpha.numel()
    stats['valid'] += int(result.valid.sum().item())
    for value in config.alphas:
        stats['alpha_counts'][str(value)] += int((alpha == value).sum().item())
    diagnostics = result.diagnostics
    stats['basis_fallback_rows'] += int(
        diagnostics.basis_fallback.any(dim=1).sum().item())
    stats['max_position_error'] = max(
        stats['max_position_error'], float(diagnostics.position_error.max().item()))
    stats['max_branch_distance'] = max(
        stats['max_branch_distance'], float(diagnostics.branch_distance.max().item()))
    stats['min_cone_cosine'] = min(
        stats['min_cone_cosine'], float(diagnostics.cone_cosine.min().item()))
    stats['min_collision_margin'] = min(
        stats['min_collision_margin'],
        float(diagnostics.collision_margin.min().item()))
    stats['min_joint_margin'] = min(
        stats['min_joint_margin'], float(diagnostics.joint_margin.min().item()))


def _source_required_keys() -> tuple[str, ...]:
    return (
        'format_version', 'seed_policy', 'controller',
        'controller_config',
        'controller_state_sha256', 'controller_run_config_sha256',
        'seed_architecture', 'seed_include_ray_error',
        'seed_include_log_manip', 'seed_config', 'train_indices',
        'validation_indices', 'train_task_indices',
        'validation_task_indices', 'train_valid_mask',
        'validation_valid_mask', 'split_mode', 'args', 'provenance',
    )


def _resume_required_keys() -> tuple[str, ...]:
    return (
        'format_version', 'completed_updates', 'episodes_completed',
        'episodes_per_update', 'source_checkpoint', 'candidate_cache',
        'controller_artifacts', 'source_seed_state_sha256',
        'controller_state_sha256', 'controller_run_config_sha256',
        'residual_head_state_sha256', 'seed_return', 'return_scale',
        'residual_head', 'residual_architecture', 'shield_config',
        'bandit_config', 'optimizer', 'sampler_rng_state',
        'noise_rng_state', 'train_indices', 'validation_indices',
        'train_task_indices', 'validation_task_indices',
        'train_valid_mask', 'validation_valid_mask', 'split_mode',
        'seed_include_ray_error', 'seed_include_log_manip',
        'source_seed_architecture', 'gate_threshold', 'shield_stats',
        'numpy_rng_state', 'torch_rng_state', 'cuda_rng_state',
        'args', 'provenance',
    )


def _require_nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{label} must be a non-negative integer')
    return value


def _validate_shield_stats(stats, config: ResidualSeedConfig) -> None:
    if not isinstance(stats, dict):
        raise ValueError('resume shield_stats must be a dictionary')
    expected = set(_initial_shield_stats(config))
    if set(stats) != expected:
        raise ValueError('resume shield_stats schema differs')
    for key in ('attempts', 'valid', 'basis_fallback_rows'):
        _require_nonnegative_int(stats[key], f'resume shield_stats.{key}')
    alpha_counts = stats['alpha_counts']
    expected_alphas = {str(value) for value in config.alphas}
    if not isinstance(alpha_counts, dict) or set(alpha_counts) != expected_alphas:
        raise ValueError('resume shield_stats.alpha_counts schema differs')
    for key, value in alpha_counts.items():
        _require_nonnegative_int(
            value, f'resume shield_stats.alpha_counts[{key!r}]')
    for key in ('max_position_error', 'max_branch_distance',
                'min_cone_cosine', 'min_collision_margin',
                'min_joint_margin'):
        value = stats[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or math.isnan(float(value))):
            raise ValueError(f'resume shield_stats.{key} must be numeric')
    if stats['valid'] > stats['attempts']:
        raise ValueError('resume shield_stats.valid exceeds attempts')
    if stats['basis_fallback_rows'] > stats['attempts']:
        raise ValueError(
            'resume shield_stats.basis_fallback_rows exceeds attempts')
    if sum(alpha_counts.values()) != stats['attempts']:
        raise ValueError(
            'resume shield_stats.alpha_counts do not sum to attempts')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        help='immutable unified-bidirectional-v4 unified.pt')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--tasks', type=int, default=64,
                        help='unique-geometry-balanced tasks per update')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--std', type=float, default=0.25)
    parser.add_argument('--return-scale', type=float, default=None)
    parser.add_argument('--rho', type=float, default=0.08)
    parser.add_argument('--seed', type=int, default=24000)
    parser.add_argument('--reject-penalty', type=float, default=0.02)
    parser.add_argument('--gate-entropy-coef', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.updates < 0 or args.tasks < 1:
        raise ValueError('--updates must be non-negative and --tasks positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('--lr must be finite and positive')
    seed_global_rng(args.seed)
    device = torch.device(args.device if args.device is not None
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    source_path = Path(args.source).expanduser().resolve(strict=True)
    if source_path.name != 'unified.pt':
        raise ValueError('--source must name unified.pt')
    source = torch.load(source_path, map_location=device, weights_only=False)
    require_checkpoint_keys(
        source, _source_required_keys(), kind='source unified-v4')
    require_checkpoint_format_version(
        source, 4, kind='source unified-v4')
    if source['provenance'].get('format') != 'unified-bidirectional-v4':
        raise ValueError('source must have unified-bidirectional-v4 provenance')
    if source['split_mode'] != 'task-geometry-grouped-v1':
        raise ValueError('source must use the task-geometry-grouped-v1 split')
    source_settings = source['provenance'].get('settings', {})
    controller_kind = source.get(
        'controller_kind', source_settings.get('controller_kind', 'pure'))
    if controller_kind != 'pure':
        raise ValueError(
            'version-2 residual training supports only a pure RL controller')

    source_fingerprint = file_fingerprint(source_path)
    candidate_fingerprint = source['provenance']['candidate_cache']
    candidate_path = Path(candidate_fingerprint['path'])
    _require_equal(
        file_fingerprint(candidate_path), candidate_fingerprint,
        'candidate cache fingerprint')
    controller_dir = source_path.parent
    controller_artifacts = {
        'agent': file_fingerprint(controller_dir / 'agent.pt'),
        'config': file_fingerprint(controller_dir / 'config.yaml'),
    }
    controller_file_state = torch.load(
        controller_dir / 'agent.pt', map_location='cpu', weights_only=True)
    controller_state_sha256 = state_dict_fingerprint(source['controller'])
    _require_equal(
        controller_state_sha256, source['controller_state_sha256'],
        'source controller state hash')
    _require_equal(
        state_dict_fingerprint(controller_file_state), controller_state_sha256,
        'source agent.pt state hash')
    _require_equal(
        controller_artifacts['config']['sha256'],
        source['controller_run_config_sha256'], 'source config hash')
    run_config = load_run_config(controller_dir)
    controller_gamma = float(ppo_config_from_run(run_config).gamma)
    if not math.isfinite(controller_gamma) or not 0.0 <= controller_gamma <= 1.0:
        raise ValueError('fingerprinted controller gamma must be in [0, 1]')
    source_controller_config = source['controller_config']
    if not isinstance(source_controller_config, dict):
        raise ValueError('source controller_config must be a dictionary')
    source_gamma = source_controller_config.get('gamma')
    if (isinstance(source_gamma, bool)
            or not isinstance(source_gamma, (int, float))
            or not math.isfinite(float(source_gamma))
            or not 0.0 <= float(source_gamma) <= 1.0):
        raise ValueError('source controller_config.gamma must be in [0, 1]')
    _require_equal(
        float(source_gamma), controller_gamma,
        'source controller gamma versus fingerprinted config')
    source_seed_state_sha256 = state_dict_fingerprint(source['seed_policy'])

    dataset_all = CachedSeedCandidateDataset.from_npz(candidate_path)
    train_dataset = dataset_all.select_source_tasks(
        source['train_task_indices'].cpu())
    saved_train_mask = torch.as_tensor(
        source['train_valid_mask'], dtype=torch.bool, device='cpu')
    if saved_train_mask.shape != train_dataset.batch.valid.shape:
        raise ValueError('source training valid mask shape changed')
    if bool((saved_train_mask & ~train_dataset.batch.valid).any().item()):
        raise ValueError('source valid mask enables a cache-invalid candidate')
    train_dataset = train_dataset.with_valid(saved_train_mask)
    validation_dataset = dataset_all.select_source_tasks(
        source['validation_task_indices'].cpu())
    saved_validation_mask = torch.as_tensor(
        source['validation_valid_mask'], dtype=torch.bool, device='cpu')
    if saved_validation_mask.shape != validation_dataset.batch.valid.shape:
        raise ValueError('source validation valid mask shape changed')
    if bool((saved_validation_mask & ~validation_dataset.batch.valid).any().item()):
        raise ValueError('source validation mask enables a cache-invalid candidate')
    if set(train_dataset.task_fingerprints) & set(validation_dataset.task_fingerprints):
        raise ValueError('source grouped split leaks task geometry')
    groups = geometry_groups(train_dataset.task_fingerprints)

    episodes_per_update = 3 * args.tasks
    env = build_env_from_run(controller_dir, episodes_per_update, device)
    controller = load_controller_agent(controller_dir, env, device).eval()
    _require_equal(
        state_dict_fingerprint(controller.state_dict()),
        controller_state_sha256, 'loaded controller state hash')
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    policy_config = infer_seed_policy_config(source)
    selector = CandidateSeedActorCritic(**policy_config.to_dict()).to(device)
    selector.load_state_dict(source['seed_policy'])
    selector.eval()
    for parameter in selector.parameters():
        parameter.requires_grad_(False)
    include_ray_error = bool(source['seed_include_ray_error'])
    include_log_manip = bool(source['seed_include_log_manip'])
    seed_return = source['args'].get('seed_return')
    if seed_return not in ('undiscounted', 'discounted'):
        raise ValueError('source seed objective is missing or invalid')
    if args.return_scale is None:
        args.return_scale = float(source['seed_config']['return_scale'])
    bandit_config = ResidualBanditConfig(
        std=args.std, return_scale=args.return_scale,
        reject_penalty=args.reject_penalty,
        gate_entropy_coef=args.gate_entropy_coef,
        max_grad_norm=args.max_grad_norm)
    shield_config = ResidualSeedConfig(
        rho=args.rho, cone_deg=env.cfg.cone_deg)
    head = ResidualSeedHead(input_dim=2 * policy_config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    sampler_generator = torch.Generator().manual_seed(args.seed + 1)
    noise_generator = torch.Generator().manual_seed(args.seed + 2)
    shield_stats = _initial_shield_stats(shield_config)
    completed_updates = 0

    settings = {
        'tasks': args.tasks,
        'lr': args.lr,
        'std': args.std,
        'return_scale': args.return_scale,
        'rho': args.rho,
        'seed': args.seed,
        'reject_penalty': args.reject_penalty,
        'gate_entropy_coef': args.gate_entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'gate_threshold': GATE_THRESHOLD,
        'device': device_identity(device),
        'seed_return': seed_return,
        'controller_gamma': controller_gamma,
        'sampling': 'uniform-geometry-then-uniform-row-v1',
        'rollout': 'merged-base-plus-minus-v1',
        'residual_architecture': head.architecture,
        'shield_config': dataclasses.asdict(shield_config),
        'bandit_config': dataclasses.asdict(bandit_config),
    }
    provenance = {
        'format': FORMAT,
        'source_checkpoint': source_fingerprint,
        'candidate_cache': candidate_fingerprint,
        'controller_artifacts': controller_artifacts,
        'source_seed_state_sha256': source_seed_state_sha256,
        'controller_state_sha256': controller_state_sha256,
        'controller_run_config_sha256': source['controller_run_config_sha256'],
        'settings': settings,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / 'residual.pt'
    if args.resume is None and output_path.exists():
        raise ValueError(f'refusing to overwrite existing checkpoint: {output_path}')
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve(strict=True)
        if resume_path != output_path:
            raise ValueError('--resume must be OUT_DIR/residual.pt')
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False)
        if (checkpoint.get('format_version') == 1
                or checkpoint.get('provenance', {}).get('format')
                == 'unified-residual-seed-v1'):
            raise ValueError(
                'residual v1 checkpoints are evaluation-only; start a new '
                'v2 training run instead of resuming in place')
        require_checkpoint_keys(
            checkpoint, _resume_required_keys(), kind=FORMAT)
        require_checkpoint_format_version(
            checkpoint, FORMAT_VERSION, kind=FORMAT)
        assert_same_provenance(checkpoint['provenance'], provenance)
        mirrors = {
            'source_checkpoint': source_fingerprint,
            'candidate_cache': candidate_fingerprint,
            'controller_artifacts': controller_artifacts,
            'source_seed_state_sha256': source_seed_state_sha256,
            'controller_state_sha256': controller_state_sha256,
            'controller_run_config_sha256': source['controller_run_config_sha256'],
            'seed_return': seed_return,
            'return_scale': args.return_scale,
            'split_mode': source['split_mode'],
            'gate_threshold': GATE_THRESHOLD,
        }
        for key, value in mirrors.items():
            _require_equal(checkpoint[key], value, f'resume {key}')
        _require_equal(
            checkpoint['residual_architecture'], head.architecture,
            'resume residual architecture')
        _require_equal(
            checkpoint['shield_config'], dataclasses.asdict(shield_config),
            'resume shield config')
        _require_equal(
            checkpoint['bandit_config'], dataclasses.asdict(bandit_config),
            'resume bandit config')
        for key in ('train_indices', 'validation_indices',
                    'train_task_indices', 'validation_task_indices',
                    'train_valid_mask', 'validation_valid_mask'):
            if not torch.equal(checkpoint[key].cpu(), source[key].cpu()):
                raise ValueError(f'resume {key} differs from source')
        _require_equal(
            state_dict_fingerprint(checkpoint['residual_head']),
            checkpoint['residual_head_state_sha256'],
            'resume residual head state hash')
        head.load_state_dict(checkpoint['residual_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        _optimizer_to(optimizer, device)
        sampler_generator.set_state(checkpoint['sampler_rng_state'].cpu())
        noise_generator.set_state(checkpoint['noise_rng_state'].cpu())
        restore_global_rng(checkpoint, device)
        shield_stats = checkpoint['shield_stats']
        _validate_shield_stats(shield_stats, shield_config)
        completed_updates = _require_nonnegative_int(
            checkpoint['completed_updates'], 'resume completed_updates')
        expected_shield_attempts = completed_updates * 2 * args.tasks
        if shield_stats['attempts'] != expected_shield_attempts:
            raise ValueError(
                'resume shield_stats.attempts differs from the completed '
                'training budget')
        if shield_stats['valid'] != shield_stats['attempts']:
            raise ValueError(
                'resume shield_stats contains a rejected training branch')
        expected_episodes = completed_updates * episodes_per_update
        resume_episodes_per_update = _require_nonnegative_int(
            checkpoint['episodes_per_update'], 'resume episodes_per_update')
        resume_episodes_completed = _require_nonnegative_int(
            checkpoint['episodes_completed'], 'resume episodes_completed')
        if (resume_episodes_per_update != episodes_per_update
                or resume_episodes_completed != expected_episodes):
            raise ValueError('resume episode budget is inconsistent')
    if args.updates < completed_updates:
        raise ValueError('--updates is behind the resume checkpoint')

    frozen_controller = FrozenRLController(controller)

    def save() -> None:
        head_state = head.state_dict()
        state = {
            'format_version': FORMAT_VERSION,
            'completed_updates': completed_updates,
            'episodes_completed': completed_updates * episodes_per_update,
            'episodes_per_update': episodes_per_update,
            'source_checkpoint': source_fingerprint,
            'candidate_cache': candidate_fingerprint,
            'controller_artifacts': controller_artifacts,
            'source_seed_state_sha256': source_seed_state_sha256,
            'controller_state_sha256': controller_state_sha256,
            'controller_run_config_sha256': source['controller_run_config_sha256'],
            'residual_head_state_sha256': state_dict_fingerprint(head_state),
            'seed_return': seed_return,
            'return_scale': args.return_scale,
            'residual_head': head_state,
            'residual_architecture': head.architecture,
            'shield_config': dataclasses.asdict(shield_config),
            'bandit_config': dataclasses.asdict(bandit_config),
            'optimizer': optimizer.state_dict(),
            'sampler_rng_state': sampler_generator.get_state(),
            'noise_rng_state': noise_generator.get_state(),
            'train_indices': source['train_indices'].cpu(),
            'validation_indices': source['validation_indices'].cpu(),
            'train_task_indices': source['train_task_indices'].cpu(),
            'validation_task_indices': source['validation_task_indices'].cpu(),
            'train_valid_mask': source['train_valid_mask'].cpu(),
            'validation_valid_mask': source['validation_valid_mask'].cpu(),
            'split_mode': source['split_mode'],
            'seed_include_ray_error': include_ray_error,
            'seed_include_log_manip': include_log_manip,
            'source_seed_architecture': source['seed_architecture'],
            'gate_threshold': GATE_THRESHOLD,
            'shield_stats': shield_stats,
            'args': vars(args),
            'provenance': provenance,
        }
        state.update(global_rng_state(device))
        atomic_torch_save(state, output_path)

    if args.updates == completed_updates and args.resume is None:
        save()
    for update in range(completed_updates + 1, args.updates + 1):
        row_index = sample_group_balanced_indices(
            groups, args.tasks, sampler_generator)
        candidates = train_dataset.batch.index_select(row_index).to(
            device, dtype=env.kin.dtype)
        with torch.no_grad():
            features = initial_observation_features(
                env.kin, candidates,
                include_log_manip=include_log_manip,
                include_ray_error=include_ray_error)
            candidate_index = selector.select(features, candidates.valid)
            representation = selector.selected_representation(
                features, candidates.valid, candidate_index)
        base = candidates.select(candidate_index)
        # The source mask is immutable provenance, but safety is rechecked on
        # every actually sampled base before it reaches the environment.  The
        # disabled/zero shield path is bitwise identity, so this cannot alter
        # the baseline branch.
        base_safety = apply_residual_seed(
            env.kin, env.collision, base.q0, base.p0,
            base.line_dir, base.n_target,
            torch.zeros((args.tasks, 4), device=device, dtype=base.q0.dtype),
            enabled=False, config=shield_config)
        if not bool(base_safety.valid.all().item()):
            bad = torch.nonzero(
                ~base_safety.valid, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                'source mask selected an unsafe base before rollout: '
                f'{bad[:16]}')
        gate_logits, latent_mean = head(representation)
        noise = torch.randn(
            (args.tasks, 4), generator=noise_generator,
            dtype=torch.float32).to(device=device, dtype=latent_mean.dtype)
        latent, log_prob = antithetic_gaussian_actions_and_log_prob(
            latent_mean, noise, std=args.std)
        shield = apply_residual_seed(
            env.kin, env.collision,
            torch.cat([base.q0, base.q0]),
            torch.cat([base.p0, base.p0]),
            torch.cat([base.line_dir, base.line_dir]),
            torch.cat([base.n_target, base.n_target]),
            torch.cat([latent[:, 0], latent[:, 1]]),
            enabled=True, config=shield_config)
        if not bool(shield.valid.all().item()):
            bad = torch.nonzero(~shield.valid, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f'residual shield rejected safety fallback before rollout: {bad[:16]}')
        _accumulate_shield_stats(shield_stats, shield, shield_config)
        plus_q, minus_q = shield.q.split(args.tasks)
        merged = SeedSelection(
            q0=torch.cat([base.q0, plus_q, minus_q]),
            p0=torch.cat([base.p0, base.p0, base.p0]),
            line_dir=torch.cat([base.line_dir, base.line_dir, base.line_dir]),
            n_target=torch.cat([base.n_target, base.n_target, base.n_target]),
        )
        rollout = rollout_seed_selection(
            env, merged, frozen_controller, gamma=controller_gamma)
        returns = (rollout.undiscounted_return
                   if seed_return == 'undiscounted'
                   else rollout.discounted_return)
        base_return, plus_return, minus_return = returns.split(args.tasks)
        alpha_plus, alpha_minus = shield.accepted_alpha.split(args.tasks)
        loss, metrics = residual_bandit_loss(
            gate_logits, log_prob, base_return,
            torch.stack([plus_return, minus_return], dim=1),
            torch.stack([alpha_plus, alpha_minus], dim=1), bandit_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        completed_updates = update
        save()
        if update == 1 or update % 10 == 0 or update == args.updates:
            alpha_nonzero = float((shield.accepted_alpha > 0).float().mean().item())
            print(
                f'[residual] upd {update:>4}/{args.updates}  '
                f'base {metrics["base_return"]:.2f}  '
                f'resid {metrics["branch_return"]:.2f}  '
                f'gate {metrics["gate_probability"]:.3f}  '
                f'alpha>0 {alpha_nonzero:.1%}  grad {float(grad_norm):.3f}',
                flush=True)


if __name__ == '__main__':
    main()
