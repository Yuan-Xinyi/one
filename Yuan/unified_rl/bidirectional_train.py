"""Alternating forward/backward optimization for unified seed-control RL.

Forward phase:
    freeze a seed-policy snapshot and adapt continuous PPO on its reset
    distribution.
Backward phase:
    freeze the resulting controller and update the seed macro-policy from
    complete downstream returns.

The peers never move within a phase, so every PPO likelihood ratio and every
seed return has a well-defined policy version.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import shutil
from pathlib import Path

import torch
import yaml

from Yuan.RL_controller.algorithms.ppo import RewardScaler, train as ppo_train
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    adapt_controller_optimizer_observation_state,
    atomic_torch_save,
    build_env_from_run,
    load_controller_agent,
    load_controller_state_dict,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    require_checkpoint_keys,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    rollout_selected_seeds,
)
from Yuan.unified_rl.features import (
    fit_candidate_feature_normalization,
    initial_observation_features,
)
from Yuan.unified_rl.provenance import (
    assert_same_provenance,
    controller_fingerprint,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.reproducibility import (
    device_identity,
    global_rng_state,
    restore_global_rng,
    seed_global_rng,
)
from Yuan.unified_rl.seed_distribution import SeedPolicyLineDistribution
from Yuan.unified_rl.seed_gpi import (
    DenseSeedConfig,
    collect_dense_seed_rollout,
    update_dense_seed_policy,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    SeedPolicyConfig,
    infer_seed_policy_config,
)
from Yuan.unified_rl.seed_ppo import (
    SeedPPOConfig,
    collect_seed_rollout,
    update_seed_policy,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resume_position(checkpoint: dict) -> tuple[int, bool]:
    """Return next outer round and whether its controller phase is done."""
    phase = checkpoint.get('phase', 'round_complete')
    if phase not in ('warmup_complete', 'controller_complete', 'round_complete'):
        raise ValueError(f'unknown checkpoint phase: {phase!r}')
    completed_round = int(checkpoint['outer_round'])
    after_controller = phase == 'controller_complete'
    next_round = completed_round if after_controller else completed_round + 1
    return next_round, after_controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--init-controller-ckpt', required=True)
    parser.add_argument('--out-dir', required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        '--resume', default=None,
        help='continue the exact run recorded by a unified checkpoint')
    checkpoint_group.add_argument(
        '--branch-from', default=None,
        help='start a new v4 run from a phase boundary; only reset selection '
             'mode may differ')
    parser.add_argument('--allow-legacy-resume', action='store_true',
                        help='resume an early checkpoint without provenance')
    parser.add_argument('--device', default=None)
    parser.add_argument('--outer-rounds', type=int, default=4)
    parser.add_argument('--seed-warmup-updates', type=int, default=200)
    parser.add_argument('--seed-updates-per-round', type=int, default=100)
    parser.add_argument('--seed-tasks-per-update', type=int, default=28)
    parser.add_argument('--seed-samples-per-task', type=int, default=4)
    parser.add_argument('--backward-mode',
                        choices=('dense-gpi', 'sampled-ppo'),
                        default='dense-gpi')
    parser.add_argument('--no-log-manip', action='store_true',
                        help='ablate log positional manipulability feature')
    parser.add_argument(
        '--directional-dynamics', action='store_true',
        help='append 10 controller-aligned directional seed features')
    parser.add_argument('--seed-encoder', choices=('mean', 'attention'),
                        default='mean')
    parser.add_argument('--seed-hidden-dim', type=int, default=256)
    parser.add_argument('--seed-attention-heads', type=int, default=4)
    parser.add_argument('--seed-attention-layers', type=int, default=1)
    parser.add_argument('--seed-attention-ff-mult', type=int, default=2)
    parser.add_argument('--controller-n-envs', type=int, default=128)
    parser.add_argument('--controller-steps-per-round', type=int, default=1_000_000)
    parser.add_argument('--controller-lr', type=float, default=1e-5)
    parser.add_argument('--first-controller-actor-warmup', type=int, default=100)
    parser.add_argument('--policy-reset-prob', type=float, default=0.7)
    parser.add_argument('--uniform-reset-prob', type=float, default=0.2)
    parser.add_argument('--fallback-reset-prob', type=float, default=0.1)
    parser.add_argument(
        '--deterministic-policy-reset', action='store_true',
        help='use deployed argmax inside the policy component of controller resets')
    parser.add_argument('--seed', type=int, default=13000)
    parser.add_argument('--skip-physical-validation', action='store_true')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed-return', choices=('undiscounted', 'discounted'),
                        default='undiscounted')
    args = parser.parse_args()

    nonnegative = {
        '--outer-rounds': args.outer_rounds,
        '--seed-warmup-updates': args.seed_warmup_updates,
        '--seed-updates-per-round': args.seed_updates_per_round,
        '--controller-steps-per-round': args.controller_steps_per_round,
        '--first-controller-actor-warmup': args.first_controller_actor_warmup,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
    positive = {
        '--seed-tasks-per-update': args.seed_tasks_per_update,
        '--seed-samples-per-task': args.seed_samples_per_task,
        '--controller-n-envs': args.controller_n_envs,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(args.controller_lr) or args.controller_lr <= 0.0:
        raise ValueError('--controller-lr must be finite and positive')
    reset_probs = (
        args.policy_reset_prob,
        args.uniform_reset_prob,
        args.fallback_reset_prob,
    )
    if (not all(math.isfinite(value) and value >= 0.0 for value in reset_probs)
            or not math.isclose(sum(reset_probs), 1.0, abs_tol=1e-6)):
        raise ValueError('reset probabilities must be non-negative and sum to 1')
    if (not math.isfinite(args.validation_fraction)
            or not 0.0 < args.validation_fraction < 1.0):
        raise ValueError('--validation-fraction must be in (0, 1)')

    seed_global_rng(args.seed)
    device = torch.device(args.device if args.device is not None
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = CachedSeedCandidateDataset.from_npz(args.candidates)
    checkpoint_path = args.resume if args.resume is not None else args.branch_from
    checkpoint = (torch.load(
        checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_path is not None else None)
    is_branch = args.branch_from is not None
    if is_branch:
        recorded_out_dir = checkpoint.get('args', {}).get('out_dir')
        source_run = (Path(recorded_out_dir).expanduser().resolve()
                      if recorded_out_dir is not None
                      else Path(checkpoint_path).resolve().parent)
        target_run = out_dir.expanduser().resolve()
        if (target_run.is_relative_to(source_run)
                or source_run.is_relative_to(target_run)):
            raise ValueError(
                '--branch-from output directory must not overlap its source run')
    if checkpoint is not None and args.outer_rounds < int(checkpoint['outer_round']):
        raise ValueError(
            f'--outer-rounds={args.outer_rounds} is behind resume checkpoint '
            f'round {int(checkpoint["outer_round"])}')
    requested_architecture = {
        'encoder_type': args.seed_encoder,
        'hidden_dim': args.seed_hidden_dim,
        'heads': args.seed_attention_heads,
        'layers': args.seed_attention_layers,
        'ff_mult': args.seed_attention_ff_mult,
    }
    saved_provenance = checkpoint.get('provenance') if checkpoint else None
    legacy_v3_resume = (
        saved_provenance is not None
        and saved_provenance.get('format') == 'unified-bidirectional-v3')
    if legacy_v3_resume and args.deterministic_policy_reset:
        raise ValueError(
            'version-3 resume requires sampled policy resets; start a new '
            'version-4 run to change reset semantics')
    split_mode = ('task-geometry-grouped-v1' if checkpoint is None
                  else checkpoint.get('split_mode', 'row-random-v1'))
    provenance_settings = {
        'seed_warmup_updates': args.seed_warmup_updates,
        'seed_updates_per_round': args.seed_updates_per_round,
        'seed_tasks_per_update': args.seed_tasks_per_update,
        'seed_samples_per_task': (
            args.seed_samples_per_task
            if args.backward_mode == 'sampled-ppo' else None),
        'backward_mode': args.backward_mode,
        'controller_n_envs': args.controller_n_envs,
        'controller_steps_per_round': args.controller_steps_per_round,
        'controller_lr': args.controller_lr,
        'first_controller_actor_warmup': args.first_controller_actor_warmup,
        'policy_reset_prob': args.policy_reset_prob,
        'uniform_reset_prob': args.uniform_reset_prob,
        'fallback_reset_prob': args.fallback_reset_prob,
        'seed': args.seed,
        'device': device_identity(device),
        'physical_validation': not args.skip_physical_validation,
        'seed_return': args.seed_return,
        'controller_observation': 'ray-error-34d',
        'seed_log_manip': not args.no_log_manip,
        'seed_directional_dynamics': args.directional_dynamics,
    }
    if not legacy_v3_resume:
        provenance_settings['seed_architecture'] = requested_architecture
        provenance_settings['split_mode'] = split_mode
        provenance_settings['deterministic_policy_reset'] = (
            args.deterministic_policy_reset)
        provenance_settings['controller_reset_rng'] = 'independent-streams-v1'
    provenance = {
        # Bump whenever optimizer or derived-state semantics change.
        'format': ('unified-bidirectional-v3' if legacy_v3_resume
                   else 'unified-bidirectional-v4'),
        'candidate_cache': file_fingerprint(args.candidates),
        'initial_controller': controller_fingerprint(args.init_controller_ckpt),
        'settings': provenance_settings,
    }
    if checkpoint is not None:
        if saved_provenance is None:
            if not args.allow_legacy_resume:
                raise ValueError(
                    'resume checkpoint has no provenance; pass '
                    '--allow-legacy-resume only after manually verifying its '
                    'candidate cache, controller, and phase settings')
            print('[unified] WARNING: accepting legacy resume without provenance')
        else:
            if is_branch:
                if saved_provenance.get('format') != 'unified-bidirectional-v4':
                    raise ValueError('--branch-from requires a version-4 checkpoint')
                if checkpoint.get('phase') not in (
                        'warmup_complete', 'round_complete'):
                    raise ValueError(
                        '--branch-from must start before a controller phase')
                branch_provenance = copy.deepcopy(saved_provenance)
                branch_provenance['settings']['deterministic_policy_reset'] = (
                    args.deterministic_policy_reset)
                assert_same_provenance(branch_provenance, provenance)
            else:
                assert_same_provenance(saved_provenance, provenance)
            required_keys = [
                'format_version', 'outer_round', 'phase',
                'global_seed_update', 'seed_policy', 'controller',
                'controller_state_sha256', 'seed_optimizer',
                'controller_optimizer', 'controller_scaler',
                'task_generator_state', 'feature_dim',
                'seed_include_ray_error', 'seed_include_log_manip',
                'backward_mode', 'seed_config', 'controller_config',
                'controller_run_config_sha256', 'train_indices',
                'validation_indices', 'train_task_indices',
                'validation_task_indices', 'train_valid_mask',
                'validation_valid_mask', 'args', 'numpy_rng_state',
                'torch_rng_state', 'cuda_rng_state', 'provenance',
            ]
            if not legacy_v3_resume:
                required_keys.extend(('seed_architecture', 'split_mode'))
            require_checkpoint_keys(
                checkpoint, required_keys,
                kind=('version-3 bidirectional' if legacy_v3_resume
                      else 'version-4 bidirectional'))
            require_checkpoint_format_version(
                checkpoint, 3 if legacy_v3_resume else 4,
                kind=('version-3 bidirectional' if legacy_v3_resume
                      else 'version-4 bidirectional'))
    seed_include_ray_error = (
        bool(checkpoint.get(
            'seed_include_ray_error',
            int(checkpoint['feature_dim']) in (34, 35)))
        if checkpoint is not None else True)
    seed_include_log_manip = (
        bool(checkpoint.get(
            'seed_include_log_manip', int(checkpoint['feature_dim']) in (32, 35)))
        if checkpoint is not None else not args.no_log_manip)
    seed_include_directional_dynamics = (
        bool(checkpoint.get('seed_include_directional_dynamics', False))
        if checkpoint is not None else args.directional_dynamics)
    actions_per_task = (dataset.batch.n_candidates
                        if args.backward_mode == 'dense-gpi'
                        else args.seed_samples_per_task)
    seed_rollout_n = args.seed_tasks_per_update * actions_per_task
    print(
        f'[unified] backward={args.backward_mode}  '
        f'episodes/update={seed_rollout_n}')
    controller_env = build_env_from_run(
        args.init_controller_ckpt, args.controller_n_envs, device,
        env_overrides={'observe_ray_error': True})
    seed_env = build_env_from_run(
        args.init_controller_ckpt, seed_rollout_n, device,
        env_overrides={'observe_ray_error': True})
    controller = load_controller_agent(
        args.init_controller_ckpt, controller_env, device)
    if not args.skip_physical_validation:
        dataset, valid_stats = validate_cached_dataset(
            dataset, controller_env.kin, controller_env.collision,
            cone_deg=controller_env.cfg.cone_deg)
        print(f'[unified] physical candidate validity: '
              f'{valid_stats["frac_valid"]:.1%}')
    if checkpoint is not None:
        train_index = checkpoint['train_indices'].cpu()
        validation_index = checkpoint['validation_indices'].cpu()
        source_dataset = dataset
        if 'train_task_indices' in checkpoint:
            dataset = source_dataset.select_source_tasks(
                checkpoint['train_task_indices'].cpu())
            validation_dataset = source_dataset.select_source_tasks(
                checkpoint['validation_task_indices'].cpu())
            train_task_indices = checkpoint['train_task_indices'].cpu()
            validation_task_indices = checkpoint['validation_task_indices'].cpu()
        else:
            source_task_indices = source_dataset.task_indices
            dataset = source_dataset.index_select(train_index)
            validation_dataset = source_dataset.index_select(validation_index)
            train_task_indices = dataset.task_indices.clone()
            validation_task_indices = source_task_indices[validation_index].clone()
        if 'train_valid_mask' in checkpoint:
            assert_same_valid_mask(
                dataset, checkpoint['train_valid_mask'], label='training')
        if 'validation_valid_mask' in checkpoint:
            assert_same_valid_mask(
                validation_dataset, checkpoint['validation_valid_mask'],
                label='validation')
    else:
        dataset, validation_dataset, train_index, validation_index = (
            dataset.train_validation_split(args.validation_fraction, args.seed))
        train_task_indices = dataset.task_indices.clone()
        validation_task_indices = validation_dataset.task_indices.clone()
    train_valid_mask = dataset.batch.valid.clone()
    validation_valid_mask = validation_dataset.batch.valid.clone()
    if split_mode == 'task-geometry-grouped-v1':
        train_fingerprints = set(dataset.task_fingerprints)
        validation_fingerprints = set(validation_dataset.task_fingerprints)
        overlap = train_fingerprints & validation_fingerprints
        if overlap:
            raise ValueError(
                'task-grouped split contains overlapping task geometries')
        print(
            f'[unified] geometry groups: train={len(train_fingerprints)}  '
            f'validation={len(validation_fingerprints)}')
    print(f'[unified] split: train={len(train_index)}  '
          f'validation={len(validation_index)}')

    feature_probe = dataset.sample(min(8, len(dataset))).to(
        device, dtype=controller_env.kin.dtype)
    feature_dim = initial_observation_features(
        controller_env.kin, feature_probe,
        include_log_manip=seed_include_log_manip,
        include_ray_error=seed_include_ray_error,
        include_directional_dynamics=(
            seed_include_directional_dynamics)).shape[-1]
    requested_policy_config = SeedPolicyConfig(
        feature_dim=feature_dim,
        hidden_dim=args.seed_hidden_dim,
        encoder_type=args.seed_encoder,
        heads=args.seed_attention_heads,
        layers=args.seed_attention_layers,
        ff_mult=args.seed_attention_ff_mult,
    )
    if checkpoint is not None:
        saved_policy_config = infer_seed_policy_config(checkpoint)
        if saved_policy_config != requested_policy_config:
            raise ValueError(
                'requested seed architecture differs from resume checkpoint: '
                f'requested={requested_policy_config.to_dict()}, '
                f'saved={saved_policy_config.to_dict()}')
        policy_config = saved_policy_config
    else:
        policy_config = requested_policy_config
    seed_policy = CandidateSeedActorCritic(**policy_config.to_dict()).to(device)
    if checkpoint is None:
        print('[unified] fitting valid-only seed feature normalization')
        feature_mean, feature_std = fit_candidate_feature_normalization(
            controller_env.kin, dataset,
            include_log_manip=seed_include_log_manip,
            include_ray_error=seed_include_ray_error,
            include_directional_dynamics=(
                seed_include_directional_dynamics))
        seed_policy.set_feature_normalization(feature_mean, feature_std)
    controller_cfg_yaml = load_run_config(args.init_controller_ckpt)
    controller_cfg = ppo_config_from_run(
        controller_cfg_yaml,
        total_timesteps=args.controller_steps_per_round,
        learning_rate=args.controller_lr,
        anneal_lr=False,
    )
    effective_controller_cfg = dataclasses.asdict(controller_cfg)
    output_cfg_yaml = copy.deepcopy(controller_cfg_yaml)
    output_cfg_yaml.setdefault('env', {}).update({
        'n_envs': args.controller_n_envs,
        'observe_ray_error': True,
    })
    # Keep the standard agent.pt/config.yaml artifact truthful: evaluation
    # consumes gamma/architecture here, while researchers inspecting the run
    # should also see the actual per-round LR/timestep/annealing settings.
    output_cfg_yaml['ppo'] = effective_controller_cfg
    output_cfg_yaml['unified'] = {
        'controller_steps_per_round': args.controller_steps_per_round,
        'controller_n_envs': args.controller_n_envs,
        'first_controller_actor_warmup': args.first_controller_actor_warmup,
        'backward_mode': args.backward_mode,
    }
    if not legacy_v3_resume:
        output_cfg_yaml['unified']['seed_architecture'] = (
            seed_policy.architecture)
        output_cfg_yaml['unified']['split_mode'] = split_mode
        output_cfg_yaml['unified']['deterministic_policy_reset'] = (
            args.deterministic_policy_reset)
        output_cfg_yaml['unified']['controller_reset_rng'] = (
            'independent-streams-v1')
        output_cfg_yaml['unified']['seed_directional_dynamics'] = (
            seed_include_directional_dynamics)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(output_cfg_yaml, f, sort_keys=False)
    output_config_sha256 = file_fingerprint(
        out_dir / 'config.yaml')['sha256']
    if checkpoint is not None:
        saved_controller_cfg = checkpoint.get('controller_config')
        if (saved_controller_cfg is not None
                and saved_controller_cfg != effective_controller_cfg):
            raise ValueError(
                'effective controller PPO config differs from resume checkpoint')
        saved_config_sha256 = checkpoint.get('controller_run_config_sha256')
        accepted_config_hashes = {output_config_sha256}
        if legacy_v3_resume and saved_provenance is not None:
            # Early v3 checkpoints used more than one config convention: some
            # recorded the initial controller config plus only the 34-D
            # observation switch, while later ones emitted effective PPO
            # settings. Accept only content hashes tied to those artifacts.
            accepted_config_hashes.add(
                saved_provenance['initial_controller']['config']['sha256'])
            legacy_config_path = Path(checkpoint_path).resolve().parent / 'config.yaml'
            if legacy_config_path.is_file():
                accepted_config_hashes.add(
                    file_fingerprint(legacy_config_path)['sha256'])
        if (not is_branch and saved_config_sha256 is not None
                and saved_config_sha256 not in accepted_config_hashes):
            raise ValueError(
                'effective controller run config differs from resume checkpoint')
    if checkpoint is not None:
        checkpoint_mode = checkpoint.get('backward_mode', 'sampled-ppo')
        if checkpoint_mode != args.backward_mode:
            raise ValueError(
                f'resume checkpoint uses {checkpoint_mode}, but '
                f'--backward-mode={args.backward_mode}')
        if args.backward_mode == 'dense-gpi':
            seed_cfg = DenseSeedConfig(**checkpoint['seed_config'])
        else:
            seed_cfg = SeedPPOConfig(**checkpoint['seed_config'])
        checkpoint_return = checkpoint.get('args', {}).get(
            'seed_return', 'discounted')
        if checkpoint_return != args.seed_return:
            raise ValueError(
                f'resume checkpoint uses {checkpoint_return} return, '
                f'but --seed-return={args.seed_return}')
    else:
        discounted_horizon = (
            float(seed_env.max_steps) if controller_cfg.gamma == 1.0
            else ((1.0 - controller_cfg.gamma ** seed_env.max_steps)
                  / (1.0 - controller_cfg.gamma)))
        seed_return_scale = (
            float(seed_env.max_steps) if args.seed_return == 'undiscounted'
            else discounted_horizon)
        base_seed_cfg = (DenseSeedConfig()
                         if args.backward_mode == 'dense-gpi'
                         else SeedPPOConfig())
        seed_cfg = dataclasses.replace(
            base_seed_cfg, return_scale=seed_return_scale)
    seed_optimizer = torch.optim.Adam(
        seed_policy.parameters(), lr=seed_cfg.learning_rate)
    controller_optimizer = torch.optim.Adam(
        controller.parameters(), lr=args.controller_lr, eps=1e-5)
    controller_scaler = (RewardScaler(
        args.controller_n_envs, controller_cfg.gamma, device)
        if controller_cfg.normalize_returns else None)
    task_generator = torch.Generator().manual_seed(args.seed)
    start_round = 1
    resume_after_controller = False
    global_seed_update = 0

    if checkpoint is not None:
        seed_policy.load_state_dict(checkpoint['seed_policy'])
        load_controller_state_dict(controller, checkpoint['controller'])
        seed_optimizer.load_state_dict(checkpoint['seed_optimizer'])
        controller_optimizer.load_state_dict(checkpoint['controller_optimizer'])
        adapt_controller_optimizer_observation_state(
            controller_optimizer, controller)
        _optimizer_to(seed_optimizer, device)
        _optimizer_to(controller_optimizer, device)
        if controller_scaler is not None and checkpoint['controller_scaler'] is not None:
            controller_scaler.load_state_dict(checkpoint['controller_scaler'])
        task_generator.set_state(checkpoint['task_generator_state'].cpu())
        restore_global_rng(checkpoint, device)
        checkpoint_phase = checkpoint.get('phase', 'round_complete')
        start_round, resume_after_controller = resume_position(checkpoint)
        global_seed_update = int(checkpoint['global_seed_update'])
        action = 'branched' if is_branch else 'resumed'
        print(
            f'[unified] {action} phase={checkpoint_phase} at outer round '
            f'{start_round}')

    def save(outer_round: int, phase: str) -> None:
        controller_state = controller.state_dict()
        state = {
            'format_version': 3 if legacy_v3_resume else 4,
            'outer_round': outer_round,
            'phase': phase,
            'global_seed_update': global_seed_update,
            'seed_policy': seed_policy.state_dict(),
            'controller': controller_state,
            'controller_state_sha256': state_dict_fingerprint(controller_state),
            'seed_optimizer': seed_optimizer.state_dict(),
            'controller_optimizer': controller_optimizer.state_dict(),
            'controller_scaler': (controller_scaler.state_dict()
                                  if controller_scaler is not None else None),
            'task_generator_state': task_generator.get_state(),
            'feature_dim': feature_dim,
            'hidden_dim': seed_policy.hidden_dim,
            'seed_architecture': seed_policy.architecture,
            'split_mode': split_mode,
            'seed_include_ray_error': seed_include_ray_error,
            'seed_include_log_manip': seed_include_log_manip,
            'seed_include_directional_dynamics': (
                seed_include_directional_dynamics),
            'backward_mode': args.backward_mode,
            'seed_config': dataclasses.asdict(seed_cfg),
            'controller_config': effective_controller_cfg,
            'controller_run_config_sha256': output_config_sha256,
            'candidate_cache': str(args.candidates),
            'train_indices': train_index,
            'validation_indices': validation_index,
            'train_task_indices': train_task_indices,
            'validation_task_indices': validation_task_indices,
            'train_valid_mask': train_valid_mask,
            'validation_valid_mask': validation_valid_mask,
            'init_controller_ckpt': str(args.init_controller_ckpt),
            'args': vars(args),
            'provenance': provenance,
        }
        state.update(global_rng_state(device))
        atomic_torch_save(state, out_dir / 'unified.pt')
        # Standard controller checkpoint for all existing evaluation tools.
        atomic_torch_save(controller_state, out_dir / 'agent.pt')
        # Keep immutable phase artifacts for paired 2x2 attribution. The
        # top-level files remain the resume pointers, while each snapshot is
        # also a self-contained controller directory accepted by evaluate.py.
        snapshot_dir = (
            out_dir / 'snapshots'
            / f'round_{outer_round:03d}_{phase}')
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(state, snapshot_dir / 'unified.pt')
        atomic_torch_save(controller_state, snapshot_dir / 'agent.pt')
        shutil.copyfile(out_dir / 'config.yaml', snapshot_dir / 'config.yaml')

    def backward_seed_phase(n_updates: int, tag: str) -> None:
        nonlocal global_seed_update
        frozen_controller = FrozenRLController(controller.eval())
        for local_update in range(1, n_updates + 1):
            candidates = dataset.sample(
                args.seed_tasks_per_update, generator=task_generator).to(
                    device, dtype=seed_env.kin.dtype)
            features = initial_observation_features(
                seed_env.kin, candidates,
                include_log_manip=seed_include_log_manip,
                include_ray_error=seed_include_ray_error,
                include_directional_dynamics=(
                    seed_include_directional_dynamics))

            def rollout_fn(repeated, actions):
                result = rollout_selected_seeds(
                    seed_env, repeated, actions, frozen_controller,
                    gamma=controller_cfg.gamma)
                return (result.undiscounted_return
                        if args.seed_return == 'undiscounted'
                        else result.discounted_return)

            if args.backward_mode == 'dense-gpi':
                rollout = collect_dense_seed_rollout(
                    seed_policy, candidates, features, rollout_fn,
                    return_scale=seed_cfg.return_scale)
                stats = update_dense_seed_policy(
                    seed_policy, seed_optimizer, rollout, seed_cfg)
            else:
                rollout = collect_seed_rollout(
                    seed_policy, candidates, features, rollout_fn,
                    samples_per_task=args.seed_samples_per_task,
                    return_scale=seed_cfg.return_scale,
                    center_within_task=seed_cfg.center_within_task)
                stats = update_seed_policy(
                    seed_policy, seed_optimizer, rollout, seed_cfg)
            global_seed_update += 1
            if local_update == 1 or local_update % 10 == 0:
                selected_return = stats.get(
                    'seed/policy_return_mean', stats['seed/raw_return_mean'])
                oracle_return = stats.get('seed/oracle_return_mean')
                oracle_text = (f'/{oracle_return:.2f}'
                               if oracle_return is not None else '')
                print(
                    f'[{tag}] upd {local_update:>4}/{n_updates}  '
                    f'pick/oracle {selected_return:.2f}{oracle_text}  '
                    f'ent {stats["seed/entropy"]:.3f}  '
                    f'kl {stats["seed/approx_kl"]:.4f}',
                    flush=True)

    if checkpoint is None and args.seed_warmup_updates > 0:
        print('[unified] backward warmup against initial frozen controller')
        backward_seed_phase(args.seed_warmup_updates, 'seed-warmup')
        save(0, 'warmup_complete')

    for outer_round in range(start_round, args.outer_rounds + 1):
        skip_controller = resume_after_controller and outer_round == start_round
        if skip_controller:
            print(
                f'[unified] ===== outer round {outer_round} / controller '
                'already complete =====')
        else:
            print(
                f'[unified] ===== outer round {outer_round} / forward '
                'controller =====')
            frozen_seed = copy.deepcopy(seed_policy).eval()
            controller_env.line_dist = SeedPolicyLineDistribution(
                dataset, frozen_seed, controller_env.kin,
                policy_prob=args.policy_reset_prob,
                uniform_prob=args.uniform_reset_prob,
                fallback_prob=args.fallback_reset_prob,
                include_log_manip=seed_include_log_manip,
                include_ray_error=seed_include_ray_error,
                include_directional_dynamics=(
                    seed_include_directional_dynamics),
                deterministic_policy=args.deterministic_policy_reset,
                independent_rng_streams=not legacy_v3_resume,
                seed=args.seed + 1000 * outer_round)

        def controller_log(stats: dict) -> None:
            if 'update' in stats and (
                    stats['update'] == 1 or stats['update'] % 20 == 0):
                print(
                    f'[ctrl-r{outer_round}] upd {stats["update"]:>4}  '
                    f'progress {stats.get("reward/progress", 0.0):.3f}  '
                    f'kl {stats.get("train/approx_kl", 0.0):.4f}',
                    flush=True)

        if not skip_controller:
            controller.train()
            round_controller_cfg = dataclasses.replace(
                controller_cfg,
                actor_warmup_updates=(args.first_controller_actor_warmup
                                      if outer_round == 1 else 0),
            )
            ppo_train(
                round_controller_cfg, controller_env, device,
                log_fn=controller_log,
                agent=controller,
                optimizer=controller_optimizer,
                reward_scaler=controller_scaler,
            )
            # The backward block can itself be long. Save its exact starting
            # point so a restart never repeats a completed controller phase.
            save(outer_round, 'controller_complete')

        print(f'[unified] ===== outer round {outer_round} / backward seed =====')
        backward_seed_phase(args.seed_updates_per_round, f'seed-r{outer_round}')
        save(outer_round, 'round_complete')
        resume_after_controller = False
        print(f'[unified] round {outer_round} saved -> {out_dir / "unified.pt"}')


if __name__ == '__main__':
    main()
