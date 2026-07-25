"""Gate-1 trainer: learn the seed macro-policy against a frozen controller.

This is deliberately the first executable slice of the unified framework.
It validates delayed cross-stage credit before controller non-stationarity is
introduced by bidirectional co-training.

Example:
    python -m Yuan.unified_rl.train_seed \
        --candidates Yuan/seed_selection/runs/rank_train/candidates_K8.npz \
        --controller-ckpt Yuan/RL_controller/runs/exit_rounds7plus/final_avg \
        --out-dir Yuan/unified_rl/runs/seed_gate
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import torch

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
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
    FrozenHybridController,
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
)
from Yuan.unified_rl.reproducibility import (
    device_identity,
    global_rng_state,
    restore_global_rng,
    seed_global_rng,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    SeedPolicyConfig,
    infer_seed_policy_config,
)
from Yuan.unified_rl.seed_gpi import (
    DenseSeedConfig,
    collect_dense_seed_rollout,
    update_dense_seed_policy,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--allow-legacy-resume', action='store_true',
                        help='resume an early checkpoint without provenance')
    parser.add_argument('--device', default=None)
    parser.add_argument('--controller', choices=('pure', 'hybrid'), default='pure')
    parser.add_argument('--tau-enter', type=float, default=0.985)
    parser.add_argument('--tau-exit', type=float, default=0.96)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--tasks-per-update', type=int, default=28)
    parser.add_argument('--samples-per-task', type=int, default=4)
    parser.add_argument('--backward-mode',
                        choices=('dense-gpi', 'sampled-ppo'),
                        default='dense-gpi')
    parser.add_argument('--no-log-manip', action='store_true',
                        help='ablate log positional manipulability feature')
    parser.add_argument('--seed-encoder', choices=('mean', 'attention'),
                        default='mean')
    parser.add_argument('--seed-hidden-dim', type=int, default=256)
    parser.add_argument('--seed-attention-heads', type=int, default=4)
    parser.add_argument('--seed-attention-layers', type=int, default=1)
    parser.add_argument('--seed-attention-ff-mult', type=int, default=2)
    parser.add_argument('--seed', type=int, default=12000)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--skip-physical-validation', action='store_true')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--seed-return', choices=('undiscounted', 'discounted'),
                        default='undiscounted')
    args = parser.parse_args()

    if args.updates < 0:
        raise ValueError('--updates must be non-negative')
    if args.tasks_per_update < 1:
        raise ValueError('--tasks-per-update must be positive')
    if args.samples_per_task < 1:
        raise ValueError('--samples-per-task must be positive')
    if args.save_every < 1:
        raise ValueError('--save-every must be positive')
    if (not math.isfinite(args.validation_fraction)
            or not 0.0 < args.validation_fraction < 1.0):
        raise ValueError('--validation-fraction must be in (0, 1)')

    # PPO minibatch shuffling uses NumPy's global RNG; policy sampling and
    # parameter initialization use torch's global CPU/CUDA RNGs.
    seed_global_rng(args.seed)
    device = torch.device(args.device if args.device is not None
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = (torch.load(args.resume, map_location=device, weights_only=False)
                  if args.resume is not None else None)
    if checkpoint is not None and args.updates < int(checkpoint['update']):
        raise ValueError(
            f'--updates={args.updates} is behind resume checkpoint update '
            f'{int(checkpoint["update"])}')
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
        and saved_provenance.get('format') == 'unified-seed-gate-v3')
    split_mode = ('task-geometry-grouped-v1' if checkpoint is None
                  else checkpoint.get('split_mode', 'row-random-v1'))
    provenance_settings = {
        'controller_kind': args.controller,
        'tau_enter': args.tau_enter if args.controller == 'hybrid' else None,
        'tau_exit': args.tau_exit if args.controller == 'hybrid' else None,
        'tasks_per_update': args.tasks_per_update,
        'samples_per_task': (args.samples_per_task
                             if args.backward_mode == 'sampled-ppo'
                             else None),
        'backward_mode': args.backward_mode,
        'seed': args.seed,
        'device': device_identity(device),
        'physical_validation': not args.skip_physical_validation,
        'seed_return': args.seed_return,
        'controller_observation': 'historical-31d',
        'seed_log_manip': not args.no_log_manip,
    }
    if not legacy_v3_resume:
        provenance_settings['seed_architecture'] = requested_architecture
        provenance_settings['split_mode'] = split_mode
    provenance = {
        # Bump whenever the optimizer target or derived action-mask semantics
        # change; strict resume must never splice two algorithms into one run.
        'format': ('unified-seed-gate-v3' if legacy_v3_resume
                   else 'unified-seed-gate-v4'),
        'candidate_cache': file_fingerprint(args.candidates),
        'controller': controller_fingerprint(args.controller_ckpt),
        'settings': provenance_settings,
    }
    if checkpoint is not None:
        if saved_provenance is None:
            if not args.allow_legacy_resume:
                raise ValueError(
                    'resume checkpoint has no provenance; pass '
                    '--allow-legacy-resume only after manually verifying its '
                    'candidate cache, controller, and rollout settings')
            print('[unified-seed] WARNING: accepting legacy resume without provenance')
        else:
            assert_same_provenance(saved_provenance, provenance)
            required_keys = [
                'model', 'feature_dim', 'hidden_dim', 'update',
                'seed_include_log_manip', 'backward_mode', 'seed_return',
                'seed_ppo', 'dense_seed', 'optimizer', 'generator_state',
                'train_indices', 'validation_indices',
                'train_task_indices', 'validation_task_indices',
                'train_valid_mask', 'validation_valid_mask',
                'numpy_rng_state', 'torch_rng_state', 'cuda_rng_state',
                'provenance',
            ]
            if not legacy_v3_resume:
                required_keys.extend((
                    'format_version', 'seed_architecture', 'split_mode'))
            require_checkpoint_keys(
                checkpoint, required_keys,
                kind=('version-3 seed gate' if legacy_v3_resume
                      else 'version-4 seed gate'))
            if not legacy_v3_resume:
                require_checkpoint_format_version(
                    checkpoint, 4, kind='version-4 seed gate')
    dataset = CachedSeedCandidateDataset.from_npz(args.candidates)
    actions_per_task = (dataset.batch.n_candidates
                        if args.backward_mode == 'dense-gpi'
                        else args.samples_per_task)
    rollout_batch_size = args.tasks_per_update * actions_per_task
    print(
        f'[unified-seed] backward={args.backward_mode}  '
        f'episodes/update={rollout_batch_size}')
    env = build_env_from_run(args.controller_ckpt, rollout_batch_size, device)
    agent = load_controller_agent(args.controller_ckpt, env, device)
    controller_gamma = ppo_config_from_run(
        load_run_config(args.controller_ckpt)).gamma
    agent.eval()
    if not args.skip_physical_validation:
        dataset, valid_stats = validate_cached_dataset(
            dataset, env.kin, env.collision, cone_deg=env.cfg.cone_deg)
        print(f'[unified-seed] physical validity: {valid_stats["frac_valid"]:.1%}')
    if checkpoint is None:
        dataset, validation_dataset, train_index, validation_index = (
            dataset.train_validation_split(args.validation_fraction, args.seed))
        train_task_indices = dataset.task_indices.clone()
        validation_task_indices = validation_dataset.task_indices.clone()
    else:
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
            f'[unified-seed] geometry groups: train={len(train_fingerprints)}  '
            f'validation={len(validation_fingerprints)}')
    print(f'[unified-seed] split: train={len(train_index)}  '
          f'validation={len(validation_index)}')
    if args.controller == 'hybrid':
        controller = FrozenHybridController(
            agent, ClassicalNullspaceController(env.kin),
            tau_enter=args.tau_enter, tau_exit=args.tau_exit)
    else:
        controller = FrozenRLController(agent)

    probe = dataset.sample(min(args.tasks_per_update, len(dataset))).to(
        device, dtype=env.kin.dtype)
    seed_include_log_manip = not args.no_log_manip
    feature_dim = initial_observation_features(
        env.kin, probe,
        include_log_manip=seed_include_log_manip).shape[-1]
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
    policy = CandidateSeedActorCritic(**policy_config.to_dict()).to(device)
    if checkpoint is None:
        print('[unified-seed] fitting valid-only feature normalization')
        feature_mean, feature_std = fit_candidate_feature_normalization(
            env.kin, dataset,
            include_log_manip=seed_include_log_manip)
        policy.set_feature_normalization(feature_mean, feature_std)
    if checkpoint is not None:
        checkpoint_mode = checkpoint.get('backward_mode', 'sampled-ppo')
        if checkpoint_mode != args.backward_mode:
            raise ValueError(
                f'resume checkpoint uses {checkpoint_mode}, but '
                f'--backward-mode={args.backward_mode}')
        if args.backward_mode == 'dense-gpi':
            backward_cfg = DenseSeedConfig(**checkpoint['dense_seed'])
        else:
            backward_cfg = SeedPPOConfig(**checkpoint['seed_ppo'])
        checkpoint_return = checkpoint.get('seed_return', 'discounted')
        if checkpoint_return != args.seed_return:
            raise ValueError(
                f'resume checkpoint uses {checkpoint_return} return, '
                f'but --seed-return={args.seed_return}')
    else:
        discounted_horizon = (
            float(env.max_steps) if controller_gamma == 1.0
            else ((1.0 - controller_gamma ** env.max_steps)
                  / (1.0 - controller_gamma)))
        return_scale = (float(env.max_steps)
                        if args.seed_return == 'undiscounted'
                        else discounted_horizon)
        base_cfg = (DenseSeedConfig() if args.backward_mode == 'dense-gpi'
                    else SeedPPOConfig())
        backward_cfg = dataclasses.replace(base_cfg, return_scale=return_scale)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=backward_cfg.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    start_update = 1
    if checkpoint is not None:
        model_state = checkpoint.get('model', checkpoint.get('seed_policy'))
        if model_state is None:
            raise ValueError('resume checkpoint has no seed model')
        policy.load_state_dict(model_state)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        if 'generator_state' in checkpoint:
            generator.set_state(checkpoint['generator_state'].cpu())
        restore_global_rng(checkpoint, device)
        start_update = int(checkpoint['update']) + 1
        print(f'[unified-seed] resumed at update {start_update}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(update: int) -> None:
        state = {
            'format_version': 3 if legacy_v3_resume else 4,
            'model': policy.state_dict(),
            'feature_dim': feature_dim,
            'seed_include_log_manip': seed_include_log_manip,
            'hidden_dim': policy.hidden_dim,
            'seed_architecture': policy.architecture,
            'split_mode': split_mode,
            'update': update,
            'candidate_cache': str(args.candidates),
            'controller_ckpt': str(args.controller_ckpt),
            'controller_kind': args.controller,
            'backward_mode': args.backward_mode,
            'seed_return': args.seed_return,
            'seed_ppo': (dataclasses.asdict(backward_cfg)
                         if args.backward_mode == 'sampled-ppo' else None),
            'dense_seed': (dataclasses.asdict(backward_cfg)
                           if args.backward_mode == 'dense-gpi' else None),
            'optimizer': optimizer.state_dict(),
            'generator_state': generator.get_state(),
            'train_indices': train_index,
            'validation_indices': validation_index,
            'train_task_indices': train_task_indices,
            'validation_task_indices': validation_task_indices,
            'train_valid_mask': train_valid_mask,
            'validation_valid_mask': validation_valid_mask,
            'provenance': provenance,
        }
        state.update(global_rng_state(device))
        atomic_torch_save(state, out_dir / 'seed_agent.pt')

    for update in range(start_update, args.updates + 1):
        candidates = dataset.sample(
            args.tasks_per_update, generator=generator).to(
                device, dtype=env.kin.dtype)
        features = initial_observation_features(
            env.kin, candidates,
            include_log_manip=seed_include_log_manip)

        def rollout_fn(repeated, actions):
            result = rollout_selected_seeds(
                env, repeated, actions, controller, gamma=controller_gamma)
            return (result.undiscounted_return
                    if args.seed_return == 'undiscounted'
                    else result.discounted_return)

        if args.backward_mode == 'dense-gpi':
            rollout = collect_dense_seed_rollout(
                policy, candidates, features, rollout_fn,
                return_scale=backward_cfg.return_scale)
            stats = update_dense_seed_policy(
                policy, optimizer, rollout, backward_cfg)
        else:
            rollout = collect_seed_rollout(
                policy, candidates, features, rollout_fn,
                samples_per_task=args.samples_per_task,
                return_scale=backward_cfg.return_scale,
                center_within_task=backward_cfg.center_within_task)
            stats = update_seed_policy(
                policy, optimizer, rollout, backward_cfg)
        if update == 1 or update % 10 == 0:
            selected_return = stats.get(
                'seed/policy_return_mean', stats['seed/raw_return_mean'])
            oracle_return = stats.get('seed/oracle_return_mean')
            oracle_text = (f'/{oracle_return:.2f}'
                           if oracle_return is not None else '')
            print(
                f'upd {update:>5}  pick/oracle '
                f'{selected_return:.2f}{oracle_text}  '
                f'ent {stats["seed/entropy"]:.3f}  '
                f'kl {stats["seed/approx_kl"]:.4f}',
                flush=True)
        if update % args.save_every == 0:
            save(update)
    save(args.updates)
    print(f'[unified-seed] saved -> {out_dir / "seed_agent.pt"}')


if __name__ == '__main__':
    main()
