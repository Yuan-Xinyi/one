"""Paired evaluation for a unified seed policy and frozen controller."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
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
    rollout_topk_prefix_lookahead,
)
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.provenance import (
    controller_fingerprint,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    CandidateSeedPolicyEnsemble,
    infer_seed_policy_config,
    seed_policy_ensemble_states,
)
from Yuan.unified_rl.seed_deployment import (
    deployment_config_from_checkpoint,
    select_seed_deployment,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


def _geometry_macro_mean(values: np.ndarray,
                         fingerprints: Sequence[str]) -> float:
    """Average rows within a geometry, then average geometries equally."""
    grouped: dict[str, list[float]] = {}
    for fingerprint, value in zip(fingerprints, values.tolist()):
        grouped.setdefault(fingerprint, []).append(float(value))
    if not grouped:
        raise ValueError('geometry-macro metric requires at least one row')
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_seed_policy(path: str | Path, device: torch.device
                     ) -> tuple[
                         CandidateSeedActorCritic | CandidateSeedPolicyEnsemble,
                         dict,
                     ]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ensemble = seed_policy_ensemble_states(checkpoint)
    policy_config = infer_seed_policy_config(checkpoint)
    if ensemble is not None:
        states, _ = ensemble
        members = []
        for index, state in enumerate(states):
            member = CandidateSeedActorCritic(
                **policy_config.to_dict()).to(device)
            try:
                member.load_state_dict(state, strict=True)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f'invalid seed_policy_ensemble[{index}] state: {exc}') from exc
            members.append(member)
        policy = CandidateSeedPolicyEnsemble(members).to(device)
        policy.eval()
        return policy, checkpoint

    if 'seed_policy' in checkpoint:
        state = checkpoint['seed_policy']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        raise ValueError('checkpoint contains neither seed_policy nor model')
    policy = CandidateSeedActorCritic(**policy_config.to_dict()).to(device)
    policy.load_state_dict(state)
    policy.eval()
    return policy, checkpoint


def _pad_indices(start: int, end: int, total: int,
                 batch_size: int) -> tuple[torch.Tensor, int]:
    index = torch.arange(start, end)
    n_real = end - start
    if n_real < batch_size:
        index = torch.cat([index, index[-1:].expand(batch_size - n_real)])
    if end > total or n_real < 1:
        raise ValueError('invalid evaluation slice')
    return index, n_real


def _effective_probe_task_chunk(
    requested_chunk: int,
    top_k: int,
    n_candidates: int,
    max_branch_batch: int,
) -> int:
    """Cap task rows so fixed-width probe branches fit the lane budget."""
    if requested_chunk < 1 or max_branch_batch < 1:
        raise ValueError('chunk and branch batch must be positive')
    if top_k < 0 or top_k > n_candidates:
        raise ValueError('probe top_k is outside the candidate count')
    if top_k == 0:
        return requested_chunk
    width = top_k + 1
    if max_branch_batch < width:
        raise ValueError('branch batch is smaller than one shortlist')
    return min(requested_chunk, max_branch_batch // width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--controller', choices=('auto', 'pure', 'hybrid'),
                        default='auto')
    parser.add_argument('--allow-controller-mismatch', action='store_true',
                        help='allow controller/kind mismatch for an explicit ablation')
    parser.add_argument(
        '--cross-pair-controller', action='store_true',
        help='allow only controller-weight mismatch for a matched 2x2 phase decomposition')
    parser.add_argument('--tau-enter', type=float, default=0.985)
    parser.add_argument('--tau-exit', type=float, default=0.96)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--full-candidate-oracle', action='store_true')
    parser.add_argument(
        '--seed-probe-top-k', type=int, default=0,
        help='enable actor top-k union first-valid prefix lookahead; 0 disables')
    parser.add_argument(
        '--seed-probe-horizon', type=int, default=128,
        help='controller steps per shortlisted branch when probe is enabled')
    parser.add_argument(
        '--seed-probe-alive-bonus', type=float, default=100.0,
        help='additive prefix score for branches alive at the probe horizon')
    parser.add_argument(
        '--seed-probe-score',
        choices=(
            'auto', 'progress_m', 'undiscounted_return',
            'discounted_return'),
        default='auto',
        help='prefix objective; auto follows the checkpoint selector objective')
    parser.add_argument(
        '--seed-probe-max-branch-batch', type=int, default=2048,
        help='cap top-k branch lanes by reducing the effective task chunk')
    parser.add_argument('--skip-physical-validation', action='store_true')
    parser.add_argument('--all-tasks', action='store_true',
                        help='ignore validation_indices stored in the checkpoint')
    parser.add_argument('--external-holdout', action='store_true',
                        help='allow a different candidate cache and evaluate all of it')
    parser.add_argument(
        '--allow-task-overlap', action='store_true',
        help='allow task-geometry overlap only for an explicit transfer diagnostic')
    args = parser.parse_args()
    requested_chunk_size = args.chunk_size

    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be positive')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('--max-tasks must be positive')
    if args.allow_task_overlap and not args.external_holdout:
        raise ValueError('--allow-task-overlap requires --external-holdout')
    if args.seed_probe_top_k < 0:
        raise ValueError('--seed-probe-top-k must be non-negative')
    if args.seed_probe_horizon < 1:
        raise ValueError('--seed-probe-horizon must be positive')
    if args.seed_probe_max_branch_batch < 1:
        raise ValueError('--seed-probe-max-branch-batch must be positive')
    if (not np.isfinite(args.seed_probe_alive_bonus)
            or args.seed_probe_alive_bonus < 0.0):
        raise ValueError(
            '--seed-probe-alive-bonus must be finite and non-negative')

    device = torch.device(args.device if args.device is not None
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed_artifact = file_fingerprint(args.seed_checkpoint)
    candidate_artifact = file_fingerprint(args.candidates)
    controller_artifact = controller_fingerprint(args.controller_ckpt)
    dataset = CachedSeedCandidateDataset.from_npz(args.candidates)
    if args.seed_probe_top_k > 0:
        effective_chunk = _effective_probe_task_chunk(
            args.chunk_size, args.seed_probe_top_k,
            dataset.batch.n_candidates, args.seed_probe_max_branch_batch)
        if args.chunk_size > effective_chunk:
            args.chunk_size = effective_chunk
            print(
                '[unified-eval] reducing task chunk for seed probe: '
                f'{requested_chunk_size} -> {args.chunk_size} '
                f'(branch batch <= {args.seed_probe_max_branch_batch})')
    seed_policy, seed_checkpoint = load_seed_policy(args.seed_checkpoint, device)
    saved_provenance = seed_checkpoint.get('provenance')
    checkpoint_format = (saved_provenance.get('format')
                         if saved_provenance is not None else None)
    if checkpoint_format in ('unified-seed-gate-v3',
                              'unified-seed-gate-v4'):
        require_checkpoint_keys(
            seed_checkpoint,
            (
                'model', 'feature_dim', 'seed_include_log_manip',
                'seed_return', 'train_task_indices', 'train_valid_mask',
                'validation_task_indices', 'validation_valid_mask',
                'provenance',
            ),
            kind=checkpoint_format)
        if checkpoint_format == 'unified-seed-gate-v4':
            require_checkpoint_keys(
                seed_checkpoint,
                ('format_version', 'seed_architecture', 'split_mode'),
                kind=checkpoint_format)
            require_checkpoint_format_version(
                seed_checkpoint, 4, kind=checkpoint_format)
    elif checkpoint_format in ('unified-bidirectional-v3',
                                'unified-bidirectional-v4'):
        require_checkpoint_keys(
            seed_checkpoint,
            (
                'seed_policy', 'feature_dim', 'seed_include_ray_error',
                'seed_include_log_manip', 'controller_state_sha256',
                'controller_run_config_sha256', 'train_task_indices',
                'train_valid_mask', 'validation_task_indices',
                'validation_valid_mask', 'args', 'provenance',
            ),
            kind=checkpoint_format)
        if checkpoint_format == 'unified-bidirectional-v4':
            require_checkpoint_keys(
                seed_checkpoint,
                ('format_version', 'seed_architecture', 'split_mode'),
                kind=checkpoint_format)
            require_checkpoint_format_version(
                seed_checkpoint, 4, kind=checkpoint_format)
        else:
            require_checkpoint_format_version(
                seed_checkpoint, 3, kind=checkpoint_format)
    recorded_kind = seed_checkpoint.get('controller_kind')
    if recorded_kind is None and saved_provenance is not None:
        recorded_kind = saved_provenance.get('settings', {}).get(
            'controller_kind')
    if recorded_kind is None:
        recorded_kind = 'pure'
    controller_kind = recorded_kind if args.controller == 'auto' else args.controller
    if (controller_kind != recorded_kind
            and not args.allow_controller_mismatch):
        raise ValueError(
            f'seed checkpoint was trained with {recorded_kind} controller, '
            f'but --controller={controller_kind}; pass '
            '--allow-controller-mismatch for an explicit ablation')
    recorded_settings = (saved_provenance.get('settings', {})
                         if saved_provenance is not None else {})
    recorded_tau_enter = recorded_settings.get('tau_enter')
    recorded_tau_exit = recorded_settings.get('tau_exit')
    tau_enter = (recorded_tau_enter
                 if args.controller == 'auto' and recorded_tau_enter is not None
                 else args.tau_enter)
    tau_exit = (recorded_tau_exit
                if args.controller == 'auto' and recorded_tau_exit is not None
                else args.tau_exit)
    if (controller_kind == 'hybrid' and args.controller != 'auto'
            and recorded_tau_enter is not None
            and (tau_enter != recorded_tau_enter or tau_exit != recorded_tau_exit)
            and not args.allow_controller_mismatch):
        raise ValueError(
            'hybrid thresholds differ from seed-training provenance; pass '
            '--allow-controller-mismatch for an explicit ablation')
    seed_include_ray_error = bool(seed_checkpoint.get(
        'seed_include_ray_error', seed_policy.feature_dim in (34, 35)))
    seed_include_log_manip = bool(seed_checkpoint.get(
        'seed_include_log_manip', seed_policy.feature_dim in (32, 35)))
    seed_include_directional_dynamics = bool(seed_checkpoint.get(
        'seed_include_directional_dynamics', False))
    seed_return = seed_checkpoint.get('seed_return')
    if seed_return is None:
        seed_return = seed_checkpoint.get('args', {}).get(
            'seed_return', 'discounted')
    if seed_return not in ('undiscounted', 'discounted'):
        raise ValueError(f'unknown seed return objective: {seed_return!r}')
    seed_selector_objective = seed_checkpoint.get(
        'seed_selector_objective', seed_return)
    if seed_selector_objective not in (
            'progress_m', 'undiscounted', 'discounted'):
        raise ValueError(
            f'unknown seed selector objective: {seed_selector_objective!r}')
    if args.seed_probe_score == 'auto':
        seed_probe_score = {
            'progress_m': 'progress_m',
            'undiscounted': 'undiscounted_return',
            'discounted': 'discounted_return',
        }[seed_selector_objective]
    else:
        seed_probe_score = args.seed_probe_score
    seed_deployment = deployment_config_from_checkpoint(seed_checkpoint)
    env = build_env_from_run(args.controller_ckpt, args.chunk_size, device)
    valid_stats = None
    if not args.skip_physical_validation:
        dataset, valid_stats = validate_cached_dataset(
            dataset, env.kin, env.collision, cone_deg=env.cfg.cone_deg)
        print(f'[unified-eval] physical candidate validity: '
              f'{valid_stats["frac_valid"]:.1%}')
    same_candidate_cache = saved_provenance is not None
    if saved_provenance is not None:
        saved_cache = saved_provenance['candidate_cache']
        current_cache = candidate_artifact
        same_candidate_cache = (
            saved_cache['size'] == current_cache['size']
            and saved_cache['sha256'] == current_cache['sha256'])
        if not same_candidate_cache and not args.external_holdout:
            raise ValueError(
                'candidate cache differs from training; pass '
                '--external-holdout to evaluate every task without reusing '
                'training split indices')
        if same_candidate_cache and args.external_holdout:
            raise ValueError(
                '--external-holdout requires a candidate cache different '
                'from training; use --all-tasks for a training-cache '
                'diagnostic')
    elif args.external_holdout:
        raise ValueError(
            'legacy seed checkpoint has no candidate provenance, so external '
            'task overlap cannot be audited')
    elif not args.all_tasks:
        raise ValueError(
            'legacy seed checkpoint has no candidate fingerprint; pass '
            '--all-tasks for an explicit training-set diagnostic')
    if args.external_holdout:
        print(f'[unified-eval] external holdout ({len(dataset)} tasks)')
    task_overlap_rows = 0
    task_overlap_unique = 0
    task_overlap_audited = False
    if same_candidate_cache and not args.external_holdout:
        split_datasets = {}
        for split in ('train', 'validation'):
            task_key = f'{split}_task_indices'
            mask_key = f'{split}_valid_mask'
            if task_key in seed_checkpoint and mask_key in seed_checkpoint:
                split_dataset = dataset.select_source_tasks(
                    seed_checkpoint[task_key].cpu())
                assert_same_valid_mask(
                    split_dataset, seed_checkpoint[mask_key], label=split)
                split_datasets[split] = split_dataset
        if set(split_datasets) == {'train', 'validation'}:
            train_fingerprints = set(
                split_datasets['train'].task_fingerprints)
            validation_fingerprints = (
                split_datasets['validation'].task_fingerprints)
            overlap = [fingerprint for fingerprint in validation_fingerprints
                       if fingerprint in train_fingerprints]
            task_overlap_rows = len(overlap)
            task_overlap_unique = len(set(overlap))
            task_overlap_audited = True
            split_mode = seed_checkpoint.get('split_mode', 'row-random-v1')
            if task_overlap_rows and split_mode == 'task-geometry-grouped-v1':
                raise ValueError(
                    'checkpoint claims a task-grouped split, but '
                    f'{task_overlap_rows} validation rows overlap training')
            if task_overlap_rows:
                print(
                    '[unified-eval] WARNING: legacy row-heldout split has '
                    f'{task_overlap_rows} overlapping validation rows '
                    f'({task_overlap_unique} unique task geometries)')
    if args.external_holdout and saved_provenance is not None:
        saved_cache = saved_provenance['candidate_cache']
        training_cache_path = Path(saved_cache['path'])
        if not training_cache_path.is_file():
            raise ValueError(
                'cannot audit external task overlap because the recorded '
                f'training cache is unavailable: {training_cache_path}')
        current_training_artifact = file_fingerprint(training_cache_path)
        if (current_training_artifact['size'] != saved_cache['size']
                or current_training_artifact['sha256'] != saved_cache['sha256']):
            raise ValueError(
                'recorded training candidate cache changed; refusing an '
                'unaudited external evaluation')
        training_dataset = CachedSeedCandidateDataset.from_npz(
            training_cache_path)
        training_fingerprints = set(training_dataset.task_fingerprints)
        offline_ensemble = saved_provenance.get('offline_seed_ensemble')
        if offline_ensemble is not None:
            if not isinstance(offline_ensemble, dict):
                raise ValueError(
                    'offline ensemble provenance must be a dictionary')
            extra_fingerprints = seed_checkpoint.get(
                'offline_seed_ensemble_fit_task_fingerprints')
            if not isinstance(extra_fingerprints, (list, tuple)):
                raise ValueError(
                    'offline ensemble checkpoint is missing its exact fit '
                    'geometry audit set')
            for fingerprint in extra_fingerprints:
                if (not isinstance(fingerprint, str)
                        or len(fingerprint) != 64
                        or any(char not in '0123456789abcdef'
                               for char in fingerprint)):
                    raise ValueError(
                        'offline ensemble fit fingerprints must be lowercase '
                        'SHA-256 strings')
            if len(set(extra_fingerprints)) != len(extra_fingerprints):
                raise ValueError(
                    'offline ensemble fit fingerprint audit set has duplicates')
            training_fingerprints.update(extra_fingerprints)
        external_fingerprints = dataset.task_fingerprints
        overlap = [fingerprint for fingerprint in external_fingerprints
                   if fingerprint in training_fingerprints]
        task_overlap_rows = len(overlap)
        task_overlap_unique = len(set(overlap))
        task_overlap_audited = True
        if task_overlap_rows and not args.allow_task_overlap:
            raise ValueError(
                f'external cache has {task_overlap_rows} task rows '
                f'({task_overlap_unique} unique geometries) present in the '
                'training cache; pass --allow-task-overlap only for an '
                'explicit transfer diagnostic')
        if task_overlap_rows:
            print(
                '[unified-eval] WARNING: transfer diagnostic contains '
                f'{task_overlap_rows} overlapping task rows')
    if not args.all_tasks and not args.external_holdout and same_candidate_cache:
        if 'validation_task_indices' in seed_checkpoint:
            dataset = dataset.select_source_tasks(
                seed_checkpoint['validation_task_indices'].cpu())
            print(
                f'[unified-eval] using checkpoint validation split '
                f'({len(dataset)} tasks)')
        elif 'validation_indices' in seed_checkpoint:
            dataset = dataset.index_select(seed_checkpoint['validation_indices'])
            print(
                f'[unified-eval] using legacy validation split '
                f'({len(dataset)} tasks)')
        else:
            raise ValueError(
                'seed checkpoint has no validation split; pass --all-tasks '
                'only for an explicit training-set diagnostic')
    if saved_provenance is not None and 'controller' in saved_provenance:
        expected_controller = saved_provenance['controller']
        same_agent = (
            expected_controller['agent']['sha256']
            == controller_artifact['agent']['sha256'])
        same_config = (
            expected_controller['config']['sha256']
            == controller_artifact['config']['sha256'])
        if not same_config and not args.allow_controller_mismatch:
            raise ValueError(
                'controller config differs from seed-training provenance; '
                'cross-pair evaluation only permits controller weights to differ')
        if (not same_agent
                and not args.cross_pair_controller
                and not args.allow_controller_mismatch):
            raise ValueError(
                'controller weights differ from the frozen controller used '
                'for seed training; pass --cross-pair-controller for a '
                'matched phase decomposition')
    controller_agent = load_controller_agent(
        args.controller_ckpt, env, device).eval()
    expected_state_hash = seed_checkpoint.get('controller_state_sha256')
    if (expected_state_hash is not None
            and state_dict_fingerprint(controller_agent.state_dict())
            != expected_state_hash
            and not args.cross_pair_controller
            and not args.allow_controller_mismatch):
        raise ValueError(
            'controller agent does not match the controller stored in the '
            'unified checkpoint; pass --cross-pair-controller for a matched '
            'phase decomposition')
    expected_config_hash = seed_checkpoint.get(
        'controller_run_config_sha256')
    if (expected_config_hash is not None
            and controller_artifact['config']['sha256']
            != expected_config_hash
            and not args.allow_controller_mismatch):
        raise ValueError(
            'controller config does not match the unified checkpoint; pass '
            '--allow-controller-mismatch for an explicit ablation')
    controller_gamma = ppo_config_from_run(
        load_run_config(args.controller_ckpt)).gamma
    seed_probe_enabled = args.seed_probe_top_k > 0
    if seed_probe_enabled and controller_kind != 'pure':
        raise ValueError(
            'seed prefix probe supports only the pure FrozenRLController')
    if seed_probe_enabled and args.seed_probe_horizon > env.max_steps:
        raise ValueError(
            '--seed-probe-horizon cannot exceed controller max_steps')
    probe_width = args.seed_probe_top_k + 1 if seed_probe_enabled else 0
    probe_env = (
        build_env_from_run(
            args.controller_ckpt, args.chunk_size * probe_width, device)
        if seed_probe_enabled else None)

    def make_controller():
        if controller_kind == 'hybrid':
            return FrozenHybridController(
                controller_agent, ClassicalNullspaceController(env.kin),
                tau_enter, tau_exit)
        return FrozenRLController(controller_agent)

    n = len(dataset) if args.max_tasks is None else min(len(dataset), args.max_tasks)
    output_names = ['policy', 'feasibility', 'first_valid', 'fallback']
    if seed_probe_enabled:
        output_names.append('static_policy')
    outputs = {
        name: {
            'discounted_return': np.zeros(n, np.float32),
            'undiscounted_return': np.zeros(n, np.float32),
            'progress_m': np.zeros(n, np.float32),
            'episode_len': np.zeros(n, np.int64),
            'term_reason': np.zeros(n, np.int32),
            'candidate_index': np.zeros(n, np.int64),
        }
        for name in output_names
    }
    best_progress = np.zeros(n, np.float32) if args.full_candidate_oracle else None
    best_progress_candidate_index = (
        np.zeros(n, np.int64) if args.full_candidate_oracle else None)
    best_seed_return = np.zeros(n, np.float32) if args.full_candidate_oracle else None
    best_seed_candidate_index = (
        np.zeros(n, np.int64) if args.full_candidate_oracle else None)
    mean_valid_progress = (
        np.zeros(n, np.float32) if args.full_candidate_oracle else None)
    mean_valid_seed_return = (
        np.zeros(n, np.float32) if args.full_candidate_oracle else None)
    all_candidate_progress = (
        np.full(
            (n, dataset.batch.n_candidates), np.nan, dtype=np.float32)
        if args.full_candidate_oracle else None)
    all_candidate_seed_return = (
        np.full(
            (n, dataset.batch.n_candidates), np.nan, dtype=np.float32)
        if args.full_candidate_oracle else None)
    deployment_proposal_index = np.zeros(n, np.int64)
    deployment_predicted_gain = np.zeros(n, np.float32)
    deployment_accepted = np.zeros(n, np.bool_)
    if seed_probe_enabled:
        probe_shortlist_index = np.full(
            (n, probe_width), -1, dtype=np.int64)
        probe_shortlist_valid = np.zeros(
            (n, probe_width), dtype=np.bool_)
        probe_prefix_undiscounted = np.full(
            (n, probe_width), np.nan, dtype=np.float32)
        probe_prefix_discounted = np.full(
            (n, probe_width), np.nan, dtype=np.float32)
        probe_prefix_progress = np.full(
            (n, probe_width), np.nan, dtype=np.float32)
        probe_prefix_steps = np.full(
            (n, probe_width), -1, dtype=np.int64)
        probe_prefix_term_reason = np.full(
            (n, probe_width), -1, dtype=np.int32)
        probe_prefix_alive = np.zeros(
            (n, probe_width), dtype=np.bool_)
        probe_prefix_score = np.full(
            (n, probe_width), np.nan, dtype=np.float32)
        probe_selected_position = np.full(n, -1, dtype=np.int64)
        probe_active_steps = np.zeros(n, dtype=np.int64)
        probe_continuation_steps = np.zeros(n, dtype=np.int64)
        probe_total_controller_steps = np.zeros(n, dtype=np.int64)

    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        index, n_real = _pad_indices(start, end, n, args.chunk_size)
        candidates = dataset.batch.index_select(index).to(
            device, dtype=env.kin.dtype)
        features = initial_observation_features(
            env.kin, candidates,
            include_log_manip=seed_include_log_manip,
            include_ray_error=seed_include_ray_error,
            include_directional_dynamics=seed_include_directional_dynamics)
        with torch.no_grad():
            dist, _, feasibility = seed_policy.distribution_and_values(
                features, candidates.valid)
            deployment = select_seed_deployment(
                dist.logits, feasibility, candidates.valid, seed_deployment)
            policy_index = deployment.selected_index
            static_policy_index = deployment.selected_index
            feasibility_index = feasibility.masked_fill(
                ~candidates.valid, -torch.inf).argmax(dim=-1)
        first_index = candidates.valid.float().argmax(dim=1)
        deployment_proposal_index[start:end] = (
            deployment.proposal_index[:n_real].cpu().numpy())
        deployment_predicted_gain[start:end] = (
            deployment.predicted_gain[:n_real].cpu().numpy())
        deployment_accepted[start:end] = (
            deployment.accepted[:n_real].cpu().numpy())
        if dataset.fallback_index is None:
            fallback_index = first_index
        else:
            fallback_index = torch.full_like(
                first_index, dataset.fallback_index)
            row = torch.arange(candidates.n_tasks, device=device)
            fallback_index = torch.where(
                candidates.valid[row, fallback_index],
                fallback_index, first_index)

        probe_result = None
        if seed_probe_enabled:
            if probe_env is None:
                raise RuntimeError('probe environment was not constructed')
            probe_result = rollout_topk_prefix_lookahead(
                probe_env, env, candidates, dist.logits,
                FrozenRLController(controller_agent),
                top_k=args.seed_probe_top_k,
                horizon_steps=args.seed_probe_horizon,
                alive_bonus=args.seed_probe_alive_bonus,
                gamma=controller_gamma,
                restart_selected=True,
                score_objective=seed_probe_score)
            selected_position = probe_result.selected_shortlist_position
            policy_index = probe_result.shortlist_index.gather(
                1, selected_position.unsqueeze(-1)).squeeze(-1)
            sl = slice(start, end)
            probe_shortlist_index[sl] = (
                probe_result.shortlist_index[:n_real].cpu().numpy())
            probe_shortlist_valid[sl] = (
                probe_result.shortlist_valid[:n_real].cpu().numpy())
            probe_prefix_undiscounted[sl] = (
                probe_result.prefix_undiscounted_return[
                    :n_real].cpu().numpy())
            probe_prefix_discounted[sl] = (
                probe_result.prefix_discounted_return[:n_real].cpu().numpy())
            probe_prefix_progress[sl] = (
                probe_result.prefix_progress_m[:n_real].cpu().numpy())
            probe_prefix_steps[sl] = (
                probe_result.prefix_steps[:n_real].cpu().numpy())
            probe_prefix_term_reason[sl] = (
                probe_result.prefix_term_reason[:n_real].cpu().numpy())
            probe_prefix_alive[sl] = (
                probe_result.prefix_alive[:n_real].cpu().numpy())
            probe_prefix_score[sl] = (
                probe_result.prefix_score[:n_real].cpu().numpy())
            probe_selected_position[sl] = selected_position[
                :n_real].cpu().numpy()
            probe_active_steps[sl] = (
                probe_result.probe_active_steps[:n_real].cpu().numpy())
            probe_continuation_steps[sl] = (
                probe_result.continuation_steps[:n_real].cpu().numpy())
            probe_total_controller_steps[sl] = (
                probe_result.total_controller_steps[:n_real].cpu().numpy())

        rollout_actions = [
                ('policy', policy_index),
                ('feasibility', feasibility_index),
                ('first_valid', first_index),
                ('fallback', fallback_index),
        ]
        if seed_probe_enabled:
            rollout_actions.append(('static_policy', static_policy_index))
        for name, action in rollout_actions:
            if name == 'policy' and probe_result is not None:
                result = probe_result.rollout
            else:
                result = rollout_selected_seeds(
                    env, candidates, action, make_controller(),
                    gamma=controller_gamma)
            sl = slice(start, end)
            outputs[name]['discounted_return'][sl] = (
                result.discounted_return[:n_real].cpu().numpy())
            outputs[name]['undiscounted_return'][sl] = (
                result.undiscounted_return[:n_real].cpu().numpy())
            outputs[name]['progress_m'][sl] = result.progress_m[:n_real].cpu().numpy()
            outputs[name]['episode_len'][sl] = result.episode_len[:n_real].cpu().numpy()
            outputs[name]['term_reason'][sl] = result.term_reason[:n_real].cpu().numpy()
            outputs[name]['candidate_index'][sl] = action[:n_real].cpu().numpy()

        if args.full_candidate_oracle:
            slot_progress = torch.full(
                (args.chunk_size, candidates.n_candidates),
                -torch.inf, device=device)
            slot_seed_return = torch.full_like(slot_progress, -torch.inf)
            for slot in range(candidates.n_candidates):
                action = torch.full(
                    (args.chunk_size,), slot, device=device, dtype=torch.long)
                valid = candidates.valid[:, slot]
                safe_action = torch.where(valid, action, first_index)
                result = rollout_selected_seeds(
                    env, candidates, safe_action, make_controller(),
                    gamma=controller_gamma)
                slot_progress[valid, slot] = result.progress_m[valid]
                objective_return = (
                    result.undiscounted_return
                    if seed_return == 'undiscounted'
                    else result.discounted_return)
                slot_seed_return[valid, slot] = objective_return[valid]
            best = slot_progress[:n_real].max(dim=1)
            best_progress[start:end] = best.values.cpu().numpy()
            best_progress_candidate_index[start:end] = best.indices.cpu().numpy()
            best = slot_seed_return[:n_real].max(dim=1)
            best_seed_return[start:end] = best.values.cpu().numpy()
            best_seed_candidate_index[start:end] = best.indices.cpu().numpy()
            valid_float = candidates.valid[:n_real].to(slot_progress.dtype)
            valid_count = valid_float.sum(dim=1).clamp_min(1.0)
            mean_valid_progress[start:end] = (
                torch.where(
                    candidates.valid[:n_real], slot_progress[:n_real],
                    torch.zeros_like(slot_progress[:n_real])).sum(dim=1)
                / valid_count
            ).cpu().numpy()
            mean_valid_seed_return[start:end] = (
                torch.where(
                    candidates.valid[:n_real], slot_seed_return[:n_real],
                    torch.zeros_like(slot_seed_return[:n_real])).sum(dim=1)
                / valid_count
            ).cpu().numpy()
            all_candidate_progress[start:end] = torch.where(
                candidates.valid[:n_real], slot_progress[:n_real],
                torch.full_like(slot_progress[:n_real], torch.nan),
            ).cpu().numpy()
            all_candidate_seed_return[start:end] = torch.where(
                candidates.valid[:n_real], slot_seed_return[:n_real],
                torch.full_like(slot_seed_return[:n_real], torch.nan),
            ).cpu().numpy()
        print(f'[unified-eval] {end}/{n}', flush=True)

    policy_progress = outputs['policy']['progress_m']
    first_progress = outputs['first_valid']['progress_m']
    geometry_fingerprints = dataset.task_fingerprints[:n]
    evaluated_valid_mask = dataset.batch.valid[:n].numpy().copy()
    evaluated_valid_mask_sha256 = hashlib.sha256(
        np.ascontiguousarray(evaluated_valid_mask).tobytes()).hexdigest()
    print(
        f'policy progress {policy_progress.mean():.4f} m  '
        f'first-valid {first_progress.mean():.4f} m  '
        f'gain {(policy_progress - first_progress).mean():+.4f} m')
    if seed_probe_enabled:
        static_progress = outputs['static_policy']['progress_m']
        print(
            'seed probe active-step accounting '
            f'{probe_total_controller_steps.mean():.1f} steps/task '
            f'(prefix branches {probe_active_steps.mean():.1f}, '
            f'selected execution {probe_continuation_steps.mean():.1f})')
        print(
            f'static policy progress {static_progress.mean():.4f} m  '
            f'gain {(static_progress - first_progress).mean():+.4f} m')
    seed_probe_provenance = {
        'format': 'topk-controller-prefix-probe-v1',
        'enabled': seed_probe_enabled,
        'top_k': args.seed_probe_top_k if seed_probe_enabled else 0,
        'horizon_steps': (
            args.seed_probe_horizon if seed_probe_enabled else 0),
        'alive_bonus': (
            args.seed_probe_alive_bonus if seed_probe_enabled else 0.0),
        'score_objective': (
            seed_probe_score if seed_probe_enabled else 'disabled'),
        'shortlist': 'stable-actor-topk-union-first-valid-v1',
        'execution': 'virtual-probe-then-restart-selected-q0-v1',
        'controller_action': 'frozen-pure-deterministic-mean-v1',
        'cost': (
            'sum-active-probe-branches-plus-selected-full-execution-v1'),
        'requested_chunk_size': requested_chunk_size,
        'evaluation_chunk_size': args.chunk_size,
        'max_branch_batch': args.seed_probe_max_branch_batch,
        'parallel_branch_batch': (
            args.chunk_size * probe_width if seed_probe_enabled else 0),
        'seed_checkpoint_sha256': seed_artifact['sha256'],
        'candidate_cache_sha256': candidate_artifact['sha256'],
        'controller_agent_sha256': controller_artifact['agent']['sha256'],
        'controller_config_sha256': controller_artifact['config']['sha256'],
        'physical_validation': not args.skip_physical_validation,
        'evaluated_valid_mask_sha256': evaluated_valid_mask_sha256,
    }
    seed_probe_provenance_json = json.dumps(
        seed_probe_provenance, sort_keys=True, separators=(',', ':'))
    payload = {
        'task_indices': dataset.task_indices[:n].numpy(),
        'task_geometry_sha256': np.asarray(
            dataset.task_fingerprints[:n], dtype='<U64'),
        'seed_checkpoint_sha256': np.asarray(seed_artifact['sha256']),
        'candidate_cache_sha256': np.asarray(candidate_artifact['sha256']),
        'controller_agent_sha256': np.asarray(
            controller_artifact['agent']['sha256']),
        'controller_config_sha256': np.asarray(
            controller_artifact['config']['sha256']),
        'controller_state_sha256': np.asarray(
            state_dict_fingerprint(controller_agent.state_dict())),
        'checkpoint_phase': np.asarray(
            seed_checkpoint.get('phase', 'seed_gate')),
        'checkpoint_outer_round': np.int64(
            seed_checkpoint.get('outer_round', -1)),
        'seed_return_objective': np.asarray(seed_return),
        'seed_selector_objective': np.asarray(seed_selector_objective),
        'controller_kind': np.asarray(controller_kind),
        'evaluation_requested_chunk_size': np.int64(requested_chunk_size),
        'evaluation_chunk_size': np.int64(args.chunk_size),
        'seed_encoder': np.asarray(seed_policy.encoder_type),
        'seed_hidden_dim': np.int64(seed_policy.hidden_dim),
        'seed_attention_heads': np.int64(seed_policy.heads),
        'seed_attention_layers': np.int64(seed_policy.layers),
        'seed_attention_ff_mult': np.int64(seed_policy.ff_mult),
        'seed_include_directional_dynamics': np.bool_(
            seed_include_directional_dynamics),
        'seed_ensemble_size': np.int64(
            seed_policy.size
            if isinstance(seed_policy, CandidateSeedPolicyEnsemble) else 1),
        'seed_ensemble_aggregation': np.asarray(
            seed_policy.aggregation
            if isinstance(seed_policy, CandidateSeedPolicyEnsemble)
            else 'single-policy-v1'),
        'split_mode': np.asarray(
            seed_checkpoint.get('split_mode', 'row-random-v1')),
        'task_overlap_audited': np.bool_(task_overlap_audited),
        'task_overlap_rows': np.int64(task_overlap_rows),
        'task_overlap_unique': np.int64(task_overlap_unique),
        'physical_validation_enabled': np.bool_(
            not args.skip_physical_validation),
        'physical_validation_frac_valid': np.float64(
            valid_stats['frac_valid'] if valid_stats is not None else np.nan),
        'physical_validation_n_tasks': np.int64(
            valid_stats['n_tasks'] if valid_stats is not None else -1),
        'physical_validation_n_tasks_retained': np.int64(
            valid_stats['n_tasks_retained']
            if valid_stats is not None else -1),
        'physical_validation_n_tasks_rejected': np.int64(
            valid_stats['n_tasks_rejected']
            if valid_stats is not None else -1),
        'physical_validation_rejected_task_indices': np.asarray(
            valid_stats['rejected_task_indices']
            if valid_stats is not None else [], dtype=np.int64),
        'evaluated_candidate_valid': evaluated_valid_mask,
        'evaluated_candidate_valid_sha256': np.asarray(
            evaluated_valid_mask_sha256),
        'seed_deployment_mode': np.asarray(seed_deployment.mode),
        'seed_deployment_proposal_head': np.asarray(
            seed_deployment.proposal_head),
        'seed_deployment_threshold': np.float64(seed_deployment.threshold),
        'seed_deployment_comparison': np.asarray(
            seed_deployment.comparison),
        'deployment_proposal_candidate_index': deployment_proposal_index,
        'deployment_predicted_gain': deployment_predicted_gain,
        'deployment_accepted': deployment_accepted,
        'seed_probe_format': np.asarray('topk-controller-prefix-probe-v1'),
        'seed_probe_enabled': np.bool_(seed_probe_enabled),
        'seed_probe_top_k': np.int64(
            args.seed_probe_top_k if seed_probe_enabled else 0),
        'seed_probe_horizon_steps': np.int64(
            args.seed_probe_horizon if seed_probe_enabled else 0),
        'seed_probe_alive_bonus': np.float64(
            args.seed_probe_alive_bonus if seed_probe_enabled else 0.0),
        'seed_probe_shortlist_rule': np.asarray(
            'stable-actor-topk-union-first-valid-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_score': np.asarray(
            f'prefix-{seed_probe_score}-plus-alive-bonus-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_continuation': np.asarray(
            'virtual-probe-then-restart-selected-q0-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_controller': np.asarray(
            'frozen-pure-deterministic-mean-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_cost_accounting': np.asarray(
            'sum-active-probe-branches-plus-selected-full-execution-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_cost_semantics': np.asarray(
            'active-state-transitions-not-dense-kernel-invocations-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_parallel_branch_batch': np.int64(
            args.chunk_size * probe_width if seed_probe_enabled else 0),
        'seed_probe_max_branch_batch': np.int64(
            args.seed_probe_max_branch_batch if seed_probe_enabled else 0),
        'seed_probe_budget_mode': np.asarray(
            'extra-model-planning-unbounded-selected-execution-v1'
            if seed_probe_enabled else 'disabled'),
        'seed_probe_provenance_json': np.asarray(
            seed_probe_provenance_json),
        'seed_probe_provenance_sha256': np.asarray(
            _canonical_sha256(seed_probe_provenance)),
        'seed_probe_tie_break': np.asarray(
            'static-shortlist-order-v1'
            if seed_probe_enabled else 'disabled'),
    }
    if seed_deployment.proposal_head == 'actor-q':
        payload.update({
            'seed_deployment_proposal_q_weight': np.float64(
                seed_deployment.proposal_q_weight),
            'seed_deployment_proposal_q_scale_m': np.float64(
                seed_deployment.proposal_q_scale_m),
        })
    for name, metrics in outputs.items():
        for key, value in metrics.items():
            payload[f'{name}_{key}'] = value
    if seed_probe_enabled:
        payload.update({
            'seed_probe_shortlist_candidate_index': probe_shortlist_index,
            'seed_probe_shortlist_valid': probe_shortlist_valid,
            'seed_probe_prefix_undiscounted_return': (
                probe_prefix_undiscounted),
            'seed_probe_prefix_discounted_return': probe_prefix_discounted,
            'seed_probe_prefix_progress_m': probe_prefix_progress,
            'seed_probe_prefix_steps': probe_prefix_steps,
            'seed_probe_prefix_term_reason': probe_prefix_term_reason,
            'seed_probe_prefix_alive': probe_prefix_alive,
            'seed_probe_prefix_score': probe_prefix_score,
            'seed_probe_selected_shortlist_position': (
                probe_selected_position),
            'seed_probe_active_steps': probe_active_steps,
            'seed_probe_continuation_steps': probe_continuation_steps,
            'seed_probe_selected_execution_steps': (
                probe_continuation_steps),
            'seed_probe_total_controller_steps': (
                probe_total_controller_steps),
        })
    if best_progress is not None:
        progress_gain_rows = policy_progress - first_progress
        progress_headroom_rows = best_progress - first_progress
        progress_headroom = float(progress_headroom_rows.mean())
        progress_gain = float(progress_gain_rows.mean())
        progress_capture = progress_gain / max(progress_headroom, 1e-8)
        progress_gain_macro = _geometry_macro_mean(
            progress_gain_rows, geometry_fingerprints)
        progress_headroom_macro = _geometry_macro_mean(
            progress_headroom_rows, geometry_fingerprints)
        progress_capture_macro = (
            progress_gain_macro / max(progress_headroom_macro, 1e-8))
        if seed_probe_enabled:
            static_progress = outputs['static_policy']['progress_m']
            static_progress_gain_rows = static_progress - first_progress
            static_progress_gain = float(static_progress_gain_rows.mean())
            static_progress_capture = (
                static_progress_gain / max(progress_headroom, 1e-8))
            static_progress_gain_macro = _geometry_macro_mean(
                static_progress_gain_rows, geometry_fingerprints)
            static_progress_capture_macro = (
                static_progress_gain_macro
                / max(progress_headroom_macro, 1e-8))
        objective_name = f'{seed_return}_return'
        policy_seed_return = outputs['policy'][objective_name]
        first_seed_return = outputs['first_valid'][objective_name]
        seed_gain_rows = policy_seed_return - first_seed_return
        seed_headroom_rows = best_seed_return - first_seed_return
        seed_headroom = float(seed_headroom_rows.mean())
        seed_gain = float(seed_gain_rows.mean())
        seed_capture = seed_gain / max(seed_headroom, 1e-8)
        seed_gain_macro = _geometry_macro_mean(
            seed_gain_rows, geometry_fingerprints)
        seed_headroom_macro = _geometry_macro_mean(
            seed_headroom_rows, geometry_fingerprints)
        seed_capture_macro = seed_gain_macro / max(seed_headroom_macro, 1e-8)
        payload['best_progress_m'] = best_progress
        payload['candidate_valid'] = evaluated_valid_mask
        payload['candidate_progress_m'] = all_candidate_progress
        payload['candidate_seed_return'] = all_candidate_seed_return
        payload['best_progress_candidate_index'] = best_progress_candidate_index
        payload['best_seed_return'] = best_seed_return
        payload['best_seed_return_candidate_index'] = best_seed_candidate_index
        payload['mean_valid_progress_m'] = mean_valid_progress
        payload['mean_valid_seed_return'] = mean_valid_seed_return
        payload['metric_geometry_groups'] = np.int64(
            len(set(geometry_fingerprints)))
        payload['metric_progress_gain_m'] = np.float64(progress_gain)
        payload['metric_progress_headroom_m'] = np.float64(progress_headroom)
        payload['metric_progress_capture'] = np.float64(progress_capture)
        payload['metric_progress_gain_geometry_macro_m'] = np.float64(
            progress_gain_macro)
        payload['metric_progress_headroom_geometry_macro_m'] = np.float64(
            progress_headroom_macro)
        payload['metric_progress_capture_geometry_macro'] = np.float64(
            progress_capture_macro)
        if seed_probe_enabled:
            payload['metric_static_progress_gain_m'] = np.float64(
                static_progress_gain)
            payload['metric_static_progress_capture'] = np.float64(
                static_progress_capture)
            payload['metric_static_progress_gain_geometry_macro_m'] = (
                np.float64(static_progress_gain_macro))
            payload['metric_static_progress_capture_geometry_macro'] = (
                np.float64(static_progress_capture_macro))
            payload['metric_probe_incremental_progress_gain_m'] = np.float64(
                progress_gain - static_progress_gain)
        payload['metric_seed_return_gain'] = np.float64(seed_gain)
        payload['metric_seed_return_headroom'] = np.float64(seed_headroom)
        payload['metric_seed_return_capture'] = np.float64(seed_capture)
        payload['metric_seed_return_gain_geometry_macro'] = np.float64(
            seed_gain_macro)
        payload['metric_seed_return_headroom_geometry_macro'] = np.float64(
            seed_headroom_macro)
        payload['metric_seed_return_capture_geometry_macro'] = np.float64(
            seed_capture_macro)
        payload['metric_selector_capture'] = np.float64(
            progress_capture
            if seed_selector_objective == 'progress_m' else seed_capture)
        payload['metric_selector_capture_geometry_macro'] = np.float64(
            progress_capture_macro
            if seed_selector_objective == 'progress_m'
            else seed_capture_macro)
        # Keep the generic metric backward-compatible with earlier paired
        # evaluations, where the seed-return objective was canonical.
        payload['metric_capture'] = np.float64(seed_capture)
        payload['metric_capture_geometry_macro'] = np.float64(
            seed_capture_macro)
        print(
            f'best-candidate {best_progress.mean():.4f} m  '
            f'headroom {progress_headroom:.4f} m  '
            f'progress capture {100 * progress_capture:.1f}%  '
            f'geometry-macro {100 * progress_capture_macro:.1f}%')
        print(
            f'{seed_return} return policy {policy_seed_return.mean():.2f}  '
            f'first {first_seed_return.mean():.2f}  '
            f'best {best_seed_return.mean():.2f}  '
            f'objective capture {100 * seed_capture:.1f}%')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f'[unified-eval] saved -> {out}')


if __name__ == '__main__':
    main()
