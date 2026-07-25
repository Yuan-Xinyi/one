"""Training-only one-step search followed by search-free actor distillation.

The static seed ensemble and its conservative deployment rule are never
changed here.  Training trajectories start only from the deployment seed on
the ensemble's leakage-safe fit split.  At one state one-to-eight steps before
each trajectory terminates, an exact deterministic simulator teacher compares
16 one-step actions:

* slot 0 is always the frozen controller action;
* slot 1 is the classical null-space action;
* slots 2--15 are seven paired perturbations around slot 0.

Every branch executes its candidate for exactly one step and then resumes the
same frozen controller to termination.  By default, only locally perturbed
actions supported by at least two improving branches may become labels.  The
small interpolated target is itself rolled out exactly and must retain the
improvement.  Ordinary trajectory states retain the original actor at a 12:1
ratio.  Search is therefore a training-time teacher only: a published
checkpoint still performs one static seed selection and one deterministic
controller rollout, with no model-based inference.

Each supervised epoch is evaluated on immutable validation seed choices and,
when supplied, disjoint external development caches.  Promotion checks paired
mean, clipped/trimmed mean, lower-tail CVaR, win/harm balance and first-valid
coverage.  The strict single-validation-CI mode remains available.  Otherwise
the exact input controller is published.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController,
)
from Yuan.RL_controller.env.env import build_task_aligned_basis
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    atomic_torch_save,
    build_env_from_run,
    load_controller_agent,
    load_run_config,
    resolve_controller_dir,
)
from Yuan.unified_rl.controller_rollout import (
    NSRLEnvState,
    restore_env_state,
    snapshot_env_state,
)
from Yuan.unified_rl.evaluate import load_seed_policy
from Yuan.unified_rl.evaluate_residual import geometry_grouped_bootstrap_ci
from Yuan.unified_rl.joint_controller_refine import (
    _cpu_tree,
    _evaluate_fixed_seeds,
    _same_artifact,
    _save_evaluation,
    _static_seed_indices,
)
from Yuan.unified_rl.provenance import (
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.reproducibility import device_identity, seed_global_rng
from Yuan.unified_rl.seed_policy import seed_policy_ensemble_states
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


SEARCH_CANDIDATES = 16
PAIRED_PERTURBATIONS = 7
def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    """Return a symmetric trimmed mean without scipy version coupling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError('trimmed mean requires a non-empty vector')
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError('trim fraction must be finite in [0, 0.5)')
    trim = int(math.floor(fraction * values.size))
    ordered = np.sort(values)
    if trim:
        ordered = ordered[trim:-trim]
    return float(ordered.mean())


def _robust_delta_metrics(
    policy: np.ndarray,
    baseline: np.ndarray,
    *,
    trim_fraction: float,
    clip_m: float,
    cvar_fraction: float,
    harm_threshold_m: float,
) -> dict[str, float]:
    """Robust paired metrics used only for development-set promotion."""
    policy = np.asarray(policy, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if (policy.ndim != 1 or baseline.shape != policy.shape
            or policy.size < 1):
        raise ValueError('policy and baseline must be equal non-empty vectors')
    if not math.isfinite(clip_m) or clip_m <= 0.0:
        raise ValueError('clip_m must be positive')
    if not math.isfinite(cvar_fraction) or not 0.0 < cvar_fraction <= 1.0:
        raise ValueError('cvar_fraction must be finite in (0, 1]')
    delta = policy - baseline
    tail_count = max(1, int(math.ceil(cvar_fraction * policy.size)))
    policy_cvar = float(np.partition(
        policy, tail_count - 1)[:tail_count].mean())
    baseline_cvar = float(np.partition(
        baseline, tail_count - 1)[:tail_count].mean())
    win_rate = float(np.mean(delta > harm_threshold_m))
    harm_rate = float(np.mean(delta < -harm_threshold_m))
    return {
        'paired_delta_row_mean_m': float(delta.mean()),
        'paired_delta_trimmed_mean_m': _trimmed_mean(
            delta, trim_fraction),
        'paired_delta_clipped_mean_m': float(np.clip(
            delta, -clip_m, clip_m).mean()),
        'paired_win_rate': win_rate,
        'paired_harm_rate': harm_rate,
        'paired_win_minus_harm_rate': win_rate - harm_rate,
        'policy_lower_tail_cvar_m': policy_cvar,
        'baseline_lower_tail_cvar_m': baseline_cvar,
        'lower_tail_cvar_delta_m': policy_cvar - baseline_cvar,
    }


def _conservative_search_targets(
    search: dict[str, torch.Tensor],
    *,
    minimum_gain_m: float,
    blend_gain_scale_m: float,
    maximum_blend: float,
    local_only: bool,
    minimum_supporting_actions: int,
    maximum_target_action_delta: float | None,
) -> dict[str, torch.Tensor]:
    """Choose locally supported labels and cap their policy-space movement.

    Slot one is a potentially distant classical action.  In ``local_only``
    mode it remains a useful search diagnostic but cannot supervise the neural
    controller.  A label is retained only when multiple independently sampled
    candidates clear the same absolute progress threshold.
    """
    progress = search['slot_progress_m']
    actions = search['candidate_action']
    current = search['current_action']
    if (progress.ndim != 2 or actions.ndim != 3 or current.ndim != 2
            or progress.shape[:2] != actions.shape[:2]
            or actions.shape[0] != current.shape[0]
            or actions.shape[2] != current.shape[1]):
        raise ValueError('search tensors have incompatible shapes')
    if progress.shape[1] != SEARCH_CANDIDATES:
        raise ValueError('search progress has an unexpected candidate count')
    if minimum_supporting_actions < 1:
        raise ValueError('minimum_supporting_actions must be positive')

    baseline = progress[:, 0]
    eligible_slot = torch.ones_like(progress, dtype=torch.bool)
    eligible_slot[:, 0] = False
    if local_only:
        eligible_slot[:, 1] = False
    candidate_gain = progress - baseline[:, None]
    masked_progress = progress.masked_fill(~eligible_slot, -torch.inf)
    best_progress, best_index = masked_progress.max(dim=-1)
    row = torch.arange(progress.shape[0], device=progress.device)
    best_action = actions[row, best_index]
    gain = best_progress - baseline
    support_count = (
        (candidate_gain > minimum_gain_m) & eligible_slot
    ).sum(dim=-1)
    accepted = (
        (gain > minimum_gain_m)
        & (support_count >= minimum_supporting_actions))

    nominal_blend = maximum_blend * (
        gain / blend_gain_scale_m).clamp(0.0, 1.0)
    action_delta = best_action - current
    target_delta = nominal_blend[:, None] * action_delta
    if maximum_target_action_delta is not None:
        target_norm = target_delta.norm(dim=-1, keepdim=True)
        target_delta = target_delta * (
            float(maximum_target_action_delta)
            / target_norm.clamp_min(1e-12)).clamp(max=1.0)
    target = (current + target_delta).clamp(-1.0, 1.0)
    action_delta_norm = action_delta.norm(dim=-1)
    target_delta_norm = (target - current).norm(dim=-1)
    effective_blend = target_delta_norm / action_delta_norm.clamp_min(1e-12)
    return {
        'target_action': target,
        'accepted_before_verification': accepted,
        'label_best_index': best_index,
        'label_best_action': best_action,
        'label_best_progress_m': best_progress,
        'label_gain_m': gain,
        'supporting_action_count': support_count,
        'candidate_action_delta_norm': action_delta_norm,
        'target_action_delta_norm': target_delta_norm,
        'effective_blend_weight': effective_blend,
    }


def _state_from_fields(fields: dict[str, torch.Tensor]) -> NSRLEnvState:
    return NSRLEnvState(**{
        name: fields[name]
        for name in NSRLEnvState.__dataclass_fields__
    })


def _cat_states(states: Sequence[NSRLEnvState]) -> NSRLEnvState:
    if not states:
        raise ValueError('cannot concatenate an empty state sequence')
    return NSRLEnvState(**{
        name: torch.cat([getattr(state, name) for state in states], dim=0)
        for name in NSRLEnvState.__dataclass_fields__
    })


def _state_to(
    state: NSRLEnvState,
    device: torch.device,
) -> NSRLEnvState:
    return NSRLEnvState(**{
        name: getattr(state, name).to(device=device)
        for name in NSRLEnvState.__dataclass_fields__
    })


def _save_state_npz(
    path: Path,
    state: NSRLEnvState,
    **arrays: np.ndarray,
) -> None:
    payload = dict(arrays)
    payload.update({
        f'state_{name}': getattr(state, name).cpu().numpy()
        for name in NSRLEnvState.__dataclass_fields__
    })
    np.savez_compressed(path, **payload)


@torch.no_grad()
def _classical_action(
    env,
    controller: ClassicalNullspaceController,
) -> torch.Tensor:
    basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping)
    q_dot = controller.q_dot_null(
        env.q, env.line_dir, env.n_target)
    action = (
        basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)
    ).squeeze(-1)
    return torch.nan_to_num(
        action / env.a_max, nan=0.0).clamp(-1.0, 1.0)


@torch.no_grad()
def _paired_action_candidates(
    current_action: torch.Tensor,
    classical_action: torch.Tensor,
    *,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return (B,16,A), preserving the current actor as exact slot zero."""
    if current_action.shape != classical_action.shape:
        raise ValueError('current and classical actions must have equal shape')
    directions = torch.randn(
        (current_action.shape[0], PAIRED_PERTURBATIONS,
         current_action.shape[1]),
        device=current_action.device, dtype=current_action.dtype,
        generator=generator)
    directions = directions / directions.norm(
        dim=-1, keepdim=True).clamp_min(1e-8)
    delta = float(sigma) * directions
    paired = torch.stack([
        current_action[:, None, :] + delta,
        current_action[:, None, :] - delta,
    ], dim=2).reshape(current_action.shape[0], -1, current_action.shape[1])
    candidates = torch.cat([
        current_action[:, None, :],
        classical_action[:, None, :],
        paired,
    ], dim=1).clamp(-1.0, 1.0)
    if candidates.shape[1] != SEARCH_CANDIDATES:
        raise RuntimeError('internal search candidate count is not 16')
    # Clamping happens after concatenation, but slot zero was already bounded.
    if not torch.equal(candidates[:, 0], current_action):
        raise RuntimeError('search slot zero no longer equals current action')
    return candidates


@torch.no_grad()
def _collect_tail_states(
    controller,
    controller_dir: Path,
    dataset: CachedSeedCandidateDataset,
    selected_index: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int,
    seed: int,
) -> tuple[NSRLEnvState, torch.Tensor, dict[str, np.ndarray], torch.Tensor]:
    """Collect one exact pre-terminal state per static deployment task.

    The state offset is sampled uniformly from the available last 1--8
    pre-action snapshots.  All ordinary active observations are returned as a
    retention reservoir; it is reduced to exactly 3x accepted search labels
    after search gating.
    """
    if selected_index.shape != (len(dataset),):
        raise ValueError('selected_index shape does not match collection data')
    offset_generator = torch.Generator().manual_seed(int(seed))
    state_parts: list[NSRLEnvState] = []
    obs_parts: list[torch.Tensor] = []
    ordinary_parts: list[torch.Tensor] = []
    fit_local_parts: list[torch.Tensor] = []
    source_index_parts: list[torch.Tensor] = []
    offset_parts: list[torch.Tensor] = []
    terminal_step_parts: list[torch.Tensor] = []

    controller.eval()
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        n = end - start
        env = build_env_from_run(
            controller_dir, n, device,
            env_overrides={'observe_ray_error': True})
        batch = dataset.batch.index_select(
            torch.arange(start, end)).to(device, dtype=env.kin.dtype)
        action_index = selected_index[start:end].to(device)
        selection = batch.select(action_index)
        env.line_dist = ScriptedLineDistribution(selection.specs())
        env.reset()
        ring: list[tuple[NSRLEnvState, torch.Tensor]] = []
        collected = torch.zeros(n, device=device, dtype=torch.bool)

        for _ in range(env.max_steps + 1):
            active = ~env.done_persistent
            if not bool(active.any().item()):
                break
            obs = env.current_obs()
            ordinary_parts.append(obs[active].float().cpu())
            ring.append((snapshot_env_state(env), obs.detach().clone()))
            if len(ring) > 8:
                ring.pop(0)
            env_action = controller.actor_mean(obs).clamp(-1.0, 1.0)
            _, _, _, _, info = env.step(env_action, auto_reset=False)
            new_done = info['episode_done']
            if bool(new_done.any().item()):
                lanes = torch.nonzero(new_done, as_tuple=False).squeeze(-1)
                for lane_tensor in lanes:
                    lane = int(lane_tensor.item())
                    available = min(8, len(ring), int(env.t[lane].item()))
                    if available < 1:
                        raise RuntimeError('terminal task has no pre-action state')
                    offset = int(torch.randint(
                        1, available + 1, (1,),
                        generator=offset_generator).item())
                    lane_index = torch.tensor(
                        [lane], device=device, dtype=torch.long)
                    chosen_state, chosen_obs = ring[-offset]
                    state_parts.append(_state_to(
                        chosen_state.index_select(lane_index),
                        torch.device('cpu')))
                    obs_parts.append(chosen_obs.index_select(
                        0, lane_index).float().cpu())
                    global_local = start + lane
                    fit_local_parts.append(torch.tensor([global_local]))
                    source_index_parts.append(
                        dataset.task_indices[global_local:global_local + 1])
                    offset_parts.append(torch.tensor([offset]))
                    terminal_step_parts.append(
                        env.t[lane:lane + 1].long().cpu())
                    collected[lane] = True
        if not bool(collected.all().item()):
            missing = torch.nonzero(~collected).flatten().tolist()
            raise RuntimeError(
                f'failed to collect terminal states for lanes {missing[:20]}')
        print(
            f'[search-distill] collected tail states {end}/{len(dataset)}',
            flush=True)

    tail_state = _cat_states(state_parts)
    tail_obs = torch.cat(obs_parts)
    fit_local_index = torch.cat(fit_local_parts).long()
    order = torch.argsort(fit_local_index, stable=True)
    tail_state = tail_state.index_select(order)
    tail_obs = tail_obs.index_select(0, order)
    metadata = {
        'fit_local_index': fit_local_index[order].numpy(),
        'source_task_index': torch.cat(source_index_parts)[order].numpy(),
        'terminal_offset_steps': torch.cat(offset_parts)[order].numpy(),
        'terminal_step': torch.cat(terminal_step_parts)[order].numpy(),
    }
    expected = torch.arange(len(dataset))
    if not torch.equal(fit_local_index[order], expected):
        raise RuntimeError('tail-state collection did not cover fit tasks once')
    ordinary_obs = torch.cat(ordinary_parts)
    return tail_state, tail_obs, metadata, ordinary_obs


@torch.no_grad()
def _search_state_chunk(
    controller,
    controller_dir: Path,
    state: NSRLEnvState,
    device: torch.device,
    *,
    sigma: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Run one exact candidate step and frozen-policy continuation."""
    state = _state_to(state, device)
    n = state.n_envs
    probe = build_env_from_run(
        controller_dir, n, device,
        env_overrides={'observe_ray_error': True})
    restore_env_state(probe, state)
    obs = probe.current_obs()
    current = controller.actor_mean(obs).clamp(-1.0, 1.0)
    classical = _classical_action(
        probe, ClassicalNullspaceController(probe.kin))
    candidate_actions = _paired_action_candidates(
        current, classical, sigma=sigma, generator=generator)

    branches = build_env_from_run(
        controller_dir, n * SEARCH_CANDIDATES, device,
        env_overrides={'observe_ray_error': True})
    repeat_index = torch.arange(
        n, device=device).repeat_interleave(SEARCH_CANDIDATES)
    restore_env_state(branches, state.index_select(repeat_index))
    flat_actions = candidate_actions.reshape(
        n * SEARCH_CANDIDATES, -1).to(branches.kin.dtype)
    branches.step(flat_actions, auto_reset=False)
    for _ in range(branches.max_steps + 1):
        if bool(branches.done_persistent.all().item()):
            break
        action = controller.actor_mean(
            branches.current_obs()).clamp(-1.0, 1.0)
        branches.step(action, auto_reset=False)
    if not bool(branches.done_persistent.all().item()):
        raise RuntimeError('search continuation exceeded environment horizon')
    p_final, _, _, _ = branches.kin.tcp_fk_jac(branches.q)
    progress = (
        (p_final - branches.p_start) * branches.line_dir
    ).sum(dim=-1).view(n, SEARCH_CANDIDATES)
    best_progress, best_index = progress.max(dim=-1)
    row = torch.arange(n, device=device)
    best_action = candidate_actions[row, best_index]
    baseline_progress = progress[:, 0]
    return {
        'obs': obs.float().cpu(),
        'candidate_action': candidate_actions.float().cpu(),
        'current_action': current.float().cpu(),
        'classical_action': classical.float().cpu(),
        'best_action': best_action.float().cpu(),
        'slot_progress_m': progress.float().cpu(),
        'best_index': best_index.long().cpu(),
        'baseline_progress_m': baseline_progress.float().cpu(),
        'best_progress_m': best_progress.float().cpu(),
        'gain_m': (best_progress - baseline_progress).float().cpu(),
    }


@torch.no_grad()
def _search_tail_states(
    controller,
    controller_dir: Path,
    states: NSRLEnvState,
    device: torch.device,
    *,
    chunk_size: int,
    sigma: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    parts: list[dict[str, torch.Tensor]] = []
    for start in range(0, states.n_envs, chunk_size):
        end = min(start + chunk_size, states.n_envs)
        index = torch.arange(start, end)
        parts.append(_search_state_chunk(
            controller, controller_dir, states.index_select(index), device,
            sigma=sigma, generator=generator))
        print(
            f'[search-distill] searched states {end}/{states.n_envs}',
            flush=True)
    return {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in parts[0]
    }


@torch.no_grad()
def _rollout_state_actions(
    controller,
    controller_dir: Path,
    states: NSRLEnvState,
    actions: torch.Tensor,
    device: torch.device,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Execute one supplied action then the frozen controller to termination."""
    if actions.ndim != 2 or actions.shape[0] != states.n_envs:
        raise ValueError('actions must have shape (states.n_envs, action_dim)')
    parts: list[torch.Tensor] = []
    controller.eval()
    for start in range(0, states.n_envs, chunk_size):
        end = min(start + chunk_size, states.n_envs)
        index = torch.arange(start, end)
        state = _state_to(states.index_select(index), device)
        env = build_env_from_run(
            controller_dir, end - start, device,
            env_overrides={'observe_ray_error': True})
        restore_env_state(env, state)
        env.step(
            actions[start:end].to(device=device, dtype=env.kin.dtype),
            auto_reset=False)
        for _ in range(env.max_steps + 1):
            if bool(env.done_persistent.all().item()):
                break
            action = controller.actor_mean(
                env.current_obs()).clamp(-1.0, 1.0)
            env.step(action, auto_reset=False)
        if not bool(env.done_persistent.all().item()):
            raise RuntimeError(
                'target verification exceeded environment horizon')
        p_final, _, _, _ = env.kin.tcp_fk_jac(env.q)
        progress = ((p_final - env.p_start) * env.line_dir).sum(dim=-1)
        parts.append(progress.float().cpu())
    return torch.cat(parts)


def _evaluation_summary(
    epoch: int,
    outputs: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    fingerprints: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    first_valid_tolerance_m: float,
    first_coverage_tolerance: float,
    harm_threshold_m: float,
    max_paired_harm_rate: float,
    selected_harm_tolerance: float,
    robust_trim_fraction: float = 0.05,
    robust_clip_m: float = 0.05,
    robust_cvar_fraction: float = 0.05,
    robust_trim_tolerance_m: float = 1e-5,
    robust_cvar_tolerance_m: float = 0.001,
    require_positive_ci: bool = True,
) -> dict[str, Any]:
    policy = outputs['policy_progress_m']
    first = outputs['first_valid_progress_m']
    baseline_policy = baseline['policy_progress_m']
    baseline_first = baseline['first_valid_progress_m']
    delta = policy - baseline_policy
    estimate, low, high, groups = geometry_grouped_bootstrap_ci(
        delta, fingerprints, seed=bootstrap_seed,
        samples=bootstrap_samples)
    coverage = float(np.mean(policy >= first))
    baseline_coverage = float(np.mean(baseline_policy >= baseline_first))
    robust = _robust_delta_metrics(
        policy, baseline_policy,
        trim_fraction=robust_trim_fraction,
        clip_m=robust_clip_m,
        cvar_fraction=robust_cvar_fraction,
        harm_threshold_m=harm_threshold_m)
    paired_harm = robust['paired_harm_rate']
    selected_harm = float(np.mean(policy < first - harm_threshold_m))
    baseline_selected_harm = float(np.mean(
        baseline_policy < baseline_first - harm_threshold_m))
    first_mean = float(first.mean())
    baseline_first_mean = float(baseline_first.mean())
    common_eligible = bool(
        epoch > 0
        and first_mean >= baseline_first_mean - first_valid_tolerance_m
        and coverage >= baseline_coverage - first_coverage_tolerance
        and paired_harm <= max_paired_harm_rate
        and selected_harm <= (
            baseline_selected_harm + selected_harm_tolerance))
    robust_eligible = bool(
        common_eligible
        and estimate > 0.0
        and robust['paired_delta_clipped_mean_m'] > 0.0
        and robust['paired_delta_trimmed_mean_m'] >= -robust_trim_tolerance_m
        and robust['paired_win_minus_harm_rate'] >= 0.0
        and robust['lower_tail_cvar_delta_m'] >= -robust_cvar_tolerance_m)
    eligible = bool(
        robust_eligible and (not require_positive_ci or low > 0.0))
    result = {
        'epoch': int(epoch),
        'policy_progress_mean_m': float(policy.mean()),
        'first_valid_progress_mean_m': first_mean,
        'policy_episode_len_mean': float(
            outputs['policy_episode_len'].mean()),
        'gain_vs_baseline_geometry_macro_m': estimate,
        'gain_vs_baseline_ci95_low_m': low,
        'gain_vs_baseline_ci95_high_m': high,
        'geometry_groups': groups,
        'first_valid_progress_delta_m': (
            first_mean - baseline_first_mean),
        'first_coverage_rate': coverage,
        'baseline_first_coverage_rate': baseline_coverage,
        'paired_harm_rate': paired_harm,
        'selected_vs_first_harm_rate': selected_harm,
        'baseline_selected_vs_first_harm_rate': baseline_selected_harm,
        'robust_development_eligible': robust_eligible,
        'positive_ci_required': bool(require_positive_ci),
        'promotion_eligible': eligible,
    }
    result.update(robust)
    return result


def _validate_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'--{name} must be finite in [0, 1]')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Distill a training-only exact tail-search teacher into the same '
            'single-rollout controller actor.'))
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument(
        '--external-dev-candidates', action='append', default=[],
        help=(
            'optional disjoint development cache; repeatable. With '
            '--development-promotion every cache must pass the robust gate'))
    parser.add_argument('--controller-ckpt', default=None)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--collection-chunk-size', type=int, default=512)
    parser.add_argument('--search-chunk-size', type=int, default=128)
    parser.add_argument('--eval-chunk-size', type=int, default=1024)
    parser.add_argument(
        '--max-fit-tasks', type=int, default=None,
        help='optional deterministic fit-task subsample for a search gate')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument(
        '--train-scope', choices=('mean-head', 'actor'),
        default='mean-head',
        help='mean-head limits collateral drift on the small Q-filtered dataset')
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--perturbation-sigma', type=float, default=0.08)
    parser.add_argument('--minimum-search-gain-m', type=float, default=0.001)
    parser.add_argument('--blend-gain-scale-m', type=float, default=0.020)
    parser.add_argument('--maximum-blend', type=float, default=0.25)
    parser.add_argument(
        '--allow-classical-search-labels', action='store_true',
        help=(
            'allow the potentially distant classical slot to supervise the '
            'actor; the conservative default uses local perturbations only'))
    parser.add_argument(
        '--minimum-supporting-actions', type=int, default=2,
        help='independent local actions that must clear the search-gain gate')
    parser.add_argument(
        '--maximum-target-action-delta', type=float, default=0.01,
        help='maximum L2 movement of a distilled action label from C0')
    parser.add_argument(
        '--skip-target-verification', action='store_true',
        help='skip exact rollout verification of the final interpolated label')
    parser.add_argument(
        '--retention-multiplier', type=int, default=12,
        help='frozen-controller trajectory labels per accepted search label')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--first-valid-tolerance-m', type=float, default=0.001)
    parser.add_argument('--first-coverage-tolerance', type=float, default=0.01)
    parser.add_argument('--harm-threshold-m', type=float, default=0.001)
    parser.add_argument('--max-paired-harm-rate', type=float, default=0.10)
    parser.add_argument('--selected-harm-tolerance', type=float, default=0.01)
    parser.add_argument('--robust-trim-fraction', type=float, default=0.05)
    parser.add_argument('--robust-clip-m', type=float, default=0.05)
    parser.add_argument('--robust-cvar-fraction', type=float, default=0.05)
    parser.add_argument(
        '--robust-trim-tolerance-m', type=float, default=1e-5)
    parser.add_argument(
        '--robust-cvar-tolerance-m', type=float, default=0.001)
    parser.add_argument(
        '--development-promotion', action='store_true',
        help=(
            'use robust point metrics on development data instead of a '
            'single-split positive CI; a fresh sealed test remains required'))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=61000)
    args = parser.parse_args()

    for name in (
            'collection_chunk_size', 'search_chunk_size', 'eval_chunk_size',
            'epochs', 'batch_size', 'bootstrap_samples',
            'minimum_supporting_actions', 'retention_multiplier'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.max_fit_tasks is not None and args.max_fit_tasks < 1:
        raise ValueError('--max-fit-tasks must be positive')
    if args.development_promotion and not args.external_dev_candidates:
        raise ValueError(
            '--development-promotion requires --external-dev-candidates')
    for name in (
            'actor_lr', 'perturbation_sigma', 'minimum_search_gain_m',
            'blend_gain_scale_m', 'harm_threshold_m'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    for name in ('weight_decay', 'first_valid_tolerance_m'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{name.replace("_", "-")} must be non-negative')
    for name in (
            'maximum-blend', 'first-coverage-tolerance',
            'max-paired-harm-rate', 'selected-harm-tolerance'):
        _validate_probability(
            getattr(args, name.replace('-', '_')), name)
    if args.maximum_blend > 0.25 + 1e-12:
        raise ValueError('--maximum-blend cannot exceed conservative cap 0.25')
    if (not math.isfinite(args.maximum_target_action_delta)
            or args.maximum_target_action_delta <= 0.0):
        raise ValueError('--maximum-target-action-delta must be positive')
    if (not math.isfinite(args.robust_clip_m)
            or args.robust_clip_m <= 0.0):
        raise ValueError('--robust-clip-m must be positive')
    for name in ('robust_trim_tolerance_m', 'robust_cvar_tolerance_m'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{name.replace("_", "-")} must be non-negative')
    if (not math.isfinite(args.robust_trim_fraction)
            or not 0.0 <= args.robust_trim_fraction < 0.5):
        raise ValueError('--robust-trim-fraction must be in [0, 0.5)')
    if (not math.isfinite(args.robust_cvar_fraction)
            or not 0.0 < args.robust_cvar_fraction <= 1.0):
        raise ValueError('--robust-cvar-fraction must be in (0, 1]')
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        raise ValueError('--gamma must be finite in [0, 1]')

    source_path = Path(args.source_checkpoint).expanduser().resolve(strict=True)
    candidate_path = Path(args.candidates).expanduser().resolve(strict=True)
    external_paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in args.external_dev_candidates
    ]
    if len(set(external_paths)) != len(external_paths):
        raise ValueError('--external-dev-candidates contains duplicates')
    source_dir = source_path.parent
    controller_dir = resolve_controller_dir(
        source_dir if args.controller_ckpt is None else args.controller_ckpt)
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if os.path.lexists(out_dir):
        raise FileExistsError(
            f'refusing to overwrite output directory: {out_dir}')
    out_dir.mkdir(parents=True)
    search_dir = out_dir / 'search'
    eval_dir = out_dir / 'eval'
    snapshot_dir = out_dir / 'snapshots'
    search_dir.mkdir()
    eval_dir.mkdir()
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
    if 'offline_ensemble_fit_local_indices' not in source:
        raise ValueError(
            'source lacks leakage-safe offline_ensemble_fit_local_indices')
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
    validation_probe = build_env_from_run(
        controller_dir, args.eval_chunk_size, device,
        env_overrides={'observe_ray_error': True})
    dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    dataset, validity = validate_cached_dataset(
        dataset, validation_probe.kin, validation_probe.collision,
        cone_deg=validation_probe.cfg.cone_deg)
    train_dataset = dataset.select_source_tasks(
        torch.as_tensor(source['train_task_indices']).cpu())
    validation_dataset = dataset.select_source_tasks(
        torch.as_tensor(source['validation_task_indices']).cpu())
    assert_same_valid_mask(
        train_dataset, source['train_valid_mask'], label='training')
    assert_same_valid_mask(
        validation_dataset, source['validation_valid_mask'],
        label='validation')
    fit_indices = torch.as_tensor(
        source['offline_ensemble_fit_local_indices']).long().cpu()
    if (fit_indices.ndim != 1 or fit_indices.numel() < 1
            or fit_indices.unique().numel() != fit_indices.numel()
            or bool(((fit_indices < 0)
                     | (fit_indices >= len(train_dataset))).any().item())):
        raise ValueError('offline ensemble fit indices are invalid')
    if (args.max_fit_tasks is not None
            and args.max_fit_tasks < fit_indices.numel()):
        fit_generator = torch.Generator().manual_seed(args.seed + 17)
        chosen = torch.randperm(
            fit_indices.numel(), generator=fit_generator
        )[:args.max_fit_tasks].sort().values
        fit_indices = fit_indices.index_select(0, chosen)
    fit_dataset = train_dataset.index_select(fit_indices)
    if set(fit_dataset.task_fingerprints) & set(
            validation_dataset.task_fingerprints):
        raise ValueError('fit and validation geometries overlap')

    external_datasets: list[CachedSeedCandidateDataset] = []
    external_validity: list[dict[str, float | list[int]]] = []
    occupied_fingerprints = set(dataset.task_fingerprints)
    for external_path in external_paths:
        external = CachedSeedCandidateDataset.from_npz(external_path)
        external, external_stats = validate_cached_dataset(
            external, validation_probe.kin, validation_probe.collision,
            cone_deg=validation_probe.cfg.cone_deg)
        overlap = occupied_fingerprints & set(external.task_fingerprints)
        if overlap:
            raise ValueError(
                f'external development cache overlaps {len(overlap)} '
                'training/validation geometries')
        occupied_fingerprints.update(external.task_fingerprints)
        external_datasets.append(external)
        external_validity.append(external_stats)

    controller = load_controller_agent(controller_dir, validation_probe, device)
    if 'controller' not in source or 'controller_state_sha256' not in source:
        raise ValueError('source checkpoint lacks embedded controller state')
    source_controller_hash = state_dict_fingerprint(source['controller'])
    if source_controller_hash != source['controller_state_sha256']:
        raise ValueError('source controller hash metadata is inconsistent')
    if state_dict_fingerprint(controller.state_dict()) != source_controller_hash:
        raise ValueError('controller checkpoint differs from source controller')
    teacher = copy.deepcopy(controller).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    baseline_state = _cpu_tree(controller.state_dict())
    atomic_torch_save(baseline_state, snapshot_dir / 'agent_epoch_000.pt')

    fit_selected, _ = _static_seed_indices(
        policy, source, fit_dataset, validation_probe.kin,
        chunk_size=args.eval_chunk_size)
    validation_selected, validation_first = _static_seed_indices(
        policy, source, validation_dataset, validation_probe.kin,
        chunk_size=args.eval_chunk_size)
    external_seed_choices = [
        _static_seed_indices(
            policy, source, external, validation_probe.kin,
            chunk_size=args.eval_chunk_size)
        for external in external_datasets
    ]
    np.savez_compressed(
        search_dir / 'static_seed_choices.npz',
        fit_source_task_index=fit_dataset.task_indices.numpy(),
        fit_candidate_index=fit_selected.numpy(),
        validation_source_task_index=validation_dataset.task_indices.numpy(),
        validation_candidate_index=validation_selected.numpy(),
        validation_first_valid_candidate_index=validation_first.numpy())
    for index, ((selected, first), external_path) in enumerate(zip(
            external_seed_choices, external_paths)):
        np.savez_compressed(
            search_dir / f'external_static_seed_choices_{index:02d}.npz',
            candidate_cache_sha256=np.asarray(
                file_fingerprint(external_path)['sha256']),
            candidate_index=selected.numpy(),
            first_valid_candidate_index=first.numpy())

    tail_state, tail_obs, tail_metadata, ordinary_obs = _collect_tail_states(
        teacher, controller_dir, fit_dataset, fit_selected, device,
        chunk_size=args.collection_chunk_size, seed=args.seed + 1)
    tail_metadata['train_local_index'] = fit_indices[
        torch.from_numpy(tail_metadata['fit_local_index'])].numpy()
    _save_state_npz(
        search_dir / 'tail_states.npz', tail_state,
        obs=tail_obs.numpy(), **tail_metadata)
    search = _search_tail_states(
        teacher, controller_dir, tail_state, device,
        chunk_size=args.search_chunk_size,
        sigma=args.perturbation_sigma, seed=args.seed + 2)
    label = _conservative_search_targets(
        search,
        minimum_gain_m=args.minimum_search_gain_m,
        blend_gain_scale_m=args.blend_gain_scale_m,
        maximum_blend=args.maximum_blend,
        local_only=not args.allow_classical_search_labels,
        minimum_supporting_actions=args.minimum_supporting_actions,
        maximum_target_action_delta=args.maximum_target_action_delta)
    search.update(label)
    accepted = label['accepted_before_verification'].clone()
    verified_progress = torch.full_like(label['label_best_progress_m'], torch.nan)
    verified_gain = torch.full_like(label['label_gain_m'], torch.nan)
    if not args.skip_target_verification and bool(accepted.any().item()):
        accepted_index = torch.nonzero(
            accepted, as_tuple=False).squeeze(-1)
        verified_progress_subset = _rollout_state_actions(
            teacher, controller_dir,
            tail_state.index_select(accepted_index),
            label['target_action'].index_select(0, accepted_index),
            device, chunk_size=args.search_chunk_size)
        verified_gain_subset = (
            verified_progress_subset
            - search['baseline_progress_m'].index_select(0, accepted_index))
        verified_progress[accepted_index] = verified_progress_subset
        verified_gain[accepted_index] = verified_gain_subset
        accepted[accepted_index] &= (
            verified_gain_subset > args.minimum_search_gain_m)
    elif args.skip_target_verification:
        verified_progress = label['label_best_progress_m'].clone()
        verified_gain = label['label_gain_m'].clone()
    search['verified_target_progress_m'] = verified_progress
    search['verified_target_gain_m'] = verified_gain
    search['accepted'] = accepted
    np.savez_compressed(
        search_dir / 'search_results.npz',
        **{key: value.numpy() for key, value in search.items()},
        **tail_metadata)
    n_accepted = int(accepted.sum().item())
    n_preverified = int(
        label['accepted_before_verification'].sum().item())
    print(
        '[search-distill] search gate: '
        f'{n_accepted}/{n_preverified}/{len(accepted)} labels pass '
        'verification/pre-gate/all; '
        f'mean local best gain='
        f'{float(label["label_gain_m"].mean()) * 1000.0:.3f} mm',
        flush=True)

    search_obs = search['obs'][accepted]
    search_target = label['target_action'][accepted]
    retention_count = args.retention_multiplier * n_accepted
    cpu_generator = torch.Generator().manual_seed(args.seed + 3)
    if retention_count > 0:
        if retention_count <= ordinary_obs.shape[0]:
            retention_index = torch.randperm(
                ordinary_obs.shape[0],
                generator=cpu_generator)[:retention_count]
        else:
            # Extremely short one-step trajectories cannot provide three
            # distinct observations per accepted search state. Sampling their
            # ordinary observations with replacement preserves the exact 3:1
            # retention weighting without inventing off-trajectory states.
            retention_index = torch.randint(
                ordinary_obs.shape[0], (retention_count,),
                generator=cpu_generator)
        retention_obs = ordinary_obs.index_select(0, retention_index)
        retention_target_parts = []
        for start in range(0, retention_count, args.batch_size):
            end = min(start + args.batch_size, retention_count)
            retention_target_parts.append(
                teacher.actor_mean(
                    retention_obs[start:end].to(device)
                ).clamp(-1.0, 1.0).float().cpu())
        retention_target = torch.cat(retention_target_parts)
    else:
        retention_obs = ordinary_obs[:0]
        retention_target = torch.empty(
            (0, validation_probe.act_dim), dtype=torch.float32)
    if retention_count != args.retention_multiplier * n_accepted:
        raise RuntimeError(
            'ordinary trajectories did not provide the required 3x '
            'actor-retention states')
    np.savez_compressed(
        search_dir / 'retention_states.npz',
        obs=retention_obs.numpy(), target_action=retention_target.numpy(),
        multiplier=np.asarray(args.retention_multiplier))
    if n_accepted > 0:
        train_obs = torch.cat([search_obs, retention_obs], dim=0)
        train_target = torch.cat([search_target, retention_target], dim=0)
        train_kind = torch.cat([
            torch.ones(n_accepted, dtype=torch.int8),
            torch.zeros(retention_count, dtype=torch.int8),
        ])
    else:
        train_obs = torch.empty((0, validation_probe.obs_dim))
        train_target = torch.empty((0, validation_probe.act_dim))
        train_kind = torch.empty((0,), dtype=torch.int8)
    np.savez_compressed(
        search_dir / 'distill_dataset.npz',
        obs=train_obs.numpy(), target_action=train_target.numpy(),
        is_search_label=train_kind.numpy())

    baseline_outputs = _evaluate_fixed_seeds(
        controller, validation_probe, validation_dataset,
        validation_selected, validation_first,
        chunk_size=args.eval_chunk_size, gamma=args.gamma)
    fingerprints = validation_dataset.task_fingerprints
    baseline_summary = _evaluation_summary(
        0, baseline_outputs, baseline_outputs, fingerprints,
        bootstrap_seed=args.seed + 100,
        bootstrap_samples=args.bootstrap_samples,
        first_valid_tolerance_m=args.first_valid_tolerance_m,
        first_coverage_tolerance=args.first_coverage_tolerance,
        harm_threshold_m=args.harm_threshold_m,
        max_paired_harm_rate=args.max_paired_harm_rate,
        selected_harm_tolerance=args.selected_harm_tolerance,
        robust_trim_fraction=args.robust_trim_fraction,
        robust_clip_m=args.robust_clip_m,
        robust_cvar_fraction=args.robust_cvar_fraction,
        robust_trim_tolerance_m=args.robust_trim_tolerance_m,
        robust_cvar_tolerance_m=args.robust_cvar_tolerance_m,
        require_positive_ci=not args.development_promotion)
    _save_evaluation(
        eval_dir / 'eval_epoch_000.npz', baseline_outputs,
        baseline_summary, validation_selected, validation_first,
        fingerprints)
    external_baselines: list[dict[str, np.ndarray]] = []
    external_evaluations: list[list[dict[str, Any]]] = []
    for index, (external, (selected, first)) in enumerate(zip(
            external_datasets, external_seed_choices)):
        outputs = _evaluate_fixed_seeds(
            controller, validation_probe, external, selected, first,
            chunk_size=args.eval_chunk_size, gamma=args.gamma)
        summary = _evaluation_summary(
            0, outputs, outputs, external.task_fingerprints,
            bootstrap_seed=args.seed + 1000 + index,
            bootstrap_samples=args.bootstrap_samples,
            first_valid_tolerance_m=args.first_valid_tolerance_m,
            first_coverage_tolerance=args.first_coverage_tolerance,
            harm_threshold_m=args.harm_threshold_m,
            max_paired_harm_rate=args.max_paired_harm_rate,
            selected_harm_tolerance=args.selected_harm_tolerance,
            robust_trim_fraction=args.robust_trim_fraction,
            robust_clip_m=args.robust_clip_m,
            robust_cvar_fraction=args.robust_cvar_fraction,
            robust_trim_tolerance_m=args.robust_trim_tolerance_m,
            robust_cvar_tolerance_m=args.robust_cvar_tolerance_m,
            require_positive_ci=False)
        external_baselines.append(outputs)
        external_evaluations.append([summary])
        _save_evaluation(
            eval_dir / f'external_{index:02d}_eval_epoch_000.npz',
            outputs, summary, selected, first,
            external.task_fingerprints)
    evaluations = [baseline_summary]
    train_log: list[dict[str, Any]] = []

    actor_parameters = list(controller._mean_head.parameters())
    if args.train_scope == 'actor':
        actor_parameters = [
            *controller._actor_trunk.parameters(),
            *actor_parameters,
        ]
    optimizer = torch.optim.AdamW(
        actor_parameters, lr=args.actor_lr,
        weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        if n_accepted == 0:
            print(
                '[search-distill] no accepted labels; skipping supervised '
                'epochs and retaining baseline controller', flush=True)
            break
        controller.train()
        order = torch.randperm(train_obs.shape[0], generator=cpu_generator)
        epoch_loss = 0.0
        epoch_search_loss = 0.0
        epoch_retention_loss = 0.0
        n_seen = 0
        n_search_seen = 0
        n_retention_seen = 0
        for start in range(0, train_obs.shape[0], args.batch_size):
            end = min(start + args.batch_size, train_obs.shape[0])
            index = order[start:end]
            obs_batch = train_obs[index].to(device)
            target_batch = train_target[index].to(device)
            kind = train_kind[index].to(device=device, dtype=torch.bool)
            prediction = controller.actor_mean(obs_batch)
            per_row = F.mse_loss(
                prediction, target_batch, reduction='none').mean(dim=-1)
            loss = per_row.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            count = end - start
            epoch_loss += float(loss.item()) * count
            n_seen += count
            if bool(kind.any().item()):
                count_search = int(kind.sum().item())
                epoch_search_loss += float(per_row[kind].sum().item())
                n_search_seen += count_search
            if bool((~kind).any().item()):
                count_retention = int((~kind).sum().item())
                epoch_retention_loss += float(per_row[~kind].sum().item())
                n_retention_seen += count_retention

        outputs = _evaluate_fixed_seeds(
            controller, validation_probe, validation_dataset,
            validation_selected, validation_first,
            chunk_size=args.eval_chunk_size, gamma=args.gamma)
        summary = _evaluation_summary(
            epoch, outputs, baseline_outputs, fingerprints,
            bootstrap_seed=args.seed + 100 + epoch,
            bootstrap_samples=args.bootstrap_samples,
            first_valid_tolerance_m=args.first_valid_tolerance_m,
            first_coverage_tolerance=args.first_coverage_tolerance,
            harm_threshold_m=args.harm_threshold_m,
            max_paired_harm_rate=args.max_paired_harm_rate,
            selected_harm_tolerance=args.selected_harm_tolerance,
            robust_trim_fraction=args.robust_trim_fraction,
            robust_clip_m=args.robust_clip_m,
            robust_cvar_fraction=args.robust_cvar_fraction,
            robust_trim_tolerance_m=args.robust_trim_tolerance_m,
            robust_cvar_tolerance_m=args.robust_cvar_tolerance_m,
            require_positive_ci=not args.development_promotion)
        summary.update({
            'train_loss': epoch_loss / max(n_seen, 1),
            'train_search_loss': (
                epoch_search_loss / max(n_search_seen, 1)),
            'train_retention_loss': (
                epoch_retention_loss / max(n_retention_seen, 1)),
        })
        epoch_external: list[dict[str, Any]] = []
        for index, (external, (selected, first), baseline) in enumerate(zip(
                external_datasets, external_seed_choices,
                external_baselines)):
            external_outputs = _evaluate_fixed_seeds(
                controller, validation_probe, external, selected, first,
                chunk_size=args.eval_chunk_size, gamma=args.gamma)
            external_summary = _evaluation_summary(
                epoch, external_outputs, baseline,
                external.task_fingerprints,
                bootstrap_seed=args.seed + 1000 + 100 * epoch + index,
                bootstrap_samples=args.bootstrap_samples,
                first_valid_tolerance_m=args.first_valid_tolerance_m,
                first_coverage_tolerance=args.first_coverage_tolerance,
                harm_threshold_m=args.harm_threshold_m,
                max_paired_harm_rate=args.max_paired_harm_rate,
                selected_harm_tolerance=args.selected_harm_tolerance,
                robust_trim_fraction=args.robust_trim_fraction,
                robust_clip_m=args.robust_clip_m,
                robust_cvar_fraction=args.robust_cvar_fraction,
                robust_trim_tolerance_m=args.robust_trim_tolerance_m,
                robust_cvar_tolerance_m=args.robust_cvar_tolerance_m,
                require_positive_ci=False)
            epoch_external.append(external_summary)
            external_evaluations[index].append(external_summary)
            _save_evaluation(
                eval_dir / (
                    f'external_{index:02d}_eval_epoch_{epoch:03d}.npz'),
                external_outputs, external_summary, selected, first,
                external.task_fingerprints)
        if args.development_promotion:
            summary['promotion_eligible'] = bool(
                summary['robust_development_eligible']
                and all(item['robust_development_eligible']
                        for item in epoch_external))
        development_gains = [
            summary['gain_vs_baseline_geometry_macro_m'],
            *[
                item['gain_vs_baseline_geometry_macro_m']
                for item in epoch_external
            ],
        ]
        summary['external_development'] = copy.deepcopy(epoch_external)
        summary['development_minimax_gain_m'] = float(min(
            development_gains))
        evaluations.append(summary)
        train_log.append(copy.deepcopy(summary))
        state = _cpu_tree(controller.state_dict())
        atomic_torch_save(
            state, snapshot_dir / f'agent_epoch_{epoch:03d}.pt')
        _save_evaluation(
            eval_dir / f'eval_epoch_{epoch:03d}.npz', outputs,
            summary, validation_selected, validation_first, fingerprints)
        print(
            f'[search-distill] epoch {epoch}/{args.epochs}: '
            f'policy={summary["policy_progress_mean_m"]:.6f} m, '
            f'delta={summary["gain_vs_baseline_geometry_macro_m"]:+.6f} m '
            f'CI=[{summary["gain_vs_baseline_ci95_low_m"]:+.6f}, '
            f'{summary["gain_vs_baseline_ci95_high_m"]:+.6f}], '
            f'eligible={summary["promotion_eligible"]}', flush=True)

    eligible = [
        item for item in evaluations if item['promotion_eligible']]
    if eligible:
        selection_key = (
            (lambda item: item['development_minimax_gain_m'])
            if args.development_promotion
            else (lambda item: item['policy_progress_mean_m']))
        best_summary = max(eligible, key=selection_key)
        best_epoch = int(best_summary['epoch'])
    else:
        best_summary = baseline_summary
        best_epoch = 0
    best_state = torch.load(
        snapshot_dir / f'agent_epoch_{best_epoch:03d}.pt',
        map_location='cpu', weights_only=True)
    controller.load_state_dict(best_state)
    promoted = best_epoch > 0

    settings = {
        'format': 'static-seed-tail-search-distill-v2',
        'source_checkpoint': str(source_path),
        'fit_tasks_only': True,
        'fit_tasks_used': int(len(fit_dataset)),
        'max_fit_tasks': args.max_fit_tasks,
        'tail_offset_steps': [1, 8],
        'search_candidates': SEARCH_CANDIDATES,
        'search_slot_zero': 'frozen-current-controller-action',
        'candidate_protocol': (
            'one-candidate-step-then-frozen-controller-to-termination'),
        'perturbation_sigma': args.perturbation_sigma,
        'minimum_search_gain_m': args.minimum_search_gain_m,
        'blend': (
            f'{args.maximum_blend}*clip(gain_m/'
            f'{args.blend_gain_scale_m},0,1)'),
        'local_search_labels_only': not args.allow_classical_search_labels,
        'minimum_supporting_actions': args.minimum_supporting_actions,
        'maximum_target_action_delta': args.maximum_target_action_delta,
        'exact_target_verification': not args.skip_target_verification,
        'preverification_search_labels': n_preverified,
        'retention_multiplier': args.retention_multiplier,
        'epochs': args.epochs,
        'train_scope': args.train_scope,
        'actor_lr': args.actor_lr,
        'weight_decay': args.weight_decay,
        'accepted_search_labels': n_accepted,
        'best_epoch': best_epoch,
        'promoted': promoted,
        'promotion_gate': {
            'mode': (
                'robust-development-v1' if args.development_promotion
                else 'robust-positive-ci-v1'),
            'geometry_ci95_low_strictly_positive': (
                not args.development_promotion),
            'first_valid_tolerance_m': args.first_valid_tolerance_m,
            'first_coverage_tolerance': args.first_coverage_tolerance,
            'harm_threshold_m': args.harm_threshold_m,
            'max_paired_harm_rate': args.max_paired_harm_rate,
            'selected_harm_tolerance': args.selected_harm_tolerance,
            'trim_fraction': args.robust_trim_fraction,
            'clip_m': args.robust_clip_m,
            'cvar_fraction': args.robust_cvar_fraction,
            'trim_tolerance_m': args.robust_trim_tolerance_m,
            'cvar_tolerance_m': args.robust_cvar_tolerance_m,
        },
        'inference': 'one-static-seed-one-controller-rollout-v1',
        'inference_search_steps': 0,
        'inference_model_rollouts': 0,
    }
    source_config = load_run_config(controller_dir)
    output_config = copy.deepcopy(source_config)
    output_config.setdefault('unified', {})[
        'joint_controller_search_distill'] = copy.deepcopy(settings)
    with open(out_dir / 'config.yaml', 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
    output_config_hash = file_fingerprint(
        out_dir / 'config.yaml')['sha256']
    controller_state = _cpu_tree(controller.state_dict())
    controller_hash = state_dict_fingerprint(controller_state)

    result = copy.deepcopy(source)
    result['outer_round'] = int(source.get('outer_round', 0)) + 1
    result['phase'] = 'round_complete'
    result['controller'] = controller_state
    result['controller_state_sha256'] = controller_hash
    result['controller_run_config_sha256'] = output_config_hash
    result['search_distill_optimizer'] = _cpu_tree(optimizer.state_dict())
    # The source PPO Adam moments no longer correspond to the distilled actor.
    # Publish a structurally compatible, fresh all-parameter optimizer so a
    # later bidirectional controller phase cannot silently reuse stale moments.
    resume_lr = float(source.get('args', {}).get('controller_lr', 3e-4))
    resume_optimizer = torch.optim.Adam(
        controller.parameters(), lr=resume_lr, eps=1e-5)
    result['controller_optimizer'] = _cpu_tree(resume_optimizer.state_dict())
    result['joint_controller_search_distill'] = {
        'format': 'static-seed-tail-search-distill-v2',
        'source_checkpoint': file_fingerprint(source_path),
        'candidate_cache': file_fingerprint(candidate_path),
        'device': device_identity(device),
        'physical_validation': validity,
        'settings': copy.deepcopy(settings),
        'evaluations': copy.deepcopy(evaluations),
        'baseline_controller_state_sha256': source_controller_hash,
        'promoted_controller_state_sha256': controller_hash,
        'validation_used_for_controller_promotion': True,
        'external_holdout_used': False,
        'external_development_used_for_promotion': bool(external_paths),
        'external_development_candidate_caches': [
            file_fingerprint(path) for path in external_paths
        ],
        'external_development_physical_validation': external_validity,
        'external_development_evaluations': copy.deepcopy(
            external_evaluations),
        'deployment_uses_search': False,
    }
    result_provenance = copy.deepcopy(result['provenance'])
    result_provenance['joint_controller_search_distill'] = copy.deepcopy(
        result['joint_controller_search_distill'])
    result['provenance'] = result_provenance
    atomic_torch_save(controller_state, out_dir / 'agent.pt')
    atomic_torch_save(result, out_dir / 'unified.pt')
    np.savez_compressed(
        out_dir / 'training_log.npz',
        records=np.asarray(train_log, dtype=object),
        evaluations=np.asarray(evaluations, dtype=object))
    print(
        f'[search-distill] selected epoch {best_epoch}/{args.epochs}; '
        f'promoted={promoted}; '
        f'policy={best_summary["policy_progress_mean_m"]:.6f} m; '
        f'controller_sha256={controller_hash}', flush=True)


if __name__ == '__main__':
    main()
