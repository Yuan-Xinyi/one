"""Safe loaders for historical controller checkpoints."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import yaml

from Yuan.IJRR.stage2_traj.ppo import Agent, PPOConfig
from Yuan.IJRR.env.env import EnvConfig, NSRLBatchedEnv


_CONTROLLER_INPUT_WEIGHT_KEYS = (
    'critic.0.weight',
    '_actor_trunk.0.weight',
)


def resolve_controller_dir(path: str | Path) -> Path:
    """Resolve either a controller run directory or its ``agent.pt`` file."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_file():
        if resolved.name != 'agent.pt':
            raise ValueError(
                f'controller checkpoint file must be named agent.pt: {resolved}')
        resolved = resolved.parent
    if not resolved.is_dir():
        raise ValueError(f'controller checkpoint is not a directory: {resolved}')
    return resolved


def load_run_config(ckpt_dir: str | Path) -> dict:
    with open(resolve_controller_dir(ckpt_dir) / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


def env_config_from_run(cfg_yaml: dict, n_envs: int,
                        **overrides) -> EnvConfig:
    valid = {field.name for field in dataclasses.fields(EnvConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f'unknown environment overrides: {sorted(unknown)}')
    values = {key: value for key, value in cfg_yaml['env'].items()
              if key in valid}
    values.update(overrides)
    return EnvConfig(**{**values, 'n_envs': int(n_envs)})


def ppo_config_from_run(cfg_yaml: dict, **overrides) -> PPOConfig:
    valid = {field.name for field in dataclasses.fields(PPOConfig)}
    values = {key: value for key, value in cfg_yaml['ppo'].items()
              if key in valid}
    values.update(overrides)
    return PPOConfig(**values)


def build_env_from_run(ckpt_dir: str | Path, n_envs: int,
                       device: torch.device,
                       env_overrides: dict | None = None) -> NSRLBatchedEnv:
    cfg_yaml = load_run_config(ckpt_dir)
    return NSRLBatchedEnv(
        env_config_from_run(cfg_yaml, n_envs, **(env_overrides or {})),
        line_dist=None, device=device)


def adapt_controller_observation_state_dict(
        state: dict[str, torch.Tensor], target_obs_dim: int
        ) -> dict[str, torch.Tensor]:
    """Zero-expand a historical controller's observation input weights.

    Only the actor and critic input layers depend on observation width. Zero
    columns preserve both outputs exactly while allowing newly appended state
    to be learned during subsequent PPO updates.
    """
    missing = [key for key in _CONTROLLER_INPUT_WEIGHT_KEYS if key not in state]
    if missing:
        raise ValueError(f'controller checkpoint is missing input weights: {missing}')
    source_dims = {int(state[key].shape[1]) for key in _CONTROLLER_INPUT_WEIGHT_KEYS}
    if len(source_dims) != 1:
        raise ValueError(
            f'controller actor/critic input dimensions disagree: {sorted(source_dims)}')
    source_obs_dim = source_dims.pop()
    if source_obs_dim == target_obs_dim:
        return state
    if (source_obs_dim, target_obs_dim) != (31, 34):
        raise ValueError(
            f'unsupported controller observation migration '
            f'{source_obs_dim}-D -> {target_obs_dim}-D; only 31-D -> 34-D '
            'zero expansion is defined')

    adapted = dict(state)
    for key in _CONTROLLER_INPUT_WEIGHT_KEYS:
        old_weight = state[key]
        weight = old_weight.new_zeros((old_weight.shape[0], target_obs_dim))
        weight[:, :source_obs_dim] = old_weight
        adapted[key] = weight
    return adapted


def load_controller_state_dict(agent: Agent,
                               state: dict[str, torch.Tensor]) -> None:
    """Load controller weights, adapting historical observation width."""
    target_obs_dim = int(agent.critic[0].in_features)
    agent.load_state_dict(
        adapt_controller_observation_state_dict(state, target_obs_dim))


def load_controller_agent(ckpt_dir: str | Path, env: NSRLBatchedEnv,
                          device: torch.device) -> Agent:
    ckpt_dir = resolve_controller_dir(ckpt_dir)
    cfg_yaml = load_run_config(ckpt_dir)
    ppo_cfg = ppo_config_from_run(cfg_yaml)
    agent = Agent(
        env.obs_dim, env.act_dim,
        hidden_dim=ppo_cfg.hidden_dim,
        init_log_std=ppo_cfg.init_log_std,
        squashed_entropy=ppo_cfg.squashed_entropy,
    ).to(device)
    state = torch.load(ckpt_dir / 'agent.pt', map_location=device, weights_only=True)
    load_controller_state_dict(agent, state)
    return agent
