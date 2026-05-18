"""PPO adapted from cleanrl/ppo_continuous_action.py for our torch-batched env.

Differences from cleanrl reference:
  - Env returns torch tensors directly (no gym.vector wrapper). Storage lives
    on the same device as the env.
  - Bootstrap on truncation uses `info["terminal_obs"]` rather than the
    auto-reset next obs (PPO-correct truncation handling, per rules.md).
  - Single-file, no CLI: callable from train.py via `train(cfg, env, ...)`.

Source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 32
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    anneal_ent_coef: bool = False  # linearly decay ent_coef from initial to 0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_dim: int = 256
    init_log_std: float = -0.5
    normalize_returns: bool = True


class _RunningMeanStd:
    """Welford's online scalar mean/var, torch."""
    def __init__(self, device, epsilon: float = 1e-4):
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean = self.mean + delta * batch_count / tot
        self.var = M2 / tot
        self.count = tot


class RewardScaler:
    """Divide rewards by running std of discounted returns.

    Mirrors sb3 VecNormalize / OpenAI baselines NormalizeReward. The V function
    then learns in scaled (z-score-magnitude) space, sidestepping the MLP-with-
    standard-init-doesn't-like-targets-of-magnitude-1000 problem.
    """
    def __init__(self, n_envs: int, gamma: float, device, epsilon: float = 1e-4):
        self.rms = _RunningMeanStd(device, epsilon)
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_acc = torch.zeros(n_envs, device=device)

    @torch.no_grad()
    def step(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        self.return_acc = self.return_acc * self.gamma + rewards
        self.rms.update(self.return_acc)
        # Reset return accumulator on done (so next-step return starts fresh)
        self.return_acc = self.return_acc * (1.0 - dones.to(self.return_acc.dtype))
        return rewards / torch.sqrt(self.rms.var + self.epsilon)

    @property
    def scale(self) -> float:
        return float(torch.sqrt(self.rms.var + self.epsilon).item())


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-critic with tanh-squashed Gaussian policy.

    Action representation:
      - actor outputs (μ(s), log σ(s)) ∈ ℝ⁴, state-dependent log σ
      - pre-squash z ~ Normal(μ, σ) is the "action" stored in PPO buffer
      - env receives tanh(z) ∈ (-1, 1)⁴ (bounded, no clipping bias)
      - log_prob includes Jacobian correction: log π(a) = log N(z) - Σ log(1 - tanh²(z))

    Without the squash, PPO would push μ unbounded (because clip-then-step
    gives biased gradient): we observed μ growing to 3.5+ on runs7/10, which
    produced perpetually-saturated actions and explained why deterministic
    eval matched random-Gaussian baseline.
    """
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 0.0  # σ ≤ 1.0 (was 7.4); state-dep log_std grew unboundedly otherwise

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 init_log_std: float = -0.5):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self._actor_trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self._mean_head = _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self._logstd_head = _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        with torch.no_grad():
            self._logstd_head.bias.fill_(init_log_std)

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action: tanh(μ(s)) ∈ (-1, 1). Used by eval / viz."""
        return torch.tanh(self._mean_head(self._actor_trunk(x)))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def _actor_dist(self, x: torch.Tensor) -> Normal:
        h = self._actor_trunk(x)
        mean = self._mean_head(h)
        log_std = self._logstd_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def get_action_and_value(self, x: torch.Tensor,
                             action: torch.Tensor | None = None):
        """Action representation is PRE-SQUASH z. Caller is responsible for
        `torch.tanh(action)` before sending to env.

        Returns (z, log_prob, entropy_of_underlying_Normal, value).
        """
        dist = self._actor_dist(x)
        if action is None:
            action = dist.sample()  # z ~ Normal(μ, σ)
        z = action
        log_prob_normal = dist.log_prob(z).sum(-1)
        squashed = torch.tanh(z)
        # Tanh Jacobian correction: log|∂a/∂z| = Σ log(1 - tanh²(z))
        log_prob = log_prob_normal - torch.log(
            (1.0 - squashed.pow(2)).clamp(min=1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)  # underlying Normal entropy (proxy)
        return action, log_prob, entropy, self.critic(x).squeeze(-1)


def train(cfg: PPOConfig, env, device: torch.device,
          eval_fn=None, eval_every: int = 10_000,
          log_fn=None, ckpt_path: str | None = None,
          ckpt_every_n_updates: int = 10):
    """Train PPO on `env`.

    `env` must expose: `n_envs`, `obs_dim`, `act_dim`, `device`, `reset()`,
    `step(actions)` returning `(obs, rew, term, trunc, info)` where `info`
    contains `"terminal_obs"` (obs BEFORE auto-reset) and `"episode_done"`.

    `eval_fn(agent)` is called every `eval_every` env steps and may return a
    dict logged via `log_fn`. `log_fn(dict)` is called after each update.
    """
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    n_envs = env.n_envs
    batch_size = n_envs * cfg.n_steps
    minibatch_size = batch_size // cfg.n_minibatches
    n_updates = cfg.total_timesteps // batch_size

    agent = Agent(obs_dim, act_dim, hidden_dim=cfg.hidden_dim,
                  init_log_std=cfg.init_log_std).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Storage
    obs_buf = torch.zeros((cfg.n_steps, n_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.n_steps, n_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    rewards_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    terminated_buf = torch.zeros((cfg.n_steps, n_envs), device=device)  # for bootstrap mask
    truncated_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    values_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    terminal_obs_buf = torch.zeros((cfg.n_steps, n_envs, obs_dim), device=device)

    reward_scaler = (RewardScaler(n_envs, cfg.gamma, device)
                     if cfg.normalize_returns else None)

    next_obs = env.reset()
    next_done = torch.zeros(n_envs, device=device)
    global_step = 0
    next_eval = eval_every
    # Per-term reward accumulators (averaged per update)
    _reward_term_keys = ("r_progress_mean", "r_jl_mean", "r_cone_mean", "r_dm_mean",
                         "w_u_mean")
    # Episode-finish aggregates (weighted by n_episodes_done across the rollout)
    _episode_keys = ("ep_reward_mean", "ep_len_mean", "r_terminal_mean",
                     "ep_progress_mean")

    for update in range(1, n_updates + 1):
        frac = 1.0 - (update - 1.0) / n_updates  # 1 → 0 over training
        if cfg.anneal_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = frac * cfg.learning_rate
        ent_coef_now = cfg.ent_coef * frac if cfg.anneal_ent_coef else cfg.ent_coef

        rollout_term_accum = {k: 0.0 for k in _reward_term_keys}
        rollout_term_n = 0
        ep_accum = {k: 0.0 for k in _episode_keys}
        ep_total_finished = 0
        for step in range(cfg.n_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            # Squash the unsquashed z before sending to env (env expects (-1, 1)^4)
            squashed_action = torch.tanh(action)
            next_obs, reward, term, trunc, info = env.step(squashed_action)
            done_now = (term | trunc).to(device)
            r_dev = reward.to(device)
            if reward_scaler is not None:
                r_dev = reward_scaler.step(r_dev, done_now)
            rewards_buf[step] = r_dev
            terminated_buf[step] = term.float()
            truncated_buf[step] = trunc.float()
            terminal_obs_buf[step] = info["terminal_obs"]
            next_done = done_now.float()
            for k in _reward_term_keys:
                if k in info:
                    rollout_term_accum[k] += info[k]
            rollout_term_n += 1
            n_done = int(info.get("n_episodes_done", 0))
            if n_done > 0:
                for k in _episode_keys:
                    v = info.get(k, float("nan"))
                    if v == v:  # not NaN
                        ep_accum[k] += float(v) * n_done
                ep_total_finished += n_done

        # Bootstrap: V(next_obs) for ongoing envs; V(terminal_obs) for truncated;
        # 0 for terminated. We compute V(next_obs) for the boundary and
        # within-rollout truncations get bootstrap during GAE backward pass.
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(n_envs, device=device)
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    nextnonterminal_term = 1.0 - terminated_buf[t]
                    nextnonterminal_trunc = 1.0 - truncated_buf[t]
                    # For the boundary step: bootstrap from next_obs if not terminated
                    bootstrap_val = next_value * nextnonterminal_term
                else:
                    nextnonterminal_term = 1.0 - terminated_buf[t]
                    nextnonterminal_trunc = 1.0 - truncated_buf[t]
                    bootstrap_val = values_buf[t + 1] * nextnonterminal_term
                # If truncated (and not terminated), bootstrap from terminal_obs
                truncated_only = truncated_buf[t] * (1.0 - terminated_buf[t])
                if truncated_only.any():
                    term_val = agent.get_value(terminal_obs_buf[t])
                    bootstrap_val = torch.where(
                        truncated_only.bool(), term_val, bootstrap_val,
                    )
                # GAE: episode boundary resets lastgaelam (both term and trunc)
                episode_continues = (1.0 - terminated_buf[t]) * (1.0 - truncated_buf[t])
                delta = rewards_buf[t] + cfg.gamma * bootstrap_val - values_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * episode_continues * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl_value = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    approx_kl_value = approx_kl
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * ent_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl_value > cfg.target_kl:
                break

        if log_fn is not None:
            log_dict = {
                "update": update,
                "global_step": global_step,
                "train/pg_loss": float(pg_loss.item()),
                "train/v_loss": float(v_loss.item()),
                "train/entropy": float(ent_loss.item()),
                "train/approx_kl": float(approx_kl_value),
                "train/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/ent_coef": ent_coef_now,
                "train/reward_scale": (reward_scaler.scale
                                       if reward_scaler is not None else 1.0),
            }
            for k in _reward_term_keys:
                short = k.replace("_mean", "").replace("r_", "")  # progress, jl, cone, dm, w_u
                log_dict[f"reward/{short}"] = (rollout_term_accum[k] / rollout_term_n
                                                if rollout_term_n > 0 else 0.0)
            # Episode-finish stats (only emit if any episodes finished in this rollout)
            if ep_total_finished > 0:
                log_dict["episode/reward_mean"] = ep_accum["ep_reward_mean"] / ep_total_finished
                log_dict["episode/length_mean"] = ep_accum["ep_len_mean"] / ep_total_finished
                log_dict["episode/progress_mean_m"] = ep_accum["ep_progress_mean"] / ep_total_finished
                log_dict["episode/terminal_bonus_mean"] = ep_accum["r_terminal_mean"] / ep_total_finished
                log_dict["episode/n_finished"] = ep_total_finished
            log_fn(log_dict)

        if eval_fn is not None and global_step >= next_eval:
            eval_stats = eval_fn(agent)
            if log_fn is not None and eval_stats is not None:
                log_fn({"eval_at_step": global_step, **eval_stats})
            next_eval += eval_every

        if ckpt_path is not None and update % ckpt_every_n_updates == 0:
            torch.save(agent.state_dict(), ckpt_path)

    if ckpt_path is not None:
        torch.save(agent.state_dict(), ckpt_path)
    return agent
