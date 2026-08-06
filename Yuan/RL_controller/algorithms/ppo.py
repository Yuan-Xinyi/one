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
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_dim: int = 256
    init_log_std: float = -0.5
    normalize_returns: bool = True
    # Freeze the actor for the first N updates (critic + reward-scaler warmup).
    # Essential when resuming from a distilled ckpt whose critic is random —
    # garbage advantages would erode the distilled actor before the critic
    # calibrates.
    actor_warmup_updates: int = 0
    # Use the TRUE entropy of the tanh-squashed distribution (single-sample
    # rsample estimate) instead of the underlying Normal's entropy. The Normal
    # proxy is degenerate: inflating sigma earns unbounded log-sigma bonus
    # while tanh saturates the behavior to bang-bang — every 30M run ended
    # with log_std pinned at the +0.5 clamp (sigma=1.65), which destroys the
    # fine-grained exploration the danger zone requires.
    squashed_entropy: bool = False


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

    def state_dict(self) -> dict:
        return {
            'mean': self.mean.detach().clone(),
            'var': self.var.detach().clone(),
            'count': self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        mean = torch.as_tensor(state['mean']).to(self.mean)
        var = torch.as_tensor(state['var']).to(self.var)
        count = float(state['count'])
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError('running mean/variance shape mismatch')
        if (not bool(torch.isfinite(mean).all().item())
                or not bool(torch.isfinite(var).all().item())
                or bool((var < 0.0).any().item())):
            raise ValueError('running mean/variance must be finite with variance >= 0')
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError('running-stat count must be finite and positive')
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count = count


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

    def state_dict(self) -> dict:
        return {
            'rms': self.rms.state_dict(),
            'return_acc': self.return_acc.detach().clone(),
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state: dict) -> None:
        gamma = float(state.get('gamma', self.gamma))
        epsilon = float(state.get('epsilon', self.epsilon))
        if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError(f'invalid RewardScaler gamma: {gamma}')
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f'invalid RewardScaler epsilon: {epsilon}')
        return_acc = state['return_acc']
        if tuple(return_acc.shape) != tuple(self.return_acc.shape):
            raise ValueError(
                f'RewardScaler return_acc shape {tuple(return_acc.shape)} '
                f'does not match {tuple(self.return_acc.shape)}')
        self.rms.load_state_dict(state['rms'])
        self.return_acc.copy_(return_acc.to(self.return_acc))
        self.gamma = gamma
        self.epsilon = epsilon


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
    gives biased gradient), producing perpetually-saturated actions whose
    deterministic eval matches the random-Gaussian baseline.
    """
    # State-independent log_std (issue 3 fix): the state-dep Linear head saturated
    # at upper clamp during 50M run; PPO continuous-control standard is a single
    # learnable Parameter per action dim, much easier for the entropy bonus
    # to compress without state-routing the gradient through the trunk.
    LOG_STD_MIN = -2.5  # σ ≥ exp(-2.5) ≈ 0.082 (v4: was -2.0)
    LOG_STD_MAX =  0.5  # σ ≤ exp(0.5) ≈ 1.65 (safety cap; nn.Parameter rarely saturates)

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 init_log_std: float = -0.5, squashed_entropy: bool = False):
        super().__init__()
        self.squashed_entropy = squashed_entropy
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
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action: tanh(μ(s)) ∈ (-1, 1). Used by eval / viz."""
        return torch.tanh(self._mean_head(self._actor_trunk(x)))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def _actor_dist(self, x: torch.Tensor) -> Normal:
        h = self._actor_trunk(x)
        mean = self._mean_head(h)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
        return Normal(mean, log_std.exp())

    def get_action_and_value(self, x: torch.Tensor,
                             action: torch.Tensor | None = None):
        """Action representation is PRE-SQUASH z. Caller is responsible for
        `torch.tanh(action)` before sending to env.

        Returns (z, log_prob, entropy_of_underlying_Normal, value, log_std).
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
        if self.squashed_entropy:
            # Single-sample estimate of H[tanh(z)], z~N(mu,sigma):
            # H = -E[log pi(a)] with the tanh Jacobian correction. rsample
            # keeps it differentiable; inflating sigma past saturation now
            # yields no free bonus (correction cancels the log-sigma growth).
            z_e = dist.rsample()
            log_p_e = dist.log_prob(z_e).sum(-1) - torch.log(
                (1.0 - torch.tanh(z_e).pow(2)).clamp(min=1e-6)).sum(-1)
            entropy = -log_p_e
        else:
            entropy = dist.entropy().sum(-1)  # underlying Normal entropy (proxy)
        log_std = dist.scale.log()
        return action, log_prob, entropy, self.critic(x).squeeze(-1), log_std


def train(cfg: PPOConfig, env, device: torch.device,
          eval_fn=None, eval_every: int = 10_000,
          log_fn=None, ckpt_path: str | None = None,
          ckpt_every_n_updates: int = 10,
          resume_from_ckpt: str | None = None,
          bc_obs: torch.Tensor | None = None,
          bc_actions: torch.Tensor | None = None,
          bc_coef: float = 0.0,
          bc_anneal: bool = True,
          guide_action_fn=None,
          guide_tau_enter: float = 0.98,
          guide_tau_exit: float = 0.94,
          guide_anneal_start: float | None = None,
          guide_anneal_end: float = 0.9,
          critic_floor_fn=None,
          critic_floor_coef: float = 0.5,
          anchor_agent: Agent | None = None,
          actor_anchor_coef: float = 0.0,
          actor_anchor_anneal: bool = False,
          agent: Agent | None = None,
          optimizer: torch.optim.Optimizer | None = None,
          reward_scaler: RewardScaler | None = None):
    """Train PPO on `env`.

    `env` must expose: `n_envs`, `obs_dim`, `act_dim`, `device`, `reset()`,
    `step(actions)` returning `(obs, rew, term, trunc, info)` where `info`
    contains `"terminal_obs"` (obs BEFORE auto-reset) and `"episode_done"`.

    `eval_fn(agent)` is called every `eval_every` env steps and may return a
    dict logged via `log_fn`. `log_fn(dict)` is called after each update.

    BC auxiliary loss (self-improvement distillation, DAPG-style):
    when `bc_obs`/`bc_actions` (fixed dataset of verified rescue steps, see
    self_improve/collect.py) are given with `bc_coef > 0`, each minibatch adds
        bc_coef_now * || tanh(mu(s)) - a_cls(s) ||^2
    on a random BC minibatch. `bc_anneal=True` decays the coefficient linearly
    to 0 over the run so PPO exploration owns the endgame of each round.
    Targets must already be clamped strictly inside (-1, 1).

    Guided switch (RL-native self-improvement, switch-in-the-loop):
    when `guide_action_fn(env) -> (B, act_dim) in [-1,1]` is given, training
    rollouts run under the variant-B hysteresis switch on max|q_norm| (read
    straight from obs[:, :7]): envs past `guide_tau_enter` execute the guide
    (classical) action instead of the policy sample, until back under
    `guide_tau_exit`. Guide steps enter the SAME PPO buffer as the policy's
    own experience (stored action = atanh(a_guide), logprob under pi_old), so
    the clipped objective reinforces rescues exactly where their advantage is
    positive — advantage-gated internalization, no auxiliary imitation loss.
    As the policy learns to avoid/handle the danger zone, guide engagement
    (`train/guide_frac`) decays to 0 on its own — state-driven annealing.

    Curriculum (anti-moral-hazard): a free safety net makes leaning on the
    rescue optimal (progress reward keeps flowing during classical control),
    so `guide_frac` grows instead of shrinking. With `guide_anneal_start` set,
    each episode gets a safety net only with probability p: p=1 until
    `guide_anneal_start` (fraction of training), then linear to 0 at
    `guide_anneal_end` — the policy increasingly faces real termination, and
    V(danger) is corrected by genuine deaths while rescue demonstrations are
    still available in the protected episodes.

    Critic floor (classical value as a lower bound, see self_improve/vcls.py):
    `critic_floor_fn(obs) -> G_cls` in RAW reward units adds
        critic_floor_coef * relu(G_cls/reward_scale - V(s))^2
    to each minibatch. V*(s) >= V_cls(s) always holds (classical is a valid
    policy), so the critic may exceed the floor but never sit below it —
    danger-zone states that classical can survive keep positive advantage for
    classical-like actions even in unprotected curriculum episodes.

    Phase-start actor anchor (stable alternating optimization): when a frozen
    ``anchor_agent`` and ``actor_anchor_coef > 0`` are supplied, every PPO
    minibatch also minimizes the deterministic-action MSE

        ||tanh(mu(s)) - tanh(mu_anchor(s))||^2.

    The states are the current on-policy rollout states, so the constraint
    follows the seed-induced reset distribution without a separate replay
    buffer.  Only the deterministic actor is anchored; the critic and
    exploration variance remain free to adapt.  The anchor is training-only
    and adds no deployment inference cost.
    """
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    n_envs = env.n_envs
    batch_size = n_envs * cfg.n_steps
    minibatch_size = batch_size // cfg.n_minibatches
    n_updates = cfg.total_timesteps // batch_size

    if agent is None:
        agent = Agent(obs_dim, act_dim, hidden_dim=cfg.hidden_dim,
                      init_log_std=cfg.init_log_std,
                      squashed_entropy=cfg.squashed_entropy).to(device)
    else:
        agent = agent.to(device)
    if resume_from_ckpt is not None:
        if optimizer is not None:
            raise ValueError(
                'resume_from_ckpt cannot be combined with an injected optimizer')
        print(f"[ppo] resuming policy weights from {resume_from_ckpt}")
        agent.load_state_dict(torch.load(resume_from_ckpt, map_location=device))
    if optimizer is None:
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    else:
        optimizer_parameters = {
            id(parameter) for group in optimizer.param_groups
            for parameter in group['params']}
        agent_parameters = {id(parameter) for parameter in agent.parameters()}
        if optimizer_parameters != agent_parameters:
            raise ValueError('injected optimizer does not own exactly agent parameters')

    if not np.isfinite(actor_anchor_coef) or actor_anchor_coef < 0.0:
        raise ValueError('actor_anchor_coef must be finite and non-negative')
    use_actor_anchor = anchor_agent is not None and actor_anchor_coef > 0.0
    if use_actor_anchor:
        anchor_agent = anchor_agent.to(device).eval()
        anchor_parameters = tuple(anchor_agent.parameters())
        if any(parameter.requires_grad for parameter in anchor_parameters):
            for parameter in anchor_parameters:
                parameter.requires_grad_(False)
        if (anchor_agent._actor_trunk[0].in_features != obs_dim
                or anchor_agent._mean_head.out_features != act_dim):
            raise ValueError('anchor_agent observation/action dimensions mismatch')

    # Storage
    obs_buf = torch.zeros((cfg.n_steps, n_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.n_steps, n_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    rewards_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    terminated_buf = torch.zeros((cfg.n_steps, n_envs), device=device)  # for bootstrap mask
    truncated_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    values_buf = torch.zeros((cfg.n_steps, n_envs), device=device)
    terminal_obs_buf = torch.zeros((cfg.n_steps, n_envs, obs_dim), device=device)

    if cfg.normalize_returns and reward_scaler is None:
        reward_scaler = RewardScaler(n_envs, cfg.gamma, device)
    if not cfg.normalize_returns:
        reward_scaler = None
    if (reward_scaler is not None
            and reward_scaler.return_acc.shape != (n_envs,)):
        raise ValueError(
            f'reward scaler has {reward_scaler.return_acc.shape[0]} envs, '
            f'but training env has {n_envs}')
    if reward_scaler is not None and not np.isclose(
            reward_scaler.gamma, cfg.gamma):
        raise ValueError(
            f'reward scaler gamma {reward_scaler.gamma} does not match '
            f'PPO gamma {cfg.gamma}')

    use_bc = (bc_obs is not None and bc_actions is not None
              and bc_coef > 0.0 and bc_obs.shape[0] > 0)
    if use_bc:
        bc_obs = bc_obs.to(device=device, dtype=torch.float32)
        bc_actions = bc_actions.to(device=device, dtype=torch.float32)

    next_obs = env.reset()
    # An injected scaler may carry long-running RMS statistics across outer
    # training phases, but reset() starts fresh episodes for every env. Do not
    # leak partial discounted-return accumulators across that hard boundary.
    if reward_scaler is not None:
        reward_scaler.return_acc.zero_()
    next_done = torch.zeros(n_envs, device=device)
    global_step = 0
    next_eval = eval_every
    # Guided-switch hysteresis state (obs[:, :7] is q_norm, so the switch
    # criterion max|q_norm| needs no env internals).
    if guide_action_fn is not None:
        guide_using_rl = (next_obs[:, :7].abs().max(dim=-1).values
                          < guide_tau_enter)
        # Per-episode safety-net mask (curriculum); resampled on episode reset.
        guide_net_on = torch.ones(n_envs, dtype=torch.bool, device=device)
    # Per-term reward accumulators (averaged per update)
    _reward_term_keys = ("r_progress_mean",)
    # Episode-finish aggregates (weighted by n_episodes_done across the rollout)
    _episode_keys = ("ep_reward_mean", "ep_len_mean", "ep_progress_mean")
    # MGS fallback rate per anchor column (rollout-averaged)
    _fb_keys = ("fb_rate_e0", "fb_rate_e1", "fb_rate_e2")

    for update in range(1, n_updates + 1):
        progress = (update - 1.0) / n_updates  # 0 → 1 over training
        if cfg.anneal_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = (1.0 - progress) * cfg.learning_rate

        rollout_term_accum = {k: 0.0 for k in _reward_term_keys}
        rollout_term_n = 0
        ep_accum = {k: 0.0 for k in _episode_keys}
        ep_total_finished = 0
        rollout_fb_accum = {k: 0.0 for k in _fb_keys}
        rollout_sigma_sum = 0.0          # Σ over steps of (mean σ across env×dim)
        rollout_sigma_clamp_sum = 0.0    # Σ over steps of fraction at log_std min
        rollout_guide_frac_sum = 0.0     # Σ over steps of fraction of guided envs
        for step in range(cfg.n_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value, log_std = agent.get_action_and_value(next_obs)
                rollout_sigma_sum += float(log_std.exp().mean().item())
                rollout_sigma_clamp_sum += float(
                    (log_std <= Agent.LOG_STD_MIN + 1e-6).float().mean().item())

            if guide_action_fn is not None:
                # Curriculum: p(safety net) anneals 1 -> 0 between
                # guide_anneal_start and guide_anneal_end (fractions of run).
                if guide_anneal_start is not None:
                    span = max(guide_anneal_end - guide_anneal_start, 1e-8)
                    guide_p = min(max(1.0 - (progress - guide_anneal_start)
                                      / span, 0.0), 1.0)
                else:
                    guide_p = 1.0
                # Hysteresis update on current obs; envs whose episode was
                # auto-reset last step re-init from tau_enter and redraw
                # their safety net with the current p.
                reinit = next_done.bool()
                if bool(reinit.any().item()):
                    redraw = torch.rand(n_envs, device=device) < guide_p
                    guide_net_on = torch.where(reinit, redraw, guide_net_on)
                qn = next_obs[:, :7].abs().max(dim=-1).values
                stay = torch.where(guide_using_rl,
                                   qn < guide_tau_enter, qn < guide_tau_exit)
                guide_using_rl = torch.where(reinit,
                                             qn < guide_tau_enter, stay)
                guided = guide_net_on & ~guide_using_rl
                rollout_guide_frac_sum += float(guided.float().mean().item())
                if bool(guided.any().item()):
                    with torch.no_grad():
                        # nan_to_num: classical q_dot can NaN near singular
                        # configs (linalg.inv); zero action = pure task-space
                        # feed-forward, a safe fallback. Clamp at 0.995
                        # (z=atanh->3.0) keeps guide actions out of the extreme
                        # Gaussian tail where logprob is hypersensitive.
                        a_guide = torch.nan_to_num(
                            guide_action_fn(env), nan=0.0).clamp(-0.995, 0.995)
                        z_guide = torch.atanh(a_guide).to(action.dtype)
                        action = torch.where(guided.unsqueeze(-1), z_guide, action)
                        # Re-evaluate logprob of the executed action under
                        # pi_old (value is obs-only, unchanged).
                        _, logprob, _, value, _ = agent.get_action_and_value(
                            next_obs, action=action)

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
            for k in _fb_keys:
                if k in info:
                    rollout_fb_accum[k] += info[k]
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
                # bootstrap_val is masked only by `terminated` (zero if terminated);
                # truncation is handled below by overwriting with V(terminal_obs),
                # so we don't need a separate nextnonterminal_trunc factor here.
                if t == cfg.n_steps - 1:
                    nextnonterminal_term = 1.0 - terminated_buf[t]
                    bootstrap_val = next_value * nextnonterminal_term
                else:
                    nextnonterminal_term = 1.0 - terminated_buf[t]
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

        bc_coef_now = (bc_coef * (1.0 - progress) if bc_anneal else bc_coef) \
            if use_bc else 0.0
        anchor_coef_now = (
            actor_anchor_coef * (1.0 - progress)
            if use_actor_anchor and actor_anchor_anneal
            else actor_anchor_coef if use_actor_anchor else 0.0)
        bc_loss_value = 0.0
        anchor_loss_value = 0.0
        floor_pen_value = 0.0
        floor_viol_frac = 0.0

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl_value = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                # Truncated IS ratio (V-trace style): guide-action transitions
                # sit in the far Gaussian tail where small (mu, sigma) shifts
                # move logprob by tens of nats — raw exp() overflows and one
                # bad minibatch NaNs the net. Clamping bounds pg_loss and
                # zeroes the gradient of runaway samples; ordinary on-policy
                # samples (|logratio| << 1) are unaffected.
                ratio = logratio.clamp(-20.0, 2.0).exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = ((mb_adv - mb_adv.mean())
                              / (mb_adv.std(unbiased=False) + 1e-8))

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
                if update <= cfg.actor_warmup_updates:
                    # Critic-only warmup: no policy/entropy gradient.
                    loss = cfg.vf_coef * v_loss
                else:
                    loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                if critic_floor_fn is not None:
                    with torch.no_grad():
                        floor_raw = critic_floor_fn(b_obs[mb_inds])
                        scale = (reward_scaler.scale
                                 if reward_scaler is not None else 1.0)
                        floor_v = floor_raw / scale
                    floor_gap = torch.relu(floor_v - newvalue)
                    floor_pen = floor_gap.pow(2).mean()
                    floor_pen_value = float(floor_pen.item())
                    floor_viol_frac = float((floor_gap > 0).float().mean().item())
                    loss = loss + critic_floor_coef * floor_pen

                if use_bc and bc_coef_now > 0.0:
                    bc_idx = torch.randint(0, bc_obs.shape[0], (minibatch_size,),
                                           device=device)
                    bc_pred = torch.tanh(agent._mean_head(
                        agent._actor_trunk(bc_obs[bc_idx])))
                    bc_loss = ((bc_pred - bc_actions[bc_idx]) ** 2).mean()
                    bc_loss_value = float(bc_loss.item())
                    loss = loss + bc_coef_now * bc_loss

                if use_actor_anchor and anchor_coef_now > 0.0:
                    current_action = agent.actor_mean(b_obs[mb_inds])
                    with torch.no_grad():
                        anchor_action = anchor_agent.actor_mean(b_obs[mb_inds])
                    anchor_loss = (
                        (current_action - anchor_action).square().mean())
                    anchor_loss_value = float(anchor_loss.item())
                    loss = loss + anchor_coef_now * anchor_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(),
                                                     cfg.max_grad_norm)
                # Last-line defense: a non-finite grad norm means clipping
                # already poisoned the grads (inf * 0 = nan) — skip the step.
                if bool(torch.isfinite(grad_norm).item()):
                    optimizer.step()

            # Measure the final policy produced by this epoch against the
            # rollout policy. Minibatch-local pre-step KL can be identically
            # zero with one minibatch and systematically underreports the
            # completed epoch update.
            with torch.no_grad():
                _, epoch_logprob, _, _, _ = agent.get_action_and_value(
                    b_obs, b_actions)
                epoch_logratio = (
                    epoch_logprob - b_logprobs).clamp(-20.0, 20.0)
                approx_kl_value = float((
                    torch.expm1(epoch_logratio) - epoch_logratio
                ).mean().item())
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
                "train/ent_coef": cfg.ent_coef,
                "train/reward_scale": (reward_scaler.scale
                                       if reward_scaler is not None else 1.0),
                "train/sigma_mean": rollout_sigma_sum / max(rollout_term_n, 1),
                "train/sigma_clamp_frac":
                    rollout_sigma_clamp_sum / max(rollout_term_n, 1),
            }
            if use_bc:
                log_dict["train/bc_loss"] = bc_loss_value
                log_dict["train/bc_coef"] = bc_coef_now
            if use_actor_anchor:
                log_dict["train/actor_anchor_loss"] = anchor_loss_value
                log_dict["train/actor_anchor_coef"] = anchor_coef_now
            if guide_action_fn is not None:
                log_dict["train/guide_frac"] = (
                    rollout_guide_frac_sum / max(rollout_term_n, 1))
                log_dict["train/guide_p"] = guide_p
            if critic_floor_fn is not None:
                log_dict["train/floor_pen"] = floor_pen_value
                log_dict["train/floor_viol_frac"] = floor_viol_frac
            for k in _reward_term_keys:
                short = k.replace("_mean", "").replace("r_", "")  # progress
                log_dict[f"reward/{short}"] = (rollout_term_accum[k] / rollout_term_n
                                                if rollout_term_n > 0 else 0.0)
            for k in _fb_keys:
                log_dict[f"train/{k}"] = (rollout_fb_accum[k] / rollout_term_n
                                          if rollout_term_n > 0 else 0.0)
            # Episode-finish stats (only emit if any episodes finished in this rollout)
            if ep_total_finished > 0:
                log_dict["episode/reward_mean"] = ep_accum["ep_reward_mean"] / ep_total_finished
                log_dict["episode/length_mean"] = ep_accum["ep_len_mean"] / ep_total_finished
                log_dict["episode/progress_mean_m"] = ep_accum["ep_progress_mean"] / ep_total_finished
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
