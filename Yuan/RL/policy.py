"""Tanh-squashed Gaussian policy (state -> q_seed) and a value baseline.

Action mapping
--------------
Network outputs (mu, log_std) for a pre-squash variable u in R^ndof.
    u    ~ N(mu, sigma)
    a    = q_mid + tanh(u) * q_half               (-> q in [lmt_lo, lmt_up])

log_prob compensates for the tanh + linear scaling:
    log p(a) = log N(u; mu, sigma)
               - sum log(1 - tanh(u)^2)
               - sum log(q_half)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import Yuan.RL.config as cfg


def _mlp(in_dim: int, out_dim: int, hidden: int = cfg.HIDDEN_DIM) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


def _log_tanh_linear_jac(u: torch.Tensor, q_half: torch.Tensor) -> torch.Tensor:
    log_jac_tanh = (2.0 * (torch.log(torch.tensor(2.0, device=u.device))
                           - u - F.softplus(-2.0 * u))).sum(-1)
    log_jac_lin = torch.log(q_half).sum()
    return log_jac_tanh + log_jac_lin


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim: int, action_dim: int,
                 q_mid: torch.Tensor, q_half: torch.Tensor,
                 state_dep_log_std: bool | None = None):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, cfg.HIDDEN_DIM), nn.Tanh(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM), nn.Tanh(),
        )
        self.mu_head = nn.Linear(cfg.HIDDEN_DIM, action_dim)
        if state_dep_log_std is None:
            state_dep_log_std = bool(cfg.STATE_DEP_LOG_STD)
        self.state_dep_log_std = state_dep_log_std
        if state_dep_log_std:
            # state-dep std lets sigma adapt to task difficulty
            self.log_std_head = nn.Linear(cfg.HIDDEN_DIM, action_dim)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, float(cfg.LOG_STD_INIT))
            # log_std attribute kept for backward-compatible save/load probes
            self.register_buffer("log_std",
                torch.full((action_dim,), float(cfg.LOG_STD_INIT)))
        else:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), float(cfg.LOG_STD_INIT)))
        # action scaling buffers (move with .to(device))
        self.register_buffer("q_mid", q_mid)
        self.register_buffer("q_half", q_half)
        # zero mu head so initial mean(u)=0 -> action = q_mid (~ FR3 home)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

    def _dist(self, s: torch.Tensor) -> torch.distributions.Normal:
        h = self.trunk(s)
        mu = self.mu_head(h)
        if self.state_dep_log_std:
            log_std = self.log_std_head(h).clamp(
                cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        return torch.distributions.Normal(mu, log_std.exp())

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, pre_tanh u)."""
        dist = self._dist(s)
        u = dist.mean if deterministic else dist.rsample()
        a = self.q_mid + torch.tanh(u) * self.q_half
        return a, u

    def log_prob(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Log-density of the squashed action whose pre-tanh latent is u."""
        dist = self._dist(s)
        log_p_u = dist.log_prob(u).sum(-1)
        return log_p_u - _log_tanh_linear_jac(u, self.q_half)

    def entropy(self, s: torch.Tensor) -> torch.Tensor:
        return self._dist(s).entropy().sum(-1)


class MixtureGaussianPolicy(nn.Module):
    """Tanh-squashed mixture of diagonal Gaussians.

    The PPO ratio is evaluated under the full mixture density:
        log pi(u|s) = logsumexp_k [log w_k(s) + log N_k(u; mu_k, sigma_k)]
    """

    def __init__(self, state_dim: int, action_dim: int,
                 q_mid: torch.Tensor, q_half: torch.Tensor,
                 n_components: int | None = None,
                 state_dep_log_std: bool | None = None):
        super().__init__()
        if n_components is None:
            n_components = int(cfg.MIXTURE_COMPONENTS)
        if state_dep_log_std is None:
            state_dep_log_std = bool(cfg.STATE_DEP_LOG_STD)
        self.action_dim = action_dim
        self.n_components = int(n_components)
        self.state_dep_log_std = state_dep_log_std
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, cfg.HIDDEN_DIM), nn.Tanh(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM), nn.Tanh(),
        )
        self.logit_head = nn.Linear(cfg.HIDDEN_DIM, self.n_components)
        self.mu_head = nn.Linear(cfg.HIDDEN_DIM,
                                 self.n_components * action_dim)
        if state_dep_log_std:
            self.log_std_head = nn.Linear(
                cfg.HIDDEN_DIM, self.n_components * action_dim)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, float(cfg.LOG_STD_INIT))
            self.register_buffer("log_std",
                torch.full((self.n_components, action_dim),
                           float(cfg.LOG_STD_INIT)))
        else:
            self.log_std = nn.Parameter(torch.full(
                (self.n_components, action_dim), float(cfg.LOG_STD_INIT)))
        self.register_buffer("q_mid", q_mid)
        self.register_buffer("q_half", q_half)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

    def _params(self, s: torch.Tensor):
        h = self.trunk(s)
        logits = self.logit_head(h)
        mu = self.mu_head(h).view(
            -1, self.n_components, self.action_dim)
        if self.state_dep_log_std:
            log_std = self.log_std_head(h).view(
                -1, self.n_components, self.action_dim)
            log_std = log_std.clamp(cfg.LOG_STD_MIN, cfg.LOG_STD_MAX)
        else:
            log_std = self.log_std.clamp(
                cfg.LOG_STD_MIN, cfg.LOG_STD_MAX).unsqueeze(0)
            log_std = log_std.expand_as(mu)
        return logits, mu, log_std

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, pre_tanh u)."""
        logits, mu, log_std = self._params(s)
        if deterministic:
            comp = logits.argmax(dim=-1)
            row = torch.arange(s.shape[0], device=s.device)
            u = mu[row, comp]
        else:
            cat = torch.distributions.Categorical(logits=logits)
            comp = cat.sample()
            row = torch.arange(s.shape[0], device=s.device)
            dist = torch.distributions.Normal(
                mu[row, comp], log_std[row, comp].exp())
            u = dist.sample()
        a = self.q_mid + torch.tanh(u) * self.q_half
        return a, u

    @torch.no_grad()
    def component_actions(self, s: torch.Tensor) -> torch.Tensor:
        """Return all component mean actions, shaped (B, K, action_dim)."""
        _, mu, _ = self._params(s)
        return self.q_mid + torch.tanh(mu) * self.q_half

    def log_prob(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        logits, mu, log_std = self._params(s)
        u_k = u.unsqueeze(1)
        log_p = (-0.5 * (((u_k - mu) / log_std.exp()) ** 2
                         + 2.0 * log_std
                         + torch.log(torch.tensor(2.0 * torch.pi,
                                                  device=u.device)))).sum(-1)
        log_w = torch.log_softmax(logits, dim=-1)
        log_p_u = torch.logsumexp(log_w + log_p, dim=-1)
        return log_p_u - _log_tanh_linear_jac(u, self.q_half)

    def entropy(self, s: torch.Tensor) -> torch.Tensor:
        logits, _, log_std = self._params(s)
        comp_entropy = (0.5 * (1.0 + torch.log(torch.tensor(
            2.0 * torch.pi, device=s.device))) + log_std).sum(-1)
        w = torch.softmax(logits, dim=-1)
        cat_entropy = torch.distributions.Categorical(logits=logits).entropy()
        return (w * comp_entropy).sum(-1) + cat_entropy


def make_policy(state_dim: int, action_dim: int,
                q_mid: torch.Tensor, q_half: torch.Tensor,
                policy_type: str | None = None) -> nn.Module:
    if policy_type is None:
        policy_type = cfg.POLICY_TYPE
    if policy_type == "gaussian":
        return GaussianPolicy(state_dim, action_dim, q_mid, q_half)
    if policy_type == "mixture":
        return MixtureGaussianPolicy(state_dim, action_dim, q_mid, q_half)
    raise ValueError(f"Unknown policy type: {policy_type}")


class ValueNet(nn.Module):

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)
