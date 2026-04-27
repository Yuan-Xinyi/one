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
        # tanh correction:  - sum log(1 - tanh(u)^2)  (numerically stable form)
        log_jac_tanh = (2.0 * (torch.log(torch.tensor(2.0, device=u.device))
                               - u - F.softplus(-2.0 * u))).sum(-1)
        # constant scaling correction:  - sum log(q_half)
        log_jac_lin = torch.log(self.q_half).sum()
        return log_p_u - log_jac_tanh - log_jac_lin

    def entropy(self, s: torch.Tensor) -> torch.Tensor:
        return self._dist(s).entropy().sum(-1)


class ValueNet(nn.Module):

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)
