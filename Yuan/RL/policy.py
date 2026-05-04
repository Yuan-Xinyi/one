"""Tanh-squashed Gaussian policy and a value baseline.

Action mapping
--------------
Network outputs (mu, log_std) for a pre-squash variable u in R^action_dim.
    u    ~ N(mu, sigma)
    a    = q_mid + tanh(u) * q_half

log_prob compensates for the tanh + linear scaling:
    log p(a) = log N(u; mu, sigma)
               - sum log(1 - tanh(u)^2)
               - sum log(q_half)
"""
from __future__ import annotations
import numpy as np
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

    def rsample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SAC reparameterised sample. Returns (action, u, log_prob_u)."""
        dist = self._dist(s)
        u = dist.rsample()
        a = self.q_mid + torch.tanh(u) * self.q_half
        log_p = dist.log_prob(u).sum(-1) - _log_tanh_linear_jac(u, self.q_half)
        return a, u, log_p

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

    def rsample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SAC reparameterised sample.

        - Component index sampled with stop-gradient (Categorical is not
          reparameterisable; we treat it as discrete latent).
        - Within the chosen component, u = mu + sigma * eps (reparam, grad
          flows to mu_k and log_std_k of the SELECTED component).
        - log_prob uses the full mixture density => gradient flows back to
          the categorical logits and all components, which is what trains the
          mixture weights to match Q(c, q).
        Returns (action, u, log_prob_u).
        """
        logits, mu, log_std = self._params(s)
        with torch.no_grad():
            comp = torch.distributions.Categorical(logits=logits).sample()
        row = torch.arange(s.shape[0], device=s.device)
        mu_sel = mu[row, comp]
        log_std_sel = log_std[row, comp]
        eps = torch.randn_like(mu_sel)
        u = mu_sel + log_std_sel.exp() * eps                      # reparam
        a = self.q_mid + torch.tanh(u) * self.q_half
        log_p = self._mixture_log_prob(logits, mu, log_std, u)
        return a, u, log_p

    @torch.no_grad()
    def component_actions(self, s: torch.Tensor) -> torch.Tensor:
        """Return all component mean actions, shaped (B, K, action_dim)."""
        _, mu, _ = self._params(s)
        return self.q_mid + torch.tanh(mu) * self.q_half

    def _mixture_log_prob(self, logits: torch.Tensor, mu: torch.Tensor,
                          log_std: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u_k = u.unsqueeze(1)
        log_p = (-0.5 * (((u_k - mu) / log_std.exp()) ** 2
                         + 2.0 * log_std
                         + torch.log(torch.tensor(2.0 * torch.pi,
                                                  device=u.device)))).sum(-1)
        log_w = torch.log_softmax(logits, dim=-1)
        log_p_u = torch.logsumexp(log_w + log_p, dim=-1)
        return log_p_u - _log_tanh_linear_jac(u, self.q_half)

    def log_prob(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        logits, mu, log_std = self._params(s)
        return self._mixture_log_prob(logits, mu, log_std, u)

    def entropy(self, s: torch.Tensor) -> torch.Tensor:
        logits, _, log_std = self._params(s)
        comp_entropy = (0.5 * (1.0 + torch.log(torch.tensor(
            2.0 * torch.pi, device=s.device))) + log_std).sum(-1)
        w = torch.softmax(logits, dim=-1)
        cat_entropy = torch.distributions.Categorical(logits=logits).entropy()
        return (w * comp_entropy).sum(-1) + cat_entropy


class CouplingLayer(nn.Module):
    """One affine coupling layer of a conditional RealNVP flow.

    Splits the action vector into two halves; transforms one half conditioned
    on (state, frozen half). Forward & inverse have analytic Jacobian
    determinants (sum of log-scales over transformed dims).
    """

    def __init__(self, action_dim: int, state_dim: int, hidden: int,
                 transform_first: bool):
        super().__init__()
        assert action_dim % 2 == 0, "FlowPolicy expects even action_dim"
        self.action_dim = action_dim
        self.d_half = action_dim // 2
        self.transform_first = transform_first
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.d_half, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * self.d_half),
        )
        # Initialize last layer near zero so initial transform ~= identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, z):
        if self.transform_first:
            return z[:, :self.d_half], z[:, self.d_half:]   # (xa to transform, xb frozen)
        return z[:, self.d_half:], z[:, :self.d_half]

    def _join(self, xa, xb):
        if self.transform_first:
            return torch.cat([xa, xb], dim=-1)
        return torch.cat([xb, xa], dim=-1)

    def forward(self, z, s):
        """z -> u; returns (u, log|det(du/dz)|)."""
        xa, xb = self._split(z)
        h = self.net(torch.cat([s, xb], dim=-1))
        shift, log_scale = h[:, :self.d_half], h[:, self.d_half:]
        log_scale = torch.tanh(log_scale)                     # bound for stability
        xa_new = xa * torch.exp(log_scale) + shift
        u = self._join(xa_new, xb)
        log_det = log_scale.sum(dim=-1)
        return u, log_det

    def inverse(self, u, s):
        """u -> z; returns (z, log|det(dz/du)|) = -log_det_forward."""
        xa, xb = self._split(u)
        h = self.net(torch.cat([s, xb], dim=-1))
        shift, log_scale = h[:, :self.d_half], h[:, self.d_half:]
        log_scale = torch.tanh(log_scale)
        xa_new = (xa - shift) * torch.exp(-log_scale)
        z = self._join(xa_new, xb)
        log_det_inv = -log_scale.sum(dim=-1)
        return z, log_det_inv


class FlowPolicy(nn.Module):
    """Conditional RealNVP normalising flow with tanh action squashing.

    Base distribution: standard Normal in R^action_dim.
    Forward (sampling): z ~ N(0, I) -> coupling layers (state-conditioned) -> u
    Action: a = q_mid + tanh(u) * q_half
    Density:
        log p(a|s) = log p_z(z) - log|det(du/dz)| - log_jac_tanh - log_jac_lin

    Multi-modality is naturally supported (no fixed K, no mode-collapse).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 q_mid: torch.Tensor, q_half: torch.Tensor,
                 n_layers: int | None = None,
                 hidden: int = cfg.HIDDEN_DIM):
        super().__init__()
        if n_layers is None:
            n_layers = int(getattr(cfg, "FLOW_LAYERS", 4))
        self.action_dim = action_dim
        self.layers = nn.ModuleList([
            CouplingLayer(action_dim, state_dim, hidden,
                          transform_first=(i % 2 == 0))
            for i in range(n_layers)
        ])
        self.register_buffer("q_mid", q_mid)
        self.register_buffer("q_half", q_half)

    def _flow_forward(self, z, s):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer.forward(z, s)
            log_det = log_det + ld
        return z, log_det

    def _flow_inverse(self, u, s):
        log_det_inv = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for layer in reversed(self.layers):
            u, ld = layer.inverse(u, s)
            log_det_inv = log_det_inv + ld
        return u, log_det_inv

    def _log_normal(self, z):
        """Standard normal log density, summed over dims."""
        return (-0.5 * (z ** 2).sum(dim=-1)
                - 0.5 * self.action_dim
                * float(np.log(2.0 * np.pi)))

    def rsample(self, s):
        """SAC reparameterised sample: returns (action, u, log_prob)."""
        z = torch.randn(s.shape[0], self.action_dim,
                        device=s.device, dtype=s.dtype)
        u, log_det_fwd = self._flow_forward(z, s)
        a = self.q_mid + torch.tanh(u) * self.q_half
        log_pz = self._log_normal(z)
        log_pu = log_pz - log_det_fwd
        log_pa = log_pu - _log_tanh_linear_jac(u, self.q_half)
        return a, u, log_pa

    @torch.no_grad()
    def act(self, s, deterministic: bool = False):
        if deterministic:
            z = torch.zeros(s.shape[0], self.action_dim,
                            device=s.device, dtype=s.dtype)
        else:
            z = torch.randn(s.shape[0], self.action_dim,
                            device=s.device, dtype=s.dtype)
        u, _ = self._flow_forward(z, s)
        a = self.q_mid + torch.tanh(u) * self.q_half
        return a, u

    def log_prob(self, s, u):
        z, log_det_inv = self._flow_inverse(u, s)
        log_pz = self._log_normal(z)
        log_pu = log_pz + log_det_inv
        return log_pu - _log_tanh_linear_jac(u, self.q_half)

    def log_prob_action(self, s, a):
        """Density at the *squashed* action a. Inverts a = mid + tanh(u)*half
        to recover u, then calls log_prob(s, u). Used at deploy for
        mode-seeking selection: argmax_k log_prob_action(s, a_k)."""
        x = ((a - self.q_mid) / self.q_half).clamp(-1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(x)
        return self.log_prob(s, u)

    def entropy(self, s):
        # No closed-form; one-sample MC estimate (used only for logging).
        _, _, log_pa = self.rsample(s)
        return -log_pa


def make_policy(state_dim: int, action_dim: int,
                q_mid: torch.Tensor, q_half: torch.Tensor,
                policy_type: str | None = None) -> nn.Module:
    if policy_type is None:
        policy_type = cfg.POLICY_TYPE
    if policy_type == "gaussian":
        return GaussianPolicy(state_dim, action_dim, q_mid, q_half)
    if policy_type == "mixture":
        return MixtureGaussianPolicy(state_dim, action_dim, q_mid, q_half)
    if policy_type == "flow":
        return FlowPolicy(state_dim, action_dim, q_mid, q_half)
    raise ValueError(f"Unknown policy type: {policy_type}")


class ValueNet(nn.Module):

    def __init__(self, state_dim: int):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)
