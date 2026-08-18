"""Policy whose action space is the vertex set of the admissible command box.

Maximising the arc length a stroke survives is a maximum-time problem. The
joint velocity is affine in the null-space command and the command is
box-bounded, so the Hamiltonian is linear in it and the optimum lies at a
vertex of the box except on singular arcs. Three measurements on the deployed
continuous policy agree with that:

    classical law      median |a| = 0.14, on the boundary 0.1% of steps
    learned policy     median |a| = 1.00, on the boundary 89% of steps
    hard-thresholding the learned command to the nearest vertex costs nothing
        (+0.004 in the ratio to the classical law, CI spanning zero)

So the intermediate magnitudes the continuous parameterisation can express
carry no information here, and the policy can act directly on the 2^4 vertices.
Doing so removes the tanh saturation the continuous version relies on, and with
it the pathology that the entropy bonus drove log_std to its ceiling in every
previous run.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from Yuan.IJRR.stage2_traj.ppo import _layer_init


class VertexAgent(nn.Module):
    """Actor-critic over the 2**act_dim vertices of [-1, 1]^act_dim.

    Interface-compatible with ``ppo.train``: ``get_action_and_value`` returns
    the same tuple, with the action being a category index carried as a float
    so that it fits the existing buffer, and ``log_std`` returned as None since
    a categorical policy has no scale parameter. ``to_env`` expands the index
    to the vertex the environment executes.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 **_ignored):
        super().__init__()
        self.act_dim = act_dim
        self.action_store_dim = 1   # a category index
        self.n_actions = 2 ** act_dim
        grid = np.stack(np.meshgrid(*[[-1.0, 1.0]] * act_dim, indexing='ij'), -1)
        self.register_buffer('vertices',
                             torch.tensor(grid.reshape(-1, act_dim),
                                          dtype=torch.float32))
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
        self._logits_head = _layer_init(nn.Linear(hidden_dim, self.n_actions),
                                        std=0.01)

    # ---- interface used by ppo.train ------------------------------------
    def to_env(self, action: torch.Tensor) -> torch.Tensor:
        return self.vertices[action.squeeze(-1).long()]

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor,
                             action: torch.Tensor | None = None,
                             mask: torch.Tensor | None = None):
        logits = self._logits_head(self._actor_trunk(x))
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        idx = dist.sample() if action is None else action.squeeze(-1).long()
        return (idx.unsqueeze(-1).float(), dist.log_prob(idx), dist.entropy(),
                self.critic(x).squeeze(-1), None)

    # ---- deterministic action, used by every evaluation ------------------
    @torch.no_grad()
    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.vertices[self._logits_head(self._actor_trunk(x)).argmax(-1)]


class SpeedVertexAgent(VertexAgent):
    """Vertex policy that also picks the tangential speed each step.

    Action space = 2**act_dim vertices x len(speed_levels); ``to_env``
    appends the chosen fraction of cfg.v as a trailing channel, consumed by
    the environment when ``EnvConfig.speed_levels`` is set.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 speed_levels: tuple = (1.0, 0.5), **_ignored):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim,
                         hidden_dim=hidden_dim)
        lv = torch.tensor(speed_levels, dtype=torch.float32)
        base = self.vertices                                  # (2^m, m)
        self.vertices = torch.cat(
            [base.repeat_interleave(len(lv), 0),
             lv.repeat(base.shape[0]).unsqueeze(-1)], dim=-1)
        self.n_actions = self.vertices.shape[0]
        self._logits_head = _layer_init(
            nn.Linear(hidden_dim, self.n_actions), std=0.01)


class PriorVertexAgent(VertexAgent):
    """Vertex policy whose logits ride on the analytic margin prior.

        z(q, v) = alpha * sigma_margin(q)^T v + dz_theta(q, v)

    The environment appends the 2^m prior scores to the observation; the
    first term reproduces the margin-gradient law (strong from step one),
    and the learned head starts near zero, so training begins at the
    analytic controller's level and PPO learns when to deviate from it.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 **_ignored):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim,
                         hidden_dim=hidden_dim)
        self.n_prior = 2 ** act_dim
        self.alpha = nn.Parameter(torch.tensor(5.0))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        prior = x[..., -self.n_prior:]
        return self.alpha * prior + self._logits_head(self._actor_trunk(x))

    def get_action_and_value(self, x: torch.Tensor,
                             action: torch.Tensor | None = None):
        dist = Categorical(logits=self._logits(x))
        idx = dist.sample() if action is None else action.squeeze(-1).long()
        return (idx.unsqueeze(-1).float(), dist.log_prob(idx), dist.entropy(),
                self.critic(x).squeeze(-1), None)

    @torch.no_grad()
    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.vertices[self._logits(x).argmax(-1)]


class _SeqVertexBase(nn.Module):
    """Shared scaffolding for sequence-backbone vertex agents.

    The observation is a K-step history window flattened to K*D; the
    backbone maps it to a feature vector from which the categorical vertex
    head and the value head read. Interface-identical to VertexAgent.
    """

    def __init__(self, obs_dim: int, act_dim: int, history: int):
        super().__init__()
        assert obs_dim % history == 0, (obs_dim, history)
        self.history = history
        self.base_dim = obs_dim // history
        self.act_dim = act_dim
        self.action_store_dim = 1
        self.n_actions = 2 ** act_dim
        grid = np.stack(np.meshgrid(*[[-1.0, 1.0]] * act_dim,
                                    indexing='ij'), -1)
        self.register_buffer('vertices',
                             torch.tensor(grid.reshape(-1, act_dim),
                                          dtype=torch.float32))

    def _feat(self, x):
        raise NotImplementedError

    def _logits(self, x):
        return self._logits_head(self._feat(x))

    def critic(self, x):
        return self._value_head(self._feat(x))

    def to_env(self, action):
        return self.vertices[action.squeeze(-1).long()]

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None, mask=None):
        f = self._feat(x)
        logits = self._logits_head(f)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        idx = dist.sample() if action is None else action.squeeze(-1).long()
        return (idx.unsqueeze(-1).float(), dist.log_prob(idx), dist.entropy(),
                self._value_head(f).squeeze(-1), None)

    @torch.no_grad()
    def actor_mean(self, x):
        return self.vertices[self._logits(x).argmax(-1)]


class LSTMVertexAgent(_SeqVertexBase):
    """LSTM backbone over the K-step observation window (last hidden)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 history: int = 8, lstm_hidden: int = 256, **_ignored):
        super().__init__(obs_dim, act_dim, history)
        self.lstm = nn.LSTM(self.base_dim, lstm_hidden, num_layers=1,
                            batch_first=True)
        self._logits_head = nn.Sequential(
            _layer_init(nn.Linear(lstm_hidden, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, self.n_actions), std=0.01))
        self._value_head = nn.Sequential(
            _layer_init(nn.Linear(lstm_hidden, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0))

    def _feat(self, x):
        B = x.shape[0]
        seq = x.view(B, self.history, self.base_dim)
        out, _ = self.lstm(seq)
        return out[:, -1]


class TransformerVertexAgent(_SeqVertexBase):
    """Causal-free encoder over the K-step window (history-only context),
    reading the last token."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 512,
                 history: int = 8, d_model: int = 128, nhead: int = 4,
                 n_layers: int = 2, **_ignored):
        super().__init__(obs_dim, act_dim, history)
        self.embed = _layer_init(nn.Linear(self.base_dim, d_model))
        self.pos = nn.Parameter(torch.zeros(1, history, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._logits_head = nn.Sequential(
            _layer_init(nn.Linear(d_model, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, self.n_actions), std=0.01))
        self._value_head = nn.Sequential(
            _layer_init(nn.Linear(d_model, hidden_dim)), nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0))

    def _feat(self, x):
        B = x.shape[0]
        seq = self.embed(x.view(B, self.history, self.base_dim)) + self.pos
        return self.encoder(seq)[:, -1]
