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
                             action: torch.Tensor | None = None):
        dist = Categorical(logits=self._logits_head(self._actor_trunk(x)))
        idx = dist.sample() if action is None else action.squeeze(-1).long()
        return (idx.unsqueeze(-1).float(), dist.log_prob(idx), dist.entropy(),
                self.critic(x).squeeze(-1), None)

    # ---- deterministic action, used by every evaluation ------------------
    @torch.no_grad()
    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.vertices[self._logits_head(self._actor_trunk(x)).argmax(-1)]


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
