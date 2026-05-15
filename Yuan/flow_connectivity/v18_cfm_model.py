"""v18 Conditional Flow Matching model — backward generative for q_curr.

Learns p(q_curr | q_next, x_curr, x_next, plane_normal, direction)
via Tong/Lipman-style CFM:

  v_τ = (1 - τ) z + τ q_curr,   z ~ N(0, I_7),   τ ~ U(0, 1)
  target velocity = q_curr - z
  loss = ||v_θ(v_τ, τ, cond) - (q_curr - z)||²

At inference: ODE-integrate from z to v_1 = sampled q_curr.

Conditioning vector (19-D):
  [q_next (7), x_curr (3), x_next (3), plane_normal (3), direction (3)]
"""
from __future__ import annotations
import torch
import torch.nn as nn


COND_DIM = 19


class CFMFlowModel(nn.Module):
    """Conditional Flow Matching: learns p(q_curr | cond)."""

    def __init__(self, q_dim: int = 7, cond_dim: int = COND_DIM,
                 hidden: int = 512, depth: int = 6):
        super().__init__()
        # input = v_tau (7) + tau (1) + cond (19) = 27
        in_dim = q_dim + 1 + cond_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, q_dim)]
        self.net = nn.Sequential(*layers)
        self.q_dim = q_dim

    def forward(self, v_tau: torch.Tensor, tau: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        x = torch.cat([v_tau, tau, cond], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_steps: int = 16,
               cfg_scale: float = 1.0, device=None) -> torch.Tensor:
        """ODE integration (Heun midpoint) from N(0, I) to learned p(q_curr | cond).
        cfg_scale > 1 amplifies velocity field (CFG-style)."""
        if device is None:
            device = cond.device
        B = cond.shape[0]
        v = torch.randn(B, self.q_dim, device=device, dtype=torch.float32)
        dtau = 1.0 / n_steps
        for k in range(n_steps):
            tau = torch.full((B,), (k + 0.5) * dtau, device=device,
                             dtype=torch.float32)
            velocity = self.forward(v, tau, cond)
            v = v + dtau * cfg_scale * velocity
        return v
