"""Q-network and replay buffer for v8 SAC.

For our single-step contextual bandit:
    Q(c, q) ≈ L(c, q)
where L is the deterministic rollout length (in [0, 1] after normalising by T).
No Bellman bootstrap, no target net — just regression on ground-truth labels.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

import Yuan.RL.config as cfg


class QNet(nn.Module):
    """Two-layer MLP critic Q(c, q) -> scalar reward prediction."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int = cfg.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class QEnsemble(nn.Module):
    """Bootstrap Q ensemble for uncertainty estimation (NeuralUCB-style).

    Each member is an independent QNet trained on a randomly subsampled
    minibatch. Disagreement std across members estimates epistemic
    uncertainty σ̂(c, q), used by active task sampling.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 m: int = cfg.Q_ENSEMBLE_M,
                 hidden: int = cfg.HIDDEN_DIM):
        super().__init__()
        self.m = int(m)
        self.members = nn.ModuleList(
            [QNet(state_dim, action_dim, hidden) for _ in range(self.m)])

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Stack predictions across members. Returns (M, B)."""
        return torch.stack([m(s, a) for m in self.members], dim=0)

    def mean(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self(s, a).mean(dim=0)

    def std(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self(s, a).std(dim=0)


class ReplayBuffer:
    """FIFO buffer for (state, action, reward) transitions.

    Tensors live on CPU; minibatches are moved to the training device when
    sampled. Reward is the per-task-normalised rollout length (L / T).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 capacity: int = cfg.SAC_REPLAY_SIZE):
        self.capacity = int(capacity)
        self.size = 0
        self.cursor = 0
        self.s  = torch.zeros((capacity, state_dim),  dtype=torch.float32)
        self.a  = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.r  = torch.zeros((capacity,),            dtype=torch.float32)
        self.T  = torch.zeros((capacity,),            dtype=torch.float32)
        self.L  = torch.zeros((capacity,),            dtype=torch.float32)

    def add_batch(self, s_np: np.ndarray, a_np: np.ndarray,
                  L_np: np.ndarray, T_np: np.ndarray,
                  r_np: np.ndarray | None = None):
        """Insert a batch of (state, action, raw rollout length, T) tuples."""
        n = s_np.shape[0]
        s_t = torch.as_tensor(s_np, dtype=torch.float32)
        a_t = torch.as_tensor(a_np, dtype=torch.float32)
        T_t = torch.as_tensor(T_np, dtype=torch.float32)
        L_t = torch.as_tensor(L_np, dtype=torch.float32)
        if r_np is None:
            r_t = L_t / T_t.clamp_min(1.0)
        else:
            r_t = torch.as_tensor(r_np, dtype=torch.float32)
        # FIFO insertion: handle wrap-around
        end = self.cursor + n
        if end <= self.capacity:
            self.s[self.cursor:end] = s_t
            self.a[self.cursor:end] = a_t
            self.r[self.cursor:end] = r_t
            self.T[self.cursor:end] = T_t
            self.L[self.cursor:end] = L_t
        else:
            tail = self.capacity - self.cursor
            self.s[self.cursor:] = s_t[:tail];  self.s[:n - tail] = s_t[tail:]
            self.a[self.cursor:] = a_t[:tail];  self.a[:n - tail] = a_t[tail:]
            self.r[self.cursor:] = r_t[:tail];  self.r[:n - tail] = r_t[tail:]
            self.T[self.cursor:] = T_t[:tail];  self.T[:n - tail] = T_t[tail:]
            self.L[self.cursor:] = L_t[:tail];  self.L[:n - tail] = L_t[tail:]
        self.cursor = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, device: torch.device):
        idx = torch.randint(0, self.size, (batch_size,))
        return (self.s[idx].to(device, non_blocking=True),
                self.a[idx].to(device, non_blocking=True),
                self.r[idx].to(device, non_blocking=True))

    def __len__(self) -> int:
        return self.size
