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
    """Two-layer MLP critic Q(c, q) -> scalar reward prediction.

    Hidden width defaults to ``cfg.Q_HIDDEN_DIM`` (or ``cfg.HIDDEN_DIM`` if
    that field isn't defined). Q usually wants more capacity than the
    policy because it has to model the full reward landscape, not just
    output a good action.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = int(getattr(cfg, "Q_HIDDEN_DIM", cfg.HIDDEN_DIM))
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class ResidualNet(nn.Module):
    """Residual predictor R(s, a) → real_L - phantom_L (both as ratios in [0, 1]).

    Used in v15 residual-corrected bandit framework: at deploy, candidate
    selection is argmax over (phantom_L_pred + R(s, a)), where phantom_L_pred
    comes from a cheap analytic forward simulator and R captures the
    surrogate's structured bias. R is supervised — no Bellman bootstrap.

    Output range is small (~ ±0.1 typical) because phantom is already 96%
    accurate on geometric tasks, BUT can be much larger (~±0.3) on contact
    tasks where phantom is blind to force failures. Tanh-bounded with
    learnable scale to allow the network to express the full range while
    keeping outputs well-conditioned.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden: int | None = None,
                 max_residual: float = 0.5):
        super().__init__()
        if hidden is None:
            hidden = int(getattr(cfg, "Q_HIDDEN_DIM", cfg.HIDDEN_DIM))
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.max_residual = float(max_residual)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
        return self.max_residual * torch.tanh(raw)


class QEnsemble(nn.Module):
    """Bootstrap Q ensemble for uncertainty estimation (NeuralUCB-style).

    Each member is an independent QNet trained on a randomly subsampled
    minibatch. Disagreement std across members estimates epistemic
    uncertainty σ̂(c, q), used at deploy as ``mean(Q) - λ·std(Q)`` for
    pessimistic ranking.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 m: int = cfg.Q_ENSEMBLE_M,
                 hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = int(getattr(cfg, "Q_HIDDEN_DIM", cfg.HIDDEN_DIM))
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

    Optionally supports Prioritized Experience Replay (PER, Schaul 2016):
      sample probability  p_i ∝ priority_i ** alpha
      IS bias correction  w_i = (1 / (N * p_i)) ** beta   (normalized by max)

    With cfg.PER_ENABLE=False, falls back to uniform random sampling and
    weights are all 1.0 — behaviorally identical to the legacy buffer.

    Tensors live on CPU; minibatches are moved to the training device when
    sampled. Reward is the per-task-normalised rollout length (L / T).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 capacity: int = cfg.SAC_REPLAY_SIZE,
                 use_per: bool | None = None,
                 per_alpha: float | None = None,
                 per_eps: float | None = None):
        self.capacity = int(capacity)
        self.size = 0
        self.cursor = 0
        self.s  = torch.zeros((capacity, state_dim),  dtype=torch.float32)
        self.a  = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.r  = torch.zeros((capacity,),            dtype=torch.float32)
        self.T  = torch.zeros((capacity,),            dtype=torch.float32)
        self.L  = torch.zeros((capacity,),            dtype=torch.float32)

        # PER state
        self.use_per   = bool(getattr(cfg, "PER_ENABLE", False)
                              if use_per is None else use_per)
        self.per_alpha = float(getattr(cfg, "PER_ALPHA", 0.6)
                               if per_alpha is None else per_alpha)
        self.per_eps   = float(getattr(cfg, "PER_EPS", 1e-3)
                               if per_eps is None else per_eps)
        self.priorities    = np.zeros((capacity,), dtype=np.float64)
        self.max_priority  = 1.0   # priority assigned to fresh samples

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

        # collect indices of inserted slots so we can mark their priority
        inserted_idx = (np.arange(n, dtype=np.int64) + self.cursor) % self.capacity

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

        # New samples get max priority so they are guaranteed to be drawn
        # at least once before being downweighted by their actual TD error.
        self.priorities[inserted_idx] = self.max_priority

    def sample(self, batch_size: int, device: torch.device,
               beta: float = 1.0):
        """Draw a minibatch. Returns (s, a, r, idx, weights).

        idx is needed by the caller so they can call update_priorities(idx,
        |TD-error|) after the gradient step. weights are IS bias corrections
        — multiply your per-sample loss by these (and average) before
        backward(). Under PER_ENABLE=False, idx is uniform and weights=1.
        """
        if self.use_per and self.size > 0:
            p = self.priorities[:self.size] ** self.per_alpha
            p_sum = p.sum()
            if p_sum <= 0:
                # fallback (e.g. all-zero priorities at warm start)
                p_norm = np.full(self.size, 1.0 / self.size)
            else:
                p_norm = p / p_sum
            idx_np = np.random.choice(self.size, size=batch_size,
                                      p=p_norm, replace=True)
            w = (1.0 / (self.size * p_norm[idx_np])) ** float(beta)
            w_max = w.max() if w.size else 1.0
            w = (w / w_max).astype(np.float32)
        else:
            idx_np = np.random.randint(0, self.size, size=batch_size).astype(np.int64)
            w = np.ones(batch_size, dtype=np.float32)

        idx_t = torch.as_tensor(idx_np, dtype=torch.long)
        w_t   = torch.as_tensor(w,      dtype=torch.float32, device=device)
        return (self.s[idx_t].to(device, non_blocking=True),
                self.a[idx_t].to(device, non_blocking=True),
                self.r[idx_t].to(device, non_blocking=True),
                idx_np, w_t)

    def update_priorities(self, idx_np: np.ndarray, abs_td_errors: np.ndarray):
        """Refresh the priorities of the just-trained samples. No-op if PER off."""
        if not self.use_per:
            return
        new_p = np.abs(np.asarray(abs_td_errors, dtype=np.float64)) + self.per_eps
        self.priorities[idx_np] = new_p
        if new_p.size:
            self.max_priority = max(self.max_priority, float(new_p.max()))

    def __len__(self) -> int:
        return self.size
