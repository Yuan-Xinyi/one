"""Minimal c → q0 diffusion model.

Replaces fr3_dit's token-sequence Transformer with a 9-d MLP conditioner.
Reuses fr3_dit's DDPM cosine schedule, v-prediction, joint-limit normalization,
and sinusoidal timestep embedding so the training loop stays familiar.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from Yuan.fr3_dit.training.task_cond_dit_q0 import sinusoidal_timestep_embedding


@dataclass
class SeedQ0Config:
    c_dim: int = 9            # (p0, line_dir, n_target) = 3+3+3
    q_dim: int = 7
    d_model: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 1000


class SeedQ0DiT(nn.Module):
    """MLP-based diffusion network. Predicts v ∈ ℝ^7 from (x_t, t, c)."""

    def __init__(self, cfg: SeedQ0Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Input LayerNorm for c stabilizes mixed scales (p0 in m vs unit vectors).
        self.c_ln = nn.LayerNorm(cfg.c_dim)
        self.c_mlp = nn.Sequential(
            nn.Linear(cfg.c_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        # Learnable null condition for classifier-free guidance. When
        # `uncond_mask[i]` is True, c[i] is replaced with this vector before
        # the c-MLP. Trained jointly via dropout in train_q0.
        self.null_c = nn.Parameter(torch.zeros(cfg.c_dim))
        self.t_mlp = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        self.q_in = nn.Linear(cfg.q_dim, d)

        # Residual MLP backbone.
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d * 4), nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d * 4, d),
            ) for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.q_dim)

    def forward(
        self,
        x: torch.Tensor,                       # (B, q_dim)
        t: torch.Tensor,                       # (B,)
        c: torch.Tensor,                       # (B, c_dim)
        uncond_mask: torch.Tensor | None = None,   # (B,) bool. True → replace c[i] with null_c
    ) -> torch.Tensor:
        if uncond_mask is not None and uncond_mask.any():
            c = torch.where(uncond_mask.unsqueeze(-1),
                            self.null_c.to(c.dtype).expand_as(c),
                            c)
        c_emb = self.c_mlp(self.c_ln(c))
        t_emb = self.t_mlp(sinusoidal_timestep_embedding(t, self.cfg.d_model))
        h = self.q_in(x) + c_emb + t_emb
        for blk in self.blocks:
            h = h + blk(h)
        h = self.ln_f(h)
        return self.head(h)


if __name__ == "__main__":
    cfg = SeedQ0Config()
    m = SeedQ0DiT(cfg)
    B = 4
    x = torch.randn(B, 7)
    t = torch.randint(0, cfg.diffusion_steps, (B,))
    c = torch.randn(B, 9)
    y = m(x, t, c)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"shape={tuple(y.shape)} params={n_params:.3f}M")
