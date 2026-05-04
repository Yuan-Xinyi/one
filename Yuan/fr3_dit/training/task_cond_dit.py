#!/usr/bin/env python3
"""Task-conditioned DiT for FR3 joint-trajectory denoising.

Input: variable-length task-token sequence (B, T_tok, D_tok)
Denoising target: joint-space trajectory (B, T_q, 7)

Architecture: TransformerDecoder. The query side is the noisy q-trajectory with
learned positional embedding; the memory side is the task-token sequence with
learned positional embedding and a diffusion-timestep token prepended.

Both tgt and memory carry padding masks (variable lengths).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding used by DDPM. t: (B,) int or float → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass
class DiTConfig:
    act_dim: int = 7
    token_dim: int = 32
    d_model: int = 256
    n_head: int = 4
    n_layers: int = 6
    dim_ff_mult: int = 4
    dropout: float = 0.1
    max_qsteps: int = 1500
    max_tokens: int = 16
    diffusion_steps: int = 1000


class TaskCondDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg

        # Target (q-trajectory) side
        self.act_in = nn.Linear(cfg.act_dim, cfg.d_model)
        self.pos_emb_q = nn.Parameter(torch.zeros(1, cfg.max_qsteps, cfg.d_model))
        nn.init.normal_(self.pos_emb_q, std=0.02)

        # Condition (tokens + timestep) side
        self.token_in = nn.Linear(cfg.token_dim, cfg.d_model)
        self.pos_emb_cond = nn.Parameter(torch.zeros(1, cfg.max_tokens + 1, cfg.d_model))
        nn.init.normal_(self.pos_emb_cond, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.SiLU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

        # TransformerDecoder: query=q-traj, memory=[time_tok, tokens...]
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.d_model * cfg.dim_ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.act_dim)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, T_q, 7) noisy q-trajectory
        t: torch.Tensor,           # (B,) diffusion timestep, int in [0, T)
        tokens: torch.Tensor,      # (B, T_tok, D_tok)
        token_mask: torch.Tensor,  # (B, T_tok) float/bool, 1 for valid
        qtraj_mask: torch.Tensor,  # (B, T_q)  float/bool, 1 for valid
    ) -> torch.Tensor:
        B, T_q, _ = x.shape
        T_tok = tokens.shape[1]
        d = self.cfg.d_model

        # Condition side: time token + task tokens
        t_emb = sinusoidal_timestep_embedding(t, d)       # (B, d)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)         # (B, 1, d)
        tok_emb = self.token_in(tokens)                   # (B, T_tok, d)
        cond = torch.cat([t_emb, tok_emb], dim=1)         # (B, 1+T_tok, d)
        cond = cond + self.pos_emb_cond[:, : 1 + T_tok]
        cond = self.dropout(cond)

        # Memory padding mask: time token always valid, then token_mask
        time_tok_valid = torch.ones(B, 1, device=x.device, dtype=token_mask.dtype)
        mem_valid = torch.cat([time_tok_valid, token_mask.to(dtype=time_tok_valid.dtype)], dim=1)
        # PyTorch expects True = IGNORE in key_padding_mask
        mem_kpm = mem_valid <= 0.5

        # Query side: noisy q-traj with pos embedding
        q_in = self.act_in(x)                              # (B, T_q, d)
        q_in = q_in + self.pos_emb_q[:, :T_q]
        q_in = self.dropout(q_in)

        tgt_kpm = qtraj_mask <= 0.5

        y = self.decoder(
            tgt=q_in,
            memory=cond,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
        )
        y = self.ln_f(y)
        y = self.head(y)                                   # (B, T_q, 7)
        return y


class DDPMCosineSchedule:
    """Continuous-time-ish cosine beta schedule (same form as Nichol-Dhariwal)."""

    def __init__(self, T: int = 1000, s: float = 0.008, device: str = "cpu"):
        steps = torch.arange(T + 1, dtype=torch.float64, device=device)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = betas.clamp(1e-6, 0.999).to(torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.T = T
        self.device = device

    def to(self, device: str | torch.device) -> "DDPMCosineSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DDPMCosineSchedule) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward noising; returns (xt, eps)."""
    bar_alpha = schedule.alphas_cumprod.gather(0, t)
    bar_alpha = bar_alpha.view(-1, 1, 1)
    eps = torch.randn_like(x0)
    xt = bar_alpha.sqrt() * x0 + (1 - bar_alpha).sqrt() * eps
    return xt, eps


@torch.no_grad()
def ddpm_sample(
    model: TaskCondDiT,
    schedule: DDPMCosineSchedule,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    qtraj_mask: torch.Tensor,
    shape: tuple[int, int, int],
    device: str | torch.device,
    eta: float = 0.0,
    num_steps: int | None = None,
    clip_x0: float | None = 3.2,
) -> torch.Tensor:
    """DDIM-style sampling (η configurable). Far more stable than full-T DDPM at the
    high-t tail, where ``bar_alpha → 0`` amplifies any tiny ε-prediction error into
    an enormous x0 estimate.

    η = 0  → deterministic DDIM
    η = 1  → reduces to ancestral DDPM (noisier)

    ``num_steps`` picks a subset of the T timesteps uniformly; 50 is a reasonable default.
    ``clip_x0`` clamps the reconstructed x0 to a physical joint-space range (FR3 joints all
    lie within |q| ≲ 3.2 rad, so 3.2 is a safe ceiling).
    """
    model.eval()
    B = shape[0]
    T = schedule.T
    steps = int(num_steps) if num_steps is not None else 100
    steps = max(1, min(steps, T))

    # Subsample the schedule
    step_indices = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long, device=device)

    xt = torch.randn(shape, device=device)
    for i in range(steps):
        t_int = int(step_indices[i].item())
        t_prev = int(step_indices[i + 1].item())
        t_vec = torch.full((B,), t_int, device=device, dtype=torch.long)

        ba_t = schedule.alphas_cumprod[t_int]
        ba_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        eps_pred = model(xt, t_vec, tokens, token_mask, qtraj_mask)

        x0_hat = (xt - (1 - ba_t).sqrt() * eps_pred) / ba_t.sqrt()
        if clip_x0 is not None:
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

        # DDIM update (generalized with eta ∈ [0, 1]):
        #   x_{t_prev} = sqrt(ba_prev) * x0_hat + sqrt(1 - ba_prev - sigma^2) * eps_pred + sigma * z
        if t_prev >= 0 and i < steps - 1:
            sigma = eta * (((1 - ba_prev) / (1 - ba_t)) * (1 - ba_t / ba_prev)).clamp_min(0).sqrt()
            noise_coef = (1 - ba_prev - sigma ** 2).clamp_min(0).sqrt()
            xt = ba_prev.sqrt() * x0_hat + noise_coef * eps_pred + sigma * torch.randn_like(xt)
        else:
            xt = x0_hat
        xt = xt * qtraj_mask.unsqueeze(-1)
    return xt


if __name__ == "__main__":
    # Simple shape smoke-test
    cfg = DiTConfig(max_qsteps=1500, max_tokens=16)
    m = TaskCondDiT(cfg)
    B, T_q, T_tok = 2, 512, 8
    x = torch.randn(B, T_q, 7)
    t = torch.randint(0, cfg.diffusion_steps, (B,))
    tok = torch.randn(B, T_tok, 32)
    tmask = (torch.arange(T_tok).unsqueeze(0).expand(B, -1) < torch.tensor([6, 4]).unsqueeze(-1)).float()
    qmask = (torch.arange(T_q).unsqueeze(0).expand(B, -1) < torch.tensor([480, 312]).unsqueeze(-1)).float()
    y = m(x, t, tok, tmask, qmask)
    print("model out:", y.shape, "tgt_nparams:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    sched = DDPMCosineSchedule(T=cfg.diffusion_steps)
    xt, eps = q_sample(x, t, sched)
    print("noisy shape:", xt.shape, "eps std:", eps.std().item())
