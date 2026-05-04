#!/usr/bin/env python3
"""Task-conditioned DiT that predicts the **initial joint configuration q₀** only.

Given a variable-length task-token sequence describing the future stroke, sample a
q₀ ∈ ℝ⁷ from which the plane-constrained tracker can execute the whole stroke. The
rest of the trajectory is determined by the tracker, so the diffusion target is
reduced from (T_q, 7) to (7,).

Design choices (all informed by v1 training failure):
- **v-prediction** — numerically stable at both noise tails, standard for Stable
  Diffusion v2 and modern diffusion policies.
- **Classifier-free guidance dropout** — trains a single network that yields both
  conditional and unconditional ε estimates. During training we drop the condition
  with ``p=drop_prob`` and swap in a learnable null sequence. At inference we can
  amplify the conditioning via ``cfg_w``.
- **Joint-limit normalization** — FR3 has q4 ≈ -1.6 mean and q6 ≈ +2.5 mean,
  which violates the DDPM assumption that data is zero-centered. We scale every
  joint into [-1, 1] via its mechanical limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Joint-space normalization (FR3 hardware limits) --------------------------
FR3_JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],   # q1
    [-1.8326,  1.8326],   # q2
    [-2.8972,  2.8972],   # q3
    [-3.0718, -0.1222],   # q4 — heavily offset
    [-2.8798,  2.8798],   # q5
    [ 0.4364,  4.6251],   # q6 — heavily offset
    [-3.0543,  3.0543],   # q7
], dtype=np.float32)
Q_CENTER = FR3_JOINT_LIMITS.mean(axis=1)                           # (7,)
Q_HALF = (FR3_JOINT_LIMITS[:, 1] - FR3_JOINT_LIMITS[:, 0]) / 2.0   # (7,)


def normalize_q(q: np.ndarray | torch.Tensor):
    if isinstance(q, torch.Tensor):
        c = torch.as_tensor(Q_CENTER, dtype=q.dtype, device=q.device)
        h = torch.as_tensor(Q_HALF, dtype=q.dtype, device=q.device)
        return (q - c) / h
    return (q - Q_CENTER) / Q_HALF


def denormalize_q(q_norm: np.ndarray | torch.Tensor):
    if isinstance(q_norm, torch.Tensor):
        c = torch.as_tensor(Q_CENTER, dtype=q_norm.dtype, device=q_norm.device)
        h = torch.as_tensor(Q_HALF, dtype=q_norm.dtype, device=q_norm.device)
        return q_norm * h + c
    return q_norm * Q_HALF + Q_CENTER


# --- Helpers ------------------------------------------------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# --- Diffusion schedule -------------------------------------------------------
class DDPMCosineSchedule:
    """Nichol & Dhariwal cosine β-schedule."""

    def __init__(self, T: int = 1000, s: float = 0.008, device: str = "cpu"):
        steps = torch.arange(T + 1, dtype=torch.float64, device=device)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        bar = f / f[0]
        betas = 1 - (bar[1:] / bar[:-1])
        self.betas = betas.clamp(1e-6, 0.999).to(torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.T = int(T)
        self.device = torch.device(device)

    def to(self, device) -> "DDPMCosineSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DDPMCosineSchedule) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward noising. ``x0``: (B, 7). Returns (xt, eps) of same shape."""
    bar = schedule.alphas_cumprod.gather(0, t).view(-1, 1)
    eps = torch.randn_like(x0)
    xt = bar.sqrt() * x0 + (1 - bar).sqrt() * eps
    return xt, eps


def v_target_from(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, schedule: DDPMCosineSchedule) -> torch.Tensor:
    """v = α·ε − σ·x₀, where α = √bar_alpha, σ = √(1 − bar_alpha)."""
    bar = schedule.alphas_cumprod.gather(0, t).view(-1, 1)
    a = bar.sqrt()
    s = (1 - bar).sqrt()
    return a * eps - s * x0


# --- Model --------------------------------------------------------------------
@dataclass
class DiTq0Config:
    act_dim: int = 7
    token_dim: int = 32
    max_tokens: int = 11
    d_model: int = 256
    n_head: int = 4
    n_enc_layers: int = 4          # transformer encoder depth (token side)
    n_dec_layers: int = 2          # transformer decoder depth (cross-attn to tokens)
    dropout: float = 0.1
    diffusion_steps: int = 1000
    pred_type: str = "v"           # "v" or "eps"


class TaskCondDiTq0(nn.Module):
    """Outputs v-prediction (or ε-prediction) for a 7-dim q₀ given variable-length tokens."""

    def __init__(self, cfg: DiTq0Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Learnable null token for CFG (a single "empty condition" memory slot).
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null_token, std=0.02)

        # Token encoder
        self.token_in = nn.Linear(cfg.token_dim, d)
        self.pos_emb_tok = nn.Parameter(torch.zeros(1, cfg.max_tokens, d))
        nn.init.normal_(self.pos_emb_tok, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=4 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)

        # q + timestep — combined into a single query token
        self.q_in = nn.Linear(cfg.act_dim, d)
        self.time_mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d),
        )

        # Decoder: (x + t) attends to encoded tokens
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=4 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers)

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.act_dim)

    def _encode_condition(
        self,
        tokens: torch.Tensor,        # (B, T_tok, token_dim)
        token_mask: torch.Tensor,    # (B, T_tok) float/bool, 1=valid
        uncond_mask: torch.Tensor | None = None,  # (B,) bool, True → replace with null
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_tok, _ = tokens.shape
        d = self.cfg.d_model

        tok_emb = self.token_in(tokens) + self.pos_emb_tok[:, :T_tok]
        kpm = token_mask <= 0.5  # True = pad, matches PyTorch MHA convention

        if uncond_mask is not None and uncond_mask.any():
            null_mem = self.null_token.expand(B, T_tok, d)
            # null mask: first slot valid, rest padded → effectively a length-1 condition
            null_kpm = torch.ones(B, T_tok, device=tokens.device, dtype=torch.bool)
            null_kpm[:, 0] = False
            expand_b = uncond_mask.view(B, 1, 1)
            tok_emb = torch.where(expand_b, null_mem, tok_emb)
            kpm = torch.where(uncond_mask.view(B, 1), null_kpm, kpm)

        memory = self.token_encoder(tok_emb, src_key_padding_mask=kpm)
        return memory, kpm

    def forward(
        self,
        x: torch.Tensor,             # (B, 7) noisy q₀ (normalized)
        t: torch.Tensor,             # (B,) diffusion step
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory, kpm = self._encode_condition(tokens, token_mask, uncond_mask)

        t_emb = sinusoidal_timestep_embedding(t, self.cfg.d_model)
        t_emb = self.time_mlp(t_emb)
        x_emb = self.q_in(x)
        query = (x_emb + t_emb).unsqueeze(1)          # (B, 1, d)
        y = self.decoder(tgt=query, memory=memory, memory_key_padding_mask=kpm)
        y = self.ln_f(y).squeeze(1)                   # (B, d)
        return self.head(y)                            # (B, 7)


# --- Sampling -----------------------------------------------------------------
@torch.no_grad()
def ddim_sample_q0(
    model: TaskCondDiTq0,
    schedule: DDPMCosineSchedule,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    shape: tuple[int, int],          # (B, 7)
    device: str | torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    cfg_w: float = 0.0,
    clip_x0: float | None = 1.2,
) -> torch.Tensor:
    """DDIM sampling with v-prediction and optional classifier-free guidance.

    Returns the sampled q₀ in **normalized** space. Caller should denormalize.
    """
    model.eval()
    B = shape[0]
    T = schedule.T
    steps = max(1, min(int(num_steps), T))
    step_indices = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long, device=device)

    xt = torch.randn(shape, device=device)
    for i in range(steps):
        t_int = int(step_indices[i].item())
        t_prev = int(step_indices[i + 1].item())
        t_vec = torch.full((B,), t_int, device=device, dtype=torch.long)

        ba_t = schedule.alphas_cumprod[t_int]
        ba_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        alpha_t = ba_t.sqrt()
        sigma_t = (1 - ba_t).sqrt()

        v_cond = model(xt, t_vec, tokens, token_mask, uncond_mask=None)
        if cfg_w != 0.0:
            umask = torch.ones(B, device=device, dtype=torch.bool)
            v_uncond = model(xt, t_vec, tokens, token_mask, uncond_mask=umask)
            v_pred = v_uncond + (1.0 + cfg_w) * (v_cond - v_uncond)
        else:
            v_pred = v_cond

        # v → (x0, eps)
        x0_hat = alpha_t * xt - sigma_t * v_pred
        eps_pred = sigma_t * xt + alpha_t * v_pred
        if clip_x0 is not None:
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

        if t_prev >= 0 and i < steps - 1:
            sigma_step = eta * (
                ((1 - ba_prev) / (1 - ba_t)) * (1 - ba_t / ba_prev)
            ).clamp_min(0).sqrt()
            noise_coef = (1 - ba_prev - sigma_step ** 2).clamp_min(0).sqrt()
            xt = ba_prev.sqrt() * x0_hat + noise_coef * eps_pred + sigma_step * torch.randn_like(xt)
        else:
            xt = x0_hat

    return xt


if __name__ == "__main__":
    cfg = DiTq0Config()
    m = TaskCondDiTq0(cfg)
    B, T_tok = 4, 11
    x = torch.randn(B, 7)
    t = torch.randint(0, cfg.diffusion_steps, (B,))
    tok = torch.randn(B, T_tok, cfg.token_dim)
    tm = (torch.arange(T_tok).unsqueeze(0).expand(B, -1) < torch.tensor([3, 5, 7, 11]).unsqueeze(-1)).float()
    u = torch.tensor([True, False, True, False])
    y = m(x, t, tok, tm, uncond_mask=u)
    print("shape:", y.shape, "params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")

    sched = DDPMCosineSchedule()
    xt, eps = q_sample(x, t, sched)
    v = v_target_from(x, eps, t, sched)
    print("v target shape:", v.shape)

    out = ddim_sample_q0(m, sched, tok, tm, shape=(B, 7), device="cpu", num_steps=20, cfg_w=2.0)
    print("sampled q0_norm:", out.shape, "range:", out.min().item(), out.max().item())
