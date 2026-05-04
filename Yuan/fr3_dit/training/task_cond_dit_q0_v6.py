"""v6: token-aligned per-keypoint q prediction.

Each token gets a predicted q (B, T_tok, 7) instead of a single q0 (B, 7). The
keypoint at token position i is the joint configuration at the *vertex* of the
polyline associated with that token:
   START token         → vertex 0 (stroke start)
   SEGMENT_i token     → vertex i (end of segment i)
   CORNER_i token      → vertex i (same vertex as adjacent SEGMENT_i, redundant
                          but kept so consecutive-position diffs in the smoothness
                          loss naturally capture inter-vertex transitions).

At inference the model returns a full keypoint sequence; downstream the IK
interpolator generates dense per-frame q's between consecutive keypoints, fully
replacing the tracker rollout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    FR3_JOINT_LIMITS,
    Q_CENTER,
    Q_HALF,
    sinusoidal_timestep_embedding,
    denormalize_q,
    normalize_q,
    q_sample,
    v_target_from,
)


@dataclass
class DiTq0Config_v6:
    act_dim: int = 7
    token_dim: int = 32
    max_tokens: int = 11             # also = max keypoint sequence length
    d_model: int = 256
    n_head: int = 4
    n_enc_layers: int = 4
    n_dec_layers: int = 2
    dropout: float = 0.1
    diffusion_steps: int = 1000


class TaskCondDiTq0_v6(nn.Module):
    """Sequence-to-sequence v-prediction: per-token keypoint q.

    Input  x       (B, T, 7) — noisy keypoint sequence (positions filled per token,
                                CORNER positions repeat adjacent SEGMENT vertex)
    Input  t       (B,)      — diffusion step
    Input  tokens  (B, T, 32)
    Input  mask    (B, T)    — 1 = valid token (any kind), 0 = pad
    Output v_pred  (B, T, 7) — same shape as x
    """

    def __init__(self, cfg: DiTq0Config_v6):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null_token, std=0.02)

        self.token_in = nn.Linear(cfg.token_dim, d)
        self.pos_emb_tok = nn.Parameter(torch.zeros(1, cfg.max_tokens, d))
        nn.init.normal_(self.pos_emb_tok, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=4 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_enc_layers)

        # Per-keypoint query side: project noisy q into model dim, add positional + time embedding.
        self.q_in = nn.Linear(cfg.act_dim, d)
        self.pos_emb_kp = nn.Parameter(torch.zeros(1, cfg.max_tokens, d))
        nn.init.normal_(self.pos_emb_kp, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d),
        )

        # Decoder: each query position cross-attends to encoded tokens, AND self-attends
        # to other query positions → captures inter-keypoint structure (smoothness etc.).
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.n_head, dim_feedforward=4 * d,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_dec_layers)

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.act_dim)

    def _encode_condition(self, tokens, token_mask, uncond_mask=None):
        B, T_tok, _ = tokens.shape
        d = self.cfg.d_model
        tok_emb = self.token_in(tokens) + self.pos_emb_tok[:, :T_tok]
        kpm = token_mask <= 0.5
        if uncond_mask is not None and uncond_mask.any():
            null_mem = self.null_token.expand(B, T_tok, d)
            null_kpm = torch.ones(B, T_tok, device=tokens.device, dtype=torch.bool)
            null_kpm[:, 0] = False
            expand_b = uncond_mask.view(B, 1, 1)
            tok_emb = torch.where(expand_b, null_mem, tok_emb)
            kpm = torch.where(uncond_mask.view(B, 1), null_kpm, kpm)
        memory = self.token_encoder(tok_emb, src_key_padding_mask=kpm)
        return memory, kpm

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, 7)
        t: torch.Tensor,                 # (B,)
        tokens: torch.Tensor,            # (B, T, 32)
        token_mask: torch.Tensor,        # (B, T)
        uncond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        memory, kpm = self._encode_condition(tokens, token_mask, uncond_mask)

        t_emb = sinusoidal_timestep_embedding(t, self.cfg.d_model)        # (B, d)
        t_emb = self.time_mlp(t_emb)
        x_emb = self.q_in(x)                                                # (B, T, d)
        query = x_emb + t_emb.unsqueeze(1) + self.pos_emb_kp[:, :T]         # (B, T, d)

        # Self-attn on query (inter-keypoint), cross-attn to memory (per-position).
        # Use the same kpm as cross-attn key-padding (memory side).
        # For the query side, also key-padding-mask out pad positions so they don't
        # bleed into self-attention.
        y = self.decoder(
            tgt=query, memory=memory,
            tgt_key_padding_mask=kpm,                # mask own pad in self-attn
            memory_key_padding_mask=kpm,
        )
        y = self.ln_f(y)
        return self.head(y)                           # (B, T, 7)


# --- Sampling ----------------------------------------------------------------

@torch.no_grad()
def ddim_sample_keypoints(
    model: TaskCondDiTq0_v6,
    schedule: DDPMCosineSchedule,
    tokens: torch.Tensor,                # (B, T, 32)
    token_mask: torch.Tensor,            # (B, T)
    device: str | torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    cfg_w: float = 0.0,
    clip_x0: float | None = 1.2,
) -> torch.Tensor:
    """DDIM sampling with v-prediction returning normalized keypoint sequence.

    Returns shape (B, T, 7) — caller masks/extracts valid positions and denormalizes.
    """
    model.eval()
    B, T, _ = tokens.shape
    T_step = schedule.T
    steps = max(1, min(int(num_steps), T_step))
    step_indices = torch.linspace(T_step - 1, 0, steps + 1, dtype=torch.long, device=device)

    xt = torch.randn(B, T, 7, device=device)
    for i in range(steps):
        t_int = int(step_indices[i].item())
        t_prev = int(step_indices[i + 1].item())
        t_vec = torch.full((B,), t_int, device=device, dtype=torch.long)

        ba_t = schedule.alphas_cumprod[t_int]
        ba_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        alpha_t = ba_t.sqrt(); sigma_t = (1 - ba_t).sqrt()

        v_cond = model(xt, t_vec, tokens, token_mask, uncond_mask=None)
        if cfg_w != 0.0:
            umask = torch.ones(B, device=device, dtype=torch.bool)
            v_uncond = model(xt, t_vec, tokens, token_mask, uncond_mask=umask)
            v_pred = v_uncond + (1.0 + cfg_w) * (v_cond - v_uncond)
        else:
            v_pred = v_cond

        x0_hat = alpha_t * xt - sigma_t * v_pred
        eps_pred = sigma_t * xt + alpha_t * v_pred
        if clip_x0 is not None:
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

        if t_prev >= 0 and i < steps - 1:
            sigma_step = eta * (((1 - ba_prev) / (1 - ba_t)) * (1 - ba_t / ba_prev)).clamp_min(0).sqrt()
            noise = torch.randn_like(xt) if sigma_step.item() > 0 else 0.0
            xt = ba_prev.sqrt() * x0_hat + (1 - ba_prev - sigma_step ** 2).clamp_min(0).sqrt() * eps_pred + sigma_step * noise
        else:
            xt = x0_hat
    return xt
