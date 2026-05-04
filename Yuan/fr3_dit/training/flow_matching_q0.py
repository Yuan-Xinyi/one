#!/usr/bin/env python3
"""Conditional Flow Matching (CFM) helpers for the q₀-DiT.

The denoiser architecture in ``task_cond_dit_q0.TaskCondDiTq0`` outputs a 7-D
vector per sample. We reinterpret that vector as the **velocity field** of a
straight-path probability flow ODE between standard Gaussian noise and the
clean q₀ distribution. Same model, same conditioning — different objective and
sampler.

Convention used here:
    t ∈ [0, 1],  t=0 is noise, t=1 is data
    x_t = (1 − t) · noise + t · x_data
    target velocity  u_t = x_data − noise   (constant along the line for a CFM
                                              straight-path Gaussian-to-Gaussian transport)

Training loss: MSE(model_output, u_target).
Sampling: forward Euler from t=0 (Gaussian noise) → t=1 in N steps.

The model's timestep embedding still expects an integer ``t`` index. We rescale
``t ∈ [0, 1]`` to the same ``[0, 1000)`` range used during DDPM training so the
sinusoidal embedding has comparable resolution. This is just a parameterization
of the position embedding — semantically still continuous time.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# Arbitrary integer scale we feed into the sinusoidal timestep embedding.
# 1000 is convenient because it matches the DDPM diffusion_steps the model has
# already been wired for, but anything smooth works.
T_SCALE = 1000


def cfm_sample(x_data: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear path interpolation. Returns (x_t, u_target).

    Args:
        x_data: (B, ...) clean data point.
        t:      (B,) values in [0, 1]. t=0 is noise, t=1 is data.
    """
    noise = torch.randn_like(x_data)
    while t.dim() < x_data.dim():
        t = t.unsqueeze(-1)
    x_t = (1.0 - t) * noise + t * x_data
    u_target = x_data - noise
    return x_t, u_target


def x_data_from_velocity(x_t: torch.Tensor, t: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
    """Recover the predicted clean data point from x_t and the predicted velocity.

    From the linear path:  x_data = x_t + (1 − t) · u_pred
    """
    while t.dim() < u_pred.dim():
        t = t.unsqueeze(-1)
    return x_t + (1.0 - t) * u_pred


def t_to_model_index(t_continuous: torch.Tensor) -> torch.Tensor:
    """Map continuous t ∈ [0, 1] to the integer index the model's sinusoidal
    timestep embedding was originally indexed by."""
    return (t_continuous * float(T_SCALE)).long().clamp(0, T_SCALE - 1)


@torch.no_grad()
def euler_sample_cfm(
    model,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    shape: tuple[int, int],
    device: str | torch.device,
    num_steps: int = 10,
    cfg_w: float = 0.0,
    clip_x_data: float | None = None,
) -> torch.Tensor:
    """Forward Euler integration of the velocity field from noise (t=0) to data (t=1).

    Returns the sampled point at t=1 (in normalized q-space). Caller is responsible
    for denormalizing.
    """
    model.eval()
    B = shape[0]
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_continuous = torch.full((B,), float(i) / num_steps, device=device)
        t_idx = t_to_model_index(t_continuous)

        u_cond = model(x, t_idx, tokens, token_mask, uncond_mask=None)
        if cfg_w != 0.0:
            umask = torch.ones(B, dtype=torch.bool, device=device)
            u_uncond = model(x, t_idx, tokens, token_mask, uncond_mask=umask)
            u_pred = u_uncond + (1.0 + cfg_w) * (u_cond - u_uncond)
        else:
            u_pred = u_cond

        if clip_x_data is not None:
            x_hat = x_data_from_velocity(x, t_continuous, u_pred)
            x_hat = x_hat.clamp(-clip_x_data, clip_x_data)
            # Re-derive a clipped velocity consistent with the clipped x_hat
            # u = (x_hat − x) / (1 − t)   for t < 1
            denom = (1.0 - t_continuous).view(-1, 1).clamp_min(1e-6)
            u_pred = (x_hat - x) / denom

        x = x + dt * u_pred
    return x


if __name__ == "__main__":
    # Smoke test
    B = 4
    x_data = torch.randn(B, 7)
    t = torch.rand(B)
    x_t, u = cfm_sample(x_data, t)
    print("xt", x_t.shape, "u", u.shape)
    x_recovered = x_data_from_velocity(x_t, t, u)
    err = (x_recovered - x_data).abs().max().item()
    print(f"recovery err: {err:.6e} (should be ~0)")
