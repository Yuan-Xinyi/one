"""DDIM sampler for c-conditioned SeedQ0DiT.

Adapted from fr3_dit's `ddim_sample_q0`: drops the (tokens, token_mask, CFG)
pieces since our SeedQ0DiT conditions on a single 9-d vector `c`.
"""
from __future__ import annotations

from pathlib import Path

import torch

from Yuan.fr3_dit.training.task_cond_dit_q0 import DDPMCosineSchedule
from Yuan.seed_selection.model_q0 import SeedQ0Config, SeedQ0DiT


@torch.no_grad()
def ddim_sample_q0(
    model: SeedQ0DiT,
    schedule: DDPMCosineSchedule,
    c: torch.Tensor,                  # (B, 9)
    device: str | torch.device,
    num_steps: int = 50,
    eta: float = 0.0,
    clip_x0: float | None = 1.2,
    cfg_w: float = 1.0,
) -> torch.Tensor:
    """DDIM sampling with v-prediction and classifier-free guidance.

    cfg_w:
      1.0 → pure conditional sampling (no guidance, equivalent to no-CFG path).
      0.0 → unconditional sampling (uses null condition).
      >1  → amplify the conditional signal:
              v = v_uncond + cfg_w * (v_cond - v_uncond)
    Returns q0 in NORMALIZED space (B, 7).
    """
    model.eval()
    B = c.shape[0]
    T = schedule.T
    steps = max(1, min(int(num_steps), T))
    step_indices = torch.linspace(T - 1, 0, steps + 1, dtype=torch.long, device=device)

    use_cfg = (cfg_w != 1.0)
    if use_cfg:
        uncond_mask_pos = torch.zeros(B, dtype=torch.bool, device=device)
        uncond_mask_neg = torch.ones(B, dtype=torch.bool, device=device)

    xt = torch.randn(B, model.cfg.q_dim, device=device)
    for i in range(steps):
        t_int = int(step_indices[i].item())
        t_prev = int(step_indices[i + 1].item())
        t_vec = torch.full((B,), t_int, device=device, dtype=torch.long)

        ba_t = schedule.alphas_cumprod[t_int]
        ba_prev = schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        alpha_t = ba_t.sqrt()
        sigma_t = (1 - ba_t).sqrt()

        if use_cfg:
            # Single batched forward over [cond, uncond] for efficiency.
            xt2 = torch.cat([xt, xt], dim=0)
            t2  = torch.cat([t_vec, t_vec], dim=0)
            c2  = torch.cat([c, c], dim=0)
            um2 = torch.cat([uncond_mask_pos, uncond_mask_neg], dim=0)
            v_both = model(xt2, t2, c2, uncond_mask=um2)
            v_cond, v_uncond = v_both[:B], v_both[B:]
            v_pred = v_uncond + cfg_w * (v_cond - v_uncond)
        else:
            v_pred = model(xt, t_vec, c)

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


def load_ckpt(ckpt_path: str | Path, device: str | torch.device, use_ema: bool = True):
    """Load (model, schedule) from a training ckpt. Defaults to EMA weights."""
    ck = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    cfg = SeedQ0Config(**ck['cfg'])
    model = SeedQ0DiT(cfg).to(device)
    state = ck['ema'] if use_ema else ck['model']
    # EMA dict is parameter dict, not full state_dict — load it via parameter handles.
    if use_ema:
        own = dict(model.named_parameters())
        for n, p in state.items():
            own[n].data.copy_(p)
        # buffers (e.g. LayerNorm running stats — none here) stay from init / model state
        # Note: SeedQ0DiT has no running buffers; nothing else to copy.
    else:
        model.load_state_dict(state)
    schedule = DDPMCosineSchedule(T=cfg.diffusion_steps).to(device)
    return model, schedule, cfg, ck.get('step', None)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--num-steps', type=int, default=50)
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, schedule, cfg, step = load_ckpt(args.ckpt, device)
    print(f'[sample] loaded ckpt step={step}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    c = torch.randn(8, 9, device=device)
    q_norm = ddim_sample_q0(model, schedule, c, device=device, num_steps=args.num_steps)
    print(f'[sample] q_norm shape={tuple(q_norm.shape)} range=[{q_norm.min():.3f}, {q_norm.max():.3f}]')
