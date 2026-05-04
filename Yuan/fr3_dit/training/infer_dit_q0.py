#!/usr/bin/env python3
"""Inference + evaluation for the q₀-predicting DiT.

Given a task (tokens), sample one or more q₀ candidates and optionally evaluate them
by running the GPU-batched plane-constrained tracker from each predicted q₀ to see
how far along the stroke it can go.

Outputs (under fr3_dit/experiments/outputs/):
  infer_q0_task<idx>_bars.svg         — per-joint GT vs predicted q₀ bar plot
  infer_q0_task<idx>_meta.json        — metrics + prediction values
  infer_q0_task<idx>_q0_pred.npy      — (n_samples, 7) candidate q₀ array
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    DiTq0Config,
    TaskCondDiTq0,
    FR3_JOINT_LIMITS,
    ddim_sample_q0,
    denormalize_q,
)


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10.hdf5"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_q0_ckpts" / "final.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of q₀ candidates to draw (diffusion is multi-modal).")
    p.add_argument("--sampler-steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--cfg-w", type=float, default=3.0,
                   help="Classifier-free guidance weight; 0 = no guidance.")
    p.add_argument("--clip-x0", type=float, default=1.2)
    p.add_argument("--no-snap-q7", action="store_true", default=False,
                   help="Disable q7→0 post-processing snap (training canonicalizes q7=0).")
    p.add_argument("--out-prefix", type=str, default="infer_q0",
                   help="Filename prefix for outputs (e.g. 'infer_q0_v5' to avoid clobbering older runs).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_task(h5_path: Path, task_idx: int, max_tokens: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        tok_off = ts["token_offset"][()]
        ss_off = ts["subseg_offset"][()]
        if task_idx < 0 or task_idx >= len(tok_off) - 1:
            raise IndexError(f"task_idx={task_idx} out of range [0,{len(tok_off)-2}]")
        t_lo, t_hi = int(tok_off[task_idx]), int(tok_off[task_idx + 1])
        s_lo, s_hi = int(ss_off[task_idx]), int(ss_off[task_idx + 1])
        n_tok = t_hi - t_lo
        token_dim = int(f["meta"].attrs["token_dim"])

        tokens = np.zeros((max_tokens, token_dim), dtype=np.float32)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_mask = np.zeros((max_tokens,), dtype=np.float32); token_mask[:n_tok] = 1.0

        start_q = np.asarray(ts["start_q"][task_idx], dtype=np.float32)
        total_len = float(ts["total_length"][task_idx])
        seg_count = int(ts["seg_count"][task_idx])
        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)

    return {
        "tokens": tokens, "token_mask": token_mask, "n_tokens": n_tok,
        "start_q": start_q, "total_length": total_len, "seg_count": seg_count,
        "subseg_meta": subseg_meta,
    }


def load_ckpt(ckpt_path: Path, device: torch.device, use_ema: bool) -> tuple[TaskCondDiTq0, DiTq0Config, DDPMCosineSchedule, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DiTq0Config(**ckpt["cfg"])
    model = TaskCondDiTq0(cfg).to(device)
    if use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ckpt["ema"]:
                    p.copy_(ckpt["ema"][n].to(device))
    else:
        model.load_state_dict(ckpt["model"])
    schedule = DDPMCosineSchedule(T=int(ckpt["T"])).to(device)
    return model.eval(), cfg, schedule, int(ckpt.get("step", -1))


def plot_q0_bars(gt: np.ndarray, preds: np.ndarray, out: Path, title: str) -> None:
    """preds: (n_samples, 7) in raw rad."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(7)
    # Joint limits as grey envelope
    lower = FR3_JOINT_LIMITS[:, 0]
    upper = FR3_JOINT_LIMITS[:, 1]
    ax.fill_between(xs, lower, upper, color="#cccccc", alpha=0.35, label="joint limit")
    # Prediction scatter
    for i, p in enumerate(preds):
        ax.scatter(xs + np.linspace(-0.25, 0.25, preds.shape[0])[i], p,
                   s=36, alpha=0.75, color="#e24a33")
    # GT
    ax.scatter(xs, gt, s=90, color="#0044aa", marker="X", label="GT", zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels([f"q{j+1}" for j in range(7)])
    ax.set_ylabel("joint angle (rad)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[ckpt] loading {args.ckpt}")
    model, cfg, schedule, step = load_ckpt(args.ckpt, device, args.use_ema)
    print(f"[ckpt] step={step} d_model={cfg.d_model} enc={cfg.n_enc_layers} dec={cfg.n_dec_layers} T={schedule.T}")

    task = load_task(args.data, args.task_idx, cfg.max_tokens)
    print(
        f"[task] idx={args.task_idx} seg_count={task['seg_count']} "
        f"total_len={task['total_length']*100:.1f}cm n_tokens={task['n_tokens']}"
    )

    tokens_t = torch.from_numpy(task["tokens"]).unsqueeze(0).expand(args.n_samples, -1, -1).contiguous().to(device)
    token_mask_t = torch.from_numpy(task["token_mask"]).unsqueeze(0).expand(args.n_samples, -1).contiguous().to(device)

    print(f"[sample] DDIM steps={args.sampler_steps} eta={args.eta} cfg_w={args.cfg_w} n={args.n_samples}")
    q0_norm = ddim_sample_q0(
        model, schedule, tokens_t, token_mask_t,
        shape=(args.n_samples, 7), device=device,
        num_steps=args.sampler_steps, eta=args.eta,
        cfg_w=args.cfg_w, clip_x0=args.clip_x0,
    )
    q0_raw = denormalize_q(q0_norm).cpu().numpy()  # (n, 7)

    # q7 (pen self-rotation) is null-space — training canonicalizes it to 0,
    # so snap predictions to 0 as a safety to eliminate OOL violations on this joint.
    snap_q7 = not args.no_snap_q7
    if snap_q7:
        q7_pre = q0_raw[:, 6].copy()
        q0_raw[:, 6] = 0.0
        print(f"[snap]  q7 pre-snap range=[{q7_pre.min():.3f},{q7_pre.max():.3f}] → 0.0")

    # Per-sample error against GT
    gt = task["start_q"]
    mse = np.mean((q0_raw - gt[None, :]) ** 2, axis=1)          # (n,)
    per_sample_rmse = np.sqrt(mse)
    print(f"[metric] per-sample q0 RMSE (rad): {np.round(per_sample_rmse, 3).tolist()}")
    print(f"[metric] best-of-{args.n_samples} RMSE={per_sample_rmse.min():.3f}  mean={per_sample_rmse.mean():.3f}")
    # Out-of-limit check
    out_of_limit = np.any(
        (q0_raw < FR3_JOINT_LIMITS[:, 0]) | (q0_raw > FR3_JOINT_LIMITS[:, 1]), axis=1
    )
    print(f"[metric] out-of-limit samples: {int(out_of_limit.sum())}/{args.n_samples}")

    out_bars = args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_bars.svg"
    out_meta = args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_meta.json"
    out_npy = args.out_dir / f"{args.out_prefix}_task{args.task_idx:06d}_q0_pred.npy"
    plot_q0_bars(
        gt, q0_raw, out_bars,
        title=(f"q₀ prediction — task {args.task_idx} | seg={task['seg_count']} | "
               f"n_tokens={task['n_tokens']} | best RMSE={per_sample_rmse.min():.3f} rad")
    )
    np.save(out_npy, q0_raw.astype(np.float32))
    with open(out_meta, "w") as f:
        json.dump({
            "task_idx": int(args.task_idx),
            "seg_count": int(task["seg_count"]),
            "n_tokens": int(task["n_tokens"]),
            "total_length_m": float(task["total_length"]),
            "ckpt_step": step,
            "sampler": {"steps": args.sampler_steps, "eta": args.eta,
                        "cfg_w": args.cfg_w, "clip_x0": args.clip_x0, "seed": args.seed},
            "gt_q0_rad": gt.tolist(),
            "pred_q0_rad": q0_raw.tolist(),
            "per_sample_rmse_rad": per_sample_rmse.tolist(),
            "out_of_limit_count": int(out_of_limit.sum()),
        }, f, indent=2)
    print(f"[saved] {out_bars}\n[saved] {out_npy}\n[saved] {out_meta}")


if __name__ == "__main__":
    main()
