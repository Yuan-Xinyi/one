#!/usr/bin/env python3
"""Inference + demo: sample a q-trajectory from a DiT checkpoint given task tokens.

Given a task_idx in the composite HDF5 dataset, load its tokens, run DDPM
sampling conditioned on those tokens, and produce three demo artifacts:

  experiments/outputs/infer_task<idx>_q_compare.svg   — 7 joint curves: GT vs predicted
  experiments/outputs/infer_task<idx>_tcp3d.svg       — 3-D TCP path: GT vs predicted (via FK)
  experiments/outputs/infer_task<idx>_meta.json       — task metadata dump

Also prints a ready-to-run visualize_composite_task command with the predicted q played
through the pen-FR3 robot in Panda3D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch

from Yuan.fr3_dit.training.task_cond_dit import (
    DDPMCosineSchedule,
    DiTConfig,
    TaskCondDiT,
    ddpm_sample,
)


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k.hdf5"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_ckpts" / "final.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"

SEGMENT_COLORS = [
    (0.10, 0.85, 0.20),
    (0.10, 0.45, 1.00),
    (0.95, 0.55, 0.15),
    (0.85, 0.20, 0.70),
    (0.65, 0.80, 0.20),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--sampler-steps", type=int, default=None,
                   help="Number of DDPM denoise steps (default: schedule.T).")
    p.add_argument("--eta", type=float, default=1.0,
                   help="Ancestral noise coefficient (1.0 = full DDPM, 0.0 = deterministic).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_task(h5_path: Path, task_idx: int, max_tokens: int, target_qsteps: int) -> dict:
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        tok_off = ts["token_offset"][()]
        q_off = ts["qtraj_offset"][()]
        ss_off = ts["subseg_offset"][()]
        if task_idx < 0 or task_idx >= len(tok_off) - 1:
            raise IndexError(f"task_idx={task_idx} out of range [0,{len(tok_off)-2}]")
        t_lo, t_hi = int(tok_off[task_idx]), int(tok_off[task_idx + 1])
        q_lo, q_hi = int(q_off[task_idx]), int(q_off[task_idx + 1])
        s_lo, s_hi = int(ss_off[task_idx]), int(ss_off[task_idx + 1])
        n_tok = t_hi - t_lo

        tokens = np.zeros((max_tokens, int(f["meta"].attrs["token_dim"])), dtype=np.float32)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_mask = np.zeros((max_tokens,), dtype=np.float32); token_mask[:n_tok] = 1.0

        q_gt_raw = np.asarray(ts["qtraj_flat"][q_lo:q_hi], dtype=np.float32)  # (T_raw, 7)
        tcp_gt_raw = np.asarray(ts["tcp_flat"][q_lo:q_hi], dtype=np.float32)
        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)
        seg_counts = np.asarray(ts["seg_step_counts_flat"][s_lo:s_hi], dtype=np.int32)
        start_q = np.asarray(ts["start_q"][task_idx], dtype=np.float32)
        total_len = float(ts["total_length"][task_idx])
        seg_count = int(ts["seg_count"][task_idx])
        dt = float(f["meta"].attrs.get("source_dt", 0.01))

        # Resample ground-truth q-traj to target_qsteps (match training)
        T_raw = q_gt_raw.shape[0]
        src = np.linspace(0.0, T_raw - 1, target_qsteps, dtype=np.float32)
        lo = np.clip(np.floor(src).astype(np.int64), 0, T_raw - 1)
        hi = np.clip(lo + 1, 0, T_raw - 1)
        frac = (src - lo.astype(np.float32))[:, None]
        q_gt = (1 - frac) * q_gt_raw[lo] + frac * q_gt_raw[hi]

    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "n_tokens": n_tok,
        "q_gt": q_gt.astype(np.float32),            # (target, 7)
        "q_gt_raw": q_gt_raw,                       # (T_raw, 7)
        "tcp_gt_raw": tcp_gt_raw,                   # (T_raw, 3)
        "subseg_meta": subseg_meta,
        "seg_counts": seg_counts,
        "start_q": start_q,
        "total_length": total_len,
        "seg_count": seg_count,
        "dt": dt,
    }


def fk_tcp_batch(q: np.ndarray, device: torch.device) -> np.ndarray:
    """Run the GPU-batched FK on a 2-D (T, 7) joint trajectory; return (T, 3) TCP positions."""
    from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3GPU
    fr3 = PenFrankaResearch3GPU(device)
    qt = torch.from_numpy(q).to(device, dtype=torch.float32)
    tcp_pos, _ = fr3.robot.fk_batch(qt)
    return tcp_pos.detach().cpu().numpy()


def plot_q_compare(q_gt: np.ndarray, q_pred: np.ndarray, dt: float, seg_counts: np.ndarray, out: Path, title: str):
    T = q_gt.shape[0]
    t_axis = np.arange(T) * (dt * max(1, int(round(q_gt.shape[0] * dt / (T * dt)))))
    # Simpler: produce a "normalized time [0,1]" axis since we resampled
    t_axis = np.linspace(0, 1, T)
    # Map seg_counts (in raw step units) to positions on the resampled grid.
    raw_total = int(seg_counts.sum())
    seam_frac = np.cumsum(seg_counts)[:-1] / max(raw_total, 1)
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 10))
    for j in range(7):
        ax = axes[j]
        ax.plot(t_axis, q_gt[:, j], color="#333", linewidth=1.8, label="GT" if j == 0 else None)
        ax.plot(t_axis, q_pred[:, j], color="#e24a33", linewidth=1.4, linestyle="--", label="DiT" if j == 0 else None)
        for s in seam_frac:
            ax.axvline(float(s), color="k", linestyle=":", linewidth=0.6, alpha=0.4)
        ax.set_ylabel(f"q{j+1} (rad)"); ax.grid(True, alpha=0.25)
        if j == 0: ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("normalized time")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); plt.close(fig)


def plot_tcp_3d(tcp_gt: np.ndarray, tcp_pred: np.ndarray, seg_counts: np.ndarray, out: Path, title: str):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    # GT path colored by segments
    start = 0
    for i, n in enumerate(seg_counts):
        end = start + int(n)
        c = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        ax.plot(tcp_gt[start:end, 0], tcp_gt[start:end, 1], tcp_gt[start:end, 2],
                color=c, linewidth=2.2, label=f"GT seg{i+1}")
        start = end
    # Predicted path in dashed red
    ax.plot(tcp_pred[:, 0], tcp_pred[:, 1], tcp_pred[:, 2],
            color="#e24a33", linewidth=1.6, linestyle="--", label="DiT pred")
    ax.scatter(*tcp_gt[0], s=50, color="#0044ff", label="start")
    ax.scatter(*tcp_gt[-1], s=50, color="#00aa00", label="goal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def save_q_npy(q_pred: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, q_pred.astype(np.float32))


def load_ckpt(ckpt_path: Path, device: torch.device, use_ema: bool) -> tuple[TaskCondDiT, DiTConfig, DDPMCosineSchedule, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = DiTConfig(**cfg_dict)
    model = TaskCondDiT(cfg).to(device)
    if use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in ckpt["ema"]:
                    p.copy_(ckpt["ema"][name].to(device))
    else:
        model.load_state_dict(ckpt["model"])
    schedule = DDPMCosineSchedule(T=int(ckpt["T"])).to(device)
    return model.eval(), cfg, schedule, int(ckpt.get("step", -1))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[ckpt] loading {args.ckpt}")
    model, cfg, schedule, step = load_ckpt(args.ckpt, device, args.use_ema)
    print(f"[ckpt] step={step} d_model={cfg.d_model} layers={cfg.n_layers} T={schedule.T} qsteps={cfg.max_qsteps}")

    task = load_task(args.data, args.task_idx, cfg.max_tokens, cfg.max_qsteps)
    print(f"[task] idx={args.task_idx} seg_count={task['seg_count']} "
          f"total_len={task['total_length']*100:.1f}cm n_tokens={task['n_tokens']}")

    tokens_t = torch.from_numpy(task["tokens"]).unsqueeze(0).to(device)       # (1, T_tok, D)
    token_mask_t = torch.from_numpy(task["token_mask"]).unsqueeze(0).to(device)
    qtraj_mask_t = torch.ones(1, cfg.max_qsteps, device=device)

    # Sample
    print(f"[sample] DDPM steps={args.sampler_steps or schedule.T} eta={args.eta}")
    q_pred = ddpm_sample(
        model, schedule, tokens_t, token_mask_t, qtraj_mask_t,
        shape=(1, cfg.max_qsteps, 7), device=device, eta=args.eta, num_steps=args.sampler_steps,
    )[0].cpu().numpy()  # (T_q, 7)

    # FK to get TCP
    tcp_pred = fk_tcp_batch(q_pred, device)
    # Also FK on GT (resampled) for a fair comparison
    tcp_gt = fk_tcp_batch(task["q_gt"], device)

    # Metrics
    mse_q = float(np.mean((q_pred - task["q_gt"]) ** 2))
    mse_tcp = float(np.mean(np.linalg.norm(tcp_pred - tcp_gt, axis=1)))
    print(f"[metric] mse_q={mse_q:.4f}  mean_tcp_err={mse_tcp*100:.2f}cm")

    title_base = (
        f"DiT task {args.task_idx} | seg_count={task['seg_count']} "
        f"| tokens={task['n_tokens']} | tcp_err={mse_tcp*100:.2f}cm"
    )

    out_q = args.out_dir / f"infer_task{args.task_idx:06d}_q_compare.svg"
    out_3d = args.out_dir / f"infer_task{args.task_idx:06d}_tcp3d.svg"
    out_meta = args.out_dir / f"infer_task{args.task_idx:06d}_meta.json"
    out_qnpy = args.out_dir / f"infer_task{args.task_idx:06d}_q_pred.npy"

    plot_q_compare(task["q_gt"], q_pred, task["dt"], task["seg_counts"], out_q, title_base)
    plot_tcp_3d(tcp_gt, tcp_pred, task["seg_counts"], out_3d, title_base)
    save_q_npy(q_pred, out_qnpy)

    with open(out_meta, "w") as f:
        json.dump({
            "task_idx": int(args.task_idx),
            "seg_count": int(task["seg_count"]),
            "n_tokens": int(task["n_tokens"]),
            "total_length_m": float(task["total_length"]),
            "ckpt_step": step,
            "sampler": {"steps": args.sampler_steps or schedule.T, "eta": args.eta, "seed": args.seed},
            "metrics": {"mse_q": mse_q, "mean_tcp_err_cm": mse_tcp * 100},
            "subseg_meta": task["subseg_meta"].tolist(),
        }, f, indent=2)

    print(f"[saved] {out_q}")
    print(f"[saved] {out_3d}")
    print(f"[saved] {out_qnpy}")
    print(f"[saved] {out_meta}")
    print(
        "\nTo animate the predicted trajectory in Panda3D, run:\n"
        f"  cd /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser && \\\n"
        f"  PYTHONPATH=/home/lqin/wrs_xinyi python -m fr3_dit.visualization.visualize_predicted_q "
        f"--q-npy {out_qnpy}\n"
    )


if __name__ == "__main__":
    main()
