#!/usr/bin/env python3
"""Render a train/val loss curve from a train_dit.py log file."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_ckpts" / "train.log"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "train_loss_curve.svg"


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_loss = re.compile(r"\[step +(\d+)/\d+\] loss=([\d.eE+-]+)")
    val_loss = re.compile(r"\[val +step +(\d+)\] val_loss=([\d.eE+-]+)")
    train_steps, train_loss = [], []
    val_steps, val_vals = [], []
    for ln in path.read_text().splitlines():
        m = step_loss.search(ln)
        if m:
            train_steps.append(int(m.group(1))); train_loss.append(float(m.group(2))); continue
        m = val_loss.search(ln)
        if m:
            val_steps.append(int(m.group(1))); val_vals.append(float(m.group(2)))
    return (np.asarray(train_steps), np.asarray(train_loss),
            np.asarray(val_steps), np.asarray(val_vals))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smooth", type=int, default=20, help="Rolling-mean window for train loss.")
    args = p.parse_args()

    ts, tl, vs, vv = parse_log(args.log)
    if len(ts) == 0:
        raise RuntimeError(f"No [step] lines found in {args.log}")

    fig, ax = plt.subplots(figsize=(9, 5))
    if args.smooth > 1 and len(tl) > args.smooth:
        k = args.smooth
        kernel = np.ones(k) / k
        smooth = np.convolve(tl, kernel, mode="valid")
        ax.plot(ts[k - 1:], smooth, color="#333", linewidth=1.8, label=f"train loss (smoothed, k={k})")
    else:
        ax.plot(ts, tl, color="#333", linewidth=1.5, label="train loss")
    if len(vs) > 0:
        ax.plot(vs, vv, color="#e24a33", marker="o", linewidth=1.5, label="val loss")
    ax.set_xlabel("step"); ax.set_ylabel("ε-MSE loss"); ax.set_yscale("log")
    ax.set_title(f"DiT training: {args.log.name}")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out); plt.close(fig)
    print(f"[saved] {args.out}  (train points={len(ts)}, val points={len(vs)})")


if __name__ == "__main__":
    main()
