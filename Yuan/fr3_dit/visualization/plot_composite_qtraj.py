#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_H5 = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks.hdf5"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"

SEGMENT_COLORS = [
    (0.10, 0.85, 0.20),
    (0.10, 0.45, 1.00),
    (0.95, 0.55, 0.15),
    (0.85, 0.20, 0.70),
    (0.65, 0.80, 0.20),
    (0.20, 0.80, 0.80),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the 7 joint-space curves of a composite task.")
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (.svg/.png). Defaults to experiments/outputs/qtraj_task<idx>.svg.")
    return p.parse_args()


def save_qtraj_plot(h5_path: Path, task_idx: int, out: Path | None = None) -> Path:
    """Render 7 joint-space curves of a composite task and save as SVG/PNG."""
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        q_off = ts["qtraj_offset"][()]
        ss_off = ts["subseg_offset"][()]
        i = int(task_idx)
        q_lo, q_hi = int(q_off[i]), int(q_off[i + 1])
        s_lo, s_hi = int(ss_off[i]), int(ss_off[i + 1])
        q = np.asarray(ts["qtraj_flat"][q_lo:q_hi], dtype=np.float32)
        seg_counts = np.asarray(ts["seg_step_counts_flat"][s_lo:s_hi], dtype=np.int64)
        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)
        dt = float(f["meta"].attrs.get("source_dt", 0.01))
        seg_count = int(ts["seg_count"][i])
        total_len = float(ts["total_length"][i])

    T = q.shape[0]
    t_axis = np.arange(T) * dt
    seam_steps = np.cumsum(seg_counts)[:-1]

    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(9, 10))
    for j in range(7):
        ax = axes[j]
        start = 0
        for seg_i, n in enumerate(seg_counts):
            color = SEGMENT_COLORS[seg_i % len(SEGMENT_COLORS)]
            end = start + int(n)
            ax.plot(t_axis[start:end], q[start:end, j], color=color, linewidth=1.6)
            start = end
        for s in seam_steps:
            ax.axvline(t_axis[int(s)], color="k", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(f"q{j+1} (rad)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"composite task {i} | seg_count={seg_count} | total_len={total_len*100:.1f} cm | "
        f"traj_ids={[int(r[0]) for r in subseg_meta]}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = DEFAULT_OUT_DIR / f"qtraj_task{i:04d}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[qtraj] saved {out}  (T={T}, duration={t_axis[-1]:.2f}s)")
    return out


def main() -> None:
    args = parse_args()
    save_qtraj_plot(args.h5, int(args.task_idx), args.out)


if __name__ == "__main__":
    main()
