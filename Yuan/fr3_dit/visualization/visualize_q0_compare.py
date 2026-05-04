#!/usr/bin/env python3
"""Side-by-side render of GT vs DiT-predicted initial joint configurations.

Shows the FR3 in two poses simultaneously:
- GT q₀ in green (semitransparent)
- DiT-predicted q₀ candidates in red (also overlaid translucently)
- desk plane + GT path TCP marker so you can see the spatial context
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, required=True,
                   help="Task index (must have run infer_dit_q0 already).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-prefix", type=str, default="infer_q0_v5",
                   help="Filename prefix used at inference time (e.g. 'infer_q0_v5', 'infer_q0', 'infer_q0_fm_v5').")
    p.add_argument("--n-show", type=int, default=4,
                   help="How many of the 8 candidates to render.")
    return p.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(helper, z); x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float32)


def main() -> None:
    args = parse_args()
    idx = int(args.task_idx)

    npy_path = args.out_dir / f"{args.out_prefix}_task{idx:06d}_q0_pred.npy"
    meta_path = args.out_dir / f"{args.out_prefix}_task{idx:06d}_meta.json"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"{npy_path} not found. Run inference first, e.g.\n"
            f"  python -m fr3_dit.training.infer_dit_q0 --task-idx {idx} --out-prefix {args.out_prefix} \\\n"
            f"      --ckpt fr3_dit/experiments/outputs/dit_q0_v5_ckpts/final.pt"
        )

    q0_preds = np.load(npy_path).astype(np.float32)  # (n, 7)
    meta = json.loads(meta_path.read_text())
    gt_q0 = np.asarray(meta["gt_q0_rad"], dtype=np.float32)
    rmses = np.asarray(meta["per_sample_rmse_rad"], dtype=np.float32)
    order = np.argsort(rmses)
    show_idx = order[: args.n_show]
    print(f"[task {idx}] seg_count={meta['seg_count']} total_len={meta['total_length_m']*100:.1f}cm")
    print(f"[gt q0]  {gt_q0.round(3).tolist()}")
    print(f"[best pred] rmse={rmses[order[0]]:.3f}  q0={q0_preds[order[0]].round(3).tolist()}")

    # Pull desk + path geometry from the HDF5 for spatial context
    with h5py.File(args.data, "r") as f:
        ts = f["tasks"]
        ss_off = ts["subseg_offset"][()]
        s_lo, s_hi = int(ss_off[idx]), int(ss_off[idx + 1])
        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)
        local_origin = np.asarray(ts["local_origin"][idx], dtype=np.float32)

        raw = f["raw_trajs"]
        raw_off = raw["offset"][()]
        raw_tcp = raw["tcp_flat"]
        gt_tcp_chunks = []
        for traj_id, st, en in subseg_meta:
            r_lo = int(raw_off[int(traj_id)])
            tcp_seg = np.asarray(raw_tcp[r_lo + int(st) : r_lo + int(en)], dtype=np.float32)
            gt_tcp_chunks.append(tcp_seg)

        ma = f["meta"].attrs
        if str(ma.get("source_sampling_mode", "")) == "fixed_desk":
            desk_center = np.asarray(ma["source_desk_center"], dtype=np.float32)
            desk_normal = np.asarray(ma["source_desk_normal"], dtype=np.float32)
        else:
            desk_center = None
            desk_normal = None

    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    if desk_center is not None:
        rot = rotation_matrix_from_normal(desk_normal)
        ossop.box(
            half_extents=(0.85, 0.85, 0.0015),
            pos=desk_center, rotmat=rot,
            rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32),
            alpha=0.4,
        ).attach_to(world.scene)

    # Draw GT TCP path (segmented) for context
    SEG_COLORS = np.array([
        [0.10, 0.85, 0.20], [0.10, 0.45, 1.00], [0.95, 0.55, 0.15],
        [0.85, 0.20, 0.70], [0.65, 0.80, 0.20],
    ], dtype=np.float32)
    for k, tcp_seg in enumerate(gt_tcp_chunks):
        c = SEG_COLORS[k % SEG_COLORS.shape[0]]
        for j in range(tcp_seg.shape[0] - 1):
            ossop.cylinder(
                spos=tcp_seg[j], epos=tcp_seg[j + 1],
                radius=0.0035, rgb=c, alpha=0.85,
            ).attach_to(world.scene)

    # GT robot pose — solid green
    robot = PenFrankaResearch3(name="pen", enable_cc=False)
    robot.goto_given_conf(gt_q0)
    robot.gen_meshmodel(rgb=np.array([0.10, 0.80, 0.20], dtype=np.float32),
                        alpha=0.85, toggle_tcp_frame=True).attach_to(world)

    # Predicted robot poses — translucent red, ordered by RMSE
    for rank, ci in enumerate(show_idx):
        alpha = max(0.18, 0.55 - 0.10 * rank)  # best one = densest
        robot.goto_given_conf(q0_preds[int(ci)].astype(np.float32))
        robot.gen_meshmodel(rgb=np.array([0.95, 0.30, 0.20], dtype=np.float32),
                            alpha=alpha, toggle_tcp_frame=False).attach_to(world)
        print(f"[pred rank {rank+1}] sample={int(ci)} rmse={rmses[int(ci)]:.3f} alpha={alpha:.2f}")

    print(
        f"\n[legend] green = GT q0, red (densest = best of {args.n_show}) = DiT predicted q0\n"
        "[legend] colored TCP curve = GT future stroke that the predicted q0 must enable"
    )

    world.run()


if __name__ == "__main__":
    main()
