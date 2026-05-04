#!/usr/bin/env python3
"""Keep composite tasks where every sub-segment's TCP length ≥ --min-m.

Reads one composite HDF5 and rewrites only the tasks that pass. Leaves the raw_trajs
and meta groups intact (meta gets two extra attrs recording the filter).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


DEFAULT_IN = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k.hdf5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--output", type=Path, default=None,
                   help="Output HDF5 path (default: alongside input with _minseg suffix).")
    p.add_argument("--min-m", type=float, default=0.10,
                   help="Minimum per-subseg TCP length in meters (default 0.10 = 10cm).")
    p.add_argument("--batch", type=int, default=1024,
                   help="Task batch size for streaming writes.")
    return p.parse_args()


def compute_kept_mask(f_in: h5py.File, min_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Scan every task, compute each subseg's TCP length, keep if min length ≥ min_m."""
    ts = f_in["tasks"]
    raw = f_in["raw_trajs"]
    ss_meta = ts["subseg_meta_flat"][()]           # (S_total, 3)
    ss_off = ts["subseg_offset"][()]               # (M+1,)
    raw_off = raw["offset"][()]                    # (N_raw+1,)

    print("[filter] loading raw_trajs/tcp_flat into RAM ...")
    raw_tcp = raw["tcp_flat"][()]                  # full read, typically <400MB
    print(f"[filter] raw_tcp in RAM: shape={raw_tcp.shape} dtype={raw_tcp.dtype}")

    M = ss_off.shape[0] - 1
    S_total = ss_meta.shape[0]
    traj_ids = ss_meta[:, 0].astype(np.int64)
    starts = ss_meta[:, 1].astype(np.int64)
    ends = ss_meta[:, 2].astype(np.int64)
    idx_start = raw_off[traj_ids] + starts
    idx_end = raw_off[traj_ids] + ends - 1

    pts_start = raw_tcp[idx_start]
    pts_end = raw_tcp[idx_end]
    subseg_lens = np.linalg.norm(pts_end - pts_start, axis=1)  # (S_total,)
    print(
        f"[filter] subseg_lens (cm): "
        f"min={subseg_lens.min()*100:.2f} median={np.median(subseg_lens)*100:.2f} "
        f"max={subseg_lens.max()*100:.2f}"
    )

    # Per-task min via np.minimum.reduceat — O(S_total) vectorized instead of Python loop.
    task_start_idx = ss_off[:-1].astype(np.int64)
    # reduceat requires strictly increasing indices; ss_off is already strictly increasing
    per_task_min = np.minimum.reduceat(subseg_lens, task_start_idx)
    # Any task with zero subsegs (shouldn't happen) keeps inf via fallback
    zero_tasks = (ss_off[1:] - ss_off[:-1]) == 0
    per_task_min = np.where(zero_tasks, np.inf, per_task_min)

    kept = per_task_min >= min_m
    kept_idx = np.where(kept)[0]
    print(
        f"[filter] tasks total={M}  kept={int(kept.sum())} ({kept.mean()*100:.2f}%) "
        f"dropped={int((~kept).sum())}"
    )
    return kept, kept_idx


def build_new_offsets(old_off: np.ndarray, kept_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given old_off (M+1,) and kept_idx (K,), return (new_off(K+1,), src_starts(K,), src_lengths(K,))."""
    src_starts = old_off[kept_idx].astype(np.int64)
    src_ends = old_off[kept_idx + 1].astype(np.int64)
    lens = src_ends - src_starts
    new_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return new_off, src_starts, lens


def write_filtered(f_in: h5py.File, f_out: h5py.File, kept_idx: np.ndarray, min_m: float) -> None:
    ts_in = f_in["tasks"]
    raw_in = f_in["raw_trajs"]
    meta_in = f_in["meta"]

    # /meta : copy attrs + add filter record
    meta_out = f_out.create_group("meta")
    for k, v in meta_in.attrs.items():
        meta_out.attrs[k] = v
    # Copy token_layout subgroup
    if "token_layout" in meta_in:
        lay_out = meta_out.create_group("token_layout")
        for k, v in meta_in["token_layout"].attrs.items():
            lay_out.attrs[k] = v
    meta_out.attrs["filter_min_subseg_m"] = float(min_m)
    meta_out.attrs["filter_num_kept"] = int(kept_idx.shape[0])
    meta_out.attrs["filter_num_dropped"] = int(meta_in.attrs.get("num_composites", 0)) - int(kept_idx.shape[0])
    meta_out.attrs["num_composites"] = int(kept_idx.shape[0])

    # /raw_trajs : passthrough clone (raw trajectories are still reachable)
    raw_out = f_out.create_group("raw_trajs")
    for name in raw_in.keys():
        raw_in.copy(name, raw_out)

    # /tasks : scalar per-task fields
    ts_out = f_out.create_group("tasks")

    for name in ["start_q", "local_frame", "local_origin", "plane_normal",
                 "seg_count", "total_length"]:
        arr = ts_in[name][()]
        if arr.ndim == 1:
            ts_out.create_dataset(name, data=arr[kept_idx])
        else:
            ts_out.create_dataset(name, data=arr[kept_idx], compression="gzip")

    # Re-offset packed arrays
    tok_off_new, tok_starts, tok_lens = build_new_offsets(ts_in["token_offset"][()], kept_idx)
    q_off_new, q_starts, q_lens = build_new_offsets(ts_in["qtraj_offset"][()], kept_idx)
    ss_off_new, ss_starts, ss_lens = build_new_offsets(ts_in["subseg_offset"][()], kept_idx)

    token_flat_in = ts_in["token_flat"]
    token_kind_in = ts_in["token_kind"]
    qtraj_flat_in = ts_in["qtraj_flat"]
    tcp_flat_in = ts_in["tcp_flat"]
    ssmeta_flat_in = ts_in["subseg_meta_flat"]
    segstep_flat_in = ts_in["seg_step_counts_flat"]

    def allocate(name: str, total: int, inner, dtype, chunks=True, comp="gzip"):
        if inner is None:
            shape = (total,)
        else:
            shape = (total, inner)
        return ts_out.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compression=comp)

    token_dim = token_flat_in.shape[1]
    token_flat_out = allocate("token_flat", int(tok_off_new[-1]), token_dim, token_flat_in.dtype)
    token_kind_out = allocate("token_kind", int(tok_off_new[-1]), None, token_kind_in.dtype, comp=None)
    qtraj_flat_out = allocate("qtraj_flat", int(q_off_new[-1]), 7, qtraj_flat_in.dtype)
    tcp_flat_out = allocate("tcp_flat", int(q_off_new[-1]), 3, tcp_flat_in.dtype)
    ssmeta_flat_out = allocate("subseg_meta_flat", int(ss_off_new[-1]), 3, ssmeta_flat_in.dtype)
    segstep_flat_out = allocate("seg_step_counts_flat", int(ss_off_new[-1]), None, segstep_flat_in.dtype, comp=None)

    # Stream-copy each kept task
    M_kept = kept_idx.shape[0]
    for k in range(M_kept):
        # tokens
        s_src, n_src = int(tok_starts[k]), int(tok_lens[k])
        s_dst = int(tok_off_new[k])
        token_flat_out[s_dst : s_dst + n_src] = token_flat_in[s_src : s_src + n_src]
        token_kind_out[s_dst : s_dst + n_src] = token_kind_in[s_src : s_src + n_src]
        # qtraj + tcp
        s_src, n_src = int(q_starts[k]), int(q_lens[k])
        s_dst = int(q_off_new[k])
        qtraj_flat_out[s_dst : s_dst + n_src] = qtraj_flat_in[s_src : s_src + n_src]
        tcp_flat_out[s_dst : s_dst + n_src] = tcp_flat_in[s_src : s_src + n_src]
        # subseg
        s_src, n_src = int(ss_starts[k]), int(ss_lens[k])
        s_dst = int(ss_off_new[k])
        ssmeta_flat_out[s_dst : s_dst + n_src] = ssmeta_flat_in[s_src : s_src + n_src]
        segstep_flat_out[s_dst : s_dst + n_src] = segstep_flat_in[s_src : s_src + n_src]
        if (k + 1) % 10000 == 0 or k + 1 == M_kept:
            print(f"[copy] {k + 1}/{M_kept} tasks written")

    ts_out.create_dataset("token_offset", data=tok_off_new)
    ts_out.create_dataset("qtraj_offset", data=q_off_new)
    ts_out.create_dataset("subseg_offset", data=ss_off_new)


def main() -> None:
    args = parse_args()
    if args.output is None:
        stem = args.input.stem
        args.output = args.input.with_name(f"{stem}_minseg{int(args.min_m*100):02d}.hdf5")
    print(f"[filter] input  = {args.input}")
    print(f"[filter] output = {args.output}")
    print(f"[filter] min subseg length = {args.min_m:.3f}m ({args.min_m*100:.0f}cm)")

    with h5py.File(args.input, "r") as f_in:
        kept, kept_idx = compute_kept_mask(f_in, args.min_m)
        if kept_idx.shape[0] == 0:
            raise RuntimeError("No tasks survived the filter.")
        with h5py.File(args.output, "w") as f_out:
            write_filtered(f_in, f_out, kept_idx, args.min_m)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
