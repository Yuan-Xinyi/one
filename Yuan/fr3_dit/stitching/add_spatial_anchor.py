#!/usr/bin/env python3
"""Stamp the path-start XY (in desk-frame coords) into every START token.

The original ``dir_local`` slot of the start token was a redundant [1, 0, 0]
(by construction, the local frame's x-axis is the first-segment direction).
We repurpose those three numbers to carry the path's start TCP location on
the desk, normalized to [-1, 1]:

    start.dir_local = [lx_norm, ly_norm, 0]
    where (lx, ly) = projection of (local_origin - desk_center) onto desk basis,
    normalized by (desk_x_half, desk_y_half).

This gives the q₀-DiT direct knowledge of WHERE on the desk the task happens,
which v2 had to guess (and got wrong, biasing q1 by ~0.4 rad).

Reads composite HDF5 with packed `tasks/token_flat`, writes a new HDF5 with the
same layout but updated start tokens. Everything else (qtraj, raw_trajs, meta)
copied verbatim. ``meta.attrs["spatial_anchor"] = True`` is added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# Token layout offsets (must match stitch_composite_tasks.TOKEN_LAYOUT)
KIND_ONEHOT_DIM = 3
DIR_LOCAL_OFFSET = 3   # right after kind_onehot
DIR_LOCAL_DIM = 3
TOKEN_KIND_START = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_anchored.hdf5")
    print(f"[anchor] input  = {args.input}")
    print(f"[anchor] output = {args.output}")

    with h5py.File(args.input, "r") as f_in:
        meta_in = f_in["meta"]
        ts_in = f_in["tasks"]
        raw_in = f_in["raw_trajs"]

        # Desk geometry (same for every task in this dataset)
        desk_center = np.asarray(meta_in.attrs["source_desk_center"], dtype=np.float32)
        desk_normal = np.asarray(meta_in.attrs["source_desk_normal"], dtype=np.float32)
        desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
        x_half = float(meta_in.attrs["source_desk_x_half"])
        y_half = float(meta_in.attrs["source_desk_y_half"])
        # Build desk basis (dx, dy)
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(desk_normal[0])) < 0.9 \
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dx = np.cross(helper, desk_normal); dx /= max(float(np.linalg.norm(dx)), 1e-12)
        dy = np.cross(desk_normal, dx); dy /= max(float(np.linalg.norm(dy)), 1e-12)
        print(f"[anchor] desk_center={desk_center.tolist()} half=({x_half:.2f}, {y_half:.2f})")

        token_offset = ts_in["token_offset"][()]
        local_origins = ts_in["local_origin"][()]  # (M, 3) world coords
        token_kind = ts_in["token_kind"][()]
        token_flat = ts_in["token_flat"][()]
        M = len(token_offset) - 1
        print(f"[anchor] total tasks={M}, total tokens={token_flat.shape[0]}")

        # For each task, find the start token (always index = token_offset[i]) and
        # rewrite its dir_local slot.
        start_idxs = token_offset[:-1].astype(np.int64)  # (M,)
        # Sanity check: those indices should all be START tokens
        assert (token_kind[start_idxs] == TOKEN_KIND_START).all(), \
            "expected token_offset[i] to point at a START token for every task"

        # Compute per-task (lx_norm, ly_norm)
        offsets = local_origins - desk_center[None, :]                  # (M, 3)
        lx = (offsets * dx[None, :]).sum(axis=1) / x_half               # (M,) in [-1, 1] roughly
        ly = (offsets * dy[None, :]).sum(axis=1) / y_half
        # Clip just in case (a few tasks may be just outside due to pos_tol)
        lx = np.clip(lx, -1.5, 1.5).astype(np.float32)
        ly = np.clip(ly, -1.5, 1.5).astype(np.float32)
        print(
            f"[anchor] anchor stats: lx min={lx.min():.3f} max={lx.max():.3f} mean={lx.mean():.3f}"
            f" | ly min={ly.min():.3f} max={ly.max():.3f} mean={ly.mean():.3f}"
        )

        # Stamp into start tokens
        token_flat_new = token_flat.copy()
        token_flat_new[start_idxs, DIR_LOCAL_OFFSET + 0] = lx
        token_flat_new[start_idxs, DIR_LOCAL_OFFSET + 1] = ly
        token_flat_new[start_idxs, DIR_LOCAL_OFFSET + 2] = 0.0

        # Write
        with h5py.File(args.output, "w") as f_out:
            # meta
            meta_out = f_out.create_group("meta")
            for k, v in meta_in.attrs.items():
                meta_out.attrs[k] = v
            if "token_layout" in meta_in:
                lay_out = meta_out.create_group("token_layout")
                for k, v in meta_in["token_layout"].attrs.items():
                    lay_out.attrs[k] = v
            meta_out.attrs["spatial_anchor"] = True

            # raw_trajs (passthrough)
            raw_out = f_out.create_group("raw_trajs")
            for name in raw_in.keys():
                raw_in.copy(name, raw_out)

            # tasks: copy everything except token_flat which we rewrote
            ts_out = f_out.create_group("tasks")
            for name in ts_in.keys():
                if name == "token_flat":
                    ts_out.create_dataset(name, data=token_flat_new, compression="gzip")
                else:
                    arr = ts_in[name][()]
                    ds = ts_out.create_dataset(name, data=arr,
                                               compression="gzip" if arr.ndim > 1 else None)
                    del ds  # silence unused-var

    print(f"[anchor] done — wrote {args.output}")


if __name__ == "__main__":
    main()
