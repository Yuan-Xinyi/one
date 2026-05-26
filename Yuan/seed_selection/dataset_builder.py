"""Module 8: batch SMM-aware label dataset generator.

Iterates over a list of ``(c, q0_seed)`` pairs, calls
``build_labels_for_one_task`` on each, and serializes the result to an NPZ.

Resume + crash-safety:
  * partial NPZs are written atomically (tmp file + rename) so a kill in the
    middle of a checkpoint write never leaves a corrupted partial on disk.
  * SIGINT/SIGTERM is caught: the in-progress task completes, the current
    partial is flushed, and the function returns (caller signaled by absence
    of the final ``<cache>.npz``). Re-running the same command resumes from
    that partial without re-doing finished tasks.
  * Per-task seed = ``label_seed_base + i`` is threaded into
    ``build_labels_for_one_task`` so a resumed task produces the same labels
    as in the original (uninterrupted) run — i.e. resume is bit-exact.

Storage format (NPZ, fixed-shape padded for vectorization):
    cs_p0                (N, 3)        task spec
    cs_line_dir          (N, 3)
    cs_n_target          (N, 3)
    q0_seeds             (N, 7)        random feasible seed
    labels_q0            (N, k, 7)     padded: unused slots = q0_seed[i]
    labels_L_clean       (N, k)        padded with NaN
    labels_L_robust_mean (N, k)        padded with NaN
    labels_L_robust_min  (N, k)        padded with NaN
    L_seed               (N,)
    n_labels             (N,)          actual label count, 1 ≤ n_labels ≤ k
    status               (N,) U16      ASCII string array
    fallback_used        (N,) bool

Diagnostics + run config go to a sibling JSON file.

Cache key: md5 of all hyperparameters (including module-7 thresholds). Same
key → identical dataset, so caller hits the on-disk cache and skips work.

Incremental checkpoints: every ``checkpoint_interval`` c's, we serialize the
partial dataset under ``<out>.partial-<i>.npz``. On resume, we load the
latest partial and continue from there. Lets a crashed 8-hour run pick up
without restarting from zero.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import NSRLBatchedEnv

from Yuan.seed_selection.label_builder import (
    build_labels_for_one_task,
    MAX_BRANCHES_PERSIST,
)


def cache_key(hyperparams: dict) -> str:
    """Deterministic 10-char md5 of a sorted-json hyperparameters dict."""
    blob = json.dumps(hyperparams, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def _partial_paths(out_path: Path) -> list[Path]:
    """All <out>.partial-<i>.npz files matching the given final path."""
    pattern = re.compile(rf"^{re.escape(out_path.stem)}\.partial-(\d+)\.npz$")
    out_dir = out_path.parent
    if not out_dir.exists():
        return []
    matches = []
    for p in out_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    matches.sort()
    return [p for _, p in matches]


def _save_partial(out_path: Path, i_end: int, rows: list[dict],
                  hyperparams: dict, k: int, K_prime: int = 6) -> None:
    """Atomically serialize ``rows[0:i_end]`` to ``<out>.partial-<i_end>.npz``.

    Writes to a sibling ``.tmp`` file first, then ``os.replace`` to the final
    name. ``os.replace`` is atomic on POSIX, so even SIGKILL during the write
    will leave either the previous partial or no partial — never a torn file.
    """
    save_path = out_path.parent / f"{out_path.stem}.partial-{i_end}.npz"
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    arrs = _rows_to_arrays(rows, k=k, K_prime=K_prime)
    # Open the file ourselves so np.savez treats it as a file-object and
    # doesn't append ".npz" to the suffix (which would corrupt the rename).
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrs, hyperparams_json=np.array(
            json.dumps(hyperparams, sort_keys=True, default=str)))
    os.replace(tmp_path, save_path)


SEED_STRENGTH_THRESHOLD = 0.20   # L_seed below this → "weak" (used only as a TAG, not a filter)


def _default_extras(k: int, K_prime: int) -> dict:
    """Default-valued extras dict, matching label_builder's `extras` schema.
    Used for error rows or other paths that didn't populate extras."""
    mb = MAX_BRANCHES_PERSIST
    return {
        "top_Kprime_q":             np.full((K_prime, 7), np.nan, dtype=np.float32),
        "top_Kprime_L_clean":       np.full((K_prime,),   np.nan, dtype=np.float32),
        "top_Kprime_valid_mask":    np.zeros((K_prime,),  dtype=bool),
        "top_Kprime_branch_ids":    np.full((K_prime,),   -1, dtype=np.int32),
        "n_candidates_total":       0,
        "n_branches":               0,
        "branch_sizes":             np.full((mb,), -1, dtype=np.int32),
        "branch_L_distribution":    np.full((mb, 5), np.nan, dtype=np.float32),
        "branch_q_centroids":       np.full((mb, 7), np.nan, dtype=np.float32),
        "q0_seed_branch_id":        -1,
        "q0_seed_branch_rank_by_L": -1,
        "q0_seed_term_reason":      "unknown",
        "q0_seed_n_steps":          -1,
        "q0_seed_max_q_norm":       float("nan"),
        "labels_term_reason":       np.array(["unknown"] * k, dtype="<U16"),
        "labels_n_steps":           np.full((k,), -1, dtype=np.int32),
        "labels_max_q_norm":        np.full((k,), np.nan, dtype=np.float32),
        "cone_ik_n_attempts":       0,
        "cone_ik_n_successes":      0,
        "walk_steps_per_branch":    np.full((mb,), -1, dtype=np.int32),
        "walk_branch_closed":       np.zeros((mb,), dtype=bool),
        "time_cone_ik_sec":         float("nan"),
        "time_smm_walk_sec":        float("nan"),
        "time_rollout_sec":         float("nan"),
        "time_robust_filter_sec":   float("nan"),
        "time_total_sec":           float("nan"),
    }


def _rows_to_arrays(rows: list[dict], *, k: int, K_prime: int = 6) -> dict:
    """Pack a list of per-c result dicts into the NPZ-friendly arrays."""
    N = len(rows)
    mb = MAX_BRANCHES_PERSIST
    cs_p0 = np.zeros((N, 3), dtype=np.float32)
    cs_d = np.zeros((N, 3), dtype=np.float32)
    cs_n = np.zeros((N, 3), dtype=np.float32)
    q0_seeds = np.zeros((N, 7), dtype=np.float32)
    labels_q0 = np.zeros((N, k, 7), dtype=np.float32)
    labels_Lc = np.full((N, k), np.nan, dtype=np.float32)
    labels_Lrm = np.full((N, k), np.nan, dtype=np.float32)
    labels_Lri = np.full((N, k), np.nan, dtype=np.float32)
    L_seed = np.zeros(N, dtype=np.float32)
    n_labels = np.zeros(N, dtype=np.int32)
    status = np.empty(N, dtype="<U16")
    fallback = np.zeros(N, dtype=bool)
    seed_strength = np.empty(N, dtype="<U8")  # "weak" or "strong"

    # Day-6/7 extras (per-task analytics / diagnostics).
    top_Kprime_q             = np.full((N, K_prime, 7), np.nan, dtype=np.float32)
    top_Kprime_L_clean       = np.full((N, K_prime),    np.nan, dtype=np.float32)
    top_Kprime_valid_mask    = np.zeros((N, K_prime),   dtype=bool)
    top_Kprime_branch_ids    = np.full((N, K_prime),    -1, dtype=np.int32)
    n_candidates_total       = np.zeros((N,), dtype=np.int32)
    n_branches_arr           = np.zeros((N,), dtype=np.int32)
    branch_sizes             = np.full((N, mb), -1, dtype=np.int32)
    branch_L_distribution    = np.full((N, mb, 5), np.nan, dtype=np.float32)
    branch_q_centroids       = np.full((N, mb, 7), np.nan, dtype=np.float32)
    q0_seed_branch_id        = np.full((N,), -1, dtype=np.int32)
    q0_seed_branch_rank_by_L = np.full((N,), -1, dtype=np.int32)
    q0_seed_term_reason      = np.array(["unknown"] * N, dtype="<U16")
    q0_seed_n_steps          = np.full((N,), -1, dtype=np.int32)
    q0_seed_max_q_norm       = np.full((N,), np.nan, dtype=np.float32)
    labels_term_reason       = np.array([["unknown"] * k for _ in range(N)], dtype="<U16")
    labels_n_steps           = np.full((N, k), -1, dtype=np.int32)
    labels_max_q_norm        = np.full((N, k), np.nan, dtype=np.float32)
    cone_ik_n_attempts       = np.zeros((N,), dtype=np.int32)
    cone_ik_n_successes      = np.zeros((N,), dtype=np.int32)
    walk_steps_per_branch    = np.full((N, mb), -1, dtype=np.int32)
    walk_branch_closed       = np.zeros((N, mb), dtype=bool)
    time_cone_ik_sec         = np.full((N,), np.nan, dtype=np.float32)
    time_smm_walk_sec        = np.full((N,), np.nan, dtype=np.float32)
    time_rollout_sec         = np.full((N,), np.nan, dtype=np.float32)
    time_robust_filter_sec   = np.full((N,), np.nan, dtype=np.float32)
    time_total_sec           = np.full((N,), np.nan, dtype=np.float32)

    for i, row in enumerate(rows):
        cs_p0[i] = row["c_np"]["p0"]
        cs_d[i] = row["c_np"]["line_dir"]
        cs_n[i] = row["c_np"]["n_target"]
        q0_seeds[i] = row["q0_seed_np"]
        nl = int(row["out"]["n_labels"])
        n_labels[i] = nl
        labels_q0_np = row["out"]["labels_q0"].detach().cpu().numpy().astype(np.float32)
        labels_q0[i, :nl] = labels_q0_np
        # Pad remaining slots with q0_seed (so model still gets a valid q
        # if it accidentally reads a padded slot — defense in depth).
        if nl < k:
            labels_q0[i, nl:] = q0_seeds[i]
        labels_Lc[i, :nl] = np.asarray(row["out"]["labels_L_clean"], dtype=np.float32)
        labels_Lrm[i, :nl] = np.asarray(row["out"]["labels_L_robust_mean"], dtype=np.float32)
        labels_Lri[i, :nl] = np.asarray(row["out"]["labels_L_robust_min"], dtype=np.float32)
        L_seed[i] = float(row["out"]["L_seed"])
        status[i] = str(row["out"]["status"])
        fallback[i] = bool(row["out"]["fallback_used"])
        seed_strength[i] = "weak" if L_seed[i] < SEED_STRENGTH_THRESHOLD else "strong"

        ex = row["out"].get("extras") or _default_extras(k, K_prime)
        top_Kprime_q[i]             = ex["top_Kprime_q"]
        top_Kprime_L_clean[i]       = ex["top_Kprime_L_clean"]
        top_Kprime_valid_mask[i]    = ex["top_Kprime_valid_mask"]
        top_Kprime_branch_ids[i]    = ex["top_Kprime_branch_ids"]
        n_candidates_total[i]       = int(ex["n_candidates_total"])
        n_branches_arr[i]           = int(ex["n_branches"])
        branch_sizes[i]             = ex["branch_sizes"]
        branch_L_distribution[i]    = ex["branch_L_distribution"]
        branch_q_centroids[i]       = ex["branch_q_centroids"]
        q0_seed_branch_id[i]        = int(ex["q0_seed_branch_id"])
        q0_seed_branch_rank_by_L[i] = int(ex["q0_seed_branch_rank_by_L"])
        q0_seed_term_reason[i]      = str(ex["q0_seed_term_reason"])
        q0_seed_n_steps[i]          = int(ex["q0_seed_n_steps"])
        q0_seed_max_q_norm[i]       = float(ex["q0_seed_max_q_norm"])
        labels_term_reason[i]       = ex["labels_term_reason"]
        labels_n_steps[i]           = ex["labels_n_steps"]
        labels_max_q_norm[i]        = ex["labels_max_q_norm"]
        cone_ik_n_attempts[i]       = int(ex["cone_ik_n_attempts"])
        cone_ik_n_successes[i]      = int(ex["cone_ik_n_successes"])
        walk_steps_per_branch[i]    = ex["walk_steps_per_branch"]
        walk_branch_closed[i]       = ex["walk_branch_closed"]
        time_cone_ik_sec[i]         = float(ex["time_cone_ik_sec"])
        time_smm_walk_sec[i]        = float(ex["time_smm_walk_sec"])
        time_rollout_sec[i]         = float(ex["time_rollout_sec"])
        time_robust_filter_sec[i]   = float(ex["time_robust_filter_sec"])
        time_total_sec[i]           = float(ex["time_total_sec"])

    return {
        "cs_p0": cs_p0,
        "cs_line_dir": cs_d,
        "cs_n_target": cs_n,
        "q0_seeds": q0_seeds,
        "labels_q0": labels_q0,
        "labels_L_clean": labels_Lc,
        "labels_L_robust_mean": labels_Lrm,
        "labels_L_robust_min": labels_Lri,
        "L_seed": L_seed,
        "seed_strength": seed_strength,
        "n_labels": n_labels,
        "status": status,
        "fallback_used": fallback,
        # Day-6/7 extras:
        "top_Kprime_q": top_Kprime_q,
        "top_Kprime_L_clean": top_Kprime_L_clean,
        "top_Kprime_valid_mask": top_Kprime_valid_mask,
        "top_Kprime_branch_ids": top_Kprime_branch_ids,
        "n_candidates_total": n_candidates_total,
        "n_branches_per_task": n_branches_arr,
        "branch_sizes": branch_sizes,
        "branch_L_distribution": branch_L_distribution,
        "branch_q_centroids": branch_q_centroids,
        "q0_seed_branch_id": q0_seed_branch_id,
        "q0_seed_branch_rank_by_L": q0_seed_branch_rank_by_L,
        "q0_seed_term_reason": q0_seed_term_reason,
        "q0_seed_n_steps": q0_seed_n_steps,
        "q0_seed_max_q_norm": q0_seed_max_q_norm,
        "labels_term_reason": labels_term_reason,
        "labels_n_steps": labels_n_steps,
        "labels_max_q_norm": labels_max_q_norm,
        "cone_ik_n_attempts": cone_ik_n_attempts,
        "cone_ik_n_successes": cone_ik_n_successes,
        "walk_steps_per_branch": walk_steps_per_branch,
        "walk_branch_closed": walk_branch_closed,
        "time_cone_ik_sec": time_cone_ik_sec,
        "time_smm_walk_sec": time_smm_walk_sec,
        "time_rollout_sec": time_rollout_sec,
        "time_robust_filter_sec": time_robust_filter_sec,
        "time_total_sec": time_total_sec,
    }


def _compute_norm_stats(arrs: dict) -> dict:
    """Compute per-axis mean/std of the conditioning inputs and labels_q0.
    Stats are over ALL entries (regardless of status); downstream consumers can
    filter by status before loading. Stored in the NPZ as scalar-shape arrays
    so reload + reuse at inference time is one-liner."""
    p0_m = arrs["cs_p0"].mean(axis=0).astype(np.float32)
    p0_s = (arrs["cs_p0"].std(axis=0) + 1e-8).astype(np.float32)
    d_m  = arrs["cs_line_dir"].mean(axis=0).astype(np.float32)
    d_s  = (arrs["cs_line_dir"].std(axis=0) + 1e-8).astype(np.float32)
    n_m  = arrs["cs_n_target"].mean(axis=0).astype(np.float32)
    n_s  = (arrs["cs_n_target"].std(axis=0) + 1e-8).astype(np.float32)
    # q0 stats over VALID label slots (mask of `labels_q0` against n_labels).
    nl = arrs["n_labels"]
    k = arrs["labels_q0"].shape[1]
    valid = np.arange(k)[None, :] < nl[:, None]   # (N, k)
    q_flat = arrs["labels_q0"][valid]              # (sum_nl, 7)
    if q_flat.shape[0] == 0:
        q_m = np.zeros(7, dtype=np.float32); q_s = np.ones(7, dtype=np.float32)
    else:
        q_m = q_flat.mean(axis=0).astype(np.float32)
        q_s = (q_flat.std(axis=0) + 1e-8).astype(np.float32)
    return {
        "norm_p0_mean": p0_m, "norm_p0_std": p0_s,
        "norm_line_dir_mean": d_m, "norm_line_dir_std": d_s,
        "norm_n_target_mean": n_m, "norm_n_target_std": n_s,
        "norm_labels_q0_mean": q_m, "norm_labels_q0_std": q_s,
    }


def _load_partial_rows(path: Path) -> list[dict]:
    """Inverse of _rows_to_arrays — returns the row list (with torch tensors
    restored on CPU; caller can move to device if needed). Tolerant to old
    partials that lack the Day-6/7 extras: missing fields become defaults."""
    z = np.load(path, allow_pickle=False)
    N = int(z["L_seed"].shape[0])
    k = int(z["labels_q0"].shape[1])
    # K_prime varies; read from file if available, else default 6.
    K_prime = int(z["top_Kprime_q"].shape[1]) if "top_Kprime_q" in z.files else 6
    has_extras = "top_Kprime_q" in z.files
    rows = []
    for i in range(N):
        nl = int(z["n_labels"][i])
        if has_extras:
            extras = {
                "top_Kprime_q":             z["top_Kprime_q"][i].copy(),
                "top_Kprime_L_clean":       z["top_Kprime_L_clean"][i].copy(),
                "top_Kprime_valid_mask":    z["top_Kprime_valid_mask"][i].copy(),
                "top_Kprime_branch_ids":    z["top_Kprime_branch_ids"][i].copy(),
                "n_candidates_total":       int(z["n_candidates_total"][i]),
                "n_branches":               int(z["n_branches_per_task"][i]),
                "branch_sizes":             z["branch_sizes"][i].copy(),
                "branch_L_distribution":    z["branch_L_distribution"][i].copy(),
                "branch_q_centroids":       z["branch_q_centroids"][i].copy(),
                "q0_seed_branch_id":        int(z["q0_seed_branch_id"][i]),
                "q0_seed_branch_rank_by_L": int(z["q0_seed_branch_rank_by_L"][i]),
                "q0_seed_term_reason":      str(z["q0_seed_term_reason"][i]),
                "q0_seed_n_steps":          int(z["q0_seed_n_steps"][i]),
                "q0_seed_max_q_norm":       float(z["q0_seed_max_q_norm"][i]),
                "labels_term_reason":       z["labels_term_reason"][i].copy(),
                "labels_n_steps":           z["labels_n_steps"][i].copy(),
                "labels_max_q_norm":        z["labels_max_q_norm"][i].copy(),
                "cone_ik_n_attempts":       int(z["cone_ik_n_attempts"][i]),
                "cone_ik_n_successes":      int(z["cone_ik_n_successes"][i]),
                "walk_steps_per_branch":    z["walk_steps_per_branch"][i].copy(),
                "walk_branch_closed":       z["walk_branch_closed"][i].copy(),
                "time_cone_ik_sec":         float(z["time_cone_ik_sec"][i]),
                "time_smm_walk_sec":        float(z["time_smm_walk_sec"][i]),
                "time_rollout_sec":         float(z["time_rollout_sec"][i]),
                "time_robust_filter_sec":   float(z["time_robust_filter_sec"][i]),
                "time_total_sec":           float(z["time_total_sec"][i]),
            }
        else:
            extras = _default_extras(k, K_prime)
        rows.append({
            "c_np": {
                "p0": z["cs_p0"][i].copy(),
                "line_dir": z["cs_line_dir"][i].copy(),
                "n_target": z["cs_n_target"][i].copy(),
            },
            "q0_seed_np": z["q0_seeds"][i].copy(),
            "out": {
                "labels_q0": torch.as_tensor(z["labels_q0"][i, :nl].copy()),
                "labels_L_clean": list(map(float, z["labels_L_clean"][i, :nl])),
                "labels_L_robust_mean": list(map(float, z["labels_L_robust_mean"][i, :nl])),
                "labels_L_robust_min": list(map(float, z["labels_L_robust_min"][i, :nl])),
                "labels_L_perturbed": [[] for _ in range(nl)],   # not persisted
                "L_seed": float(z["L_seed"][i]),
                "n_labels": nl,
                "status": str(z["status"][i]),
                "fallback_used": bool(z["fallback_used"][i]),
                "diagnostics": {},                                  # not persisted
                "extras": extras,
            },
        })
    return rows


def build_dataset(
    cs: list[dict],
    q0_seeds: list[torch.Tensor],
    *,
    kin: BatchedFR3Kinematics,
    collision: FR3SphereCollision,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    out_dir: str | Path,
    cache_name: str | None = None,
    hyperparams: dict | None = None,
    label_kwargs: dict | None = None,
    checkpoint_interval: int = 100,
    label_seed_base: int = 0,
    verbose: bool = True,
) -> Path | None:
    """Build an SMM-aware label dataset and save it under ``out_dir``.

    Args:
        cs: list of task dicts (length N).
        q0_seeds: list of (7,) torch tensors (length N), the random
            feasible seed used to construct each c.
        kin, collision, env, controller: shared FR3 setup; ``env.n_envs``
            should be 32-128 for good rollout batching.
        out_dir: directory to write the dataset + checkpoints + meta.json.
        cache_name: filename stem (defaults to ``"dataset_<md5>"``).
        hyperparams: dict of everything that determines the dataset content.
            Goes into the md5 cache key + meta.json. Caller is responsible
            for putting all label_kwargs that affect content into this dict.
        label_kwargs: passed through to ``build_labels_for_one_task``.
        checkpoint_interval: every N c's, save a partial NPZ (resume-safe).
        label_seed_base: per-task numpy seed = ``label_seed_base + i``. Two
            chunks must use disjoint ranges to avoid resampling the same RNG
            state for different tasks (in practice, set this to the same
            value as the line_distribution start_seed for the chunk).

    Returns:
        Path to the final NPZ on completion, or ``None`` if the run was
        interrupted by SIGINT/SIGTERM (in that case a partial NPZ is on disk
        and a re-run with the same cache_name will resume).
    """
    if len(cs) != len(q0_seeds):
        raise ValueError(f"cs has {len(cs)} entries but q0_seeds has {len(q0_seeds)}")
    N = len(cs)
    label_kwargs = dict(label_kwargs or {})
    hyperparams = dict(hyperparams or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(label_kwargs.get("k", 3))
    K_prime = int(label_kwargs.get("K_prime", 6))

    if cache_name is None:
        cache_name = f"dataset_{cache_key(hyperparams)}"
    out_path = out_dir / f"{cache_name}.npz"
    meta_path = out_dir / f"{cache_name}.meta.json"

    if out_path.exists():
        if verbose:
            print(f"[build_dataset] cache hit: {out_path}")
        return out_path

    # Resume from latest partial if any.
    partials = _partial_paths(out_path)
    rows: list[dict] = []
    start_idx = 0
    if partials:
        last = partials[-1]
        rows = _load_partial_rows(last)
        # Filename is "<stem>.partial-<i_end>.npz" → i_end is the count of rows saved.
        m = re.match(rf"^{re.escape(out_path.stem)}\.partial-(\d+)\.npz$", last.name)
        start_idx = int(m.group(1))
        if verbose:
            print(f"[build_dataset] resume from {last.name}: {start_idx}/{N} done")

    # Per-status counters for the run summary.
    counters: dict[str, int] = {}
    errors: list[dict] = []

    # Graceful-shutdown flag. SIGINT/SIGTERM sets it; the main loop polls it
    # after each task. We DO NOT abort mid-task — that would risk torch / env
    # state corruption — instead the current task completes then we flush a
    # partial and return. Previously installed handlers are restored on exit.
    shutdown_state = {"requested": False, "signum": 0}

    def _handler(signum, frame):
        shutdown_state["requested"] = True
        shutdown_state["signum"] = int(signum)
        # Don't print from inside the handler (re-entrant tqdm); print on exit.

    prev_int = signal.signal(signal.SIGINT, _handler)
    prev_term = signal.signal(signal.SIGTERM, _handler)

    # `seed` may already be in label_kwargs (e.g. caller passed a fixed value);
    # we override with the deterministic per-task seed below.
    label_kwargs.pop("seed", None)

    t_start = time.time()
    pbar = tqdm(range(start_idx, N), desc="build_dataset",
                initial=start_idx, total=N, disable=not verbose)
    interrupted = False
    for i in pbar:
        if shutdown_state["requested"]:
            interrupted = True
            if verbose:
                tqdm.write(f"  [shutdown] signal {shutdown_state['signum']} received; flushing partial at i={i}/{N}.")
            break
        c = cs[i]
        q0_seed = q0_seeds[i]
        task_seed = int(label_seed_base) + int(i)
        try:
            out = build_labels_for_one_task(
                c, q0_seed,
                kin=kin, collision=collision,
                env=env, controller=controller,
                seed=task_seed,
                **label_kwargs,
            )
        except Exception as e:
            # Per the user's spec: single-c failures should not kill the
            # whole pipeline. Record + skip via a placeholder row that
            # marks status as 'infeasible' so downstream filtering excludes it.
            errors.append({"i": i, "err": repr(e)})
            if verbose:
                tqdm.write(f"  [error] task {i}: {e!r}")
            out = {
                "labels_q0": q0_seed.unsqueeze(0).clone(),
                "labels_L_clean": [float("nan")],
                "labels_L_robust_mean": [float("nan")],
                "labels_L_robust_min": [float("nan")],
                "labels_L_perturbed": [[]],
                "L_seed": float("nan"),
                "n_labels": 1,
                "status": "infeasible",
                "fallback_used": True,
                "diagnostics": {"error": repr(e)},
            }
        counters[out["status"]] = counters.get(out["status"], 0) + 1
        rows.append({
            "c_np": {
                "p0": c["p0"].detach().cpu().numpy().astype(np.float32),
                "line_dir": c["line_dir"].detach().cpu().numpy().astype(np.float32),
                "n_target": c["n_target"].detach().cpu().numpy().astype(np.float32),
            },
            "q0_seed_np": q0_seed.detach().cpu().numpy().astype(np.float32),
            "out": out,
        })
        pbar.set_postfix({k: v for k, v in counters.items()})
        # Checkpoint
        if ((i + 1) % checkpoint_interval == 0) or (i + 1 == N):
            _save_partial(out_path, i + 1, rows, hyperparams, k=k, K_prime=K_prime)

    # Restore previous signal handlers regardless of how we exit the loop.
    signal.signal(signal.SIGINT, prev_int)
    signal.signal(signal.SIGTERM, prev_term)

    if interrupted:
        # Flush whatever's in rows (covers the rows since the last checkpoint
        # that were committed by build_labels_for_one_task before the signal).
        if len(rows) > 0:
            _save_partial(out_path, len(rows), rows, hyperparams, k=k, K_prime=K_prime)
        if verbose:
            print(f"[build_dataset] interrupted at i={len(rows)}/{N}; "
                  f"partial saved. Re-run same command to resume.")
        return None

    # Final save: rename last partial to the canonical path.
    final_partials = _partial_paths(out_path)
    if not final_partials:
        # Edge case: N==0 or first iteration crashed before checkpoint.
        arrs = _rows_to_arrays(rows, k=k, K_prime=K_prime)
        norm_stats = _compute_norm_stats(arrs)
        np.savez(out_path, **arrs, **norm_stats, hyperparams_json=np.array(
            json.dumps(hyperparams, sort_keys=True, default=str)))
    else:
        final_partials[-1].rename(out_path)
        # Clean up older partials (keep just the final).
        for p in final_partials[:-1]:
            try:
                p.unlink()
            except OSError:
                pass
        # Re-save with norm_stats appended (per-axis mean/std over all entries).
        _z = np.load(out_path, allow_pickle=False)
        merged = {k_: _z[k_] for k_ in _z.files}
        norm_stats = _compute_norm_stats(merged)
        merged.update(norm_stats)
        np.savez(out_path, **merged)

    elapsed = time.time() - t_start
    meta = {
        "n_tasks": N,
        "k": k,
        "hyperparams": hyperparams,
        "label_kwargs": {kk: str(vv) for kk, vv in label_kwargs.items()},
        "status_counts": counters,
        "n_errors": len(errors),
        "errors": errors[:50],   # cap to avoid bloating meta
        "elapsed_seconds": elapsed,
        "tasks_per_second": N / elapsed if elapsed > 0 else float("nan"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    if verbose:
        print(f"[build_dataset] done: {N} tasks in {elapsed:.1f}s "
              f"({N/elapsed:.2f}/s)  status={counters}  errors={len(errors)}")
        print(f"  dataset: {out_path}")
        print(f"  meta:    {meta_path}")
    return out_path
