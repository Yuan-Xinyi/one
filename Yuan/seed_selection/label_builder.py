"""Module 7: per-c label builder.

End-to-end: given a task ``c`` and the random feasible seed ``q0_seed`` that
was used to construct it, enumerate SMM candidates → score by clean rollout
→ filter by robustness → return top-k labels for diffusion training.

Pipeline:
    1. Cone-IK enumerate (5-DOF cone, broad search across branches)
    2. Prepend q0_seed; project onto strict 6-DOF target (z = n_target
       exactly) via `project_and_filter`
    3. `enumerate_branches` walks the 1D SMM around each seed and groups
       nearby seeds into branches
    4. Sample ``sample_per_branch`` arc-length-uniform points per branch
    5. Score each sample with `rollout_one` → L_clean
    6. Drop L < L_min_abs (noise floor)
    7. Sort by L_clean desc, take top-K'
    8. `filter_robust_candidates` → keep those with L_robust ≥ τ × L_clean
    9. Top-k of the filtered = labels
   10. If filter eliminates all candidates → fallback to q0_seed
   11. Status flag + diagnostics

q0_seed is ALWAYS forced into the candidate pool (Step 2). This is the
safety net the user emphasized: even if cone-IK misses the seed's branch,
the seed itself is evaluated and may end up as the top label.

L_seed is reported separately so downstream eval can compute the headline
relative-improvement metric: ``(L_pred - L_seed) / (L_max - L_seed)``.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEFAULT_H, DEDUP_RAD, JOINT_MARGIN,
    enumerate_branches, project_and_filter,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import NSRLBatchedEnv, TERM_NAMES
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics


# Schema caps for the persisted per-task metadata (chosen so the cap is well
# above empirically observed values: pilot_1k shows n_branches max 5).
MAX_BRANCHES_PERSIST = 10


def _term_str(t_int: int) -> str:
    return TERM_NAMES.get(int(t_int), f"int{int(t_int)}")

from Yuan.seed_selection.batched_rollout import batched_rollout_many
from Yuan.seed_selection.cone_ik import cone_constrained_ik_enumerate
from Yuan.seed_selection.robustness import filter_robust_candidates
from Yuan.seed_selection.rollout import DEFAULT_TARGET_DISTANCE_M, rollout_one


# Status taxonomy:
STATUS_KEPT = "kept"                # normal: ≥1 label found, max(L) ≥ L_min_acceptable
STATUS_EDGE = "edge"                # very few branches or candidates surfaced
STATUS_INFEASIBLE = "infeasible"    # NO candidate had L ≥ L_min_abs (task essentially impossible)
STATUS_LOW_QUALITY = "low_quality"  # labels exist but max(L) < L_min_acceptable (weak task)
# NB: STATUS_LOW_SEED + L_seed_min pre-filter were tried in Day-5 pilot v2 and
# REMOVED. Pre-filtering low-L_seed tasks dropped kept-only ratio median from
# 1.29 to 1.12 because the SMM-aware labels' biggest absolute gains are exactly
# on weak-seed tasks (L_seed ∈ [0.10, 0.20): ratio med 1.7-2.8, abs gain 0.13-0.23m).


def _build_R_target_strict(n_target_np: np.ndarray,
                            line_dir_np: np.ndarray) -> np.ndarray:
    """Build the 6-DOF R_target with TCP_z = n_target and TCP_x derived from
    line_dir via Gram-Schmidt. ``line_dir`` must already be ⊥ ``n_target``
    (true by construction in our task spec), so the x column is essentially
    line_dir itself."""
    z = n_target_np / max(np.linalg.norm(n_target_np), 1e-12)
    x = line_dir_np - (line_dir_np @ z) * z
    n_x = np.linalg.norm(x)
    if n_x < 1e-6:
        # Pathological: line_dir nearly parallel to n_target. Fall back to
        # world x or y, whichever is more ⊥ to z.
        wx = np.array([1.0, 0.0, 0.0], dtype=z.dtype)
        wy = np.array([0.0, 1.0, 0.0], dtype=z.dtype)
        x_seed = wx if abs(wx @ z) < abs(wy @ z) else wy
        x = x_seed - (x_seed @ z) * z
        n_x = np.linalg.norm(x)
    x = x / max(n_x, 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1).astype(np.float32)
    return R


def _arc_lengths(traj: np.ndarray) -> np.ndarray:
    """Cumulative arc length (joint-space Euclidean) along a (T, 7) trajectory.
    Returns (T,) with arc[0] = 0."""
    if traj.shape[0] <= 1:
        return np.zeros(traj.shape[0], dtype=np.float32)
    step = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)


def _sample_arc_uniform(traj: np.ndarray, n: int) -> np.ndarray:
    """Sample ``n`` points uniformly along the arc-length parameterization of
    a (T, 7) trajectory. Includes both endpoints when n ≥ 2."""
    T = traj.shape[0]
    if T == 0:
        return traj
    if n >= T or n <= 1:
        # Trivial: return everything (or just first point).
        return traj if n >= T else traj[:1]
    arc = _arc_lengths(traj)
    total = float(arc[-1])
    if total <= 1e-9:
        # Degenerate: branch is a single point repeated.
        return traj[:1]
    targets = np.linspace(0.0, total, n)
    out_idx = np.searchsorted(arc, targets, side="left").clip(0, T - 1)
    return traj[out_idx]


def build_labels_for_one_task(
    c: dict[str, torch.Tensor],
    q0_seed: torch.Tensor,
    *,
    kin: BatchedFR3Kinematics,
    collision: FR3SphereCollision,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    # Cone-IK enumeration:
    cone_angle_deg: float = 5.0,
    n_orientations: int = 10,
    n_ik_restarts: int = 5,
    ik_dedup_rad: float = 0.08,
    # SMM walk:
    walk_h: float = DEFAULT_H,
    smm_joint_margin: float = JOINT_MARGIN,
    smm_dedup_rad: float = DEDUP_RAD,
    sample_per_branch: int = 5,
    # Robust filter (Day-5 pilot showed: at perturb 2-3°/5-8mm, classical_nullspace
    # is essentially insensitive — 0/134 labels were "fragile" by L_robust/L_clean
    # < 0.7. Default tau_robust=0.0 disables filtering; parameter hook is preserved
    # in case a stronger perturbation regime later differentiates candidates).
    k: int = 3,
    K_prime: int = 6,
    tau_robust: float = 0.0,
    n_perturb: int = 4,
    perturb_d_deg: float = 5.0,
    perturb_n_deg: float = 5.0,
    perturb_p0_mm: float = 10.0,
    # Quality thresholds:
    L_min_abs: float = 0.05,
    L_min_acceptable: float = 0.30,
    edge_branch_threshold: int = 1,   # ≤ this many branches → 'edge'
    # Misc:
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M,
    seed: int | None = None,
    verbose: bool = False,
    return_all_candidates: bool = False,
) -> dict:
    """Build a top-k label set for one task (c, q0_seed).

    Returns dict with:
        labels_q0           torch.Tensor (n_labels, 7) — kept seeds (≤ k)
        labels_L_clean      list[float]
        labels_L_robust_mean list[float]
        labels_L_robust_min list[float]
        labels_L_perturbed  list[list[float]]  — full per-perturb L for each label
        L_seed              float | None       — clean rollout L of q0_seed
        n_labels            int                — len(labels_q0); may be 1 (fallback) or 0 (infeasible)
        status              str (kept / edge / infeasible / low_quality)
        fallback_used       bool — True if labels = [q0_seed] due to empty robust filter
        diagnostics         dict — see below
    """
    device = kin.device
    rng = np.random.default_rng(seed)

    p0_np = c["p0"].detach().cpu().numpy().astype(np.float32)
    n_np = c["n_target"].detach().cpu().numpy().astype(np.float32)
    d_np = c["line_dir"].detach().cpu().numpy().astype(np.float32)
    R_tgt_np = _build_R_target_strict(n_np, d_np)
    q0_seed_np = q0_seed.detach().cpu().numpy().astype(np.float32)

    diag: dict = {}
    q0_seed_t = q0_seed.to(device=device, dtype=kin.dtype)

    # Persisted-by-dataset_builder per-task fields. Filled progressively;
    # any field still NaN/-1 at return time means "phase didn't reach here".
    extras: dict = {
        "top_Kprime_q":               np.full((K_prime, 7), np.nan, dtype=np.float32),
        "top_Kprime_L_clean":         np.full((K_prime,),   np.nan, dtype=np.float32),
        "top_Kprime_valid_mask":      np.zeros((K_prime,),  dtype=bool),
        "top_Kprime_branch_ids":      np.full((K_prime,),   -1, dtype=np.int32),
        "n_candidates_total":         0,
        "n_branches":                 0,
        "branch_sizes":               np.full((MAX_BRANCHES_PERSIST,), -1, dtype=np.int32),
        "branch_L_distribution":      np.full((MAX_BRANCHES_PERSIST, 5), np.nan, dtype=np.float32),
        "branch_q_centroids":         np.full((MAX_BRANCHES_PERSIST, 7), np.nan, dtype=np.float32),
        "q0_seed_branch_id":          -1,
        "q0_seed_branch_rank_by_L":   -1,
        "q0_seed_term_reason":        "unknown",
        "q0_seed_n_steps":            -1,
        "q0_seed_max_q_norm":         float("nan"),
        "labels_term_reason":         np.array(["unknown"] * k, dtype="<U16"),
        "labels_n_steps":             np.full((k,), -1, dtype=np.int32),
        "labels_max_q_norm":          np.full((k,), np.nan, dtype=np.float32),
        "cone_ik_n_attempts":         int(n_orientations * n_ik_restarts),
        "cone_ik_n_successes":        0,
        "walk_steps_per_branch":      np.full((MAX_BRANCHES_PERSIST,), -1, dtype=np.int32),
        "walk_branch_closed":         np.zeros((MAX_BRANCHES_PERSIST,), dtype=bool),
        "time_cone_ik_sec":           float("nan"),
        "time_smm_walk_sec":          float("nan"),
        "time_rollout_sec":           float("nan"),
        "time_robust_filter_sec":     0.0,
        "time_total_sec":             float("nan"),
    }
    t_pipeline_start = time.time()

    # ---------- 1. Cone-IK enumeration ----------
    t_step = time.time()
    Q_ik = cone_constrained_ik_enumerate(
        p0=c["p0"], n_target=c["n_target"], line_dir=c["line_dir"],
        kin=kin, collision=collision,
        cone_angle_deg=cone_angle_deg,
        n_orientations=n_orientations,
        n_ik_restarts=n_ik_restarts,
        joint_margin=smm_joint_margin,
        dedup_rad=ik_dedup_rad,
        rng=rng,
    )
    diag["n_cone_ik"] = int(Q_ik.shape[0])
    extras["cone_ik_n_successes"] = int(Q_ik.shape[0])
    extras["time_cone_ik_sec"] = float(time.time() - t_step)
    if verbose:
        print(f"  [labels] cone_ik: {Q_ik.shape[0]} candidates")

    # ---------- 2. Prepend q0_seed, refine to strict 6-DOF ----------
    t_step = time.time()
    Q_seed_pool = np.concatenate([q0_seed_np[None, :],
                                   Q_ik.detach().cpu().numpy().astype(np.float32)],
                                  axis=0)
    lo_np = kin.lmt_lo.detach().cpu().numpy()
    hi_np = kin.lmt_up.detach().cpu().numpy()
    Q_clean = project_and_filter(
        kin, Q_seed_pool, p0_np, R_tgt_np, lo_np, hi_np,
        joint_margin=smm_joint_margin, dedup_rad=smm_dedup_rad, verbose=False)
    diag["n_after_project_filter"] = int(Q_clean.shape[0])
    if verbose:
        print(f"  [labels] project_and_filter: {Q_clean.shape[0]} clean seeds")

    # ---------- 3. Walk SMM branches ----------
    if Q_clean.shape[0] == 0:
        # Couldn't even project q0_seed to strict 6-DOF set. Fallback path.
        branches: list = []
        assigned = np.zeros(0, dtype=np.int32)
    else:
        branches, assigned = enumerate_branches(
            kin, Q_clean, p0_np, R_tgt_np, h=walk_h)
    n_branches = len(branches)
    diag["n_branches"] = n_branches
    diag["branch_lengths"] = [int(b["traj"].shape[0]) for b in branches]
    diag["branch_closed"] = [bool(b["closed"]) for b in branches]
    extras["n_branches"] = int(n_branches)
    cap = min(n_branches, MAX_BRANCHES_PERSIST)
    for bi in range(cap):
        extras["walk_steps_per_branch"][bi] = int(branches[bi]["traj"].shape[0])
        extras["walk_branch_closed"][bi]    = bool(branches[bi]["closed"])
        extras["branch_q_centroids"][bi]    = branches[bi]["traj"].mean(axis=0).astype(np.float32)
    if verbose:
        print(f"  [labels] enumerate_branches: {n_branches} branches "
              f"lengths={diag['branch_lengths']}")

    # q0_seed_branch_id: find Q_clean row closest to q0_seed; use its assigned branch
    # if the match is tight (within 2× the SMM dedup radius). -1 means q0_seed is
    # an isolated point not joined to any walked branch.
    if Q_clean.shape[0] > 0 and n_branches > 0:
        d2seed = np.linalg.norm(Q_clean - q0_seed_np[None, :], axis=1)
        closest = int(d2seed.argmin())
        if float(d2seed[closest]) < 2.0 * smm_dedup_rad:
            extras["q0_seed_branch_id"] = int(assigned[closest])

    # ---------- 4. Sample points per branch ----------
    if n_branches == 0:
        # Pure fallback: only q0_seed as a candidate.
        Q_candidates_np = q0_seed_np[None, :].copy()
        candidate_branch_ids = np.array([-1], dtype=np.int32)  # q0_seed prepended; samples=none
    else:
        samples = [_sample_arc_uniform(b["traj"], sample_per_branch)
                   for b in branches]
        Q_candidates_np = np.concatenate(samples, axis=0)
        candidate_branch_ids = np.concatenate([
            np.full(s.shape[0], bi, dtype=np.int32) for bi, s in enumerate(samples)
        ])
    Q_candidates_np = Q_candidates_np.astype(np.float32)
    # Note: +1 because we ALSO prepend q0_seed to the scoring batch in Step 5.
    diag["n_candidates_to_score"] = int(Q_candidates_np.shape[0]) + 1
    extras["time_smm_walk_sec"] = float(time.time() - t_step)

    # ---------- 5. Score each candidate (batched rollout — Day 4) ----------
    # Prepend q0_seed so the seed is always scored (its branch may not be
    # in the SMM walks if cone_ik missed it AND project_and_filter merged
    # it with another seed by dedup_rad). q0_seed_t was built above in Step 0.
    Q_candidates_t = torch.cat([
        q0_seed_t.unsqueeze(0),
        torch.as_tensor(Q_candidates_np, device=device, dtype=kin.dtype),
    ], dim=0)
    # candidate index 0 is q0_seed (no branch); the rest map to sampled branches.
    candidate_branch_full = np.concatenate([
        np.array([-1], dtype=np.int32),     # q0_seed has no branch yet
        candidate_branch_ids,
    ])
    # Single batched call: N rollouts on (q_i, c) pairs (same c repeated).
    cs_rep = [c] * Q_candidates_t.shape[0]
    t_step = time.time()
    score_res = batched_rollout_many(
        Q_candidates_t, cs_rep,
        env=env, controller=controller,
        target_distance_m=target_distance_m,
    )
    extras["time_rollout_sec"] = float(time.time() - t_step)
    # Optional max_q_norm tracking (batched_rollout fills it when enabled).
    max_q_norm_arr = score_res.get("max_q_norm")
    candidates = [
        {"q0": Q_candidates_t[j],
         "L_clean": float(score_res["L"][j]),
         "term_reason": int(score_res["term_reason"][j]),
         "n_steps": int(score_res["episode_len"][j]),
         "max_q_norm": (float(max_q_norm_arr[j]) if max_q_norm_arr is not None else float("nan")),
         "branch_id": int(candidate_branch_full[j])}
        for j in range(Q_candidates_t.shape[0])
    ]
    # q0_seed is at index 0 by construction → its L_clean is L_seed.
    L_seed = float(score_res["L"][0])
    diag["L_seed"] = float(L_seed)
    extras["n_candidates_total"] = int(len(candidates))
    extras["q0_seed_term_reason"] = _term_str(candidates[0]["term_reason"])
    extras["q0_seed_n_steps"] = int(candidates[0]["n_steps"])
    extras["q0_seed_max_q_norm"] = float(candidates[0]["max_q_norm"])

    # Per-branch L distribution (over the SAMPLED candidates only, not q0_seed).
    if n_branches > 0:
        branch_L_lists: list[list[float]] = [[] for _ in range(n_branches)]
        for cd in candidates[1:]:
            bi = cd["branch_id"]
            if 0 <= bi < n_branches:
                branch_L_lists[bi].append(cd["L_clean"])
        branch_max_L = np.full(n_branches, -np.inf, dtype=np.float32)
        for bi in range(min(n_branches, MAX_BRANCHES_PERSIST)):
            Ls = np.asarray(branch_L_lists[bi], dtype=np.float32)
            extras["branch_sizes"][bi] = int(len(Ls))
            if len(Ls) > 0:
                extras["branch_L_distribution"][bi] = np.array([
                    float(np.min(Ls)),
                    float(np.percentile(Ls, 25)),
                    float(np.median(Ls)),
                    float(np.percentile(Ls, 75)),
                    float(np.max(Ls)),
                ], dtype=np.float32)
                branch_max_L[bi] = float(np.max(Ls))
        # Rank branches by max(L) descending. q0_seed_branch_rank = 0-indexed
        # position of q0_seed_branch_id in this ranking.
        if extras["q0_seed_branch_id"] >= 0 and np.any(np.isfinite(branch_max_L)):
            order = np.argsort(-branch_max_L)  # high-to-low
            rank_of = {int(bi): r for r, bi in enumerate(order)}
            extras["q0_seed_branch_rank_by_L"] = int(rank_of.get(extras["q0_seed_branch_id"], -1))

    if verbose:
        L_arr = [cd["L_clean"] for cd in candidates]
        print(f"  [labels] scored {len(candidates)} candidates  "
              f"L range [{min(L_arr):.3f}, {max(L_arr):.3f}]  L_seed={L_seed:.3f}")

    # ---------- 6. Filter L < L_min_abs ----------
    above_floor = [cd for cd in candidates if cd["L_clean"] >= L_min_abs]
    diag["n_above_L_min_abs"] = len(above_floor)
    if len(above_floor) == 0:
        # Infeasible task: nothing made it past the noise floor.
        if verbose:
            print(f"  [labels] STATUS=infeasible (0/{len(candidates)} above L_min_abs={L_min_abs})")
        # Fill labels_*  for the single fallback q0_seed slot.
        extras["labels_term_reason"][0] = _term_str(candidates[0]["term_reason"])
        extras["labels_n_steps"][0]     = int(candidates[0]["n_steps"])
        extras["labels_max_q_norm"][0]  = float(candidates[0]["max_q_norm"])
        extras["time_total_sec"]        = float(time.time() - t_pipeline_start)
        result = {
            "labels_q0": q0_seed_t.unsqueeze(0).clone(),
            "labels_L_clean": [float(L_seed)],
            "labels_L_robust_mean": [float("nan")],
            "labels_L_robust_min": [float("nan")],
            "labels_L_perturbed": [[]],
            "L_seed": float(L_seed),
            "n_labels": 1,
            "status": STATUS_INFEASIBLE,
            "fallback_used": True,
            "diagnostics": diag,
            "extras": extras,
        }
        if return_all_candidates:
            result["all_candidates_q0"] = Q_candidates_t.detach().cpu().clone()
            result["all_candidates_L"] = [float(x) for x in score_res["L"].tolist()]
        return result

    # Populate top-K' info from the L-sorted above_floor (regardless of which
    # branch runs next — the robust-filter branch starts from the same sort).
    top_for_kprime = sorted(above_floor, key=lambda x: -x["L_clean"])[:K_prime]
    for ki, cd in enumerate(top_for_kprime):
        extras["top_Kprime_q"][ki]            = cd["q0"].detach().cpu().numpy().astype(np.float32)
        extras["top_Kprime_L_clean"][ki]      = float(cd["L_clean"])
        extras["top_Kprime_valid_mask"][ki]   = True
        extras["top_Kprime_branch_ids"][ki]   = int(cd["branch_id"])

    # ---------- 7-8. Sort, top-K', (optional) robust filter ----------
    # tau_robust=0 disables the robust pass: skip the K'·(1+n_perturb)
    # rollouts and take top-K' by L_clean directly. Labels' L_robust_* are
    # marked NaN. The pipeline gains ~75% speed back when the filter is
    # known to be inert (which Day-5 data showed for our perturb regime).
    t_step = time.time()
    if tau_robust > 0.0:
        evaluated = filter_robust_candidates(
            above_floor, c,
            K_prime=K_prime, tau_robust=tau_robust,
            env=env, controller=controller,
            n_perturb=n_perturb,
            perturb_d_deg=perturb_d_deg,
            perturb_n_deg=perturb_n_deg,
            perturb_p0_mm=perturb_p0_mm,
            target_distance_m=target_distance_m,
            seed=seed,
            return_all_evaluations=True,
        )
        passed = [x for x in evaluated if x["passed"]]
    else:
        # Robust filter disabled — just take top-K' by L_clean.
        # Preserve term_reason / n_steps / max_q_norm so labels can be tagged.
        top = top_for_kprime
        evaluated = [{
            "q0": cand["q0"],
            "L_clean": cand["L_clean"],
            "L_robust_mean": float("nan"),
            "L_robust_min": float("nan"),
            "L_robust_std": float("nan"),
            "L_perturbed": [],
            "passed": True,
            "term_reason": cand["term_reason"],
            "n_steps": cand["n_steps"],
            "max_q_norm": cand["max_q_norm"],
            "branch_id": cand["branch_id"],
        } for cand in top]
        passed = list(evaluated)
    extras["time_robust_filter_sec"] = float(time.time() - t_step) if tau_robust > 0.0 else 0.0
    diag["n_evaluated_robust"] = len(evaluated)
    diag["n_passed_robust"] = int(sum(1 for x in evaluated if x["passed"]))
    diag["robust_filter_active"] = bool(tau_robust > 0.0)
    if verbose:
        active = "on" if tau_robust > 0.0 else "off"
        print(f"  [labels] robust filter ({active}, τ={tau_robust}): "
              f"{diag['n_passed_robust']}/{len(evaluated)} passed (K'={K_prime})")

    # ---------- 9-10. Top-k labels, or fallback ----------
    fallback_used = False
    if len(passed) == 0:
        # Robust filter eliminated everything. Fall back to q0_seed.
        if verbose:
            print(f"  [labels] fallback to q0_seed (no candidate passed robust filter)")
        labels_q0 = q0_seed_t.unsqueeze(0).clone()
        labels_L_clean = [float(L_seed)]
        labels_L_robust_mean = [float("nan")]
        labels_L_robust_min = [float("nan")]
        labels_L_perturbed = [[]]
        fallback_used = True
        # Per-label rollout stats: only the fallback slot (q0_seed itself).
        extras["labels_term_reason"][0] = _term_str(candidates[0]["term_reason"])
        extras["labels_n_steps"][0]     = int(candidates[0]["n_steps"])
        extras["labels_max_q_norm"][0]  = float(candidates[0]["max_q_norm"])
    else:
        labels = passed[:k]
        labels_q0 = torch.stack([x["q0"] for x in labels], dim=0)
        labels_L_clean = [float(x["L_clean"]) for x in labels]
        labels_L_robust_mean = [float(x["L_robust_mean"]) for x in labels]
        labels_L_robust_min = [float(x["L_robust_min"]) for x in labels]
        labels_L_perturbed = [list(x["L_perturbed"]) for x in labels]
        for li, lab in enumerate(labels[:k]):
            # robust_filter path (tau_robust>0) does not currently propagate
            # these per-candidate fields; in that path they stay at defaults.
            tr = lab.get("term_reason"); ns = lab.get("n_steps"); mq = lab.get("max_q_norm")
            if tr is not None:
                extras["labels_term_reason"][li] = _term_str(tr)
            if ns is not None:
                extras["labels_n_steps"][li] = int(ns)
            if mq is not None:
                extras["labels_max_q_norm"][li] = float(mq)

    # ---------- 11. Status determination ----------
    max_L = max(labels_L_clean) if labels_L_clean else 0.0
    if fallback_used:
        # Could still be a usable seed; classify by L_seed.
        if max_L >= L_min_acceptable:
            status = STATUS_EDGE  # seed alone is OK but no robust alt found
        else:
            status = STATUS_LOW_QUALITY
    elif n_branches <= edge_branch_threshold:
        status = STATUS_EDGE
    elif max_L < L_min_acceptable:
        status = STATUS_LOW_QUALITY
    else:
        status = STATUS_KEPT

    extras["time_total_sec"] = float(time.time() - t_pipeline_start)
    result = {
        "labels_q0": labels_q0,
        "labels_L_clean": labels_L_clean,
        "labels_L_robust_mean": labels_L_robust_mean,
        "labels_L_robust_min": labels_L_robust_min,
        "labels_L_perturbed": labels_L_perturbed,
        "L_seed": float(L_seed),
        "n_labels": int(labels_q0.shape[0]),
        "status": status,
        "fallback_used": fallback_used,
        "diagnostics": diag,
        "extras": extras,
    }
    if return_all_candidates:
        result["all_candidates_q0"] = Q_candidates_t.detach().cpu().clone()
        result["all_candidates_L"] = [float(x) for x in score_res["L"].tolist()]
    return result
