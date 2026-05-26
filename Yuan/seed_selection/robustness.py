"""Module 5: ``evaluate_robustness`` — L_clean + L on n_perturb perturbed c's.
Module 6: ``filter_robust_candidates`` — top-K' filter by robust threshold.

Both modules are pure composition logic; they don't introduce new algorithms.

Reproducibility: same ``(q0, c, seed)`` to evaluate_robustness gives bit-exact
output. filter_robust_candidates threads its ``seed`` through to every
candidate so they're all judged against the SAME n_perturb perturbations of c
(apples-to-apples).
"""
from __future__ import annotations

from statistics import mean, pstdev

import torch

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import NSRLBatchedEnv

from Yuan.seed_selection.batched_rollout import batched_rollout_many
from Yuan.seed_selection.perturb import perturb_task
from Yuan.seed_selection.rollout import DEFAULT_TARGET_DISTANCE_M, rollout_one


def evaluate_robustness(
    q0: torch.Tensor,
    c: dict[str, torch.Tensor],
    *,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    n_perturb: int = 4,
    perturb_d_deg: float = 5.0,
    perturb_n_deg: float = 5.0,
    perturb_p0_mm: float = 10.0,
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M,
    seed: int | None = None,
) -> dict:
    """One clean rollout + ``n_perturb`` rollouts on perturbed c's.

    Returns dict with:
        L_clean              float
        L_perturbed          list[float] length n_perturb
        L_robust_mean        float (mean of L_perturbed)
        L_robust_min         float (min of L_perturbed)
        L_robust_std         float (population std of L_perturbed)
        clean_info           dict — full rollout_one return for the clean rollout
        perturbed_info       list[dict] — full rollout_one return for each perturb
    """
    clean = rollout_one(q0, c, env=env, controller=controller,
                        target_distance_m=target_distance_m)
    L_clean = clean["L"]

    gen = (torch.Generator(device=q0.device).manual_seed(int(seed))
           if seed is not None else None)

    L_perturbed: list[float] = []
    perturbed_info: list[dict] = []
    for _ in range(n_perturb):
        c_p = perturb_task(c,
                           perturb_d_deg=perturb_d_deg,
                           perturb_n_deg=perturb_n_deg,
                           perturb_p0_mm=perturb_p0_mm,
                           generator=gen)
        res = rollout_one(q0, c_p, env=env, controller=controller,
                          target_distance_m=target_distance_m)
        L_perturbed.append(res["L"])
        perturbed_info.append(res)

    if n_perturb > 0:
        L_mean = mean(L_perturbed)
        L_min = min(L_perturbed)
        L_std = pstdev(L_perturbed) if n_perturb > 1 else 0.0
    else:
        L_mean = L_clean
        L_min = L_clean
        L_std = 0.0

    return {
        "L_clean": L_clean,
        "L_perturbed": L_perturbed,
        "L_robust_mean": float(L_mean),
        "L_robust_min": float(L_min),
        "L_robust_std": float(L_std),
        "clean_info": clean,
        "perturbed_info": perturbed_info,
    }


def evaluate_robustness_batched(
    q0_list: list[torch.Tensor],
    c: dict[str, torch.Tensor],
    *,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    n_perturb: int = 4,
    perturb_d_deg: float = 5.0,
    perturb_n_deg: float = 5.0,
    perturb_p0_mm: float = 10.0,
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M,
    seed: int | None = None,
) -> list[dict]:
    """Vectorized ``evaluate_robustness`` across K candidates sharing one c.

    All ``K × (1 + n_perturb)`` (q, c) pairs are batched into ONE call to
    ``batched_rollout_many`` (chunked by ``env.n_envs``), which on this box
    is ~50-100× faster than looping ``evaluate_robustness`` K times.

    Reproducibility: same ``seed`` → same ``n_perturb`` perturbed c's, so
    all K candidates are judged against the same task perturbations.

    Args:
        q0_list: list of (7,) joint configurations (all on env.device).
        c: shared task spec.
        Other args: same as ``evaluate_robustness``.

    Returns:
        list of dicts (length K), each with the same fields as
        ``evaluate_robustness`` minus ``clean_info`` / ``perturbed_info``
        (those carry per-rollout diagnostics that aren't relevant for the
        batched call — add back if needed).
    """
    K = len(q0_list)
    if K == 0:
        return []

    # Build the (clean + n_perturb) c-list, deterministic from seed.
    device = q0_list[0].device
    gen = (torch.Generator(device=device).manual_seed(int(seed))
           if seed is not None else None)
    perturbed_cs = [
        perturb_task(c,
                     perturb_d_deg=perturb_d_deg,
                     perturb_n_deg=perturb_n_deg,
                     perturb_p0_mm=perturb_p0_mm,
                     generator=gen)
        for _ in range(n_perturb)
    ]
    all_cs = [c] + perturbed_cs                # length P = 1 + n_perturb
    P = len(all_cs)

    # Flatten the (K, P) grid into K*P (q, c) pairs.
    # Row-major: index i*P + j → (candidate_i, c_variant_j).
    qs = torch.stack(q0_list, dim=0)            # (K, 7)
    qs_rep = qs.unsqueeze(1).expand(K, P, qs.shape[-1]).reshape(K * P, qs.shape[-1])
    cs_rep = [all_cs[j] for _ in range(K) for j in range(P)]

    res = batched_rollout_many(
        qs_rep, cs_rep,
        env=env, controller=controller,
        target_distance_m=target_distance_m,
    )
    L_mat = res["L"].reshape(K, P)              # (K, P)

    out: list[dict] = []
    for i in range(K):
        L_clean = float(L_mat[i, 0])
        L_p = [float(x) for x in L_mat[i, 1:P]]
        if n_perturb > 0:
            L_mean = mean(L_p)
            L_min = min(L_p)
            L_std = pstdev(L_p) if n_perturb > 1 else 0.0
        else:
            L_mean = L_clean
            L_min = L_clean
            L_std = 0.0
        out.append({
            "L_clean": L_clean,
            "L_perturbed": L_p,
            "L_robust_mean": float(L_mean),
            "L_robust_min": float(L_min),
            "L_robust_std": float(L_std),
        })
    return out


def filter_robust_candidates(
    candidates: list[dict],
    c: dict[str, torch.Tensor],
    *,
    K_prime: int,
    tau_robust: float,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    n_perturb: int = 4,
    perturb_d_deg: float = 5.0,
    perturb_n_deg: float = 5.0,
    perturb_p0_mm: float = 10.0,
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M,
    seed: int | None = None,
    L_robust_use: str = "mean",
    return_all_evaluations: bool = False,
) -> list[dict]:
    """Take the top-K' candidates by ``L_clean``, evaluate robustness on each,
    keep those with ``L_robust >= tau_robust * L_clean``.

    Args:
        candidates: list of dicts, each with at least ``'q0'`` and
            ``'L_clean'`` keys. Not assumed pre-sorted.
        K_prime: cap on how many candidates we actually evaluate (top-K' by
            L_clean — saves rollouts for the 100+ candidates that can't be
            in top-k anyway).
        tau_robust: relative threshold; pass iff L_robust ≥ τ × L_clean.
        L_robust_use: 'mean' or 'min' — which aggregate of L_perturbed to
            compare against the threshold.
        return_all_evaluations: if True, return every evaluated candidate
            with a ``'passed'`` bool (useful for diagnostics). If False
            (default), return only the passed candidates.

    Returns:
        list of dicts with keys 'q0', 'L_clean', 'L_robust_mean',
        'L_robust_min', 'L_robust_std', 'L_perturbed', 'passed'.
        May be empty if no candidate satisfies the threshold.
    """
    if L_robust_use not in ("mean", "min"):
        raise ValueError(f"L_robust_use must be 'mean' or 'min', got {L_robust_use!r}")
    top = sorted(candidates, key=lambda x: -x["L_clean"])[:K_prime]
    if not top:
        return []

    # Single batched call across all top-K' candidates, all sharing c +
    # the same n_perturb perturbations (deterministic from seed).
    robs = evaluate_robustness_batched(
        [cand["q0"] for cand in top], c,
        env=env, controller=controller,
        n_perturb=n_perturb,
        perturb_d_deg=perturb_d_deg,
        perturb_n_deg=perturb_n_deg,
        perturb_p0_mm=perturb_p0_mm,
        target_distance_m=target_distance_m,
        seed=seed,
    )

    evaluated: list[dict] = []
    for cand, rob in zip(top, robs):
        L_robust = rob["L_robust_mean"] if L_robust_use == "mean" else rob["L_robust_min"]
        # Edge case: L_clean ≈ 0 means there's nothing to be robust about;
        # threshold τ × 0 = 0 ≤ any L_robust, so it trivially passes. The
        # caller should drop near-zero L_clean candidates upstream
        # (L_min_abs in Module 7). Note that cand["L_clean"] is the input
        # L_clean from prior scoring, NOT rob["L_clean"] — they're computed
        # from the same (q0, c) so are equal modulo rollout determinism.
        passed = L_robust >= tau_robust * cand["L_clean"]
        evaluated.append({
            "q0": cand["q0"],
            "L_clean": cand["L_clean"],
            "L_robust_mean": rob["L_robust_mean"],
            "L_robust_min": rob["L_robust_min"],
            "L_robust_std": rob["L_robust_std"],
            "L_perturbed": rob["L_perturbed"],
            "passed": bool(passed),
        })
    if return_all_evaluations:
        return evaluated
    return [x for x in evaluated if x["passed"]]
