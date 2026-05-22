"""Post-process hybrid RL+Classical evaluation on the cached N=10000 rollouts.

Variant A (episode-level switching) is pure post-processing:
  per task, choose Classical if max|q_norm(q_0)| >= tau_episode, else RL.

Variant B (step-level switching) is implemented in a separate runner
(`eval_hybrid_steplevel.py`) because once a switch happens mid-episode the
post-switch trajectory diverges from the cached one. This script does
variant A and an "oracle hybrid" upper bound only.

Usage:
    python -m Yuan.RL_controller.eval_hybrid \\
        --cache Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_10000_classical/rollouts.npz
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np


# FR3 joint limits (mirror of BatchedFR3Kinematics defaults).
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
                  dtype=np.float64)
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
                  dtype=np.float64)
Q_MID = 0.5 * (LMT_LO + LMT_UP)
Q_HALF = 0.5 * (LMT_UP - LMT_LO)

TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl",
              5: "truncated", 6: "lateral"}
TERM_ORDER = ["cone", "jl", "lateral", "collision", "truncated", "alive"]


def q_norm(q: np.ndarray) -> np.ndarray:
    """(..., 7) -> (..., 7) normalized to [-1, 1] per joint."""
    return (q.astype(np.float64) - Q_MID) / Q_HALF


def max_abs_qn(q: np.ndarray) -> np.ndarray:
    """(..., 7) -> (...) max over joints of |q_norm|."""
    return np.max(np.abs(q_norm(q)), axis=-1)


def summarize(name, T, term, T_base, dt, v):
    """Compute the standard metric block for a length array T (N,) and
    term-reason array term (N,), measured vs Classical baseline T_base."""
    N = T.shape[0]
    progress = T.astype(np.float64) * dt * v
    ratio = T.astype(np.float64) / np.maximum(T_base.astype(np.float64), 1.0)
    worse = int((T < T_base).sum())
    term_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in term)
    term_frac = {k: 100.0 * term_hist.get(k, 0) / N for k in TERM_ORDER}
    return {
        "name": name,
        "N": N,
        "mean_progress_m": float(progress.mean()),
        "median_progress_m": float(np.median(progress)),
        "mean_ratio_vs_classical": float(ratio.mean()),
        "median_ratio_vs_classical": float(np.median(ratio)),
        "frac_hybrid_worse_than_classical": 100.0 * worse / N,
        "term_frac": term_frac,
    }


def fmt_row(d, extra=None):
    cells = [
        f"{d['name']:<22s}",
        f"{d.get('n_rl', '-'):>5}",
        f"{d.get('n_cls', '-'):>5}",
        f"{d['mean_progress_m']:>5.3f}",
        f"{d['median_progress_m']:>5.3f}",
        f"{d['mean_ratio_vs_classical']:>5.3f}",
        f"{d['median_ratio_vs_classical']:>5.3f}",
        f"{d['frac_hybrid_worse_than_classical']:>5.1f}%",
    ]
    for k in TERM_ORDER:
        cells.append(f"{d['term_frac'][k]:>5.1f}")
    if extra:
        for v in extra:
            cells.append(v)
    return "  ".join(cells)


def fmt_header(extra_cols=None):
    head = [
        f"{'row':<22s}",
        f"{'nRL':>5}",
        f"{'nCls':>5}",
        f"{'meanP':>5}",
        f"{'medP':>5}",
        f"{'meanR':>5}",
        f"{'medR':>5}",
        f"{'wrs%':>6}",
    ]
    for k in TERM_ORDER:
        head.append(f"{k[:5]:>5}")
    if extra_cols:
        head.extend(extra_cols)
    return "  ".join(head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True,
                        help="path to rollouts.npz with q_traj_rl/q_traj_base etc.")
    parser.add_argument("--taus", nargs="+", type=float,
                        default=[0.75, 0.80, 0.82, 0.85, 0.88, 0.90])
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    d = np.load(cache_path, allow_pickle=True)
    T_rl = d["episode_len_rl"].astype(np.int64)
    T_base = d["episode_len_base"].astype(np.int64)
    term_rl = d["term_reason_rl"].astype(np.int64)
    term_base = d["term_reason_base"].astype(np.int64)
    q0 = d["q0"]
    dt = float(d["dt"])
    v_const = 0.2  # mirror config (cache doesn't store v; default 0.2 m/s)
    if "v" in d.files:
        v_const = float(d["v"])
    N = T_rl.shape[0]
    print(f"[hybrid] loaded {cache_path.name}  N={N}  dt={dt}  v≈{v_const}")

    # ---- init max|qn| ----
    init_max_qn = max_abs_qn(q0)
    print(f"[hybrid] init max|q_norm|  min={init_max_qn.min():.3f}  "
          f"p25={np.quantile(init_max_qn, 0.25):.3f}  "
          f"p50={np.quantile(init_max_qn, 0.50):.3f}  "
          f"p75={np.quantile(init_max_qn, 0.75):.3f}  "
          f"max={init_max_qn.max():.3f}")

    # ---- reference rows: pure RL, pure Classical ----
    rows = []
    pure_rl = summarize("pure_RL", T_rl, term_rl, T_base, dt, v_const)
    pure_rl["n_rl"], pure_rl["n_cls"] = N, 0
    rows.append(pure_rl)
    pure_cls = summarize("pure_Classical", T_base, term_base, T_base, dt, v_const)
    pure_cls["n_rl"], pure_cls["n_cls"] = 0, N
    rows.append(pure_cls)

    # ---- variant A sweep ----
    variant_a_rows = []
    for tau in args.taus:
        route_cls = init_max_qn >= tau  # use Classical
        T_hyb = np.where(route_cls, T_base, T_rl)
        term_hyb = np.where(route_cls, term_base, term_rl)
        row = summarize(f"varA_tau={tau:.2f}", T_hyb, term_hyb, T_base, dt, v_const)
        row["n_rl"] = int((~route_cls).sum())
        row["n_cls"] = int(route_cls.sum())
        variant_a_rows.append(row)

    # ---- oracle hybrid: per-task pick whichever is longer ----
    use_rl_oracle = T_rl >= T_base  # tie -> RL
    T_oracle = np.where(use_rl_oracle, T_rl, T_base)
    term_oracle = np.where(use_rl_oracle, term_rl, term_base)
    oracle = summarize("oracle_hybrid", T_oracle, term_oracle, T_base, dt, v_const)
    oracle["n_rl"] = int(use_rl_oracle.sum())
    oracle["n_cls"] = int((~use_rl_oracle).sum())

    # ---- print ----
    print()
    print("# Variant A sweep + reference rows")
    print(fmt_header())
    for r in [pure_rl, pure_cls] + variant_a_rows + [oracle]:
        print(fmt_row(r))

    # ---- best variant A ----
    best_a = max(variant_a_rows, key=lambda r: r["mean_progress_m"])
    print()
    print(f"[hybrid] best variant A (by mean_progress): "
          f"{best_a['name']}  mean_P={best_a['mean_progress_m']:.4f}  "
          f"meanR={best_a['mean_ratio_vs_classical']:.3f}  "
          f"worse={best_a['frac_hybrid_worse_than_classical']:.1f}%")
    print(f"[hybrid] pure_RL mean_P={pure_rl['mean_progress_m']:.4f}, "
          f"oracle mean_P={oracle['mean_progress_m']:.4f}")
    headroom = oracle["mean_progress_m"] - pure_rl["mean_progress_m"]
    captured = (best_a["mean_progress_m"] - pure_rl["mean_progress_m"]) / max(headroom, 1e-9)
    print(f"[hybrid] variant A captures {100*captured:.1f}% of (oracle - pure_RL) headroom")

    if args.out_csv:
        import csv
        out = Path(args.out_csv)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "n_rl", "n_cls", "mean_progress_m", "median_progress_m",
                        "mean_ratio_vs_cls", "median_ratio_vs_cls", "worse_pct"]
                       + [f"term_{k}_pct" for k in TERM_ORDER])
            for r in [pure_rl, pure_cls] + variant_a_rows + [oracle]:
                w.writerow([r["name"], r["n_rl"], r["n_cls"],
                            f"{r['mean_progress_m']:.4f}",
                            f"{r['median_progress_m']:.4f}",
                            f"{r['mean_ratio_vs_classical']:.4f}",
                            f"{r['median_ratio_vs_classical']:.4f}",
                            f"{r['frac_hybrid_worse_than_classical']:.2f}"]
                           + [f"{r['term_frac'][k]:.2f}" for k in TERM_ORDER])
        print(f"[hybrid] saved csv → {out}")


if __name__ == "__main__":
    main()
