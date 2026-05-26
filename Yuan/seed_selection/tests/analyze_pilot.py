"""Day 5 pilot diagnostics: 9 headline numbers + plots.

Reads a pilot NPZ (default: ``Yuan/seed_selection/runs/pilot_day5/pilot.npz``)
and emits:
    - 9-number summary printed to stdout + saved to ``pilot.stats.json``
    - Plots saved next to the NPZ:
        status_hist.png            status taxonomy bar chart
        candidates_hist.png        n_candidates_to_score distribution
                                   (from per-task meta — N/A here since meta
                                   is per-run, not per-task; reads from
                                   diagnostics if available)
        L_seed_vs_max_label.png    scatter, with y=x reference
        tau_sweep.png              re-filter at τ ∈ {0.3, 0.5, 0.7}
        q_pca_*.png                5 random kept-status tasks, q-space PCA of
                                   their labels + q0_seed
Run:
    python -m Yuan.seed_selection.tests.analyze_pilot [path-to-npz]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path):
    z = np.load(path, allow_pickle=False)
    meta_path = path.parent / f"{path.stem}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return z, meta


def _headline_numbers(z, meta):
    N = int(z["L_seed"].shape[0])
    status = np.array([str(s) for s in z["status"]])
    counts = {s: int((status == s).sum()) for s in
              ("kept", "edge", "infeasible", "low_quality")}
    n_errors = int(meta.get("n_errors", 0))
    # Per-task max(label_L)
    Lc = z["labels_L_clean"]                  # (N, k), NaN pad
    n_labels = z["n_labels"]                  # (N,)
    max_label = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        nl = int(n_labels[i])
        if nl == 0:
            continue
        valid = Lc[i, :nl]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            max_label[i] = float(np.max(valid))
    L_seed = z["L_seed"]
    # Relative improvement (only where both finite + L_seed > 0).
    rel = np.full(N, np.nan, dtype=np.float32)
    mask = np.isfinite(max_label) & np.isfinite(L_seed) & (L_seed > 1e-6)
    rel[mask] = max_label[mask] / L_seed[mask]

    headline = {
        "N":                      N,
        "kept_pct":               100.0 * counts["kept"] / N if N else 0.0,
        "edge_pct":               100.0 * counts["edge"] / N if N else 0.0,
        "infeasible_pct":         100.0 * counts["infeasible"] / N if N else 0.0,
        "low_quality_pct":        100.0 * counts["low_quality"] / N if N else 0.0,
        "failed_pct":             100.0 * n_errors / N if N else 0.0,
        "max_over_seed_median":   float(np.nanmedian(rel)) if np.any(np.isfinite(rel)) else float("nan"),
        "max_over_seed_p25":      float(np.nanpercentile(rel, 25)) if np.any(np.isfinite(rel)) else float("nan"),
        "max_over_seed_p75":      float(np.nanpercentile(rel, 75)) if np.any(np.isfinite(rel)) else float("nan"),
        "L_seed_mean":            float(np.nanmean(L_seed)),
        "max_label_mean":         float(np.nanmean(max_label)),
        "n_kept_with_max_label_ge_L_min_acceptable": int(
            ((status == "kept") & (max_label >= 0.20)).sum()),
    }
    return headline, status, max_label, rel


def _plot_status(status, out_path):
    cats = ["kept", "edge", "low_quality", "infeasible"]
    counts = [int((status == c).sum()) for c in cats]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(cats, counts, color=["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                ha="center", va="bottom")
    ax.set_ylabel("# tasks")
    ax.set_title("status distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_L_scatter(L_seed, max_label, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(L_seed, max_label, s=20, alpha=0.55, edgecolors="none")
    lim = max(float(np.nanmax(L_seed)) if np.any(np.isfinite(L_seed)) else 0.1,
              float(np.nanmax(max_label)) if np.any(np.isfinite(max_label)) else 0.1)
    lim = lim * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x (no improvement)")
    ax.set_xlabel("L_seed")
    ax.set_ylabel("max(labels_L_clean)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("L_seed vs max(labels_L_clean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _tau_sweep(z, taus=(0.3, 0.5, 0.7)):
    """Re-filter the saved labels at different tau_robust thresholds.
    For each label we have L_clean and L_robust_mean; passed iff
    L_robust_mean >= tau * L_clean. Reports per-tau {n_passed_labels,
    n_kept_tasks (any label passes)}."""
    Lc = z["labels_L_clean"]; Lrm = z["labels_L_robust_mean"]
    nl = z["n_labels"]
    N, k = Lc.shape
    rows = []
    for tau in taus:
        # passed[i, j] = j < nl[i] and Lrm >= tau * Lc and both finite
        passed = np.zeros((N, k), dtype=bool)
        for i in range(N):
            for j in range(int(nl[i])):
                if np.isfinite(Lc[i, j]) and np.isfinite(Lrm[i, j]):
                    if Lrm[i, j] >= tau * Lc[i, j] - 1e-9:
                        passed[i, j] = True
        n_pass_labels = int(passed.sum())
        n_kept_tasks = int((passed.any(axis=1)).sum())
        rows.append({"tau": tau, "passed_labels": n_pass_labels,
                     "kept_tasks": n_kept_tasks,
                     "kept_pct": 100.0 * n_kept_tasks / N if N else 0.0})
    return rows


def _plot_tau_sweep(rows, N, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = [r["tau"] for r in rows]
    ys = [r["kept_pct"] for r in rows]
    ax.plot(xs, ys, "-o", color="#1f77b4")
    for r in rows:
        ax.text(r["tau"], r["kept_pct"], f"{r['kept_pct']:.0f}% ({r['kept_tasks']})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("τ_robust")
    ax.set_ylabel("% tasks with ≥1 passed label")
    ax.set_title(f"tau sweep (N={N})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _q_pca(z, status, n_examples, out_dir, seed=42):
    """For ``n_examples`` random kept-status tasks, plot 2D PCA of
    (labels_q0, q0_seed) in joint space. Different branches should appear
    as distinct clusters in 7D, and PCA usually preserves that in 2D."""
    rng = np.random.default_rng(seed)
    kept_idx = np.where(status == "kept")[0]
    if kept_idx.size == 0:
        return []
    chosen = rng.choice(kept_idx, size=min(n_examples, kept_idx.size), replace=False)
    paths = []
    for idx in chosen:
        nl = int(z["n_labels"][idx])
        if nl == 0:
            continue
        labels = z["labels_q0"][idx, :nl]            # (nl, 7)
        seed_q = z["q0_seeds"][idx][None, :]          # (1, 7)
        all_q = np.concatenate([seed_q, labels], axis=0)
        # 2-component PCA via SVD.
        center = all_q.mean(axis=0)
        Xc = all_q - center
        # If all rows are equal (degenerate), skip.
        if np.linalg.norm(Xc) < 1e-6:
            continue
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:2].T
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(proj[0, 0], proj[0, 1], s=120, marker="*",
                   color="black", label=f"seed (L={z['L_seed'][idx]:.3f})", zorder=5)
        Lc = z["labels_L_clean"][idx, :nl]
        for j in range(nl):
            color = plt.cm.tab10(j % 10)
            ax.scatter(proj[1 + j, 0], proj[1 + j, 1], s=120,
                       color=color, label=f"label {j} (L={Lc[j]:.3f})",
                       edgecolors="black", linewidth=1)
        ax.set_xlabel(f"PC1  (σ={S[0]:.2f})")
        ax.set_ylabel(f"PC2  (σ={S[1] if len(S) > 1 else 0.0:.2f})")
        ax.set_title(f"task {idx}  ({nl} labels, status=kept)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        p = out_dir / f"q_pca_task{idx:03d}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "Yuan/seed_selection/runs/pilot_day5/pilot.npz")
    if not npz_path.exists():
        print(f"NPZ not found: {npz_path}", flush=True)
        return 1
    print(f"[analyze] reading {npz_path}", flush=True)
    z, meta = _load(npz_path)
    out_dir = npz_path.parent

    headline, status, max_label, rel = _headline_numbers(z, meta)
    print("\n=== headline numbers ===")
    for k, v in headline.items():
        if isinstance(v, float):
            print(f"  {k:40s}  {v:>8.3f}")
        else:
            print(f"  {k:40s}  {v!s:>8}")

    # L_seed-stratified buckets (Day 6 Analysis 1).
    # Matches the pilot 1 bucket boundaries so two pilots can be compared
    # 1:1. ratio_med + abs_gain_med are computed on the 'kept' subset only
    # (those are the only tasks with non-trivial labels).
    L_seed = z["L_seed"]
    bucket_rows = []
    print("\n=== L_seed buckets (kept tasks only) ===")
    print(f"  {'L_seed range':25s}  {'n_kept':>6}  {'ratio_med':>9}  "
          f"{'ratio_p75':>9}  {'abs_gain_med':>12}")
    bounds = [(0.00, 0.10), (0.10, 0.15), (0.15, 0.20),
              (0.20, 0.30), (0.30, 1e9)]
    for lo, hi in bounds:
        mask = (status == "kept") & (L_seed >= lo) & (L_seed < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.2f}, {hi:.2f}){'':14s}  {n:>6}  {'-':>9}  {'-':>9}  {'-':>12}")
            bucket_rows.append({"lo": lo, "hi": hi, "n": 0})
            continue
        ratio = max_label[mask] / np.clip(L_seed[mask], 1e-6, None)
        gain = max_label[mask] - L_seed[mask]
        ratio_med = float(np.median(ratio))
        ratio_p75 = float(np.percentile(ratio, 75))
        gain_med = float(np.median(gain))
        print(f"  [{lo:.2f}, {hi:.2f}){'':14s}  {n:>6}  {ratio_med:>9.3f}  "
              f"{ratio_p75:>9.3f}  {gain_med:>12.4f}")
        bucket_rows.append({"lo": lo, "hi": hi, "n": n,
                             "ratio_med": ratio_med, "ratio_p75": ratio_p75,
                             "abs_gain_med": gain_med})

    # weak/strong summary (= [0.0, 0.20) vs [0.20, inf)).
    weak_mask = (status == "kept") & (L_seed < 0.20)
    strong_mask = (status == "kept") & (L_seed >= 0.20)
    print("\n  weak vs strong (kept):")
    for label, mask in [("weak  (L_seed < 0.20)", weak_mask),
                          ("strong (L_seed ≥ 0.20)", strong_mask)]:
        n = int(mask.sum())
        if n == 0:
            print(f"    {label:25s}  n=0")
            continue
        ratio = max_label[mask] / np.clip(L_seed[mask], 1e-6, None)
        gain = max_label[mask] - L_seed[mask]
        print(f"    {label:25s}  n={n:>4}  ratio_med={float(np.median(ratio)):.3f}  "
              f"abs_gain_med={float(np.median(gain)):.4f}  "
              f"abs_gain_mean={float(np.mean(gain)):.4f}")

    # Save JSON.
    stats_path = out_dir / f"{npz_path.stem}.stats.json"
    stats_path.write_text(json.dumps({
        "headline": headline,
        "L_seed_buckets_kept": bucket_rows,
        "status_per_task": status.tolist(),
        "L_seed": z["L_seed"].tolist(),
        "max_label": [None if not np.isfinite(x) else float(x) for x in max_label],
        "rel_improvement": [None if not np.isfinite(x) else float(x) for x in rel],
        "tau_sweep": _tau_sweep(z),
    }, indent=2))
    print(f"\n  wrote {stats_path}", flush=True)

    # Plots
    _plot_status(status, out_dir / "status_hist.png")
    _plot_L_scatter(z["L_seed"], max_label, out_dir / "L_seed_vs_max_label.png")
    tau_rows = _tau_sweep(z)
    print("\n=== tau sweep ===")
    for r in tau_rows:
        print(f"  τ={r['tau']:.2f}  kept_tasks={r['kept_tasks']}/{int(z['L_seed'].shape[0])}  "
              f"({r['kept_pct']:.1f}%)  passed_labels={r['passed_labels']}")
    _plot_tau_sweep(tau_rows, int(z["L_seed"].shape[0]), out_dir / "tau_sweep.png")
    pca_paths = _q_pca(z, status, n_examples=5, out_dir=out_dir)
    if pca_paths:
        print(f"\n  q-space PCA plots: {len(pca_paths)} files in {out_dir}")
    print("\n[analyze] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
