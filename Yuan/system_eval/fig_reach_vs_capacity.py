"""Reachability is not continuous-motion capacity.

Two fields over the same (r, z) cross-section of the FR3 workspace:

  left   REACHABILITY INDEX -- the standard capability-map quantity: at each
         location, the fraction of tool-axis directions that are attainable.
         Built by binning the 201,600-entry CVT pose table (forward kinematics
         sampling), which is how capability maps are built.

  right  CONTINUOUS-MOTION CAPACITY -- at each location, the arc length the
         arm can traverse without stopping before a joint limit, the tool
         cone or a self-collision ends the motion, maximised over start
         configurations. Median over the eval tasks whose p0 falls in the
         cell; taken from the cached controller-aware oracle rollouts.

The azimuth is aggregated over, so r = sqrt(x^2 + y^2). The right field
additionally marginalises over the path direction d, which the left field has
no notion of at all -- that asymmetry is part of the point.

Both panels come from cached artifacts; this script runs no rollouts.

    python -m Yuan.system_eval.fig_reach_vs_capacity \
        --out Yuan/system_eval/runs/curvature_scan/fig_reach_vs_capacity.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TABLE = "Yuan/unified_rl/runs/iksel_clean_v1/cvt_table_201600.npz"
EVALSET = "Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
ORACLE = "Yuan/system_eval/runs/eval_10k_systematic/cell_oracle_hyb_results.npz"
TARGET_DISTANCE_M = 1.5


def fib_sphere(n: int) -> np.ndarray:
    """n roughly equal-area directions on S^2 (orientation bins)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=20)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--n-dir-bins", type=int, default=42)
    ap.add_argument("--min-tasks", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    T = np.load(REPO / TABLE)
    es = np.load(REPO / EVALSET, allow_pickle=True)
    oh = np.load(REPO / ORACLE, allow_pickle=True)

    rT = np.linalg.norm(T["pos"][:, :2], axis=1)
    zT = T["pos"][:, 2]
    rE = np.linalg.norm(es["cs_p0"][:, :2], axis=1)
    zE = es["cs_p0"][:, 2]
    ell = oh["L_best"].astype(np.float64) * TARGET_DISTANCE_M

    r_edges = np.linspace(0.0, 1.10, args.nr + 1)
    z_edges = np.linspace(-0.55, 1.40, args.nz + 1)
    ir = np.clip(np.digitize(rT, r_edges) - 1, 0, args.nr - 1)
    iz = np.clip(np.digitize(zT, z_edges) - 1, 0, args.nz - 1)
    jr = np.clip(np.digitize(rE, r_edges) - 1, 0, args.nr - 1)
    jz = np.clip(np.digitize(zE, z_edges) - 1, 0, args.nz - 1)

    # ---- left: reachability index (fraction of tool-axis bins attainable) --
    dirs = fib_sphere(args.n_dir_bins)
    dbin = np.argmax(T["zax"] @ dirs.T, axis=1)
    reach = np.full((args.nr, args.nz), np.nan)
    count_T = np.zeros((args.nr, args.nz), int)
    for a in range(args.nr):
        for b in range(args.nz):
            s = (ir == a) & (iz == b)
            count_T[a, b] = s.sum()
            if s.sum() >= 30:
                reach[a, b] = len(np.unique(dbin[s])) / args.n_dir_bins

    # ---- right: continuous-motion capacity --------------------------------
    cap = np.full((args.nr, args.nz), np.nan)
    count_E = np.zeros((args.nr, args.nz), int)
    for a in range(args.nr):
        for b in range(args.nz):
            s = (jr == a) & (jz == b) & np.isfinite(ell)
            count_E[a, b] = s.sum()
            if s.sum() >= args.min_tasks:
                cap[a, b] = np.median(ell[s])

    both = np.isfinite(reach) & np.isfinite(cap)
    from scipy.stats import spearmanr, pearsonr
    sr, _ = spearmanr(reach[both], cap[both])
    pr, _ = pearsonr(reach[both], cap[both])
    print(f"[fig] {both.sum()} cells with both quantities")
    print(f"[fig] cell-level Spearman(reachability, capacity) = {sr:+.3f}, "
          f"Pearson = {pr:+.3f}  (R^2 = {100*pr**2:.1f}%)")
    hi = np.isfinite(reach) & (reach > 0.9)
    print(f"[fig] among cells with reachability index > 0.9 "
          f"({hi.sum()} cells): capacity ranges "
          f"{np.nanmin(cap[hi]):.2f} .. {np.nanmax(cap[hi]):.2f} m "
          f"(ratio {np.nanmax(cap[hi])/max(np.nanmin(cap[hi]),1e-9):.1f}x)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6),
                           gridspec_kw={"width_ratios": [1, 1, 0.85]})
    ext = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]

    im0 = ax[0].imshow(reach.T, origin="lower", extent=ext, aspect="auto",
                       cmap="Blues", vmin=0, vmax=1)
    ax[0].set_title("Reachability index\n"
                    "$\\it{Can\\ the\\ tool\\ be\\ placed\\ here?}$", fontsize=11)
    fig.colorbar(im0, ax=ax[0], label="fraction of tool-axis\ndirections attainable")

    im1 = ax[1].imshow(cap.T, origin="lower", extent=ext, aspect="auto",
                       cmap="magma")
    ax[1].set_title("Continuous-motion capacity\n"
                    "$\\it{Starting\\ here,\\ how\\ far\\ without\\ stopping?}$",
                    fontsize=11)
    fig.colorbar(im1, ax=ax[1], label="$\\ell_{\\max}$  [m]")

    for a in ax[:2]:
        a.set_xlabel(r"$r=\sqrt{x^2+y^2}$  [m]")
        a.set_ylabel("$z$  [m]")
        a.plot(0, 0, "k^", ms=9)
        a.text(0.02, 0.03, "base", fontsize=8)

    ax[2].scatter(reach[both], cap[both], s=16, c="0.25", alpha=0.65)
    ax[2].set_xlabel("reachability index")
    ax[2].set_ylabel("$\\ell_{\\max}$  [m]")
    ax[2].set_title(f"cell-level: $R^2$ = {100*pr**2:.1f}%", fontsize=11)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"[fig] saved -> {out}")


if __name__ == "__main__":
    main()
