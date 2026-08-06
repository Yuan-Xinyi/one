"""Read a curvature_scan npz and produce the ratio tables + degradation plot.

Every number reported here is a ratio formed inside a single curvature:

    ell_ref(kappa)   largest arc length any valid seed of the curvature-agnostic
                     IK pool reaches at that curvature, under the controller
                     named in the column.

    capacity         ell_ref(kappa) / ell_ref(0), per task, then median over
                     tasks. A property of the task, not of a policy.

    seed / controller rows
                     arc length of one (seed, controller) pair divided by
                     ell_ref(kappa) of the same task at the same kappa.

Difficulty strata are fixed at kappa = 0 (a per-task label that does not move
when the curvature changes), so a small-radius arc cannot migrate into an
easier bucket just because it accumulates more arc length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl",
              5: "truncated", 6: "lateral"}
SRC_PILOT, SRC_IK = 0, 2

# Largest TCP radius the FR3 + 0.2034 m pen can reach, measured by forward
# kinematics over 20k random joint configurations. Used only to check the
# Introduction's claim that episodes end far from the reachable boundary.
REACH_MAX_M = 1.373


def arc_points(p0, d0, n_axis, kappa, s):
    """Closed form of the path point at arc length s (numpy, broadcasting)."""
    s = np.asarray(s)
    is_line = np.abs(kappa) < 1e-9
    k_safe = np.where(is_line, 1.0, kappa)
    Rs = (1.0 / k_safe)[..., None]
    m0 = np.cross(n_axis, d0)
    centre = p0 + Rs * m0
    phi = (k_safe * s)[..., None]
    r0 = -Rs * m0
    rot = r0 * np.cos(phi) + np.cross(n_axis, r0) * np.sin(phi)
    return np.where(is_line[..., None], p0 + s[..., None] * d0, centre + rot)


def reach_stats(p0, d0, n_axis, kappa, arc_len, n_samples=33):
    """Terminal radius and max radius along the travelled path."""
    u = np.linspace(0.0, 1.0, n_samples)
    pts = np.stack([arc_points(p0, d0, n_axis, kappa, arc_len * ui) for ui in u])
    rad = np.linalg.norm(pts, axis=-1)
    return rad[-1], rad.max(axis=0)


def med_iqr(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(x)),
            float(np.percentile(x, 25)), float(np.percentile(x, 75)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    d = np.load(args.scan, allow_pickle=True)
    arc = d["arc_m"]                       # (K, C, T, S)
    term = d["term_reason"]
    kappas = d["kappas"]
    ctrls = [str(c) for c in d["controllers"]]
    src = d["seed_src"]
    valid = d["seed_valid"]                # (T, S)
    cfg = json.loads(str(d["config"]))
    nK, nC, nT, nS = arc.shape
    k0 = int(np.argmin(np.abs(kappas)))
    assert abs(kappas[k0]) < 1e-9, "scan must contain kappa = 0"

    pool = (src == SRC_IK)[None, :] & valid          # (T, S) reference pool
    pilot = (src == SRC_PILOT)[None, :] & valid

    print(f"# curvature scan: {nT} tasks x {nS} seeds "
          f"({pool.sum(1).mean():.1f} valid pool seeds/task), "
          f"k_lateral={cfg['k_lateral']}, tau={cfg['tau_enter']}/{cfg['tau_exit']}")
    print(f"# reference pool = cone-IK enumeration (curvature-agnostic)\n")

    masked = np.where(pool[None, None], arc, -np.inf)     # (K,C,T,S)
    ell_ref = masked.max(axis=3)                          # (K,C,T)
    best_at_0 = masked[k0].argmax(axis=2)                 # (C,T) seed index
    ti = np.arange(nT)

    for ci, cname in enumerate(ctrls):
        ref0 = ell_ref[k0, ci]
        keep = ref0 > 0.05          # tasks the pool can move at all on a line
        print(f"## controller = {cname}   ({keep.sum()}/{nT} tasks with "
              f"ell_ref(0) > 5 cm)")
        hdr = (f"{'kappa':>7} {'R (m)':>7} | {'capacity':>21} | "
               f"{'q_jl / ref':>12} {'q*(0) / ref':>12} | "
               f"{'refTrunc%':>10} {'trunc%':>7} {'lat%':>6} {'cone%':>6} {'jl%':>6}")
        print(hdr)
        print("-" * len(hdr))
        for ki in range(nK):
            cap = ell_ref[ki, ci][keep] / ref0[keep]
            m, lo, hi = med_iqr(cap)
            ref_k = ell_ref[ki, ci]
            r_pilot = arc[ki, ci][:, 0][keep] / ref_k[keep]
            r_b0 = arc[ki, ci][ti, best_at_0[ci]][keep] / ref_k[keep]
            tt = term[ki, ci][pool]
            frac = {n: 100.0 * (tt == c).mean() for c, n in TERM_NAMES.items()}
            # ell_ref is a max, so the step cap censors it more often than it
            # censors a typical pool rollout — report it separately.
            arg = masked[ki, ci].argmax(axis=1)
            ref_trunc = 100.0 * (term[ki, ci][ti, arg][keep] == 5).mean()
            Rm = np.inf if kappas[ki] == 0 else 1.0 / kappas[ki]
            print(f"{kappas[ki]:>+7.2f} {Rm:>7.2f} | "
                  f"{m:>6.3f} [{lo:>5.3f},{hi:>5.3f}] | "
                  f"{np.nanmedian(r_pilot):>12.3f} "
                  f"{np.nanmedian(r_b0):>12.3f} | "
                  f"{ref_trunc:>10.1f} "
                  f"{frac.get('truncated', 0):>7.1f} {frac.get('lateral', 0):>6.1f} "
                  f"{frac.get('cone', 0):>6.1f} {frac.get('jl', 0):>6.1f}")
        print()

    # --- curvature magnitude view (signs pooled) --------------------------
    mags = sorted({abs(float(k)) for k in kappas})
    print("## capacity vs |kappa|, signs pooled (median over tasks x signs)")
    print(f"{'|kappa|':>8} {'R (m)':>7} | " +
          " | ".join(f"{c:>22}" for c in ctrls))
    for mg in mags:
        idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
        cells = []
        for ci in range(len(ctrls)):
            ref0 = ell_ref[k0, ci]
            keep = ref0 > 0.05
            vals = np.concatenate([ell_ref[i, ci][keep] / ref0[keep]
                                   for i in idx])
            m, lo, hi = med_iqr(vals)
            cells.append(f"{m:>6.3f} [{lo:>5.3f},{hi:>5.3f}]")
        Rm = np.inf if mg == 0 else 1.0 / mg
        print(f"{mg:>8.2f} {Rm:>7.2f} | " + " | ".join(f"{c:>22}" for c in cells))
    print()

    # --- 2x2 attribution: seed axis vs controller axis --------------------
    if len(ctrls) >= 2:
        print("## 2x2: rows = seed, cols = controller; cell = arc / ell_ref(kappa)"
              " under that controller (median)")
        print(f"{'|kappa|':>8} | " + " | ".join(
            f"{c[:9]:>9} q_jl {c[:9]:>9} q*(0)" for c in ctrls))
        for mg in mags:
            idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
            cells = []
            for ci in range(len(ctrls)):
                ref0 = ell_ref[k0, ci]
                keep = ref0 > 0.05
                a = np.concatenate([
                    arc[i, ci][:, 0][keep] / ell_ref[i, ci][keep]
                    for i in idx])
                b = np.concatenate([
                    arc[i, ci][ti, best_at_0[ci]][keep] / ell_ref[i, ci][keep]
                    for i in idx])
                cells.append(f"{np.nanmedian(a):>14.3f} {np.nanmedian(b):>15.3f}")
            print(f"{mg:>8.2f} | " + " | ".join(cells))
        print()

    # --- is the reachable envelope what stops the episode? ----------------
    # The Introduction claims episodes end well before the TCP approaches the
    # reachable boundary. Under the arc extension rule the TCP position is a
    # closed-form function of arc length, so this costs no extra rollouts.
    p0a, d0a, nta = d["p0"], d["line_dir"], d["n_target"]
    print(f"## margin to the reachable boundary at termination "
          f"(reach_max = {REACH_MAX_M:.3f} m, controller = {ctrls[0]})")
    print(f"{'kappa':>7} {'R (m)':>7} | {'r_end (m)':>10} {'margin (m)':>11} "
          f"{'within 5cm':>11} | {'max r on path':>14} {'margin':>8}")
    for ki in range(nK):
        kap = np.full(nT, float(kappas[ki]))
        ell = ell_ref[ki, 0]
        r_end, r_max = reach_stats(p0a, d0a, nta, kap, ell)
        print(f"{kappas[ki]:>+7.2f} "
              f"{(np.inf if kappas[ki] == 0 else 1.0/kappas[ki]):>7.2f} | "
              f"{np.median(r_end):>10.3f} {np.median(REACH_MAX_M-r_end):>11.3f} "
              f"{100*np.mean(REACH_MAX_M-r_end < 0.05):>10.1f}% | "
              f"{np.median(r_max):>14.3f} "
              f"{np.median(REACH_MAX_M-r_max):>8.3f}")
    print()

    # --- does the envelope explain the curvature dependence? --------------
    # If curving the path buys arc length *because* it keeps the TCP inside
    # the reachable volume, the gain must concentrate in the tasks that ran
    # closest to the boundary at kappa = 0. If the gain is flat across these
    # strata, the envelope is not the mechanism.
    r_end0, _ = reach_stats(p0a, d0a, nta, np.zeros(nT), ell_ref[k0, 0])
    margin0 = REACH_MAX_M - r_end0
    keep0 = ell_ref[k0, 0] > 0.05
    mq = np.quantile(margin0[keep0], [1 / 3, 2 / 3])
    mstrat = np.digitize(margin0, mq)
    mnames = [f"near boundary (<{mq[0]:.2f} m)",
              f"middle ({mq[0]:.2f}-{mq[1]:.2f})",
              f"deep inside (>{mq[1]:.2f} m)"]
    print("## capacity by margin-to-boundary at kappa=0 "
          f"(controller = {ctrls[0]})")
    print(f"{'|kappa|':>8} | " + " | ".join(f"{n:>24}" for n in mnames))
    for mg in mags:
        idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
        cells = []
        for s in range(3):
            m_ = keep0 & (mstrat == s)
            vals = np.concatenate([ell_ref[i, 0][m_] / ell_ref[k0, 0][m_]
                                   for i in idx])
            cells.append(f"{np.median(vals):>24.3f}")
        print(f"{mg:>8.2f} | " + " | ".join(cells))
    print()

    # --- strata fixed at kappa = 0 ---------------------------------------
    ci = 0
    ref0 = ell_ref[k0, ci]
    keep = ref0 > 0.05
    qs = np.quantile(ref0[keep], [1 / 3, 2 / 3])
    names = ["short ell_ref(0)", "medium ell_ref(0)", "long ell_ref(0)"]
    strat = np.digitize(ref0, qs)
    print(f"## capacity by difficulty stratum, fixed at kappa=0 "
          f"(controller={ctrls[ci]}; cuts at {qs[0]:.2f}/{qs[1]:.2f} m)")
    print(f"{'|kappa|':>8} | " + " | ".join(f"{n:>18}" for n in names))
    for mg in mags:
        idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
        cells = []
        for s in range(3):
            m_ = keep & (strat == s)
            vals = np.concatenate([ell_ref[i, ci][m_] / ref0[m_] for i in idx])
            cells.append(f"{np.median(vals):>18.3f}")
        print(f"{mg:>8.2f} | " + " | ".join(cells))
    print()

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ci, cname in enumerate(ctrls):
            ref0 = ell_ref[k0, ci]
            keep = ref0 > 0.05
            xs, ys, lo_, hi_ = [], [], [], []
            for mg in mags:
                idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
                vals = np.concatenate([ell_ref[i, ci][keep] / ref0[keep]
                                       for i in idx])
                m, lo, hi = med_iqr(vals)
                xs.append(mg); ys.append(m); lo_.append(lo); hi_.append(hi)
            axes[0].plot(xs, ys, "o-", label=cname)
            axes[0].fill_between(xs, lo_, hi_, alpha=0.15)
            zs = []
            for mg in mags:
                idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
                v = np.concatenate([
                    arc[i, ci][ti, best_at_0[ci]][keep] / ell_ref[i, ci][keep]
                    for i in idx])
                zs.append(np.nanmedian(v))
            axes[1].plot(xs, zs, "s-", label=f"{cname}: q*(0)/ref")
            ws = []
            for mg in mags:
                idx = [i for i in range(nK) if abs(float(kappas[i])) == mg]
                v = np.concatenate([
                    arc[i, ci][:, 0][keep] / ell_ref[i, ci][keep]
                    for i in idx])
                ws.append(np.nanmedian(v))
            axes[1].plot(xs, ws, "^--", label=f"{cname}: q_jl/ref")
        axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
        axes[0].set_xlabel(r"$|\kappa|$  [1/m]")
        axes[0].set_ylabel(r"$\ell_{\rm ref}(\kappa)\,/\,\ell_{\rm ref}(0)$")
        axes[0].set_title("reachable arc length vs curvature (capacity)")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel(r"$|\kappa|$  [1/m]")
        axes[1].set_ylabel(r"arc $/\ \ell_{\rm ref}(\kappa)$")
        axes[1].set_title("zero-shot transfer, normalised inside each curvature")
        axes[1].legend(fontsize=8)
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        print(f"[plot] -> {args.plot}")


if __name__ == "__main__":
    main()
