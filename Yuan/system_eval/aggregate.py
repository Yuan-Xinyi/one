"""Aggregate cell_{A..E,E_prime}_results.npz into summary table, report, figures.

Reporting units: absolute EE progress in METERS (= L_best * target_distance_m).
The 1.5m normalizer was an arbitrary constant; absolute meters are what
deployment cares about. Per-task ratios (L_X / L_oracle) stay meaningful.

For diffusion cells (B, D), two reductions are reported:
    progress_best_m      best-of-N with IK-fails counted as 0          (conservative)
    progress_realistic_m best-of-N; if ALL N fail IK, fallback to A    (realistic)
The realistic number is closer to actual deployment behavior.

Outputs (under <root>/):
    summary_table.csv          one row per (cell, bucket)
    summary_report.md          markdown report
    figures/
        deployment_gain_by_bucket.png
        recovery_distribution.png
        ablation_decomposition.png
        oracle_gap.png

Usage:
    python -m Yuan.system_eval.aggregate \\
        --config Yuan/system_eval/config.yaml \\
        --in-dir Yuan/system_eval/runs/eval_10k_systematic
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


CELLS = ['A', 'B', 'C', 'D', 'E', 'E_prime']
CELL_LABEL = {
    'A': 'A: baseline (q0_seed + Classical)',
    'B': 'B: seed ablation (Diffusion + Classical)',
    'C': 'C: controller ablation (q0_seed + RL hybrid)',
    'D': 'D: full method (Diffusion + RL hybrid)',
    'E': "E: seed-oracle (max-L_classical label + RL hybrid)",
    'E_prime': "E': controller-aware oracle (max over SMM top-K' + RL hybrid)",
}
CELL_COLOR = {
    'A': '#888888', 'B': '#1f77b4', 'C': '#ff7f0e',
    'D': '#2ca02c', 'E': '#d62728', 'E_prime': '#8b0000',
}
BUCKET_ORDER = ['weak', 'medium-weak', 'medium', 'strong']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--in-dir', default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--require-all', action='store_true')
    return p.parse_args()


def load_cell(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _percentile_safe(a, q):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if len(a) == 0:
        return float('nan')
    return float(np.percentile(a, q))


def _progress_best(cell_data: dict, target_distance_m: float,
                   fallback_progress: np.ndarray | None = None) -> np.ndarray:
    """Return per-task progress (m). If fallback_progress is given, replace
    NaN entries (all-IK-failed tasks for diffusion cells) with fallback."""
    progress = cell_data['L_best'].astype(np.float64) * target_distance_m
    if fallback_progress is not None:
        nan_mask = ~np.isfinite(progress)
        progress[nan_mask] = fallback_progress[nan_mask]
    return progress.astype(np.float32)


def metric_block(progress, bucket_mask, thresholds_m, catastrophic_m):
    a = progress[bucket_mask]
    finite = a[np.isfinite(a)]
    row = {
        'n_tasks': int(bucket_mask.sum()),
        'n_valid': int(len(finite)),
        'progress_mean_m': float(np.nanmean(a)) if len(finite) else float('nan'),
        'progress_median_m': _percentile_safe(a, 50),
        'progress_p10_m': _percentile_safe(a, 10),
        'progress_p25_m': _percentile_safe(a, 25),
        'progress_p75_m': _percentile_safe(a, 75),
        'progress_p90_m': _percentile_safe(a, 90),
        'catastrophic_failure_rate': (float((finite < catastrophic_m).mean())
                                      if len(finite) else float('nan')),
    }
    for t in thresholds_m:
        row[f'success_progress_geq_{t:g}m'] = (float((finite >= t).mean())
                                                if len(finite) else float('nan'))
    return row


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    in_dir = Path(args.in_dir or cfg['output']['root'])
    out_dir = Path(args.out_dir or in_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    pat = cfg['output']['cell_results_pattern']
    cells = {}
    for c in CELLS:
        p = in_dir / pat.format(cell=c)
        if not p.exists():
            msg = f'[aggregate] missing {p}'
            if args.require_all and c != 'E_prime':  # E' is optional
                raise SystemExit(msg)
            print(msg + ' — skipping')
            continue
        cells[c] = load_cell(p)
    if not cells:
        raise SystemExit('[aggregate] no cell results found')

    ref = next(iter(cells.values()))
    n_tasks = int(ref['n_tasks'])
    bucket_ref = ref['bucket']
    for c, d in cells.items():
        if not np.array_equal(d['src_idx'], ref['src_idx']):
            raise SystemExit(f'[aggregate] cell {c} src_idx differs from reference')

    target_distance_m = float(cfg['env']['target_distance_m'])
    thresholds_m = list(cfg['metrics']['progress_thresholds_m'])
    catastrophic_m = float(cfg['metrics']['catastrophic_progress_m'])
    recovery_frac = float(cfg['metrics']['recovery_fraction_oracle'])

    # ---- Build progress arrays (conservative + realistic) -------------
    # Conservative: NaN where all IK failed (excluded from finite stats)
    # Realistic: fallback to Cell A's progress on those tasks
    progress_A = _progress_best(cells['A'], target_distance_m) if 'A' in cells else None

    progress = {}             # conservative version
    progress_real = {}        # realistic version (A-fallback for B, D)
    for c, d in cells.items():
        progress[c] = _progress_best(d, target_distance_m, fallback_progress=None)
        if c in ('B', 'D') and progress_A is not None:
            progress_real[c] = _progress_best(d, target_distance_m,
                                              fallback_progress=progress_A)
        else:
            progress_real[c] = progress[c].copy()

    bucket_masks = {b: (bucket_ref == b) for b in BUCKET_ORDER
                    if (bucket_ref == b).any()}
    bucket_masks['ALL'] = np.ones(n_tasks, dtype=bool)
    bucket_iter = list(bucket_masks.keys())

    # ---- Choose the canonical oracle for ratios: E' if available else E
    oracle_cell = 'E_prime' if 'E_prime' in cells else 'E'
    progress_O = progress[oracle_cell] if oracle_cell in progress else None
    print(f'[aggregate] cells loaded: {sorted(cells.keys())}; '
          f'oracle for ratios = {oracle_cell}; n_tasks={n_tasks}')

    # ---- Main rows ----------------------------------------------------
    rows = []
    for c in cells:
        for b in bucket_iter:
            m = bucket_masks[b]
            row = {'cell': c, 'bucket': b, 'variant': 'conservative',
                   **metric_block(progress[c], m, thresholds_m, catastrophic_m)}
            if progress_A is not None:
                d = progress[c][m] - progress_A[m]
                row['gain_over_baseline_median_m'] = float(np.nanmedian(d))
                with np.errstate(invalid='ignore'):
                    r = progress[c][m] / np.where(progress_A[m] > 0, progress_A[m], np.nan)
                rf = r[np.isfinite(r)]
                row['mean_ratio_vs_baseline'] = float(rf.mean()) if len(rf) else float('nan')
            if progress_O is not None:
                with np.errstate(invalid='ignore'):
                    r = progress[c][m] / np.where(progress_O[m] > 0, progress_O[m], np.nan)
                rf = r[np.isfinite(r)]
                row['mean_ratio_vs_oracle'] = float(rf.mean()) if len(rf) else float('nan')
                row[f'recovery_vs_oracle_at_{recovery_frac}'] = (
                    float((rf >= recovery_frac).mean()) if len(rf) else float('nan'))

            if c in ('B', 'D') and 'ik_ok' in cells[c]:
                ik = cells[c]['ik_ok'][m]
                row['ik_convergence_rate_per_sample'] = float(ik.mean()) if ik.size else float('nan')
                row['mean_n_ik_success_per_task'] = float(ik.sum(axis=1).mean()) if ik.size else float('nan')
                row['ik_all_fail_rate'] = float((~ik).all(axis=1).mean()) if ik.size else float('nan')
            rows.append(row)

            # Realistic variant for B, D (A-fallback on all-IK-fail tasks)
            if c in ('B', 'D'):
                row_r = {'cell': c, 'bucket': b, 'variant': 'realistic_A_fallback',
                         **metric_block(progress_real[c], m, thresholds_m, catastrophic_m)}
                if progress_A is not None:
                    d = progress_real[c][m] - progress_A[m]
                    row_r['gain_over_baseline_median_m'] = float(np.nanmedian(d))
                if progress_O is not None:
                    with np.errstate(invalid='ignore'):
                        r = progress_real[c][m] / np.where(progress_O[m] > 0, progress_O[m], np.nan)
                    rf = r[np.isfinite(r)]
                    row_r['mean_ratio_vs_oracle'] = float(rf.mean()) if len(rf) else float('nan')
                    row_r[f'recovery_vs_oracle_at_{recovery_frac}'] = (
                        float((rf >= recovery_frac).mean()) if len(rf) else float('nan'))
                rows.append(row_r)

    # ---- CSV ---------------------------------------------------------
    head = ['cell', 'bucket', 'variant', 'n_tasks', 'n_valid',
            'progress_mean_m', 'progress_median_m',
            'progress_p10_m', 'progress_p25_m', 'progress_p75_m', 'progress_p90_m',
            'catastrophic_failure_rate',
            *[f'success_progress_geq_{t:g}m' for t in thresholds_m],
            'gain_over_baseline_median_m', 'mean_ratio_vs_baseline',
            'mean_ratio_vs_oracle', f'recovery_vs_oracle_at_{recovery_frac}',
            'ik_convergence_rate_per_sample', 'mean_n_ik_success_per_task',
            'ik_all_fail_rate']
    csv_path = out_dir / 'summary_table.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            w.writerow([_fmt(r.get(k)) for k in head])
    print(f'[aggregate] wrote {csv_path} ({len(rows)} rows)')

    # ---- Markdown report --------------------------------------------
    md_lines = build_report(cells, rows, progress, progress_real, progress_A,
                            progress_O, oracle_cell, bucket_ref, bucket_iter,
                            thresholds_m, recovery_frac, catastrophic_m, n_tasks,
                            target_distance_m)
    md_path = out_dir / 'summary_report.md'
    md_path.write_text('\n'.join(md_lines))
    print(f'[aggregate] wrote {md_path}')

    # ---- Figures -----------------------------------------------------
    plot_deployment_gain(cells, progress, progress_real, bucket_ref,
                         fig_dir / 'deployment_gain_by_bucket.png')
    plot_recovery_distribution(cells, progress, bucket_ref, oracle_cell, progress_O,
                                fig_dir / 'recovery_distribution.png')
    plot_ablation_decomposition(cells, progress, bucket_ref,
                                fig_dir / 'ablation_decomposition.png')
    plot_oracle_gap(cells, progress, bucket_ref, oracle_cell,
                    fig_dir / 'oracle_gap.png')
    print(f'[aggregate] wrote figures to {fig_dir}')


# ----------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, float):
        if not np.isfinite(v):
            return ''
        return f'{v:.4f}' if abs(v) < 100 else f'{v:.3g}'
    return v


def _fmt_pct(v, prec=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    return f'{100*v:.{prec}f}'


def build_report(cells, rows, progress, progress_real, progress_A, progress_O,
                 oracle_cell, bucket_ref, bucket_iter, thresholds_m, recovery_frac,
                 catastrophic_m, n_tasks, target_distance_m):
    out = []
    out.append(f'# System eval — 5+1 cell ablation (N={n_tasks})')
    out.append('')
    out.append(f'All progress values in **meters** (raw EE displacement along the '
               f'task line). Conservative variant: best-of-N over diffusion samples, '
               f'all-IK-fail tasks counted as progress=0. Realistic variant: '
               f'all-IK-fail tasks fall back to the q0_seed rollout (Cell A). '
               f'Oracle used for recovery ratios = **{oracle_cell}** '
               f'{"(controller-aware)" if oracle_cell == "E_prime" else "(seed-oracle under classical labels)"}.')
    out.append('')
    out.append('| cell | source |')
    out.append('|---|---|')
    for c in CELLS:
        if c in cells:
            out.append(f'| {c} | {CELL_LABEL[c]} |')
    out.append('')

    hdr = ['cell', 'variant', 'n', 'prog_med (m)', 'p25', 'p75',
           f'≥{thresholds_m[0]:g}m', f'≥{thresholds_m[1]:g}m', f'≥{thresholds_m[2]:g}m',
           'recov_vs_oracle', 'gain_med (m)']

    def row_for(c, b, variant):
        rs = [r for r in rows
              if r['cell'] == c and r['bucket'] == b and r['variant'] == variant]
        return rs[0] if rs else None

    def emit_table(b, variant_label):
        out.append('| ' + ' | '.join(hdr) + ' |')
        out.append('|' + '|'.join(['---'] * len(hdr)) + '|')
        for c in CELLS:
            if c not in cells:
                continue
            if c in ('B', 'D'):
                r = row_for(c, b, variant_label)
            else:
                r = row_for(c, b, 'conservative')   # A, C, E, E' have no IK fallback
            if r is None:
                continue
            row = [c, r['variant'][:10], str(r['n_tasks']),
                   _fmt(r['progress_median_m']),
                   _fmt(r['progress_p25_m']), _fmt(r['progress_p75_m']),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[0]:g}m')),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[1]:g}m')),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[2]:g}m')),
                   _fmt_pct(r.get(f'recovery_vs_oracle_at_{recovery_frac}')),
                   _fmt(r.get('gain_over_baseline_median_m'))]
            out.append('| ' + ' | '.join(row) + ' |')
        out.append('')

    # Overall (ALL) — both variants for B/D
    out.append('## Overall (ALL tasks) — CONSERVATIVE (IK fail = 0)')
    out.append('')
    emit_table('ALL', 'conservative')

    out.append('## Overall (ALL tasks) — REALISTIC (IK fail → q0_seed fallback)')
    out.append('')
    emit_table('ALL', 'realistic_A_fallback')

    # Per-bucket break-down (conservative only, for brevity)
    out.append('## Per-bucket break-down (conservative variant)')
    out.append('')
    for b in bucket_iter:
        if b == 'ALL':
            continue
        out.append(f'### Bucket: {b}')
        emit_table(b, 'conservative')

    # IK convergence
    if 'B' in cells:
        out.append('## Diffusion-specific (IK convergence) — cells B, D')
        out.append('')
        out.append('| cell | bucket | ik_rate | mean_n_ok | all_fail_rate |')
        out.append('|---|---|---|---|---|')
        for c in ('B', 'D'):
            if c not in cells:
                continue
            for b in bucket_iter:
                r = row_for(c, b, 'conservative')
                if r is None:
                    continue
                out.append(f'| {c} | {b} | '
                           f'{_fmt_pct(r.get("ik_convergence_rate_per_sample"))} | '
                           f'{_fmt(r.get("mean_n_ik_success_per_task"))} | '
                           f'{_fmt_pct(r.get("ik_all_fail_rate"))} |')
        out.append('')

    # Ablation decomposition (in meters, conservative)
    if all(c in cells for c in ('A', 'B', 'C', 'D')):
        out.append('## Ablation decomposition (median progress in meters, conservative)')
        out.append('')
        out.append('| bucket | n | A (m) | Δ_B (m) | Δ_C (m) | Δ_D (m) | '
                   '(Δ_B+Δ_C) | synergy |')
        out.append('|---|---|---|---|---|---|---|---|')
        for b in [bb for bb in bucket_iter if bb != 'ALL']:
            m = bucket_ref == b
            n = int(m.sum())
            pA = progress['A'][m]; pB = progress['B'][m]
            pC = progress['C'][m]; pD = progress['D'][m]
            mA = float(np.nanmedian(pA))
            dB = float(np.nanmedian(pB - pA))
            dC = float(np.nanmedian(pC - pA))
            dD = float(np.nanmedian(pD - pA))
            syn = dD - (dB + dC)
            out.append(f'| {b} | {n} | {mA:.3f} | {dB:+.3f} | {dC:+.3f} | {dD:+.3f} '
                       f'| {(dB+dC):+.3f} | **{syn:+.3f}** |')
        out.append('')

    # D vs E vs E'
    if 'D' in cells and 'E' in cells:
        out.append('## D vs E vs E\' (controller-aware oracle comparison)')
        out.append('')
        delta_DE = progress['D'] - progress['E']
        f = np.isfinite(delta_DE)
        out.append(f'**D vs E** (E = SMM-classical seed oracle, controller-mismatched):')
        out.append(f'- D > E on **{100*(delta_DE[f]>0).mean():.1f}%** of tasks; '
                   f'median(D−E) = **{np.median(delta_DE[f])*1000:+.1f} mm**')
        if 'E_prime' in cells:
            delta_DEp = progress['D'] - progress['E_prime']
            f = np.isfinite(delta_DEp)
            out.append(f'')
            out.append(f"**D vs E'** (E' = controller-aware oracle over SMM top-K' under hybrid):")
            out.append(f"- D > E' on **{100*(delta_DEp[f]>0).mean():.1f}%** of tasks; "
                       f"median(D−E') = **{np.median(delta_DEp[f])*1000:+.1f} mm**")
            delta_EpE = progress['E_prime'] - progress['E']
            f = np.isfinite(delta_EpE)
            out.append(f'')
            out.append(f"**E' vs E**:")
            out.append(f"- E' > E on **{100*(delta_EpE[f]>0).mean():.1f}%** of tasks; "
                       f"median(E'−E) = **{np.median(delta_EpE[f])*1000:+.1f} mm** "
                       "(this measures how much the SMM seed-oracle leaves on the table "
                       "by being controller-mismatched).")
        out.append('')

    return out


# ----------------------------------------------------------------------
# Figures (all in meters)
# ----------------------------------------------------------------------

def plot_deployment_gain(cells, progress, progress_real, bucket, out_path):
    n_buckets = len(BUCKET_ORDER)
    cells_present = [c for c in CELLS if c in cells]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.80 / max(len(cells_present), 1)
    x = np.arange(n_buckets)
    for i, c in enumerate(cells_present):
        # Use realistic for B/D so the bar reflects deploy behavior
        p = progress_real[c] if c in ('B', 'D') else progress[c]
        meds, p25s, p75s = [], [], []
        for b in BUCKET_ORDER:
            m = bucket == b
            a = p[m]; f = a[np.isfinite(a)]
            meds.append(float(np.median(f)) if len(f) else 0.0)
            p25s.append(float(np.percentile(f, 25)) if len(f) else 0.0)
            p75s.append(float(np.percentile(f, 75)) if len(f) else 0.0)
        meds = np.asarray(meds); p25s = np.asarray(p25s); p75s = np.asarray(p75s)
        offset = (i - (len(cells_present) - 1) / 2) * width
        is_oracle = c in ('E', 'E_prime')
        ax.bar(x + offset, meds, width, label=CELL_LABEL[c],
               color=CELL_COLOR[c],
               yerr=[np.maximum(meds - p25s, 0.0), np.maximum(p75s - meds, 0.0)],
               capsize=2,
               edgecolor='red' if is_oracle else 'none',
               linewidth=1.2 if is_oracle else 0,
               hatch='//' if c == 'E_prime' else ('\\\\' if c == 'E' else None))
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('progress (m, median, error bars = 25/75%)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title('Deployment performance (B, D use realistic IK-fallback)')
    ax.legend(loc='upper left', fontsize=7, ncol=1)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_recovery_distribution(cells, progress, bucket, oracle_cell, progress_O,
                                out_path):
    if progress_O is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in CELLS:
        if c not in cells:
            continue
        p = progress[c]
        with np.errstate(invalid='ignore'):
            r = p / np.where(np.isfinite(progress_O) & (progress_O > 0), progress_O, np.nan)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        r_sorted = np.sort(r)
        y = np.arange(1, len(r_sorted) + 1) / len(r_sorted)
        ax.plot(r_sorted, y, label=CELL_LABEL[c], color=CELL_COLOR[c],
                linewidth=2 if c == 'D' else 1.2,
                linestyle='--' if c in ('E', 'E_prime') else '-')
    ax.axvline(1.0, linestyle=':', color='black', alpha=0.5)
    ax.set_xlabel(f'progress / progress_{oracle_cell} (recovery ratio)')
    ax.set_ylabel('cumulative fraction of tasks')
    ax.set_title(f'Recovery CDF vs {oracle_cell}')
    ax.set_xlim(0, 1.5)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_ablation_decomposition(cells, progress, bucket, out_path):
    if not all(c in cells for c in ('A', 'B', 'C', 'D')):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BUCKET_ORDER))
    pA = progress['A']
    dB, dC, dD, syn = [], [], [], []
    for b in BUCKET_ORDER:
        m = bucket == b
        dB.append(float(np.nanmedian(progress['B'][m] - pA[m])))
        dC.append(float(np.nanmedian(progress['C'][m] - pA[m])))
        dD.append(float(np.nanmedian(progress['D'][m] - pA[m])))
        syn.append(dD[-1] - (dB[-1] + dC[-1]))
    ax.plot(x, dB, marker='o', label='B − A  (seed only)', color=CELL_COLOR['B'])
    ax.plot(x, dC, marker='s', label='C − A  (controller only)', color=CELL_COLOR['C'])
    ax.plot(x, dD, marker='^', label='D − A  (full method)', color=CELL_COLOR['D'], linewidth=2.5)
    ax.plot(x, np.asarray(dB) + np.asarray(dC), marker='x',
            label='(B−A) + (C−A) — additive prediction', color='gray', linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('median gain over baseline (m)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title('Ablation decomposition: seed × controller (median Δprogress)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    for xi, s in zip(x, syn):
        ax.annotate(f'syn={s*1000:+.0f}mm', xy=(xi, dD[xi]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8, color='dimgray')
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_oracle_gap(cells, progress, bucket, oracle_cell, out_path):
    if not all(c in cells for c in ('D', oracle_cell)):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BUCKET_ORDER))
    width = 0.30
    pD = progress['D']; pO = progress[oracle_cell]
    medD, medO = [], []
    for b in BUCKET_ORDER:
        m = bucket == b
        medD.append(float(np.nanmedian(pD[m])))
        medO.append(float(np.nanmedian(pO[m])))
    ax.bar(x - width / 2, medD, width, label='D (full method)', color=CELL_COLOR['D'])
    ax.bar(x + width / 2, medO, width, label=f'{oracle_cell} (oracle)',
           color=CELL_COLOR[oracle_cell], edgecolor='red', linewidth=1.5, hatch='//')
    for xi, lD, lO in zip(x, medD, medO):
        gap = lO - lD
        ax.annotate(f'gap {gap*1000:+.0f}mm', xy=(xi, max(lD, lO)),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('progress (m, median)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title(f'Oracle gap: D (deployment) vs {oracle_cell}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
