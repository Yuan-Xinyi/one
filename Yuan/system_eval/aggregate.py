"""Aggregate cell_<name>_results.npz into summary table, report, figures.

Reporting units: absolute EE progress in METERS (= L_best * target_distance_m).
The 1.5m normalizer was an arbitrary constant; absolute meters are what
deployment cares about. Per-task ratios (L_X / L_oracle) stay meaningful.

For diffusion cells (diff_cls, diff_hyb), two reductions are reported:
    progress_best_m      best-of-N with IK-fails counted as 0          (conservative)
    progress_realistic_m best-of-N; if ALL N fail IK, fallback to cls_cls (realistic)
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
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


# Cells in display order. Naming convention: <seed_source>_<controller>.
#   cls = pilot q0_seed,            classical = Yoshikawa nullspace
#   diff = diffusion seed,          hyb       = hybrid (RL + Classical, variant B)
#   oracle_cls = label-argmax seed (classical-label oracle, deployed under hybrid)
#   oracle_hyb = controller-aware oracle (max over SMM top-K' under hybrid)
CELLS = ['cls_cls', 'diff_cls', 'cls_hyb', 'diff_hyb', 'oracle_cls', 'oracle_hyb']
BASELINE_CELL = 'cls_cls'
FULL_METHOD_CELL = 'diff_hyb'
DIFFUSION_CELLS = ('diff_cls', 'diff_hyb')
ORACLE_CELLS = ('oracle_cls', 'oracle_hyb')

CELL_LABEL = {
    'cls_cls':    'cls_cls: baseline (q0_seed + Classical)',
    'diff_cls':   'diff_cls: seed ablation (Diffusion + Classical)',
    'cls_hyb':    'cls_hyb: controller ablation (q0_seed + RL hybrid)',
    'diff_hyb':   'diff_hyb: full method (Diffusion + RL hybrid)',
    'oracle_cls': 'oracle_cls: classical-label oracle (controller-mismatched)',
    'oracle_hyb': 'oracle_hyb: controller-aware oracle (true ceiling under hybrid)',
}
CELL_COLOR = {
    'cls_cls':    '#888888',
    'diff_cls':   '#1f77b4',
    'cls_hyb':    '#ff7f0e',
    'diff_hyb':   '#2ca02c',
    'oracle_cls': '#d62728',
    'oracle_hyb': '#8b0000',
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


def _best_of_N_progress(cell_data: dict, target_distance_m: float,
                        fallback_progress: np.ndarray | None = None) -> np.ndarray:
    """Per-task best-of-N progress (m). If `fallback_progress` is given,
    replace NaN entries (all-IK-failed tasks for diffusion cells) with it."""
    progress = cell_data['L_best'].astype(np.float64) * target_distance_m
    if fallback_progress is not None:
        nan_mask = ~np.isfinite(progress)
        progress[nan_mask] = fallback_progress[nan_mask]
    return progress.astype(np.float32)


def _bucket_metrics(progress, bucket_mask, thresholds_m, catastrophic_m):
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
            if args.require_all and c != 'oracle_hyb':  # oracle_hyb is optional
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
    # Realistic: fallback to baseline progress on those tasks
    progress_baseline = (_best_of_N_progress(cells[BASELINE_CELL], target_distance_m)
                         if BASELINE_CELL in cells else None)

    progress = {}             # conservative version
    progress_real = {}        # realistic version (baseline-fallback for diffusion)
    for c, d in cells.items():
        progress[c] = _best_of_N_progress(d, target_distance_m, fallback_progress=None)
        if c in DIFFUSION_CELLS and progress_baseline is not None:
            progress_real[c] = _best_of_N_progress(d, target_distance_m,
                                                    fallback_progress=progress_baseline)
        else:
            progress_real[c] = progress[c].copy()

    bucket_masks = {b: (bucket_ref == b) for b in BUCKET_ORDER
                    if (bucket_ref == b).any()}
    bucket_masks['ALL'] = np.ones(n_tasks, dtype=bool)
    bucket_iter = list(bucket_masks.keys())

    # ---- Choose the canonical oracle for ratios: oracle_hyb if available else oracle_cls
    oracle_cell = 'oracle_hyb' if 'oracle_hyb' in cells else 'oracle_cls'
    progress_O = progress[oracle_cell] if oracle_cell in progress else None
    print(f'[aggregate] cells loaded: {sorted(cells.keys())}; '
          f'oracle for ratios = {oracle_cell}; n_tasks={n_tasks}')

    # ---- Main rows ----------------------------------------------------
    rows = []
    for c in cells:
        for b in bucket_iter:
            m = bucket_masks[b]
            row = {'cell': c, 'bucket': b, 'variant': 'conservative',
                   **_bucket_metrics(progress[c], m, thresholds_m, catastrophic_m)}
            if progress_baseline is not None:
                d = progress[c][m] - progress_baseline[m]
                row['gain_over_baseline_median_m'] = float(np.nanmedian(d))
                with np.errstate(invalid='ignore'):
                    r = progress[c][m] / np.where(progress_baseline[m] > 0, progress_baseline[m], np.nan)
                rf = r[np.isfinite(r)]
                row['mean_ratio_vs_baseline'] = float(rf.mean()) if len(rf) else float('nan')
            if progress_O is not None:
                with np.errstate(invalid='ignore'):
                    r = progress[c][m] / np.where(progress_O[m] > 0, progress_O[m], np.nan)
                rf = r[np.isfinite(r)]
                row['mean_ratio_vs_oracle'] = float(rf.mean()) if len(rf) else float('nan')
                row[f'recovery_vs_oracle_at_{recovery_frac}'] = (
                    float((rf >= recovery_frac).mean()) if len(rf) else float('nan'))

            if c in DIFFUSION_CELLS and 'ik_ok' in cells[c]:
                ik = cells[c]['ik_ok'][m]
                row['ik_convergence_rate_per_sample'] = float(ik.mean()) if ik.size else float('nan')
                row['mean_n_ik_success_per_task'] = float(ik.sum(axis=1).mean()) if ik.size else float('nan')
                row['ik_all_fail_rate'] = float((~ik).all(axis=1).mean()) if ik.size else float('nan')
            rows.append(row)

            # Realistic variant for diffusion cells (baseline-fallback on all-IK-fail tasks)
            if c in DIFFUSION_CELLS:
                row_r = {'cell': c, 'bucket': b, 'variant': 'realistic_baseline_fallback',
                         **_bucket_metrics(progress_real[c], m, thresholds_m, catastrophic_m)}
                if progress_baseline is not None:
                    d = progress_real[c][m] - progress_baseline[m]
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
    md_lines = build_report(cells, rows, progress, progress_real, progress_baseline,
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


def build_report(cells, rows, progress, progress_real, progress_baseline, progress_O,
                 oracle_cell, bucket_ref, bucket_iter, thresholds_m, recovery_frac,
                 catastrophic_m, n_tasks, target_distance_m):
    out = []
    out.append(f'# System eval — seed x controller ablation (N={n_tasks})')
    out.append('')
    oracle_note = ("controller-aware" if oracle_cell == 'oracle_hyb'
                   else "classical-label oracle (controller-mismatched)")
    out.append(f'All progress values in **meters** (raw EE displacement along the '
               f'task line). Conservative variant: best-of-N over diffusion samples, '
               f'all-IK-fail tasks counted as progress=0. Realistic variant: '
               f'all-IK-fail tasks fall back to the baseline rollout ({BASELINE_CELL}). '
               f'Oracle used for recovery ratios = **{oracle_cell}** ({oracle_note}).')
    out.append('')
    out.append('| cell | source |')
    out.append('|---|---|')
    for c in CELLS:
        if c in cells:
            out.append(f'| {c} | {CELL_LABEL[c]} |')
    out.append('')

    hdr = ['cell', 'variant', 'n', 'prog_med (m)', 'p25', 'p75',
           f'>={thresholds_m[0]:g}m', f'>={thresholds_m[1]:g}m', f'>={thresholds_m[2]:g}m',
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
            if c in DIFFUSION_CELLS:
                r = row_for(c, b, variant_label)
            else:
                r = row_for(c, b, 'conservative')
            if r is None:
                continue
            row = [c, r['variant'][:14], str(r['n_tasks']),
                   _fmt(r['progress_median_m']),
                   _fmt(r['progress_p25_m']), _fmt(r['progress_p75_m']),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[0]:g}m')),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[1]:g}m')),
                   _fmt_pct(r.get(f'success_progress_geq_{thresholds_m[2]:g}m')),
                   _fmt_pct(r.get(f'recovery_vs_oracle_at_{recovery_frac}')),
                   _fmt(r.get('gain_over_baseline_median_m'))]
            out.append('| ' + ' | '.join(row) + ' |')
        out.append('')

    out.append('## Overall (ALL tasks) — CONSERVATIVE (IK fail = 0)')
    out.append('')
    emit_table('ALL', 'conservative')

    out.append(f'## Overall (ALL tasks) — REALISTIC (IK fail -> {BASELINE_CELL} fallback)')
    out.append('')
    emit_table('ALL', 'realistic_baseline_fallback')

    out.append('## Per-bucket break-down (conservative variant)')
    out.append('')
    for b in bucket_iter:
        if b == 'ALL':
            continue
        out.append(f'### Bucket: {b}')
        emit_table(b, 'conservative')

    if any(c in cells for c in DIFFUSION_CELLS):
        out.append('## Diffusion-specific (IK convergence)')
        out.append('')
        out.append('| cell | bucket | ik_rate | mean_n_ok | all_fail_rate |')
        out.append('|---|---|---|---|---|')
        for c in DIFFUSION_CELLS:
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

    ablation_cells = (BASELINE_CELL, 'diff_cls', 'cls_hyb', FULL_METHOD_CELL)
    if all(c in cells for c in ablation_cells):
        out.append('## Ablation decomposition (median progress in meters, conservative)')
        out.append('')
        out.append('| bucket | n | baseline (m) | seed_only_d | ctrl_only_d | full_d | '
                   '(seed+ctrl)_d | synergy |')
        out.append('|---|---|---|---|---|---|---|---|')
        for b in [bb for bb in bucket_iter if bb != 'ALL']:
            m = bucket_ref == b
            n = int(m.sum())
            p_base = progress[BASELINE_CELL][m]
            p_seed = progress['diff_cls'][m]
            p_ctrl = progress['cls_hyb'][m]
            p_full = progress[FULL_METHOD_CELL][m]
            mB = float(np.nanmedian(p_base))
            dS = float(np.nanmedian(p_seed - p_base))
            dC = float(np.nanmedian(p_ctrl - p_base))
            dF = float(np.nanmedian(p_full - p_base))
            syn = dF - (dS + dC)
            out.append(f'| {b} | {n} | {mB:.3f} | {dS:+.3f} | {dC:+.3f} | {dF:+.3f} '
                       f'| {(dS+dC):+.3f} | **{syn:+.3f}** |')
        out.append('')

    if FULL_METHOD_CELL in cells and 'oracle_cls' in cells:
        out.append('## Full method vs oracles')
        out.append('')
        delta = progress[FULL_METHOD_CELL] - progress['oracle_cls']
        f = np.isfinite(delta)
        out.append(f'**{FULL_METHOD_CELL} vs oracle_cls** '
                   f'(classical-label oracle, controller-mismatched):')
        out.append(f'- {FULL_METHOD_CELL} > oracle_cls on '
                   f'**{100*(delta[f]>0).mean():.1f}%** of tasks; '
                   f'median delta = **{np.median(delta[f])*1000:+.1f} mm**')
        if 'oracle_hyb' in cells:
            delta_full_hyb = progress[FULL_METHOD_CELL] - progress['oracle_hyb']
            f = np.isfinite(delta_full_hyb)
            out.append('')
            out.append(f'**{FULL_METHOD_CELL} vs oracle_hyb** '
                       f'(controller-aware oracle, true upper bound under hybrid):')
            out.append(f'- {FULL_METHOD_CELL} > oracle_hyb on '
                       f'**{100*(delta_full_hyb[f]>0).mean():.1f}%** of tasks; '
                       f'median delta = **{np.median(delta_full_hyb[f])*1000:+.1f} mm**')
            delta_oracles = progress['oracle_hyb'] - progress['oracle_cls']
            f = np.isfinite(delta_oracles)
            out.append('')
            out.append('**oracle_hyb vs oracle_cls** (oracle gap from controller mismatch):')
            out.append(f'- oracle_hyb > oracle_cls on **{100*(delta_oracles[f]>0).mean():.1f}%** of tasks; '
                       f'median delta = **{np.median(delta_oracles[f])*1000:+.1f} mm** '
                       '(this measures how much the SMM seed-oracle leaves on the table '
                       'by being controller-mismatched).')
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
        # Use realistic for diffusion cells so the bar reflects deploy behavior.
        p = progress_real[c] if c in DIFFUSION_CELLS else progress[c]
        meds, p25s, p75s = [], [], []
        for b in BUCKET_ORDER:
            m = bucket == b
            a = p[m]; f = a[np.isfinite(a)]
            meds.append(float(np.median(f)) if len(f) else 0.0)
            p25s.append(float(np.percentile(f, 25)) if len(f) else 0.0)
            p75s.append(float(np.percentile(f, 75)) if len(f) else 0.0)
        meds = np.asarray(meds); p25s = np.asarray(p25s); p75s = np.asarray(p75s)
        offset = (i - (len(cells_present) - 1) / 2) * width
        is_oracle = c in ORACLE_CELLS
        ax.bar(x + offset, meds, width, label=CELL_LABEL[c],
               color=CELL_COLOR[c],
               yerr=[np.maximum(meds - p25s, 0.0), np.maximum(p75s - meds, 0.0)],
               capsize=2,
               edgecolor='red' if is_oracle else 'none',
               linewidth=1.2 if is_oracle else 0,
               hatch='//' if c == 'oracle_hyb' else ('\\\\' if c == 'oracle_cls' else None))
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('progress (m, median, error bars = 25/75%)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title('Deployment performance (diffusion cells use realistic IK-fallback)')
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
                linewidth=2 if c == FULL_METHOD_CELL else 1.2,
                linestyle='--' if c in ORACLE_CELLS else '-')
    ax.axvline(1.0, linestyle=':', color='black', alpha=0.5)
    ax.set_xlabel(f'progress / progress_{oracle_cell} (recovery ratio)')
    ax.set_ylabel('cumulative fraction of tasks')
    ax.set_title(f'Recovery CDF vs {oracle_cell}')
    ax.set_xlim(0, 1.5)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_ablation_decomposition(cells, progress, bucket, out_path):
    ablation_cells = (BASELINE_CELL, 'diff_cls', 'cls_hyb', FULL_METHOD_CELL)
    if not all(c in cells for c in ablation_cells):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BUCKET_ORDER))
    p_base = progress[BASELINE_CELL]
    dS, dC, dF, syn = [], [], [], []
    for b in BUCKET_ORDER:
        m = bucket == b
        dS.append(float(np.nanmedian(progress['diff_cls'][m] - p_base[m])))
        dC.append(float(np.nanmedian(progress['cls_hyb'][m] - p_base[m])))
        dF.append(float(np.nanmedian(progress[FULL_METHOD_CELL][m] - p_base[m])))
        syn.append(dF[-1] - (dS[-1] + dC[-1]))
    ax.plot(x, dS, marker='o', label='diff_cls - cls_cls  (seed only)',
            color=CELL_COLOR['diff_cls'])
    ax.plot(x, dC, marker='s', label='cls_hyb - cls_cls  (controller only)',
            color=CELL_COLOR['cls_hyb'])
    ax.plot(x, dF, marker='^', label='diff_hyb - cls_cls  (full method)',
            color=CELL_COLOR[FULL_METHOD_CELL], linewidth=2.5)
    ax.plot(x, np.asarray(dS) + np.asarray(dC), marker='x',
            label='(seed-only) + (ctrl-only) — additive prediction',
            color='gray', linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('median gain over baseline (m)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title('Ablation decomposition: seed x controller (median delta-progress)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    for xi, s in zip(x, syn):
        ax.annotate(f'syn={s*1000:+.0f}mm', xy=(xi, dF[xi]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8, color='dimgray')
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_oracle_gap(cells, progress, bucket, oracle_cell, out_path):
    if not all(c in cells for c in (FULL_METHOD_CELL, oracle_cell)):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BUCKET_ORDER))
    width = 0.30
    p_full = progress[FULL_METHOD_CELL]; pO = progress[oracle_cell]
    medF, medO = [], []
    for b in BUCKET_ORDER:
        m = bucket == b
        medF.append(float(np.nanmedian(p_full[m])))
        medO.append(float(np.nanmedian(pO[m])))
    ax.bar(x - width / 2, medF, width, label=f'{FULL_METHOD_CELL} (full method)',
           color=CELL_COLOR[FULL_METHOD_CELL])
    ax.bar(x + width / 2, medO, width, label=f'{oracle_cell} (oracle)',
           color=CELL_COLOR[oracle_cell], edgecolor='red', linewidth=1.5, hatch='//')
    for xi, lF, lO in zip(x, medF, medO):
        gap = lO - lF
        ax.annotate(f'gap {gap*1000:+.0f}mm', xy=(xi, max(lF, lO)),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel('progress (m, median)')
    ax.set_xlabel('L_seed bucket')
    ax.set_title(f'Oracle gap: {FULL_METHOD_CELL} (deployment) vs {oracle_cell}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
