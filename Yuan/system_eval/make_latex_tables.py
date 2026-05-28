"""Publication-ready LaTeX tables (absolute progress in meters).

Tables produced in `summary_tables.tex`:
  1. Main 6-cell × bucket grid (A, B, C, D, E, E'); progress in meters,
     median (p25--p75), thresholded success rates, gain over baseline,
     recovery vs E'. For cells B and D also includes the *realistic*
     row (IK-all-fail tasks fall back to the q0_seed rollout).
  2. Ablation decomposition (median progress per bucket): A, Δ_B, Δ_C,
     Δ_D, (Δ_B+Δ_C), synergy. In meters.
  3. IK convergence per bucket: ok rate, mean #ok / task, all-fail rate.
  4. Oracle comparison: D vs E vs E' headline (single mini-table).

All use \\toprule / \\midrule / \\bottomrule; needs `\\usepackage{booktabs}`.

Usage:
    python -m Yuan.system_eval.make_latex_tables \\
        --in-dir Yuan/system_eval/runs/eval_10k_systematic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


CELLS = ['A', 'B', 'C', 'D', 'E', 'E_prime']
CELL_TITLE = {
    'A':       r'A \quad baseline (\(q_0^{\text{seed}}\) + Classical)',
    'B':       r'B \quad seed ablation (Diffusion + Classical)',
    'C':       r'C \quad controller ablation (\(q_0^{\text{seed}}\) + RL\(_\text{hyb}\))',
    'D':       r'D \quad \textbf{full method} (Diffusion + RL\(_\text{hyb}\))',
    'E':       r"E \quad seed-oracle (\(q_{\max\,L_{\text{cls}}}\) + RL\(_\text{hyb}\)) [\emph{controller-mismatched}]",
    'E_prime': r"E' \quad \textbf{controller-aware oracle} (\(\max\) over SMM top-\(K'\) + RL\(_\text{hyb}\))",
}
BUCKETS = ['weak', 'medium-weak', 'medium', 'strong', 'ALL']
BUCKET_LABEL = {
    'weak':        r'weak \([0.10,0.15)\)',
    'medium-weak': r'med-weak \([0.15,0.20)\)',
    'medium':      r'medium \([0.20,0.30)\)',
    'strong':      r'strong \([0.30,\infty)\)',
    'ALL':         r'\textsc{all}',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--in-dir', default=None)
    p.add_argument('--out', default=None)
    return p.parse_args()


def _f(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f'{x:.{prec}f}'


def _signed(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f'{x:+.{prec}f}'


def _pct(x, prec=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '--'
    return f'{100*x:.{prec}f}'


def _oracle_label(cell: str) -> str:
    return "E'" if cell == 'E_prime' else cell


def load_cells(in_dir: Path, pat: str):
    out = {}
    for c in CELLS:
        p = in_dir / pat.format(cell=c)
        if not p.exists():
            print(f'[latex] missing {p} — skipping cell {c}')
            continue
        z = np.load(p, allow_pickle=False)
        out[c] = {k: z[k] for k in z.files}
    return out


def _mask(cells, bucket):
    ref = next(iter(cells.values()))
    if bucket == 'ALL':
        return np.ones(int(ref['n_tasks']), dtype=bool)
    return ref['bucket'] == bucket


def _progress_best(cell_data, target_distance_m, fallback=None):
    p = cell_data['L_best'].astype(np.float64) * target_distance_m
    if fallback is not None:
        nan = ~np.isfinite(p)
        p[nan] = fallback[nan]
    return p


def metric_block(progress, mask, thr_m, prog_A, prog_O):
    a = progress[mask]
    finite = a[np.isfinite(a)]
    n = int(mask.sum())
    if not len(finite):
        return None
    row = {
        'n': n,
        'med': float(np.median(finite)),
        'p25': float(np.percentile(a[np.isfinite(a)], 25)),
        'p75': float(np.percentile(a[np.isfinite(a)], 75)),
    }
    for t in thr_m:
        row[f'succ_{t:g}'] = float((finite >= t).mean())
    # Gain & recovery
    if prog_A is not None:
        d = progress[mask] - prog_A[mask]
        d = d[np.isfinite(d)]
        row['gain_med'] = float(np.median(d)) if len(d) else float('nan')
    if prog_O is not None:
        with np.errstate(invalid='ignore'):
            r = progress[mask] / np.where(prog_O[mask] > 0, prog_O[mask], np.nan)
        r = r[np.isfinite(r)]
        row['recov_09'] = float((r >= 0.9).mean()) if len(r) else float('nan')
    return row


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    in_dir = Path(args.in_dir or cfg['output']['root'])
    out_path = Path(args.out or (in_dir / 'summary_tables.tex'))
    pat = cfg['output']['cell_results_pattern']
    cells = load_cells(in_dir, pat)
    if 'A' not in cells:
        raise SystemExit('[latex] need cell A to compute baselines / fallbacks')

    target_distance_m = float(cfg['env']['target_distance_m'])
    thr_m = list(cfg['metrics']['progress_thresholds_m'])
    n_total = int(cells['A']['n_tasks'])
    has_Eprime = 'E_prime' in cells

    # Conservative + realistic progress per cell
    progress = {}
    progress_real = {}
    prog_A = _progress_best(cells['A'], target_distance_m)
    for c, d in cells.items():
        progress[c] = _progress_best(d, target_distance_m, fallback=None)
        if c in ('B', 'D'):
            progress_real[c] = _progress_best(d, target_distance_m, fallback=prog_A)
        else:
            progress_real[c] = progress[c].copy()

    oracle_cell = 'E_prime' if has_Eprime else 'E'
    prog_O = progress[oracle_cell] if oracle_cell in progress else None

    lines = []
    lines.append('% ===========================================================')
    lines.append('% Auto-generated by Yuan/system_eval/make_latex_tables.py')
    lines.append(f'% Source: {in_dir}')
    lines.append(f'% N = {n_total} stratified safe held-out tasks')
    lines.append(f'% Oracle for recovery ratios: cell {oracle_cell}')
    lines.append(r'% Requires: \usepackage{booktabs}')
    lines.append('% ===========================================================')
    lines.append('')

    # ----- Table 1: main ablation grid (in meters) ------------------
    cols = (r'\begin{tabular}{l l r ccc ccc cc}')
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    cap = (
        r'\caption{Six-cell ablation on $N=' + str(n_total) + r'$ stratified safe '
        r'held-out tasks. \emph{Progress} = end-effector displacement along the '
        r'task line (meters). Rows for cells~B and~D report two variants: '
        r'\emph{cons.}\ (conservative: IK-all-fail tasks counted as $0$~m) and '
        r'\emph{real.}\ (realistic: IK-all-fail tasks fall back to the $q_0^{\text{seed}}$ '
        r'rollout from cell~A). Success rates are fractions of tasks with progress '
        r'$\geq \tau$ for $\tau\in\{' +
        ','.join(f'{t:g}' for t in thr_m) + r'\}$~m. '
        r'\emph{recov$_{\geq 0.9}$} is the per-task fraction with progress~$\geq 0.9 \cdot$'
        + _oracle_label(oracle_cell) +
        r'. $\Delta_A$\,med is the median per-task progress gain over the baseline '
        r'(cell~A), in meters.}'
    )
    lines.append(cap)
    lines.append(r'\label{tab:sys_eval_main}')
    lines.append(r'\footnotesize')
    lines.append(cols)
    lines.append(r'\toprule')
    lines.append(r'Cell / source & var. & $n$ & med (m) & p25 & p75 & '
                 r'$\geq\!' + f'{thr_m[0]:g}' + r'$m & $\geq\!' +
                 f'{thr_m[1]:g}' + r'$m & $\geq\!' + f'{thr_m[2]:g}' + r'$m & '
                 r'recov$_{\geq .9}$ & $\Delta_A$\,med \\')
    lines.append(r'\midrule')

    for c in CELLS:
        if c not in cells:
            continue
        lines.append(r'\multicolumn{11}{l}{\textit{' + CELL_TITLE[c] + r'}} \\')
        for b in BUCKETS:
            m = _mask(cells, b)
            n = int(m.sum())
            for variant in (('cons', 'real') if c in ('B', 'D') else ('cons',)):
                p_arr = progress[c] if variant == 'cons' else progress_real[c]
                r = metric_block(p_arr, m, thr_m,
                                 prog_A if c != 'A' else None, prog_O)
                if r is None:
                    continue
                row = [
                    r'\quad ' + BUCKET_LABEL[b],
                    variant + '.',
                    str(n),
                    _f(r['med']),
                    _f(r['p25']),
                    _f(r['p75']),
                    _pct(r.get(f'succ_{thr_m[0]:g}')),
                    _pct(r.get(f'succ_{thr_m[1]:g}')),
                    _pct(r.get(f'succ_{thr_m[2]:g}')),
                    _pct(r.get('recov_09')),
                    _signed(r.get('gain_med')) if c != 'A' else '--',
                ]
                lines.append(' & '.join(row) + r' \\')
        if c != CELLS[-1] and any(cc in cells for cc in CELLS[CELLS.index(c)+1:]):
            lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    lines.append('')

    # ----- Table 2: ablation decomposition (in meters) --------------
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation decomposition (median progress, meters, '
                 r'conservative). $\Delta_X = \mathrm{median}(\text{prog}_X - '
                 r'\text{prog}_A)$. Synergy $= \Delta_D - (\Delta_B + \Delta_C)$. '
                 r'Positive synergy means the seed-selection and controller gains '
                 r'compound rather than add.}')
    lines.append(r'\label{tab:sys_eval_ablation}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lr cccc cc}')
    lines.append(r'\toprule')
    lines.append(r'bucket & $n$ & $A$\,(m) & $\Delta_B$\,(m) & $\Delta_C$\,(m) & '
                 r'$\Delta_D$\,(m) & $\Delta_B{+}\Delta_C$ & synergy \\')
    lines.append(r'\midrule')
    for b in BUCKETS:
        if b == 'ALL':
            lines.append(r'\midrule')
        m = _mask(cells, b)
        n = int(m.sum())
        pA = progress['A'][m]; pB = progress['B'][m]
        pC = progress['C'][m]; pD = progress['D'][m]
        mA = float(np.nanmedian(pA))
        dB = float(np.nanmedian(pB - pA))
        dC = float(np.nanmedian(pC - pA))
        dD = float(np.nanmedian(pD - pA))
        sumBC = dB + dC; syn = dD - sumBC
        row = [BUCKET_LABEL[b], str(n), _f(mA),
               _signed(dB), _signed(dC), _signed(dD),
               _signed(sumBC),
               (r'\textbf{' + _signed(syn) + r'}' if syn > 0 else _signed(syn))]
        lines.append(' & '.join(row) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    # ----- Table 3: IK convergence ---------------------------------
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Diffusion + Newton-IK convergence (per-sample, '
                 r'$N{=}8$). \emph{all-fail} = fraction of tasks where every one '
                 r'of the 8 IK refinements failed; these tasks fall back to '
                 r'$q_0^{\text{seed}}$ in the \emph{real.} variant.}')
    lines.append(r'\label{tab:sys_eval_ik}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lr ccc}')
    lines.append(r'\toprule')
    lines.append(r'bucket & $n$ & IK ok (\%) & mean \#ok / task & all-fail (\%) \\')
    lines.append(r'\midrule')
    for b in BUCKETS:
        if b == 'ALL':
            lines.append(r'\midrule')
        m = _mask(cells, b)
        n = int(m.sum())
        ik = cells['B']['ik_ok'][m]
        if ik.size == 0:
            continue
        lines.append(' & '.join([
            BUCKET_LABEL[b], str(n),
            _pct(float(ik.mean())),
            _f(float(ik.sum(axis=1).mean()), prec=2),
            _pct(float((~ik).all(axis=1).mean())),
        ]) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    lines.append('')

    # ----- Table 4: D vs E vs E' oracle comparison ----------------
    if 'D' in cells and 'E' in cells:
        lines.append(r'\begin{table}[t]')
        lines.append(r'\centering')
        lines.append("\\caption{Oracle comparison. The classical-derived "
                     "seed-oracle~$E$ is \\emph{not} a true upper bound under "
                     "the hybrid deployment controller; the controller-aware "
                     "oracle~$E'$ rolls each of the $K'=6$ SMM top-$K'$ "
                     "candidates through the hybrid controller and takes the "
                     "per-task max. Median deltas in mm; \\%~is fraction of "
                     "pair-finite tasks where the row label exceeds the column.}")
        lines.append(r'\label{tab:sys_eval_oracle_cmp}')
        lines.append(r'\small')
        lines.append(r'\begin{tabular}{l rr rr}')
        lines.append(r'\toprule')
        lines.append(r'comparison & \% wins & median $\Delta$ (mm) & n \\')
        lines.append(r'\midrule')
        def _rep(label, a, b):
            d = a - b
            f = np.isfinite(d)
            return f'{label} & {100*(d[f]>0).mean():.1f}\\% & {1000*np.median(d[f]):+.1f} & {f.sum()}'
        lines.append(_rep(r"D $>$ E (old, controller-mismatched)",
                          progress['D'], progress['E']) + r' \\')
        if has_Eprime:
            lines.append(_rep(r"D $>$ $E'$ (controller-aware oracle)",
                              progress['D'], progress['E_prime']) + r' \\')
            lines.append(_rep(r"$E'$ $>$ E (oracle gap from controller mismatch)",
                              progress['E_prime'], progress['E']) + r' \\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\end{table}')
        lines.append('')

    out_path.write_text('\n'.join(lines))
    print(f'[latex] wrote {out_path} ({len(lines)} lines)')

    # Standalone driver
    standalone = (
        r'\documentclass{article}' '\n'
        r'\usepackage{booktabs,multirow,geometry,xspace}' '\n'
        r'\providecommand{\textsc}[1]{\protect\scshape #1}' '\n'
        r'\geometry{a4paper,margin=1.5cm,landscape}' '\n'
        r'\begin{document}' '\n'
        r'\input{summary_tables.tex}' '\n'
        r'\end{document}' '\n'
    )
    (out_path.parent / 'summary_tables_standalone.tex').write_text(standalone)
    print(f'[latex] standalone driver → {out_path.parent / "summary_tables_standalone.tex"}')


if __name__ == '__main__':
    main()
