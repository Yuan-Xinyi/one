"""Fill tab:horizon (MPC / MPPI vs Ours across horizons) from
runs/paper_fill/horizon/*.json. Ours = the DirFrac rows of tab:mainresult
(same tasks, same references). Missing cells print as [XX].

usage: python Yuan/IJRR/probes/horizon_table.py [--write main.tex]
"""
import json
import re
import sys
from pathlib import Path

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/horizon')
TEX = Path('/home/lqin/one/Yuan/IJRR/2026_Yuan_RAL/main.tex')
ROBOTS = ['fr3', 'xarm7', 'cobotta']
FAMS = [('straight', 'Straight'), ('serpentine', 'Serpentine'),
        ('nonplanar', 'Rot.-Axis')]
METHODS = [('mpc', 'MPC', [10, 20, 30]), ('mppi', 'MPPI', [16, 32, 64])]
# DirFrac Reactive rows of tab:mainresult (stroke, mean, p10)
OURS = {('fr3', 'straight'): (0.628, 93.3, 77.3),
        ('fr3', 'serpentine'): (0.636, 91.6, 76.3),
        ('fr3', 'nonplanar'): (0.625, 92.4, 72.8),
        ('xarm7', 'straight'): (0.540, 91.9, 68.8),
        ('xarm7', 'serpentine'): (0.543, 89.9, 67.1),
        ('xarm7', 'nonplanar'): (0.541, 91.9, 68.5)}


TEMPLATE = r"""\begin{table*}[!htbp]
\centering
\caption{Comparison of MPC and MPPI under Different Prediction Horizons across Path Families
($10{,}000$ Straight and $2{,}500$ Tasks per Curved Family per Manipulator)}
\label{tab:horizon}
\begin{threeparttable}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lllcccccc}
\toprule
& & & \multicolumn{2}{c}{Franka Research 3}
& \multicolumn{2}{c}{xArm7}
& \multicolumn{2}{c}{Cobotta} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Family
& Method
& $H$
& \shortstack{Stroke\\(m)} & \shortstack{Ratio (\%)\\mean / p10}
& \shortstack{Stroke\\(m)} & \shortstack{Ratio (\%)\\mean / p10}
& \shortstack{Stroke\\(m)} & \shortstack{Ratio (\%)\\mean / p10} \\
\midrule
%%ROWS%%
\end{tabular}
\begin{tablenotes}
    \item[Note 1] All methods share the same per-task initial configuration.
    Ratio is the per-task ratio of the executed stroke to the task's
    pointwise-reachable length $\ell^{\mathrm{pw}}$, reported in percent
    as mean / p10.
    \item[Note 2] $H$ denotes the prediction horizon in control steps;
    the proposed reactive controller has no horizon (--).
    \item[Note 3] Cobotta is evaluated at the commanded speed
    $v=0.05$\,m/s for all methods, matched to its joint-velocity limits
    ($0.39$--$1.11$\,rad/s); FR3 and xArm7 use $v=0.2$\,m/s.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""


def ours(robot, fam):
    """Ours: tab:mainresult numbers for FR3/xArm7; for Cobotta the v = 0.05
    m/s rows written by probes/cobotta_v005_ours.py."""
    if (robot, fam) in OURS:
        return OURS[(robot, fam)]
    f = OUT / f'{robot}_{fam}_ours.json'
    if not f.exists():
        return None
    r = json.load(open(f))
    return r['stroke'], r['ratio_mean'], r['ratio_p10']


def cell(robot, fam, method, H):
    f = OUT / f'{robot}_{fam}_{method}{H}.json'
    if not f.exists():
        return None
    r = json.load(open(f))
    return r['stroke'], r['ratio_mean'], r['ratio_p10'], r['ms_per_task_period']


def fmt(c, bold=False):
    if c is None:
        s = ['[XX]', '[XX]']
    else:
        s = [f'{c[0]:.3f}', f'{c[1]:.1f} / {c[2]:.1f}']
    if bold:
        s = [f'\\textbf{{{x}}}' for x in s]
    return ' & '.join(s)


def rows():
    lines = []
    for fi, (fam, fname) in enumerate(FAMS):
        first = True
        for mi, (m, mname, Hs) in enumerate(METHODS):
            for hi, H in enumerate(Hs):
                lead = (f'\\multirow{{7}}{{*}}{{{fname}}}' if first else ' ')
                first = False
                meth = f'\\multirow{{3}}{{*}}{{{mname}}}' if hi == 0 else ' '
                cells = ' & '.join(fmt(cell(r, fam, m, H)) for r in ROBOTS)
                lines.append(f'{lead} & {meth} & {H} & {cells} \\\\')
            lines.append('\\cmidrule(lr){2-9}')
        cells = ' & '.join(fmt(ours(r, fam), bold=True) for r in ROBOTS)
        lines.append(f' & \\textbf{{Ours}} & -- & {cells} \\\\')
        lines.append('\\bottomrule' if fi == len(FAMS) - 1 else '\\midrule')
    return '\n'.join(lines)


def timing():
    out = []
    for r in ROBOTS:
        for fam, _ in FAMS:
            for m, _, Hs in METHODS:
                for H in Hs:
                    c = cell(r, fam, m, H)
                    if c:
                        out.append(f'{r:6s} {fam:10s} {m}{H:<3d} '
                                   f'{c[3]:7.2f} ms/task-period')
    return '\n'.join(out)


if __name__ == '__main__':
    body = rows()
    print(body)
    print('\n% planning cost\n' + timing())
    if '--write' in sys.argv:
        tex = TEX.read_text()
        m = re.search(r'(\\label\{tab:horizon\}.*?\\midrule\n)(.*?)(\\end\{tabular\})',
                      tex, re.S)
        if m:
            tex = tex[:m.start(2)] + body + '\n' + tex[m.end(2):]
        else:
            # first fill: insert the whole table right after tab:mainresult
            table = TEMPLATE.replace('%%ROWS%%', body)
            i = tex.index('\\label{tab:mainresult}')
            j = tex.index('\\end{table*}', i) + len('\\end{table*}')
            tex = tex[:j] + '\n\n' + table + tex[j:]
        TEX.write_text(tex)
        print(f'\nwrote {TEX}')
