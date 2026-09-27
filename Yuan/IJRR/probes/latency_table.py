"""tab:latency — decision time per 50 ms control period of the horizon
baselines and the reactive policy, from runs/paper_fill/horizon/
latency_{robot}.json (probes/horizon_latency.py, idle RTX 4090).

usage: python Yuan/IJRR/probes/latency_table.py [--write]
"""
import json
import re
import sys
from pathlib import Path

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/horizon')
TEX = Path('/home/lqin/one/Yuan/IJRR/2026_Yuan_RAL/main.tex')
ROBOTS = ['fr3', 'xarm7', 'cobotta']
ROWS = [('MPC', ['mpc10', 'mpc20', 'mpc30']),
        ('MPPI', ['mppi16', 'mppi32', 'mppi64'])]


BASE = [('zero', 'Zero Null-Space'), ('classical', 'Classical Gradient'),
        ('cont', 'Continuous PPO'), ('hybrid', 'PPO + Classical')]


def load():
    L = {}
    for r in ROBOTS:
        d = {}
        for f in (f'latency_{r}.json', f'latency_base_{r}.json'):
            if (OUT / f).exists():
                d.update(json.load(open(OUT / f)))
        if d:
            L[r] = d
    return L


def f_ms(x):
    """Latency in ms: seconds above 1000 ms, else ms with sensible digits."""
    if x >= 1000:
        return f'{x / 1000:.1f}\\,s'
    if x >= 10:
        return f'{x:.0f}\\,ms'
    if x >= 1:
        return f'{x:.2f}\\,ms'
    if x >= 0.01:
        return f'{x:.2f}\\,ms'
    return f'{x * 1000:.1f}\\,$\\mu$s'


def cells(L, key, bold=False):
    out = []
    for r in ROBOTS:
        if r in L and key in L[r]:
            a = f_ms(L[r][key]['batch1_ms'])
            b = f_ms(L[r][key]['batch2500_ms_per_task'])
        else:
            a = b = '[XX]'
        if bold:
            a, b = f'\\textbf{{{a}}}', f'\\textbf{{{b}}}'
        out += [a, b]
    return ' & '.join(out)


def body(L):
    lines = []
    for key, name in BASE:
        lines.append(f'{name} & -- & {cells(L, key)} \\\\')
    lines.append('\\midrule')
    for name, keys in ROWS:
        for i, k in enumerate(keys):
            H = re.sub(r'\D', '', k)
            meth = f'\\multirow{{3}}{{*}}{{{name}}}' if i == 0 else ' '
            lines.append(f'{meth} & {H} & {cells(L, k)} \\\\')
        lines.append('\\midrule')
    lines.append(f'\\textbf{{Ours}} & -- & {cells(L, "ours", bold=True)} \\\\')
    lines.append('\\bottomrule')
    return '\n'.join(lines)


TEMPLATE = r"""\begin{table*}[!htbp]
\centering
\caption{Decision Time per $50$\,ms Control Period of All Compared
Resolution Laws (Median over $20$ Periods on One RTX 4090)}
\label{tab:latency}
\begin{threeparttable}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{2}{c}{Franka Research 3}
& \multicolumn{2}{c}{xArm7}
& \multicolumn{2}{c}{Cobotta} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
Method & $H$
& \shortstack{Single\\task} & \shortstack{Batched\\per task}
& \shortstack{Single\\task} & \shortstack{Batched\\per task}
& \shortstack{Single\\task} & \shortstack{Batched\\per task} \\
\midrule
%%ROWS%%
\end{tabular}
\begin{tablenotes}
    \item[Note 1] Single task: wall time of one decision for one
    manipulator (batch size $1$), the latency a real-time controller pays
    every period. Batched per task: wall time per task when $2{,}500$
    tasks are decided together on the GPU (the evaluation batch), i.e.,
    the throughput-amortized cost.
    \item[Note 2] All rows time the action selection only; the
    null-space projection inside the control step is shared by every
    method and excluded. Continuous PPO and PPO + Classical use the
    width-$512$ policies of Table~\ref{tab:mainresult}; MPC uses eight
    Adam iterations; MPPI uses $K=64$ rollouts.
\end{tablenotes}
\end{threeparttable}
\end{table*}
"""

if __name__ == '__main__':
    L = load()
    b = body(L)
    print(b)
    if '--write' in sys.argv:
        tex = TEX.read_text()
        m = re.search(r'(\\label\{tab:latency\}.*?\\midrule\n)(.*?)(\\end\{tabular\})',
                      tex, re.S)
        if m:
            tex = tex[:m.start(2)] + b + '\n' + tex[m.end(2):]
        else:
            i = tex.index('\\label{tab:horizon}')
            j = tex.index('\\end{table*}', i) + len('\\end{table*}')
            tex = tex[:j] + '\n\n' + TEMPLATE.replace('%%ROWS%%', b) + tex[j:]
        TEX.write_text(tex)
        print(f'wrote {TEX}')
