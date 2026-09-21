"""Aggregate the overnight run artifacts into table-ready numbers.

Reads whatever exists under runs/paper_fill (plus the selector-OOD report and
the recalibrated bound) and writes one markdown report whose sections map
one-to-one onto the paper's tables. Every section degrades gracefully: a
missing artifact produces a MISSING line, not a crash, so the report is
deliverable at any point of the queue.

Usage:
    python -m Yuan.IJRR.eval.fill_report --out Yuan/IJRR/runs/paper_fill/report.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FILL = REPO / 'Yuan/IJRR/runs/paper_fill'
TERM_NAMES = {0: 'alive', 2: 'collision', 3: 'cone', 4: 'jl',
              5: 'truncated', 6: 'lateral'}
ARM_LABELS = {
    'zero': 'No null-space (a=0)', 'classical': 'Classical law',
    'sgngrad': 'Margin law, gradient form', 'myopic': 'Margin law, one-step (ours)',
    'vertex': 'Vertex PPO', 'cem4': 'Receding CEM H=4',
    'cem8': 'Receding CEM H=8', 'cem16': 'Receding CEM H=16',
    'mcem4': 'Margin-CEM H=4', 'mcem8': 'Margin-CEM H=8',
    'mcem16': 'Margin-CEM H=16', 'beam8x8': 'Beam search H=8',
}


def sec(title):
    return [f'\n## {title}\n']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='Yuan/IJRR/runs/paper_fill/report.md')
    ap.add_argument('--n', type=int, default=10000)
    a = ap.parse_args()
    L = ['# Paper fill report (auto-generated)\n']

    # ---- bound -----------------------------------------------------------
    bound = None
    L += sec('tab:bound — recalibrated pointwise bound')
    bp = FILL / f'bound_{a.n}.npz'
    if bp.exists():
        b = np.load(bp)
        bound = b['L_hi'].astype(np.float64)
        L.append(f'- tasks: {len(bound)}; L_hi mean {bound.mean():.3f} m, '
                 f'median {np.median(bound):.3f} m')
        cp = FILL / 'bound_convergence.npz'
        if cp.exists():
            c = np.load(cp)
            n_c = len(c['L_hi'])
            d = c['L_hi'] - bound[c['idx']] if 'idx' in c.files \
                else c['L_hi'] - bound[:n_c]
            L.append(f'- budget x2 on {n_c} tasks: mean change '
                     f'{d.mean() * 1000:+.1f} mm, max {np.abs(d).max() * 1000:.1f} mm, '
                     f'changed >2 cm on {(np.abs(d) > 0.02).mean():.2%}')
        else:
            L.append('- convergence run MISSING')
    else:
        L.append('- MISSING bound npz')

    # ---- main result -----------------------------------------------------
    L += sec('tab:mainresult — realized fraction (FR3, full set)')
    arms = sorted(FILL.glob(f'arm_*_{a.n}.npz'))
    strokes = {}
    if arms and bound is not None:
        terc = np.quantile(bound, [1 / 3, 2 / 3])
        idx_b = {'L': bound >= terc[1],
                 'M': (bound >= terc[0]) & (bound < terc[1]),
                 'S': bound < terc[0]}
        L.append('| method | All m / % | L | M | S | ms |')
        L.append('|---|---|---|---|---|---|')
        for f in arms:
            name = re.sub(f'_{a.n}$', '', f.stem[4:])
            d = np.load(f)
            p = d['prog'].astype(np.float64)
            strokes[name] = p
            frac = np.clip(p, 0, None) / np.maximum(bound[:len(p)], 1e-6)
            cells = [f'{p.mean():.3f} / {100 * frac.mean():.1f}']
            for k in 'LMS':
                m = idx_b[k][:len(p)]
                cells.append(f'{p[m].mean():.3f} / {100 * frac[m].mean():.1f}')
            L.append(f'| {ARM_LABELS.get(name, name)} | ' + ' | '.join(cells)
                     + f' | {float(d["ms"]):.1f} |')
        # exceedance check for tab:bound
        allp = np.max(np.stack([v for v in strokes.values()]), 0)
        exc = allp > bound[:len(allp)] + 1e-6
        ratio = (allp / np.maximum(bound[:len(allp)], 1e-6)).max()
        L.append(f'\n- bound exceeded by any arm: {exc.mean():.2%} of tasks, '
                 f'worst ratio {ratio:.3f}')
    else:
        L.append('- MISSING arm npz or bound')

    # ---- termination causes ---------------------------------------------
    L += sec('tab:termcause — what ends the stroke (%)')
    if arms:
        L.append('| method | ' + ' | '.join(TERM_NAMES.values()) + ' |')
        L.append('|---' * (1 + len(TERM_NAMES)) + '|')
        for f in arms:
            name = re.sub(f'_{a.n}$', '', f.stem[4:])
            tm = np.load(f)['term']
            row = [f'{(tm == c).mean() * 100:.1f}' for c in TERM_NAMES]
            L.append(f'| {ARM_LABELS.get(name, name)} | ' + ' | '.join(row) + ' |')
    else:
        L.append('- MISSING')

    # ---- horizon subset --------------------------------------------------
    L += sec('tab:horizon — subset ladder (ratio to classical on subset)')
    subs = sorted(FILL.glob('arm_*_1024.npz')) + sorted(
        FILL.glob('arm_*_512.npz')) + sorted(FILL.glob('arm_*_256.npz'))
    by_n = {}
    for f in subs:
        m = re.match(r'arm_(.+)_(\d+)$', f.stem)
        by_n.setdefault(int(m.group(2)), {})[m.group(1)] = np.load(f)
    for n, d in sorted(by_n.items()):
        if 'classical' not in d:
            L.append(f'- subset {n}: no classical arm, ratios unavailable')
            continue
        base = d['classical']['prog'].astype(np.float64)
        ok = base > 1e-6
        L.append(f'\nsubset n={n}:')
        L.append('| arm | ratio | ms/decision |')
        L.append('|---|---|---|')
        for name, z in sorted(d.items()):
            r = (z['prog'][ok] / base[ok]).mean()
            L.append(f'| {ARM_LABELS.get(name, name)} | {r:.4f} '
                     f'| {float(z["ms"]):.1f} |')

    # ---- multiarm --------------------------------------------------------
    L += sec('tab:multiarm — margin law across arms (from ladder logs)')
    for robot in ('xarm7', 'cobotta'):
        lg = FILL / f'ladder_{robot}.log'
        if lg.exists():
            txt = lg.read_text()
            rows = re.findall(r'^(\w+)\s+ratio to classical ([\d.]+)',
                              txt, re.M)
            L.append(f'- {robot}: ' + ', '.join(f'{k} {v}' for k, v in rows))
        else:
            L.append(f'- {robot}: MISSING ladder log')

    # ---- full system / selection ----------------------------------------
    L += sec('tab:ablate_seed + full system — selection on the eval set')
    fp = FILL / 'fullsys_10k.npz'
    if fp.exists():
        z = np.load(fp)
        lab = z['L'].astype(np.float64)
        nf = z['n_found']
        N, K = lab.shape
        Mv = np.arange(K)[None, :] < nf[:, None]
        best = np.where(Mv, lab, -1e9).max(1)
        rand = np.where(Mv, lab, 0).sum(1) / np.maximum(nf, 1)
        first = lab[np.arange(N), np.minimum(1, nf - 1)]
        selc = lab[np.arange(N), z['pick_cond']]
        seln = lab[np.arange(N), z['pick_nocond']]
        gen = lab[:, 0]
        nz = (best - rand) > 0.02

        def cap(v):
            return ((v - rand)[nz] / (best - rand)[nz]).mean()
        L.append('| start | length m | capture |')
        L.append('|---|---|---|')
        for nm, v in (('generator config', gen), ('first candidate', first),
                      ('random candidate', rand),
                      ('selector (nocond)', seln),
                      ('selector (cond, ours)', selc),
                      ('within-pool oracle', best)):
            L.append(f'| {nm} | {v.mean():.4f} | {cap(v) * 100:.1f}% |')
        if bound is not None:
            fr = np.clip(selc, 0, None) / np.maximum(bound[:N], 1e-6)
            fo = np.clip(best, 0, None) / np.maximum(bound[:N], 1e-6)
            L.append(f'\n- FULL SYSTEM realized fraction: '
                     f'{100 * fr.mean():.1f}% (oracle-start '
                     f'{100 * fo.mean():.1f}%)')
    else:
        L.append('- MISSING fullsys npz')

    # ---- selector OOD ----------------------------------------------------
    L += sec('tab:ood — selector generalization (copied)')
    op = REPO / 'Yuan/IJRR/runs/selector_ood/v1/report.md'
    L.append(op.read_text() if op.exists() else '- MISSING OOD report')

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L) + '\n')
    print(f'[fill_report] wrote {out}')


if __name__ == '__main__':
    main()
