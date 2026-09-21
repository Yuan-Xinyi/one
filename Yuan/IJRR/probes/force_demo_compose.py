#!/usr/bin/env python3
"""Compose the implicit-force demo: for each policy, the one-viewer frame on
top and two live traces below (contact force with the +-2 N band around the
5 N set point; end-effector stiffness k_n with the 2000 N/m cap), then the
two policies side by side. Output: force_demo.mp4 (2560 x 960)."""
import sys, subprocess
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis/ompl'
SUB, HOLD = 4, 60
F_SET, F_TOL, KN_CAP = 5.0, 2.0, 2000.0
d = np.load(MAIN / 'runs/paper_fill/fam_unify/force_demo_roll.npz', allow_pickle=True)
TITLE = {'rl': 'Position-only policy (no force awareness)',
         'fo': 'Implicit-force policy (ours)'}


def interp(x):
    x = np.asarray(x, np.float32); t = np.arange(len(x))
    tt = np.linspace(0, len(x) - 1, (len(x) - 1) * SUB + 1)
    return np.interp(tt, t, x).astype(np.float32)


def panel_frames(tag, n_frames, arc_max):
    arc, kn, ferr = interp(d[f'{tag}_arc']), interp(d[f'{tag}_kn']), interp(d[f'{tag}_ferr'])
    term = str(d[f'{tag}_term'])
    K = len(arc)
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 2.4), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.26, top=0.82, wspace=0.22)
    # static parts
    ax[0].axhspan(F_SET - F_TOL, F_SET + F_TOL, color='#cfe8d4', alpha=0.8, lw=0)
    ax[0].axhline(F_SET, color='#2e7d32', lw=1.0, ls='--')
    ax[0].set_ylim(0, 10); ax[0].set_ylabel('contact force [N]')
    ax[0].set_title('realised force  (set 5 N, tolerance +-2 N)', fontsize=10)
    ax[1].axhline(KN_CAP, color='#b71c1c', lw=1.2, ls='--')
    ax[1].set_ylim(500, 2600); ax[1].set_ylabel('stiffness $k_n$ [N/m]')
    ax[1].set_title('end-effector stiffness along the pen  (cap 2000 N/m)', fontsize=10)
    for a in ax:
        a.set_xlim(0, arc_max); a.set_xlabel('arc length [m]'); a.grid(alpha=0.25)
    fig.suptitle(TITLE[tag], fontsize=12, x=0.06, ha='left', y=0.98)
    l0, = ax[0].plot([], [], color='#1a237e', lw=1.8)
    l1, = ax[1].plot([], [], color='#1a237e', lw=1.8)
    c0, = ax[0].plot([], [], 'o', color='#1a237e', ms=5)
    c1, = ax[1].plot([], [], 'o', color='#1a237e', ms=5)
    xmark = [None, None]
    fdir = OUT / f'force_demo_{tag}_panel'; fdir.mkdir(exist_ok=True)
    for t in range(n_frames):
        k = min(t, K - 1)
        l0.set_data(arc[:k + 1], F_SET + ferr[:k + 1]); l1.set_data(arc[:k + 1], kn[:k + 1])
        c0.set_data([arc[k]], [F_SET + ferr[k]]); c1.set_data([arc[k]], [kn[k]])
        if k == K - 1 and term not in ('alive', 'truncated') and xmark[0] is None:
            xmark[0] = ax[0].plot([arc[k]], [F_SET + ferr[k]], 'x', color='#b71c1c', ms=11, mew=2.5)[0]
            xmark[1] = ax[1].plot([arc[k]], [kn[k]], 'x', color='#b71c1c', ms=11, mew=2.5)[0]
            ax[1].text(arc[k], kn[k] + 120, f'terminated: {term}', color='#b71c1c',
                       fontsize=9, ha='right' if arc[k] > 0.6 * arc_max else 'left')
        fig.savefig(fdir / f'p{t:05d}.png')
    plt.close(fig)
    return fdir


n_rl = len(interp(d['rl_arc'])) + HOLD; n_fo = len(interp(d['fo_arc'])) + HOLD
N = max(n_rl, n_fo)
arc_max = float(max(d['rl_arc'][-1], d['fo_arc'][-1])) * 1.08 + 0.05
pd = {tag: panel_frames(tag, N, arc_max) for tag in ('rl', 'fo')}
print('panels done', flush=True)
cdir = OUT / 'force_demo_frames'; cdir.mkdir(exist_ok=True)
last = {}
for t in range(N):
    tiles = []
    for tag in ('rl', 'fo'):
        fr = OUT / f'force_demo_{tag}_frames' / f'f{t:05d}.png'
        if fr.exists():
            last[tag] = fr
        top = Image.open(last[tag]).convert('RGB')
        bot = Image.open(pd[tag] / f'p{t:05d}.png').convert('RGB')
        tile = Image.new('RGB', (1280, 960), (255, 255, 255))
        tile.paste(top, (0, 0)); tile.paste(bot, (0, 720))
        tiles.append(tile)
    canvas = Image.new('RGB', (2560, 960), (255, 255, 255))
    canvas.paste(tiles[0], (0, 0)); canvas.paste(tiles[1], (1280, 0))
    canvas.save(cdir / f'c{t:05d}.png')
print('composited', N, 'frames', flush=True)
subprocess.run(['ffmpeg', '-y', '-framerate', '60', '-i', str(cdir / 'c%05d.png'),
                '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                str(OUT / 'force_demo.mp4')], check=True, capture_output=True)
print('encoded', OUT / 'force_demo.mp4', flush=True)
