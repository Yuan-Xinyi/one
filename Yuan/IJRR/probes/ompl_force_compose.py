#!/usr/bin/env python3
"""Compose the OMPL-vs-policy clip for one task: the overlay render on the
left, the comparison curves (joints with hardware limits, effective
stiffness with its cap, policy force error) with a moving arc cursor on
the right. argv: task index. Output: print_analysis/ompl/ompl_force_t{i}.mp4"""
import sys, subprocess
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TI = int(sys.argv[1])
MAIN = Path('/home/lqin/one/Yuan/IJRR'); FU = MAIN / 'runs/paper_fill/fam_unify'
OUT = MAIN / 'runs/paper_fill/print_analysis/ompl'
SUB, HOLD, KN_MAX, F_TOL = 3, 45, 2000.0, 2.0
LIM_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LIM_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
d = np.load(FU / f'ompl_force_cmp_t{TI}.npz', allow_pickle=True)
grid = d['grid']; oe, pe = float(d['ompl_end']), float(d['pol_end'])
ARC = np.interp(np.linspace(0, len(grid) - 1, (len(grid) - 1) * SUB + 1), np.arange(len(grid)), grid)
K = len(ARC); N = K + HOLD
om, pm = grid <= oe + 1e-9, grid <= pe + 1e-9

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.5, wspace=0.3, left=0.06, right=0.99, top=0.9, bottom=0.08)
cursors = []
for j in range(7):
    ax = fig.add_subplot(gs[j // 4, j % 4])
    ax.plot(grid[om], d['ompl_q'][om, j], color='#1b7f3b', lw=1.6)
    ax.plot(grid[pm], d['pol_q'][pm, j], color='#3d6be0', lw=1.6)
    ax.axhline(LIM_LO[j], color='k', ls='--', lw=0.9); ax.axhline(LIM_UP[j], color='k', ls='--', lw=0.9)
    pad = 0.08 * (LIM_UP[j] - LIM_LO[j]); ax.set_ylim(LIM_LO[j] - pad, LIM_UP[j] + pad); ax.set_xlim(0, grid[-1])
    ax.set_title(f'J{j+1} [rad]', fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    cursors.append(ax.axvline(0, color='0.3', lw=0.8))
ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
ax.text(0, 0.95, f'task {TI}\nOMPL (green): {oe:.2f} m\npolicy (blue): {pe:.2f} m\nbound {float(d["bound"]):.2f} m\nmu = 0.3, cap 2000 N/m',
        fontsize=10, va='top')
ax = fig.add_subplot(gs[2, :2])
ax.plot(grid[om], d['ompl_kn'][om], color='#1b7f3b', lw=1.6, label='OMPL'); ax.plot(grid[pm], d['pol_kn'][pm], color='#3d6be0', lw=1.6, label='policy')
ax.axhline(KN_MAX, color='#b71c1c', ls='--', lw=1.1); ax.set_ylim(400, 2400); ax.set_xlim(0, grid[-1])
ax.set_ylabel('$k_{eff}$ [N/m]', fontsize=9); ax.set_xlabel('arc length [m]', fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=0.25); ax.legend(fontsize=7)
cursors.append(ax.axvline(0, color='0.3', lw=0.8))
ax = fig.add_subplot(gs[2, 2:])
ax.axhspan(-F_TOL, F_TOL, color='#cfe8d4', alpha=0.8, lw=0); ax.axhline(0, color='#2e7d32', ls='--', lw=0.9)
ax.plot(grid[pm], d['pol_ferr'][pm], color='#3d6be0', lw=1.6); ax.set_ylim(-3, 3); ax.set_xlim(0, grid[-1])
ax.set_ylabel('policy force error [N]', fontsize=9); ax.set_xlabel('arc length [m]', fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
cursors.append(ax.axvline(0, color='0.3', lw=0.8))
fig.suptitle(f'OMPL (force-aware, 300 s) vs force-aware policy -- task {TI}, same line, same start pool, mu = 0.3', fontsize=11)
pdir = OUT / f'ompl_force_t{TI}_panel'; pdir.mkdir(exist_ok=True)
for t in range(N):
    a = ARC[min(t, K - 1)]
    for c in cursors: c.set_xdata([a, a])
    fig.savefig(pdir / f'p{t:05d}.png')
plt.close(fig)
cdir = OUT / f'ompl_force_t{TI}_comp'; cdir.mkdir(exist_ok=True)
fdir = OUT / f'ompl_force_t{TI}_frames'; last = None
for t in range(N):
    fr = fdir / f'f{t:05d}.png'
    if fr.exists(): last = fr
    left = Image.open(last).convert('RGB'); right = Image.open(pdir / f'p{t:05d}.png').convert('RGB')
    canvas = Image.new('RGB', (2560, 720), (255, 255, 255)); canvas.paste(left, (0, 0)); canvas.paste(right, (1280, 0))
    canvas.save(cdir / f'c{t:05d}.png')
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '30', '-i', str(cdir / 'c%05d.png'), '-vf', 'scale=1920:540',
                '-c:v', 'libx264', '-profile:v', 'high', '-level', '4.2', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(OUT / f'ompl_force_t{TI}.mp4')], check=True)
print(f'task {TI}: {N} frames -> {OUT / f"ompl_force_t{TI}.mp4"}', flush=True)
