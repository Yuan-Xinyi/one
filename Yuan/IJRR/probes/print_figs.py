"""Figures for the fixed-base print-size analysis."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
R_COL = 0.20
plt.rcParams.update({'font.size': 11, 'axes.grid': False})

q = np.load(OUT / 'cube_query.npz')
f = np.load(OUT / 'down_field.npz')
gx, gz, STEP = q['gx'], q['gz'], float(q['step'])


def base_decor(ax):
    ax.add_patch(Circle((0, 0), R_COL, fc='0.25', ec='k', lw=1.2, zorder=5))
    ax.plot(0, 0, 'w+', ms=9, mew=2, zorder=6)
    ax.set_aspect('equal')
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')


# ---- Fig 1: placement map -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, cone in zip(axes, (5, 30, 90)):
    A = q[f'cube_c{cone}_solid']
    M = A.max(axis=2)
    im = ax.pcolormesh(gx, gx, M.T * 100, cmap='viridis', vmin=0,
                       vmax=q['cube_c90_solid'].max() * 100, shading='nearest')
    cs = ax.contour(gx, gx, M.T * 100, levels=[20, 30, 40, 50, 60],
                    colors='w', linewidths=0.8, alpha=0.75)
    ax.clabel(cs, fmt='%d', fontsize=8)
    b = np.unravel_index(A.argmax(), A.shape)
    ax.plot(gx[b[0]], gx[b[1]], 'r*', ms=18, mec='k', mew=0.8, zorder=7)
    base_decor(ax)
    ax.set_title(f'nozzle tilt tolerance {cone}$^\\circ$\n'
                 f'best {A.max()*100:.0f} cm at '
                 f'({gx[b[0]]:+.2f}, {gx[b[1]]:+.2f}), bed z={gz[b[2]]-STEP/2:+.2f} m')
fig.colorbar(im, ax=axes, shrink=0.85, label='largest solid cube side [cm]')
fig.suptitle('Where to put the bed: largest printable cube vs footprint centre '
             '(best bed height per cell), FR3 fixed base', y=0.99)
fig.savefig(OUT / 'fig1_placement.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---- Fig 2: vertical slice through the optimum ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
for ax, cone in zip(axes, (5, 30)):
    A = q[f'cube_c{cone}_solid']
    b = np.unravel_index(A.argmax(), A.shape)
    F = f[f'F{cone}'][:, b[1], :]
    ax.pcolormesh(gx, gz, F.T, cmap='Blues', vmin=0, vmax=1.6, shading='nearest')
    s, zb = float(A.max()), gz[b[2]] - STEP / 2
    ax.add_patch(Rectangle((gx[b[0]] - s / 2, zb), s, s, fill=False,
                           ec='crimson', lw=2.6, zorder=6))
    ax.axhline(zb, color='k', ls='--', lw=1.3)
    ax.axvspan(-R_COL, R_COL, color='0.35', alpha=0.45, zorder=4)
    ax.set_title(f'{cone}$^\\circ$ tolerance: slice at y={gx[b[1]]:+.2f} m\n'
                 f'cube {s*100:.0f} cm on a bed at z={zb:+.2f} m')
    ax.set_xlabel('x  [m]'); ax.set_ylabel('z  [m]'); ax.set_aspect('equal')
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gz[0], gz[-1])
fig.suptitle('Nozzle-down feasible set (blue) in a vertical plane, '
             'with the largest cube it contains; grey band is the base column',
             y=0.98)
fig.savefig(OUT / 'fig2_slice.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---- Fig 3: what each freedom is worth ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
ax = axes[0]
for cone, col in ((5, 'crimson'), (30, 'steelblue'), (90, '0.4')):
    A = q[f'cube_c{cone}_solid']
    ax.plot(gz - STEP / 2, A.max(axis=(0, 1)) * 100, lw=2.2, color=col,
            label=f'{cone}$^\\circ$')
ax.axvline(0.0, color='k', ls=':', lw=1.2)
ax.text(0.005, 4, 'bed level with the mount', rotation=90, fontsize=9, va='bottom')
ax.set_xlabel('bed height z  [m]'); ax.set_ylabel('largest cube side [cm]')
ax.set_title('bed height'); ax.legend(title='nozzle tilt'); ax.grid(alpha=0.3)

ax = axes[1]
tol = [5, 30, 90]
for tag, mk, lab in (('solid', 'o-', 'solid (full infill)'),
                     ('hollow', 's--', 'hollow shell'),
                     ('bed', '^:', 'solid, arm must clear the bed')):
    ax.plot(tol, [q[f'cube_c{c}_{tag}'].max() * 100 for c in tol], mk, lw=2,
            ms=8, label=lab)
ax.set_xscale('log'); ax.set_xticks(tol); ax.set_xticklabels([f'{t}°' for t in tol])
ax.set_xlabel('nozzle tilt tolerance'); ax.set_ylabel('largest cube side [cm]')
ax.set_title('nozzle tilt tolerance'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
k0 = int(np.argmin(np.abs(gz - STEP / 2)))
for cone, col in ((5, 'crimson'), (30, 'steelblue')):
    A = q[f'cube_c{cone}_solid']
    b = np.unravel_index(A.argmax(), A.shape)
    prof = A[b[0], b[1], :] * 100
    ax.plot(gz - STEP / 2, prof, lw=2.2, color=col,
            label=f'{cone}$^\\circ$ at its own best footprint')
ax.set_xlabel('bed height z  [m]'); ax.set_ylabel('cube side [cm]')
ax.set_title('one footprint, bed swept'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle('What each free choice is worth', y=1.0)
fig.savefig(OUT / 'fig3_freedoms.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print('wrote fig1_placement / fig2_slice / fig3_freedoms')

# ---- Fig 4: bowl ----------------------------------------------------------
bp = OUT / 'bowl_query.npz'
if bp.exists():
    from print_bowl import unit_bowl, ETA, RB_FRAC
    bq = np.load(bp)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0))
    for ax, inv in zip(axes[:2], (False, True)):
        p, a, _ = unit_bowl(inv, pitch=0.10)
        m = np.abs(p[:, 1]) < 0.06
        ax.plot(p[m, 0], p[m, 2], 'o', ms=3.5, color='0.25')
        sel = m & (p[:, 0] > 0)
        idx = np.nonzero(sel)[0][::2]
        ax.quiver(p[idx, 0], p[idx, 2], a[idx, 0], a[idx, 2],
                  color='crimson', scale=9, width=0.006)
        ax.axhline(0, color='k', lw=2)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('r / R'); ax.set_ylabel('z / R')
        ax.set_title(('upright (cavity up)' if not inv else 'inverted (dome)')
                     + '\nconformal nozzle axis, tilt reaches 84$^\\circ$')
    ax = axes[2]
    labs, vals, cols = [], [], []
    for mode in ('planar', 'conformal'):
        for cone in (5, 30):
            for inv in (False, True):
                k = f'{mode}_c{cone}_{"inv" if inv else "up"}'
                if k not in bq.files:
                    continue
                v = bq[k]
                R = float(v.max()) if mode == 'planar' else float(v[0])
                labs.append(f'{mode[:4]} {cone}° '
                            f'{"inv" if inv else "up"}')
                vals.append(200 * R)
                cols.append('steelblue' if mode == 'planar' else 'darkorange')
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('rim diameter [cm]')
    ax.set_title('largest bowl, pointwise (no part collision)')
    ax.grid(alpha=0.3, axis='x')
    fig.suptitle(f'Sushi bowl: spherical cap, depth {ETA:.2f}R, '
                 f'foot {RB_FRAC:.2f}R', y=1.0)
    fig.savefig(OUT / 'fig4_bowl.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig4_bowl')

# ---- Fig 5: the continuity gap -------------------------------------------
rp = OUT / 'raster_query.npz'
if rp.exists():
    r = np.load(rp)
    NAMES = ['classical\n@first', 'classical\n@critic',
             'policy\n@first', 'policy\n@critic']
    COLS = ['0.55', '0.35', 'steelblue', 'crimson']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for j, cone in enumerate((5, 30)):
        P, meta, has = r[f'c{cone}'], r[f'meta_c{cone}'], r[f'has_c{cone}']
        side = float(meta[0])
        ax = axes[2 * j]
        bp = ax.boxplot([p[has] for p in P], labels=NAMES, showfliers=False,
                        patch_artist=True, widths=0.6)
        for patch, c in zip(bp['boxes'], COLS):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for med in bp['medians']:
            med.set_color('k'); med.set_linewidth(1.6)
        ax.axhline(side, color='k', ls='--', lw=1.8)
        ax.text(0.55, side, f'  one pass = {side*100:.0f} cm', va='bottom',
                fontsize=10)
        ax.set_ylabel('continuous stroke [m]')
        ax.set_title(f'{cone}$^\\circ$ cone: how far one pass gets')
        ax.grid(alpha=0.3, axis='y')
        ax = axes[2 * j + 1]
        v = [np.mean(p[has] >= side - 1e-3) * 100 for p in P]
        ax.bar(range(4), v, color=COLS, alpha=0.85)
        for i, t in enumerate(v):
            ax.text(i, t + 1.5, f'{t:.0f}%', ha='center', fontsize=11)
        ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, fontsize=9)
        ax.set_ylim(0, 108); ax.set_ylabel('passes completed [%]')
        ax.set_title(f'{cone}$^\\circ$ cone: passes finished in one go')
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('The continuity gap: the pointwise field admits every point of '
                 'the cube, yet a single continuous pass is another matter',
                 y=1.0)
    fig.savefig(OUT / 'fig5_continuity.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote fig5_continuity')
