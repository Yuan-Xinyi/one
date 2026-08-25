"""Redesigned base-placement figure: three-category map + tolerance
circles, in base-relative-to-seam coordinates. Reads base_place_map.npz."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np

FU = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify')
d = np.load(FU / 'base_place_map.npz')
GRID = d['grid']
G = len(GRID)
pw = d['pw'].reshape(G, G)
cont = d['cont'].reshape(G, G)
stroke = d['stroke'].reshape(G, G)
L_REQ = float(d['L_req'])

# recompute the recommendations with the map border treated as
# infeasible (zero-padded distance transform), so the tolerance disk
# never leans on unswept territory
from scipy import ndimage


def recommend(mask):
    dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    i, j = np.unravel_index(dist.argmax(), dist.shape)
    return np.array([GRID[i], GRID[j]]), (float(dist[i, j]) - 0.5) * 0.05


rec_cont, tol_cont = recommend(cont)
rec_pw, tol_pw = recommend(pw)
print(f'cont rec {rec_cont} tol {tol_cont:.2f} | pw rec {rec_pw} '
      f'tol {tol_pw:.2f}')

# categories: 0 = pointwise-infeasible, 1 = continuous-feasible,
#             2 = pointwise-feasible but continuous-fail (protagonist)
cat = np.zeros((G, G), np.int8)
cat[cont] = 1
cat[pw & ~cont] = 2

# base position relative to the seam start = -(seam start in base frame):
# rotate the map by 180 degrees and negate the marks.
cat_b = cat[::-1, ::-1]
ext = [-GRID[-1] - 0.025, -GRID[0] + 0.025,
       -GRID[-1] - 0.025, -GRID[0] + 0.025]
star = (-rec_cont[0], -rec_cont[1])
plus = (-rec_pw[0], -rec_pw[1])
i_pw = (np.abs(GRID - rec_pw[0]).argmin(), np.abs(GRID - rec_pw[1]).argmin())
exec_pw = float(stroke[i_pw])

cmap = ListedColormap(['#f2f2f2', '#b9cde3', '#e6550d'])
fig, ax = plt.subplots(figsize=(6.6, 6.0))
ax.imshow(cat_b.T, origin='lower', cmap=cmap, vmin=0, vmax=2,
          extent=ext, interpolation='nearest')

# the seam itself: from the origin 0.5 m along +x
ax.plot([0, L_REQ], [0, 0], color='k', lw=3, solid_capstyle='butt',
        zorder=5)
ax.plot([0], [0], marker='o', ms=6, color='k', zorder=6)
ax.annotate('seam (0.5 m)', xy=(L_REQ / 2, 0),
            xytext=(L_REQ / 2 + 0.12, -0.14), ha='center', fontsize=9,
            zorder=6, arrowprops=dict(arrowstyle='-', lw=0.7))

# tolerance circles
ax.add_patch(Circle(star, tol_cont, fill=False, ec='#1a5fb4', lw=2,
                    zorder=6))
ax.add_patch(Circle(plus, tol_pw, fill=False, ec='#5e5e5e', lw=1.8,
                    ls='--', zorder=6))
ax.plot(*star, marker='*', ms=17, mec='k', mfc='#1a5fb4', ls='none',
        zorder=7)
ax.plot(*plus, marker='P', ms=11, mec='k', mfc='white', ls='none',
        zorder=7)
ax.annotate(f'continuous criterion:\nfull seam, tol {tol_cont*100:.0f} cm',
            xy=star, xytext=(star[0] - 1.12, star[1] + 0.22),
            fontsize=9, color='#1a5fb4', fontweight='bold', zorder=7,
            arrowprops=dict(arrowstyle='->', color='#1a5fb4', lw=1.1))
ax.annotate(f'pointwise criterion:\nexecutes only {exec_pw:.2f} m',
            xy=plus, xytext=(plus[0] - 1.10, plus[1] + 0.18),
            fontsize=9, color='#b03a00', fontweight='bold', zorder=7,
            arrowprops=dict(arrowstyle='->', color='#b03a00', lw=1.1))

# legend as colored proxy patches, placed in the gray corner
from matplotlib.patches import Patch
handles = [
    Patch(fc='#e6550d', label='all points reachable, seam NOT\n'
                              'executable as one stroke (55)'),
    Patch(fc='#b9cde3', label='seam executable as one stroke (523)'),
    Patch(fc='#f2f2f2', ec='#cccccc',
          label='some seam point unreachable'),
]
ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.95,
          borderpad=0.7)

ax.set_xlabel('base position relative to the seam start, x (m)')
ax.set_ylabel('base position relative to the seam start, y (m)')
ax.set_title('Where to place the base for a 0.5 m seam')
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(FU / 'base_place_map.png', dpi=220)
fig.savefig(FU / 'base_place_map.pdf')
print('written')
