"""Figure 2: four path families, thin task-plane strips; (d) shows the plane
flipping with the rotating cone axis (overlapping patches, each ⊥ n(s))."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

C = '#1f77b4'; A = '#d62728'; PL = '#9ecae1'
s = np.linspace(0, 1, 200)
OUT = ('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/'
       'Yuan/IJRR/2026_Yuan_RAL/imgs/')
K = 6


def base_ax():
    fig = plt.figure(figsize=(3.6, 2.9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.4, 0.4)
    ax.set_box_aspect((1.35, 1.0, 0.62))
    ax.view_init(elev=24, azim=-63)
    ax.set_axis_off()
    fig.subplots_adjust(left=-0.18, right=1.18, top=1.25, bottom=-0.25)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT + name, dpi=300, facecolor='white',
                pil_kwargs={'quality': 95})
    plt.close(fig)
    print('wrote', name)


def flat_strip(ax, y_lo, y_hi):
    XX, YY = np.meshgrid([-0.05, 1.05], [y_lo, y_hi])
    ax.plot_surface(XX, YY, np.zeros_like(XX), color=PL, alpha=0.30,
                    linewidth=0, shade=False)


def arrows(ax, x, y, z, dirs):
    idx = np.linspace(8, len(x) - 9, K).astype(int)
    for j in idx:
        u, v, w = dirs[:, j]
        ax.quiver(x[j], y[j], z[j], u, v, w, length=0.22, color=A,
                  arrow_length_ratio=0.35, lw=1.7)
    return idx


zz = np.zeros_like(s)
nrm = np.stack([zz, zz, np.ones_like(s)])

# (a) straight -----------------------------------------------------------
fig, ax = base_ax(); ax.set_zlim(-0.02, 0.42)
flat_strip(ax, -0.12, 0.12)
ax.plot(s, zz, zz, color=C, lw=2.8, zorder=5)
arrows(ax, s, zz, zz, nrm)
ax.text(0.02, -0.02, 0.40, 'cone axis $\\mathbf{n}(s)$', color=A, fontsize=10)
ax.text(0.72, -0.34, 0.0, 'task plane', color='#4a7fb5', fontsize=10)
save(fig, 'path_straight.jpg')

# (b) arc ----------------------------------------------------------------
th = s * np.pi * 0.5
xa = 1.05 * np.sin(th); ya = 0.42 * (1 - np.cos(th)) - 0.20
fig, ax = base_ax(); ax.set_zlim(-0.02, 0.42)
flat_strip(ax, ya.min() - 0.08, ya.max() + 0.08)
ax.plot(xa, ya, zz, color=C, lw=2.8, zorder=5)
arrows(ax, xa, ya, zz, nrm)
save(fig, 'path_arc.jpg')

# (c) serpentine ---------------------------------------------------------
ys = 0.13 * np.sin(2 * np.pi * 2.2 * s)
fig, ax = base_ax(); ax.set_zlim(-0.02, 0.42)
flat_strip(ax, ys.min() - 0.08, ys.max() + 0.08)
ax.plot(s, ys, zz, color=C, lw=2.8, zorder=5)
arrows(ax, s, ys, zz, nrm)
save(fig, 'path_serpentine.jpg')

# (d) rotating axis: local plane flips with n(s), patches stay ⊥ n(s) ----
ang = s * np.pi * 0.42
nr = np.stack([zz, np.sin(ang), np.cos(ang)])
fig, ax = base_ax(); ax.set_zlim(-0.02, 0.42)
flat_strip(ax, -0.12, 0.12)
ax.plot(s, zz, zz, color=C, lw=2.8, zorder=5)
arrows(ax, s, zz, zz, nr)
save(fig, 'path_rotating.jpg')

from PIL import Image
ims = [Image.open(OUT + f'path_{n}.jpg')
       for n in ['straight', 'arc', 'serpentine', 'rotating']]
w = sum(i.width for i in ims); h = max(i.height for i in ims)
cat = Image.new('RGB', (w, h), 'white'); xx = 0
for i in ims:
    cat.paste(i, (xx, 0)); xx += i.width
cat.save('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-'
         'vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/'
         'scratchpad/preview_row.jpg', quality=90)
print('preview updated')
