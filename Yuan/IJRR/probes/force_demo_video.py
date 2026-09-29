#!/usr/bin/env python3
"""one-viewer frames for the implicit-force demo. Usage: force_demo_video.py rl|fo

Scene: board whose top face is the contact surface; the physical pen tip
rides on the surface while the commanded target sits d(q) below it. The
reaction force is drawn as an arrow out of the board (length = realised
force, grey bead = 5 N target; green within +-2 N, red outside). Ink beads
are coloured by the end-effector stiffness k_n (dark = compliant, yellow =
near the 2000 N/m cap, red = above the cap)."""
import builtins, json, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import matplotlib; matplotlib.use('Agg')
import matplotlib.cm as cm
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

TAG = sys.argv[1]
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis/ompl'
FR = OUT / f'force_demo_{TAG}_frames'; FR.mkdir(parents=True, exist_ok=True)
SCR = Path('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad')
PEN_LEN, SUB, HOLD, CH = 0.10, 4, 60, 8
KN_LO, KN_CAP, F_SET, F_TOL = 700.0, 2000.0, 5.0, 2.0
ARROW_PER_N = 0.02          # 5 N -> 10 cm

d = np.load(MAIN / 'runs/paper_fill/fam_unify/force_demo_roll.npz', allow_pickle=True)
n = d['n_target'].astype(np.float32); n /= np.linalg.norm(n)
dd = d['line_dir'].astype(np.float32); dd /= np.linalg.norm(dd)
Q, tip, zax = d[f'{TAG}_q'], d[f'{TAG}_tip'], d[f'{TAG}_zax']
kn, ferr, depth = d[f'{TAG}_kn'], d[f'{TAG}_ferr'], d[f'{TAG}_depth']
term = str(d[f'{TAG}_term']); depth0 = float(d[f'{TAG}_depth0'])
p_start = d[f'{TAG}_p_start'].astype(np.float32)
s0 = p_start - depth0 * n                      # surface point at the start

# ---- interpolate steps for smooth motion ----
def interp(x):
    x = np.asarray(x, np.float32)
    t = np.arange(len(x)); tt = np.linspace(0, len(x) - 1, (len(x) - 1) * SUB + 1)
    if x.ndim == 1:
        return np.interp(tt, t, x).astype(np.float32)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], 1).astype(np.float32)
Qi, tipi, zaxi = interp(Q), interp(tip), interp(zax)
kni, ferri, depthi = interp(kn), interp(ferr), interp(depth)
K = len(Qi)
phys = tipi - depthi[:, None] * n              # physical tip on the surface

# ---- scene ----
ex = dd; ez = -n; ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
Rb = np.stack([ex, ey, ez], 1).astype(np.float32)
LEN = 1.9
bc = s0 + dd * (LEN / 2 - 0.1) + n * 0.012
_cp = json.loads((SCR / 'wrs2_campick_wrslogo2.json').read_text())
_off = np.array(_cp['pos']) - np.array(_cp['look_at'])
look = s0 + dd * 0.7
world = ovw.World(cam_pos=tuple(look + 1.35 * _off), cam_lookat_pos=tuple(look),
                  win_size=(1280, 720))
builtins.base = world
scene = world.scene
board = ossop.box(pos=tuple(bc), half_extents=(LEN / 2 + 0.05, 0.32, 0.012),
                  rotmat=Rb, rgb=(0.96, 0.96, 0.97), alpha=1.0)
board.attach_to(scene)
line = np.stack([s0 - n * 0.001 + dd * t for t in np.linspace(-0.1, LEN - 0.15, 60)])
ossop.linsegs(np.stack([line[:-1], line[1:]], 1), radius=0.0015,
              srgbs=np.float32([0.7, 0.72, 0.76]), alpha=0.5).attach_to(scene)

def _rot_with_z(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95:
        h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)

arm, hand = fr3_with_hand(jaw_width=0.0)
arm.attach_to(scene); arm.fk(Qi[0])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                     rgb=(0.13, 0.18, 0.45), alpha=0.98)
pen.attach_to(scene)
state = {'t': 0, 'c': 0, 'arrow': None, 'bead': None}

def ink_color(k):
    if k > KN_CAP:
        return np.float32([0.85, 0.1, 0.1])
    u = np.clip((k - KN_LO) / (KN_CAP - KN_LO), 0, 1)
    return np.float32(cm.viridis(u)[:3])

segs, cols, fr_ = [], [], []
for j in range(1, K):
    a, b = phys[j - 1] - n * 0.002, phys[j] - n * 0.002
    if np.linalg.norm(b - a) > 1e-5:
        segs.append([a, b]); cols.append(ink_color(kni[j])); fr_.append(j)
CHUNKS = [(fr_[min(i + CH, len(segs)) - 1], np.asarray(segs[i:i + CH], np.float32),
           np.asarray(cols[i:i + CH], np.float32)) for i in range(0, len(segs), CH)]

CAM_POS, CAM_LOOK = (look + 1.35 * _off).astype(np.float32), look.astype(np.float32)


def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR / f'f{t - 1:05d}.png'))
    # the window is on the user's desktop: re-pin the camera every frame so
    # a stray mouse drag cannot orbit the recording
    world.camera.set_to(CAM_POS, CAM_LOOK)
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    arm.fk(Qi[k])
    z = zaxi[k] / (np.linalg.norm(zaxi[k]) + 1e-9)
    pen.set_rotmat_pos(_rot_with_z(z), (phys[k] - PEN_LEN * z).astype(np.float32))
    # force indicators: a pressure footprint on the surface (radius = realised
    # force, grey disc = 5 N target) and a reaction-force arrow beside the pen
    # (length = force, grey bead = target) -- both green within tolerance,
    # red when the force or the stiffness cap is violated
    for key in ('arrow', 'bead', 'foot', 'ring'):
        if state.get(key) is not None:
            state[key].detach_from(scene); state[key] = None
    f = F_SET + ferri[k]
    ok = abs(ferri[k]) <= F_TOL and kni[k] <= KN_CAP
    col = (0.15, 0.65, 0.25) if ok else (0.85, 0.12, 0.12)
    base = phys[k] - n * 0.003
    r_f = 0.006 * max(float(f), 0.3)
    state['ring'] = ossop.cylinder(spos=tuple(base + n * 0.0025), epos=tuple(base + n * 0.0015),
                                   radius=0.006 * F_SET, segments=32,
                                   rgb=(0.45, 0.45, 0.5), alpha=0.5)
    state['ring'].attach_to(scene)
    state['foot'] = ossop.cylinder(spos=tuple(base + n * 0.0015), epos=tuple(base - n * 0.0005),
                                   radius=r_f, segments=32, rgb=col, alpha=0.9)
    state['foot'].attach_to(scene)
    side = base + ey * 0.06
    L = max(float(f), 0.3) * ARROW_PER_N
    state['arrow'] = ossop.arrow(spos=side, epos=side - n * L, shaft_radius=0.006,
                                 head_radius=0.013, rgb=col, alpha=0.95)
    state['arrow'].attach_to(scene)
    state['bead'] = ossop.sphere(pos=tuple(side - n * (F_SET * ARROW_PER_N)),
                                 radius=0.011, rgb=(0.45, 0.45, 0.5), alpha=0.55)
    state['bead'].attach_to(scene)
    while state['c'] < len(CHUNKS) and CHUNKS[state['c']][0] <= k:
        _, sg, cl = CHUNKS[state['c']]
        ossop.linsegs(sg, radius=0.0045, srgbs=cl, alpha=1.0).attach_to(scene)
        state['c'] += 1
    if k == K - 1 and term not in ('alive', 'truncated') and t == K:
        ossop.sphere(pos=tuple(phys[k] - n * 0.02), radius=0.03,
                     rgb=(0.9, 0.1, 0.1), alpha=0.9).attach_to(scene)
    state['t'] = t + 1

world.schedule_interval(tick, interval=1 / 60.0)
world.run()
print('frames', K + HOLD, 'term', term)
