#!/usr/bin/env python3
"""one-viewer overlay of the OMPL path and the policy rollout on one task:
two translucent arms (OMPL green, policy blue) advancing along the same
line at the same arc, each frozen with a red marker once it has ended.
argv: task index. Frames -> print_analysis/ompl/ompl_force_t{i}_frames."""
import builtins, json, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import matplotlib; matplotlib.use('Agg')
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

TI = int(sys.argv[1])
MAIN = Path('/home/lqin/one/Yuan/IJRR')
FU = MAIN / 'runs/paper_fill/fam_unify'
FR = MAIN / f'runs/paper_fill/print_analysis/ompl/ompl_force_t{TI}_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN, SUB, HOLD, ALPHA = 0.10, 3, 45, 0.45
COLS = {'ompl': (0.11, 0.50, 0.23), 'pol': (0.24, 0.42, 0.88)}
d = np.load(FU / f'ompl_force_cmp_t{TI}.npz', allow_pickle=True)
grid, p0, dd, n = d['grid'], d['p0'], d['d'], d['n']
n = n / np.linalg.norm(n); dd = dd / np.linalg.norm(dd)
ends = {'ompl': float(d['ompl_end']), 'pol': float(d['pol_end'])}

def interp(x):
    t = np.arange(len(x)); tt = np.linspace(0, len(x) - 1, (len(x) - 1) * SUB + 1)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], 1).astype(np.float32)
Q = {k: interp(d[f'{k}_q']) for k in COLS}; TIP = {k: interp(d[f'{k}_tip']) for k in COLS}
ZAX = {k: interp(d[f'{k}_zax']) for k in COLS}
ARC = np.interp(np.linspace(0, len(grid) - 1, (len(grid) - 1) * SUB + 1), np.arange(len(grid)), grid)
K = len(ARC)

# ---- scene: board with its top face on the line, line, camera from the user's pick
ex = dd; ez = -n; ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
Rb = np.stack([ex, ey, ez], 1).astype(np.float32)
LEN = max(grid[-1], 0.6) + 0.4
bc = p0 + dd * (LEN / 2 - 0.15) + n * 0.012
_cp = json.loads((FU / 'campick_offset.json').read_text())
_off = np.array(_cp['pos']) - np.array(_cp['look_at'])
look = (p0 + dd * (0.5 * grid[-1])).astype(np.float32)
CAM = (look + 1.25 * _off).astype(np.float32)
world = ovw.World(cam_pos=tuple(CAM), cam_lookat_pos=tuple(look), win_size=(1280, 720))
builtins.base = world; scene = world.scene
ossop.box(pos=tuple(bc), half_extents=(LEN / 2 + 0.05, 0.30, 0.012), rotmat=Rb,
          rgb=(0.96, 0.96, 0.97), alpha=1.0).attach_to(scene)
line = np.stack([p0 - n * 0.001 + dd * t for t in np.linspace(-0.15, LEN - 0.2, 60)])
ossop.linsegs(np.stack([line[:-1], line[1:]], 1), radius=0.0015, srgbs=np.float32([0.7, 0.72, 0.76]), alpha=0.5).attach_to(scene)

def _rot_with_z(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95: h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)

arms = {}
for tag, col in COLS.items():
    arm, hand = fr3_with_hand(jaw_width=0.0); arm.attach_to(scene); arm.fk(Q[tag][0])
    for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
        lnk.alpha = ALPHA; lnk.rgb = col
    pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007, rgb=col, alpha=0.95)
    pen.attach_to(scene); arms[tag] = (arm, pen)
state = {'t': 0, 'ink': {k: 0 for k in COLS}, 'marked': set()}
CH = 6

def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(str(FR / f'f{t - 1:05d}.png'))
    world.camera.set_to(CAM, look)
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    for tag, (arm, pen) in arms.items():
        kk = min(k, int(np.searchsorted(ARC, ends[tag] + 1e-6)))
        kk = min(kk, K - 1)
        arm.fk(Q[tag][kk])
        z = ZAX[tag][kk] / (np.linalg.norm(ZAX[tag][kk]) + 1e-9)
        tip = TIP[tag][kk]
        pen.set_rotmat_pos(_rot_with_z(z), (tip - PEN_LEN * z).astype(np.float32))
        # ink up to kk, batched
        j0 = state['ink'][tag]
        if kk - j0 >= CH:
            seg = np.stack([TIP[tag][j0:kk - 1] - n * 0.002, TIP[tag][j0 + 1:kk] - n * 0.002], 1)
            off = (ey * (0.006 if tag == 'ompl' else -0.006)).astype(np.float32)   # side-by-side inks
            ossop.linsegs(seg + off, radius=0.004, srgbs=np.float32(COLS[tag]), alpha=1.0).attach_to(scene)
            state['ink'][tag] = kk - 1
        if ARC[min(k, K - 1)] >= ends[tag] - 1e-6 and tag not in state['marked'] and ends[tag] < grid[-1] - 1e-6:
            ossop.sphere(pos=tuple(TIP[tag][kk] - n * 0.02), radius=0.025, rgb=(0.9, 0.1, 0.1), alpha=0.9).attach_to(scene)
            state['marked'].add(tag)
    state['t'] = t + 1

world.schedule_interval(tick, interval=1 / 60.0)
world.run()
print(f'task {TI}: {K + HOLD} frames', flush=True)
