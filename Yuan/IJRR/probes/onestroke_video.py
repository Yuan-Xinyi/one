#!/usr/bin/env python3
"""one-viewer clip of one figure drawn two ways, same start, same instant.

Two translucent arms share the frame: the framework in crimson, the classical
redundancy-resolution law in grey-blue. Each lays its own trace, so where the
grey one stops is exactly where that controller ran out of null space. The
target figure is drawn faintly underneath from the first frame, so "how much of
it got drawn" is readable without a caption.

argv: figure key
"""
import matplotlib; matplotlib.use('Agg')
import builtins, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

KEY = sys.argv[1]
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/onestroke')
FR = OUT / f'{KEY}_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN = 0.10
COL = {'pol': (0.86, 0.16, 0.30), 'cls': (0.42, 0.48, 0.58)}
ALPHA = 0.40
CH = 6

d = np.load(OUT / f'{KEY}_pack.npz', allow_pickle=False)
fig = d['figure']
S, cx, cy, zb = [float(v) for v in d['size']]

lo = np.minimum(fig.min(0), np.float32([-0.12, -0.12, 0.0]))
hi = np.maximum(fig.max(0), np.float32([0.12, 0.12, 0.88]))
ctr = (0.5 * (lo + hi)).astype(np.float32)
ext = float(np.linalg.norm(hi - lo))
dirv = np.array([0.52, -1.20, 0.86]); dirv /= np.linalg.norm(dirv)
world = ovw.World(cam_pos=tuple(ctr + dirv * (0.95 * ext)),
                  cam_lookat_pos=tuple(ctr), win_size=(1280, 720))
builtins.base = world
scene = world.scene

ossop.box(pos=(cx, cy, zb - 0.012),
          half_extents=(0.62 * S + 0.06, 0.62 * S + 0.06, 0.012),
          rgb=(0.90, 0.90, 0.92), alpha=1.0).attach_to(scene)
ossop.linsegs(np.stack([fig[:-1], fig[1:]], 1), radius=0.0018,
              srgbs=np.float32([0.58, 0.60, 0.64]), alpha=0.95).attach_to(scene)


def _rot_with_z(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95:
        h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)


def _pen_pose(t, z):
    z = z / (np.linalg.norm(z) + 1e-9)
    return _rot_with_z(z), (t - PEN_LEN * z).astype(np.float32)


arms, chunks = {}, {}
for tag in ('cls', 'pol'):
    arm, hand = fr3_with_hand(jaw_width=0.0)
    arm.attach_to(scene)
    arm.fk(d[f'q_{tag}'][0])
    for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
        lnk.alpha = ALPHA
        lnk.rgb = COL[tag]
    pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                         rgb=COL[tag], alpha=0.98)
    pen.set_rotmat_pos(*_pen_pose(d[f'tip_{tag}'][0], d[f'zax_{tag}'][0]))
    pen.attach_to(scene)
    arms[tag] = (arm, pen)
    # trace, batched: one scene object per short run of segments
    tip = d[f'tip_{tag}']
    lift = np.float32([0, 0, 0.003 if tag == 'pol' else 0.006])
    seg, fr_, last = [], [], tip[0]
    for i in range(1, len(tip)):
        if np.linalg.norm(tip[i] - last) > 0.004:
            seg.append([last + lift, tip[i] + lift]); fr_.append(i)
            last = tip[i]
    chunks[tag] = [(fr_[min(i + CH, len(seg)) - 1],
                    np.asarray(seg[i:i + CH], np.float32))
                   for i in range(0, len(seg), CH)]

K = max(len(d['q_pol']), len(d['q_cls']))
HOLD = 70
state = {'t': 0, 'c': {'pol': 0, 'cls': 0}}


def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR / f'f{t - 1:05d}.png'))
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    for tag, (arm, pen) in arms.items():
        i = min(k, len(d[f'q_{tag}']) - 1)
        arm.fk(d[f'q_{tag}'][i])
        pen.set_rotmat_pos(*_pen_pose(d[f'tip_{tag}'][i], d[f'zax_{tag}'][i]))
        ch = chunks[tag]
        while state['c'][tag] < len(ch) and ch[state['c'][tag]][0] <= i:
            ossop.linsegs(ch[state['c'][tag]][1], radius=0.0042,
                          srgbs=np.asarray(COL[tag], np.float32),
                          alpha=1.0).attach_to(scene)
            state['c'][tag] += 1
    state['t'] = t + 1


world.schedule_interval(tick, interval=1 / 60.0)
world.run()
