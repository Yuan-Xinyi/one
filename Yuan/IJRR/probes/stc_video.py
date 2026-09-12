#!/usr/bin/env python3
"""one-viewer clip: the proposed method drawing a Spanning Tree Coverage
cycle. One arm, the coverage cycle faint underneath, the laid bead in
crimson; every subcell of the grid is visited in one continuous stroke."""
import matplotlib; matplotlib.use('Agg')
import builtins, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/ompl')
FR = OUT / 'stc_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN, RED, CH = 0.10, (0.86, 0.16, 0.30), 8

d = np.load(OUT / 'stc_pack.npz', allow_pickle=False)
fig, tip, zax, Q = d['figure'], d['tip_b'], d['zax_b'], d['q_b']
S, cx, cy, zb = [float(v) for v in d['size']]
# thin the frames so ~5 mm of arc passes per frame
arc = d['arc_b']
keep = [0]
for i in range(1, len(arc)):
    if arc[i] - arc[keep[-1]] >= 0.005:
        keep.append(i)
keep = np.array(keep)
Q, tip, zax = Q[keep], tip[keep], zax[keep]

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
ossop.linsegs(np.stack([fig[:-1], fig[1:]], 1), radius=0.0016,
              srgbs=np.float32([0.60, 0.62, 0.66]), alpha=0.9).attach_to(scene)

def _rot_with_z(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95:
        h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)

def _pen(t, z):
    z = z / (np.linalg.norm(z) + 1e-9)
    return _rot_with_z(z), (t - PEN_LEN * z).astype(np.float32)

arm, hand = fr3_with_hand(jaw_width=0.0)
arm.attach_to(scene); arm.fk(Q[0])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                     rgb=RED, alpha=0.98)
pen.set_rotmat_pos(*_pen(tip[0], zax[0])); pen.attach_to(scene)
seg, fr_, last = [], [], tip[0]
for i in range(1, len(tip)):
    if np.linalg.norm(tip[i] - last) > 0.004:
        seg.append([last + [0, 0, 0.003], tip[i] + [0, 0, 0.003]])
        fr_.append(i); last = tip[i]
CHUNKS = [(fr_[min(i + CH, len(seg)) - 1], np.asarray(seg[i:i + CH], np.float32))
          for i in range(0, len(seg), CH)]
K, HOLD = len(Q), 70
state = {'t': 0, 'c': 0}

def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR / f'f{t - 1:05d}.png'))
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    arm.fk(Q[k])
    pen.set_rotmat_pos(*_pen(tip[k], zax[k]))
    while state['c'] < len(CHUNKS) and CHUNKS[state['c']][0] <= k:
        ossop.linsegs(CHUNKS[state['c']][1], radius=0.0042,
                      srgbs=np.asarray(RED, np.float32), alpha=1.0
                      ).attach_to(scene)
        state['c'] += 1
    state['t'] = t + 1

world.schedule_interval(tick, interval=1 / 60.0)
world.run()
