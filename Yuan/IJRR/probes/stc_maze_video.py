#!/usr/bin/env python3
"""one-viewer clip: the coverage maze at density M. The spanning-tree
skeleton lies faint red on the bed; the arm wraps it in one green stroke."""
import matplotlib; matplotlib.use('Agg')
import builtins, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

M = int(sys.argv[1])
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/ompl')
FR = OUT / f'stc_maze{M}_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN, CH, STEPF = 0.10, 10, 0.010
GREEN, RED = (0.42, 0.63, 0.17), (0.86, 0.20, 0.24)

d = np.load(OUT / f'stc_maze{M}_pack.npz', allow_pickle=False)
fig, Q, tip, zax, arc = d['figure'], d['q_b'], d['tip_b'], d['zax_b'], d['arc_b']
S, cx, cy, zb = [float(v) for v in d['size']]
treeseg = d['tree']
keep = [0]
for i in range(1, len(arc)):
    if arc[i] - arc[keep[-1]] >= STEPF:
        keep.append(i)
keep = np.array(keep)
Q, tip, zax = Q[keep], tip[keep], zax[keep]

lo = np.minimum(fig.min(0), np.float32([-0.12, -0.12, 0.0]))
hi = np.maximum(fig.max(0), np.float32([0.12, 0.12, 0.88]))
ctr = (0.5 * (lo + hi)).astype(np.float32)
ext = float(np.linalg.norm(hi - lo))
dirv = np.array([0.55, -1.15, 0.80]); dirv /= np.linalg.norm(dirv)
world = ovw.World(cam_pos=tuple(ctr + dirv * (0.92 * ext)),
                  cam_lookat_pos=tuple(ctr), win_size=(1280, 720))
builtins.base = world
scene = world.scene
ossop.box(pos=(cx, cy, zb - 0.012),
          half_extents=(0.62 * S + 0.06, 0.62 * S + 0.06, 0.012),
          rgb=(0.93, 0.93, 0.94), alpha=1.0).attach_to(scene)
# the skeleton, faint red on the bed
ossop.linsegs(treeseg + [0, 0, 0.001], radius=0.0028,
              srgbs=np.float32(RED), alpha=0.55).attach_to(scene)

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
                     rgb=GREEN, alpha=0.98)
pen.set_rotmat_pos(*_pen(tip[0], zax[0])); pen.attach_to(scene)
seg, fr_, last = [], [], tip[0]
for i in range(1, len(tip)):
    if np.linalg.norm(tip[i] - last) > 0.005:
        seg.append([last + [0, 0, 0.003], tip[i] + [0, 0, 0.003]])
        fr_.append(i); last = tip[i]
CHUNKS = [(fr_[min(i+CH, len(seg))-1], np.asarray(seg[i:i+CH], np.float32))
          for i in range(0, len(seg), CH)]
K, HOLD = len(Q), 80
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
        ossop.linsegs(CHUNKS[state['c']][1], radius=0.0038,
                      srgbs=np.float32(GREEN), alpha=1.0).attach_to(scene)
        state['c'] += 1
    state['t'] = t + 1

world.schedule_interval(tick, interval=1 / 60.0)
world.run()
