#!/usr/bin/env python3
"""one-viewer clip of the arm printing one object at its largest size.

Fixed camera, print bed drawn as a plate, the nozzle as a pen on the flange,
and the bead laid down segment by segment as the tip advances. Where the
witness chain has to change branch the pen turns grey and no material is laid:
those are the re-seats the toolpath actually needs, played as travel moves
rather than skipped.

argv: object key (cube / penholder / bowl / vase / plate / dome / mug)
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
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/render')
FR = OUT / f'{KEY}_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN = 0.10
BEAD = (0.93, 0.45, 0.16)          # filament
TRAVEL = (0.45, 0.47, 0.52)
MIN_SEG = 0.004                    # skip beads shorter than this

d = np.load(OUT / f'{KEY}_pack.npz', allow_pickle=False)
Q, tip, zax, dep = d['q'], d['tip'], d['zax'], d['dep']
s, cx, cy, zb = [float(v) for v in d['size']]
path = d['path']

# frame the object AND the arm: the base sits at the origin and can be far
# from the part, so both go into the bounding box the camera is fitted to
lo = np.minimum(path.min(0), np.float32([-0.12, -0.12, 0.0]))
hi = np.maximum(path.max(0), np.float32([0.12, 0.12, 0.62]))
ctr = (0.5 * (lo + hi)).astype(np.float32)
ext = float(np.linalg.norm(hi - lo))
world = ovw.World(cam_pos=tuple(ctr + np.array([0.92, -1.22, 0.58]) * ext),
                  cam_lookat_pos=tuple(ctr), win_size=(1280, 720))
builtins.base = world
scene = world.scene

# print bed, sized to the object with a margin
ossop.box(pos=(cx, cy, zb - 0.012), half_extents=(0.75 * s, 0.75 * s, 0.012),
          rgb=(0.22, 0.24, 0.28), alpha=1.0).attach_to(scene)
# the toolpath, faint, so the finished shape is legible from frame one
ossop.linsegs(np.stack([path[:-1], path[1:]], 1), radius=0.0012,
              srgbs=np.array([0.62, 0.64, 0.68], np.float32),
              alpha=0.30).attach_to(scene)


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


arm, hand = fr3_with_hand(jaw_width=0.0)
arm.attach_to(scene)
arm.fk(Q[0])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.0065,
                     rgb=(0.15, 0.16, 0.18), alpha=1.0)
pen.set_rotmat_pos(*_pen_pose(tip[0], zax[0]))
pen.attach_to(scene)
nozzle = ossop.cone(spos=(0, 0, 0), epos=(0, 0, 0.02), radius=0.011,
                    rgb=BEAD, alpha=1.0)
nozzle.set_rotmat_pos(*_pen_pose(tip[0], zax[0]))
nozzle.attach_to(scene)

# Beads are batched. One scene object per bead makes every later frame redraw
# every earlier bead, so the clip slows to a crawl exactly where it is longest;
# grouping them into short runs keeps the object count flat at the cost of the
# material appearing a few centimetres at a time.
CH = 8
bseg, bframe, _last = [], [], path[0].copy()
for k in range(len(path)):
    if dep[k] and np.linalg.norm(path[k] - _last) > MIN_SEG:
        bseg.append([_last.copy(), path[k].copy()]); bframe.append(k)
        _last = path[k].copy()
    elif not dep[k]:
        _last = path[k].copy()
CHUNKS = [(bframe[min(i + CH, len(bseg)) - 1],
           np.asarray(bseg[i:i + CH], np.float32))
          for i in range(0, len(bseg), CH)]

K = len(Q)
HOLD = 60
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
    pen.set_rotmat_pos(*_pen_pose(tip[k], zax[k]))
    nozzle.set_rotmat_pos(*_pen_pose(tip[k], zax[k]))
    nozzle.rgb = BEAD if dep[k] else TRAVEL
    while state['c'] < len(CHUNKS) and CHUNKS[state['c']][0] <= k:
        ossop.linsegs(CHUNKS[state['c']][1], radius=0.0038,
                      srgbs=np.asarray(BEAD, np.float32),
                      alpha=1.0).attach_to(scene)
        state['c'] += 1
    state['t'] = t + 1


world.schedule_interval(tick, interval=1 / 60.0)
world.run()
