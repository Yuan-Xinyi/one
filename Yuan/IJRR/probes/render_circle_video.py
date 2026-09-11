#!/usr/bin/env python3
"""one-viewer video of the multi-lap circle rollout: fixed camera, the
circle ring in gray, pen-tip trace colored per lap, live arm + pen."""
import matplotlib; matplotlib.use('Agg')
import builtins, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import pyglet
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

TI = int(sys.argv[1])
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/search_compare')
FR = OUT/f'circle_t{TI}_frames'; FR.mkdir(exist_ok=True)
PEN_LEN = 0.10
LAP_COLS = [(0.25, 0.41, 0.88), (0.13, 0.55, 0.13), (0.86, 0.14, 0.30),
            (0.93, 0.60, 0.10), (0.55, 0.15, 0.75), (0.05, 0.65, 0.65),
            (0.55, 0.35, 0.05), (0.35, 0.35, 0.35)]

d = np.load(OUT/f'circle_t{TI}_pack.npz')
ring, circ = d['ring'], float(d['circ'])
mid = ring.mean(0)
world = ovw.World(cam_pos=tuple(mid + np.array([1.05, -1.45, 0.9])),
                  cam_lookat_pos=tuple(mid), win_size=(1280, 720))
builtins.base = world
scene = world.scene
for i in range(len(ring) - 1):
    ossop.cylinder(spos=tuple(ring[i]), epos=tuple(ring[i + 1]),
                   radius=0.003, rgb=(0.5, 0.5, 0.52),
                   alpha=0.85).attach_to(scene)


def _rot_with_z(z):
    z = z / (np.linalg.norm(z) + 1e-9)
    h = np.array([1.0, 0.0, 0.0], np.float32)
    if abs(float(h @ z)) > 0.95:
        h = np.array([0.0, 1.0, 0.0], np.float32)
    x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)


def _pen_pose(tip, zax):
    z = zax / (np.linalg.norm(zax) + 1e-9)
    return _rot_with_z(z), (tip - PEN_LEN * z).astype(np.float32)


arm, hand = fr3_with_hand(jaw_width=0.0)
arm.attach_to(scene)
arm.fk(d['q'][0])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.006,
                     rgb=LAP_COLS[0], alpha=1.0)
pen.set_rotmat_pos(*_pen_pose(d['tip'][0], d['zax'][0]))
pen.attach_to(scene)

K = len(d['q'])
HOLD = 40
state = {'t': 0}


def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR/f'f{t - 1:04d}.png'))
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    arm.fk(d['q'][k])
    lap = int(d['s'][k] // circ)
    col = LAP_COLS[min(lap, len(LAP_COLS) - 1)]
    pen.rgb = col
    pen.set_rotmat_pos(*_pen_pose(d['tip'][k], d['zax'][k]))
    if k > 0:
        # trace slightly lifted off the ring so laps stay visible
        off = 0.004 * (lap + 1)
        n = np.array([0.0, 0.0, 1.0], np.float32)
        ossop.cylinder(spos=tuple(d['tip'][k - 1] + off * n),
                       epos=tuple(d['tip'][k] + off * n), radius=0.002,
                       rgb=col, alpha=0.9).attach_to(scene)
    state['t'] = t + 1


world.schedule_interval(tick, interval=1 / 60.0)
world.run()
print(f'circle t{TI}: {K + HOLD} frames saved', flush=True)
