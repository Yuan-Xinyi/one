#!/usr/bin/env python3
"""Fixed-camera one-viewer video of the three-way replay for one task.

Three translucent fr3 arms in the curve colors (classical crimson /
RL royal blue / search forest green), each with a matching pen; the
task seam drawn in gray. Frames captured per tick, window closes when
done. argv: task_id"""
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
FR = OUT/f't{TI}_frames'; FR.mkdir(exist_ok=True)
PEN_LEN = 0.10
COLS = {'cls': (0.86, 0.14, 0.30), 'rl': (0.25, 0.41, 0.88),
        'srch': (0.13, 0.55, 0.13)}
ALPHA = 0.42

d = np.load(OUT/f't{TI}_render_pack.npz')
p0, ray, lpw = d['p0'], d['d'], float(d['lpw'])
mid = p0 + 0.5 * lpw * ray
world = ovw.World(cam_pos=tuple(mid + np.array([1.1, -1.55, 0.95])),
                  cam_lookat_pos=tuple(mid), win_size=(1280, 720))
builtins.base = world
scene = world.scene
ossop.cylinder(spos=tuple(p0), epos=tuple(p0 + lpw * ray), radius=0.004,
               rgb=(0.45, 0.45, 0.48), alpha=0.9).attach_to(scene)
ossop.sphere(pos=tuple(p0), radius=0.014, rgb=(0.2, 0.2, 0.22),
             alpha=0.9).attach_to(scene)


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


arms = {}
for tag, col in COLS.items():
    arm, hand = fr3_with_hand(jaw_width=0.0)
    arm.attach_to(scene)
    arm.fk(d[f'{tag}_q'][0])
    for lnk in list(arm.runtime_lnks) + list(hand.runtime_lnks):
        lnk.alpha = ALPHA
        lnk.rgb = col
    pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.006,
                         rgb=col, alpha=0.95)
    pen.set_rotmat_pos(*_pen_pose(d[f'{tag}_tip'][0], d[f'{tag}_zax'][0]))
    pen.attach_to(scene)
    arms[tag] = (arm, pen)

K = max(len(d['cls_q']), len(d['rl_q']), len(d['srch_q']))
HOLD = 30
state = {'t': 0}


def tick(_dt):
    t = state['t']
    if t > 0:      # save the frame the loop just drew (poses of t-1)
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR/f'f{t - 1:04d}.png'))
    if t >= K + HOLD:
        world.close()
        pyglet.app.exit()
        return
    k = min(t, K - 1)
    for tag, (arm, pen) in arms.items():
        i = min(k, len(d[f'{tag}_q']) - 1)
        arm.fk(d[f'{tag}_q'][i])
        pen.set_rotmat_pos(*_pen_pose(d[f'{tag}_tip'][i],
                                      d[f'{tag}_zax'][i]))
    state['t'] = t + 1


world.schedule_interval(tick, interval=1 / 60.0)
world.run()
print(f't{TI}: {K + HOLD} frames saved', flush=True)
