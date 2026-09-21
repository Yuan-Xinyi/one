#!/usr/bin/env python3
"""one-viewer clip for the single-stroke packs (wrslogo2 / wrs_onestroke).

Frontal camera per user rule: the camera azimuth is aligned with the
base->pattern-centre axis, so the robot faces the lens head-on and sits
centred; only the elevation is free. One stroke means no pen lifts.
Usage: wrs2_video.py <tag>          tag in {wrslogo2, wrs_onestroke}
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

TAG = sys.argv[1]
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/ompl')
FR = OUT / f'{TAG}_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN, CH, STEPF = 0.10, 8, 0.008
INK = (0.13, 0.18, 0.45)

d = np.load(OUT / f'{TAG}_pack.npz', allow_pickle=True)
H, cx, cy, zb = [float(v) for v in d['size']]
name = str(d['names'][0])
Q, tip, zax = d[f'{name}_q'], d[f'{name}_tip'], d[f'{name}_zax']
fig = d[f'{name}_fig']

arc = np.concatenate([[0.0], np.cumsum(
    np.linalg.norm(np.diff(tip, axis=0), axis=1))])
keep = [0]
for i in range(1, len(Q)):
    if arc[i] - arc[keep[-1]] >= STEPF:
        keep.append(i)
keep = np.array(keep)

# ---- camera: user-picked pose (wrs2_campick.json), retargeted to the
# board centre so both clips share the same viewing offset ----
import json
_here = Path(__file__).resolve().parent
board = np.array([cx, cy, zb], np.float32)
# unified user-picked view: same offset for both clips, aimed at each board
_cp = json.loads((_here / 'wrs2_campick_wrslogo2.json').read_text())
_cam = board + (np.array(_cp['pos']) - np.array(_cp['look_at']))
_look = board
world = ovw.World(cam_pos=tuple(_cam), cam_lookat_pos=tuple(_look),
                  win_size=(1920, 1080))
builtins.base = world
scene = world.scene
ossop.box(pos=(cx, cy, zb - 0.012),
          half_extents=(0.55 * (fig[:, 0].max() - fig[:, 0].min()) + 0.10,
                        0.55 * (fig[:, 1].max() - fig[:, 1].min()) + 0.10,
                        0.012),
          rgb=(0.96, 0.96, 0.97), alpha=1.0).attach_to(scene)
ossop.linsegs(np.stack([fig[:-1], fig[1:]], 1), radius=0.0014,
              srgbs=np.float32([0.72, 0.74, 0.78]),
              alpha=0.45).attach_to(scene)


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
arm.attach_to(scene); arm.fk(Q[keep[0]])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                     rgb=INK, alpha=0.98)
pen.set_rotmat_pos(*_pen(tip[keep[0]], zax[keep[0]]))
pen.attach_to(scene)

segs, fr_, last = [], [], None
for j, i in enumerate(keep):
    t = tip[i]
    if last is not None and np.linalg.norm(t - last) > 0.004:
        segs.append([last + [0, 0, 0.003], t + [0, 0, 0.003]]); fr_.append(j)
    last = t
CHUNKS = [(fr_[min(i + CH, len(segs)) - 1],
           np.asarray(segs[i:i + CH], np.float32))
          for i in range(0, len(segs), CH)]
K, HOLD = len(keep), 80
state = {'t': 0, 'c': 0}


def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR / f'f{t - 1:05d}.png'))
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    i = keep[k]
    arm.fk(Q[i])
    pen.set_rotmat_pos(*_pen(tip[i], zax[i]))
    while state['c'] < len(CHUNKS) and CHUNKS[state['c']][0] <= k:
        ossop.linsegs(CHUNKS[state['c']][1], radius=0.0045,
                      srgbs=np.float32(INK), alpha=1.0).attach_to(scene)
        state['c'] += 1
    state['t'] = t + 1


world.schedule_interval(tick, interval=1 / 60.0)
world.run()
