#!/usr/bin/env python3
"""one-viewer clip: the arm writes W-R-S, one stroke per letter, pen lifted
between letters (grey nozzle, no ink)."""
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
FR = OUT / 'wrs_frames'; FR.mkdir(parents=True, exist_ok=True)
PEN_LEN, CH, STEPF, NTRAV = 0.10, 8, 0.008, 55
INK, TRAV = (0.13, 0.18, 0.45), (0.55, 0.57, 0.60)

d = np.load(OUT / 'wrs_pack.npz', allow_pickle=True)
H, cx, cy, zb = [float(v) for v in d['size']]
names = [str(n) for n in d['names']]

frames_q, frames_tip, frames_zax, dep = [], [], [], []
prev_q = None
for name in names:
    Q, tip, zax = d[f'{name}_q'], d[f'{name}_tip'], d[f'{name}_zax']
    arc = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(tip, axis=0), axis=1))])
    keep = [0]
    for i in range(1, len(Q)):
        if arc[i] - arc[keep[-1]] >= STEPF:
            keep.append(i)
    keep = np.array(keep)
    if prev_q is not None:                       # pen-lift travel
        for a in np.linspace(0, 1, NTRAV)[1:-1]:
            frames_q.append((1 - a) * prev_q + a * Q[0])
            frames_tip.append(None); frames_zax.append(None)
            dep.append(False)
    for i in keep:
        frames_q.append(Q[i]); frames_tip.append(tip[i])
        frames_zax.append(zax[i]); dep.append(True)
    prev_q = Q[keep[-1]]

figs = np.concatenate([d[f'{n}_fig'] for n in names])
lo = np.minimum(figs.min(0), np.float32([-0.12, -0.12, 0.0]))
hi = np.maximum(figs.max(0), np.float32([0.12, 0.12, 0.88]))
ctr = (0.5 * (lo + hi)).astype(np.float32)
ext = float(np.linalg.norm(hi - lo))
dirv = np.array([0.15, -1.05, 0.85]); dirv /= np.linalg.norm(dirv)
world = ovw.World(cam_pos=tuple(ctr + dirv * (0.90 * ext)),
                  cam_lookat_pos=tuple(ctr), win_size=(1280, 720))
builtins.base = world
scene = world.scene
ossop.box(pos=(cx, cy, zb - 0.012),
          half_extents=(1.12 * H + 0.08, 0.62 * H + 0.08, 0.012),
          rgb=(0.96, 0.96, 0.97), alpha=1.0).attach_to(scene)
for n in names:
    f = d[f'{n}_fig']
    ossop.linsegs(np.stack([f[:-1], f[1:]], 1), radius=0.0014,
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
arm.attach_to(scene); arm.fk(frames_q[0])
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                     rgb=INK, alpha=0.98)
pen.set_rotmat_pos(*_pen(frames_tip[0], frames_zax[0]))
pen.attach_to(scene)

segs, fr_, last = [], [], None
for i in range(len(frames_q)):
    if not dep[i]:
        last = None; continue
    t = frames_tip[i]
    if last is not None and np.linalg.norm(t - last) > 0.004:
        segs.append([last + [0, 0, 0.003], t + [0, 0, 0.003]]); fr_.append(i)
    last = t
CHUNKS = [(fr_[min(i+CH, len(segs))-1], np.asarray(segs[i:i+CH], np.float32))
          for i in range(0, len(segs), CH)]
K, HOLD = len(frames_q), 80
state = {'t': 0, 'c': 0}

def tick(_dt):
    t = state['t']
    if t > 0:
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            str(FR / f'f{t - 1:05d}.png'))
    if t >= K + HOLD:
        world.close(); pyglet.app.exit(); return
    k = min(t, K - 1)
    arm.fk(frames_q[k])
    if dep[k]:
        pen.rgb = INK
        pen.set_rotmat_pos(*_pen(frames_tip[k], frames_zax[k]))
    else:
        pen.rgb = TRAV
        import numpy as _np
        # during travel the pen follows the flange via FK of the shown q
        # (approximated by holding last pose orientation)
    while state['c'] < len(CHUNKS) and CHUNKS[state['c']][0] <= k:
        ossop.linsegs(CHUNKS[state['c']][1], radius=0.0045,
                      srgbs=np.float32(INK), alpha=1.0).attach_to(scene)
        state['c'] += 1
    state['t'] = t + 1

world.schedule_interval(tick, interval=1 / 60.0)
world.run()
