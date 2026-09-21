#!/usr/bin/env python3
"""Interactive camera picker: adjust the view with the mouse, close the
window when satisfied; the camera pose is saved to wrs2_campick.json
twice a second. Scene = board + full ink stroke + robot mid-stroke."""
import builtins, json, sys
sys.path.insert(0, "/home/lqin/one")
from pathlib import Path
import numpy as np
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis/ompl')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'wrs_onestroke'
HERE = Path(__file__).resolve().parent
SAVE = HERE / (f'wrs2_campick_{TAG}.json' if TAG != 'wrs_onestroke'
               else 'wrs2_campick.json')
PEN_LEN = 0.10
INK = (0.13, 0.18, 0.45)

d = np.load(OUT / f'{TAG}_pack.npz', allow_pickle=True)
H, cx, cy, zb = [float(v) for v in d['size']]
name = str(d['names'][0])
Q, tip, zax, fig = d[f'{name}_q'], d[f'{name}_tip'], d[f'{name}_zax'], \
    d[f'{name}_fig']

ext = float(np.linalg.norm(fig.max(0) - fig.min(0))) + 0.35
board = np.array([cx, cy, zb], np.float32)
if SAVE.exists():                       # resume from the last picked pose
    _p = json.loads(SAVE.read_text())
    cam0, look0 = np.array(_p['pos']), np.array(_p['look_at'])
elif (HERE / 'wrs2_campick.json').exists():   # seed from the word pose
    _p = json.loads((HERE / 'wrs2_campick.json').read_text())
    cam0 = board + (np.array(_p['pos']) - np.array(_p['look_at']))
    look0 = board
else:
    u = np.array([cx, cy, 0.0]); u /= (np.linalg.norm(u) + 1e-9)
    cam0 = board + u * (0.80 * ext) + np.array([0, 0, 0.62 * ext])
    look0 = board
world = ovw.World(cam_pos=tuple(cam0), cam_lookat_pos=tuple(look0),
                  win_size=(1600, 900))
builtins.base = world
scene = world.scene
ossop.box(pos=(cx, cy, zb - 0.012),
          half_extents=(0.55 * (fig[:, 0].max() - fig[:, 0].min()) + 0.10,
                        0.55 * (fig[:, 1].max() - fig[:, 1].min()) + 0.10,
                        0.012),
          rgb=(0.96, 0.96, 0.97), alpha=1.0).attach_to(scene)
ossop.linsegs(np.stack([fig[:-1], fig[1:]], 1), radius=0.0045,
              srgbs=np.float32(INK), alpha=1.0).attach_to(scene)
arm, hand = fr3_with_hand(jaw_width=0.0)
arm.attach_to(scene)
k = len(Q) * 2 // 5
arm.fk(Q[k])
z = zax[k] / (np.linalg.norm(zax[k]) + 1e-9)
h = np.array([1.0, 0, 0], np.float32)
x = h - (h @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
R = np.stack([x, np.cross(z, x), z], 1).astype(np.float32)
pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, PEN_LEN), radius=0.007,
                     rgb=INK, alpha=0.98)
pen.set_rotmat_pos(R, (tip[k] - PEN_LEN * z).astype(np.float32))
pen.attach_to(scene)


def dump(_dt):
    c = world.camera
    SAVE.write_text(json.dumps({
        'pos': [float(v) for v in np.asarray(c.pos)],
        'look_at': [float(v) for v in np.asarray(c.look_at)],
        'up': [float(v) for v in np.asarray(c.up)]}))


world.schedule_interval(dump, interval=0.5)
world.run()
print('closed; camera saved to', SAVE)
