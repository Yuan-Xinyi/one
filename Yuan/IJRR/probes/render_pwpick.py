"""Render (one/MuJoCo, real FR3 meshes) why the pointwise pick cannot
continue: the executed branch dies at 0.324 m with J2 on its limit (red
arm), while the admissible solutions 4 cm further down the seam all live
on a distant branch (blue arm) that a continuous motion cannot reach."""
import sys, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'Yuan/IJRR/figures'))
import numpy as np
import mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fr3_scene import _arm_xml, _rgba, add_polyline, MESH_ARM, MESH_HAND

FU = MAIN / 'runs/paper_fill/fam_unify'
d = np.load(FU / 'pwpick_anatomy.npz')
S, Q, TQ, TS = d['S'], d['Q'], d['TQ'], d['TS']
stroke = float(d['stroke'])
P0 = np.array([-0.25, -0.25, 0.526])
D = np.array([1.0, 0.0, 0.0])

q_dead = TQ[-1]
S_NEXT = 0.36
m = np.isclose(S, S_NEXT)
cand = Q[m]
dist = np.linalg.norm(cand - q_dead[None], axis=1)
q_next = cand[dist.argmin()]
print(f'nearest admissible at s={S_NEXT:.2f}: |dq| {dist.min():.2f} rad '
      f'(per-joint {np.abs(q_next - q_dead).round(2)})')
print(f'J2: dead {q_dead[1]:.2f} (limit 1.76), next-branch {q_next[1]:.2f}')

arms = [
    (q_dead, '#d62728', 0.60, 1.0),    # executed branch at its death
    (q_next, '#1f77b4', 0.60, 0.50),   # branch required 4 cm later
]
copies = ''
for k, (q, color, tint, alpha) in enumerate(arms):
    rgba = _rgba(color, alpha, tint)
    rgba0 = rgba if k == 0 else '0 0 0 0'
    copies += f"""
    <body name="r{k}_base" pos="0 0 0">
      <geom type="mesh" mesh="link0" rgba="{rgba0}" group="{1 + k}"/>{_arm_xml(k, rgba, 1 + k)}
    </body>"""

meshes = '\n    '.join(
    f'<mesh name="link{i}" file="{MESH_ARM}/link{i}.stl"/>' for i in range(8))
meshes += (f'\n    <mesh name="hand" file="{MESH_HAND}/hand.stl"/>'
           f'\n    <mesh name="finger" file="{MESH_HAND}/finger.stl"/>')
xml = f"""<mujoco model="pwpick">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.08 0.08 0.08"/>
    <global offwidth="1920" offheight="1400" fovy="22"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.05"/>
  </visual>
  <asset>
    {meshes}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1"
             rgb2="1 1 1" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.93"
             rgb2="0.86 0.86 0.86" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="-1.6 -1.2 3.4" dir="0.45 0.35 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="12 12 0.05" material="grid" group="0"/>
{copies}
  </worldbody>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for k, (q, *_rest) in enumerate(arms):
    for i in range(7):
        adr = model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f'r{k}_j{i + 1}')]
        data.qpos[adr] = float(q[i])
mujoco.mj_forward(model, data)

r = mujoco.Renderer(model, 1150, 1700, max_geom=20000)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [-0.02, -0.20, 0.46]
cam.distance = 1.95
cam.azimuth = -150
cam.elevation = -30

opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1
opt.geomgroup[2] = 1
r.update_scene(data, camera=cam, scene_option=opt)

# pens + seam
for k, (_q, color, *_r2) in enumerate(arms):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'r{k}_tcp')
    ps = data.site_xpos[sid].copy()
    Rz = data.site_xmat[sid].reshape(3, 3)[:, 2]
    add_polyline(r.scene, np.stack([ps, ps + 0.10 * Rz]),
                 (0.15, 0.15, 0.15, 1.0), width=0.008)
n_exec = P0 + stroke * D
add_polyline(r.scene, np.stack([P0, n_exec]), (0.1, 0.1, 0.1, 1.0),
             width=0.005)
add_polyline(r.scene, np.stack([n_exec, P0 + 0.5 * D]),
             (0.9, 0.35, 0.08, 1.0), width=0.005)
g = r.scene.geoms[r.scene.ngeom]
mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.011, 0, 0]), n_exec.astype(np.float64),
                    np.eye(3).ravel(), np.array([0.85, 0.1, 0.1, 1.0],
                                                dtype=np.float32))
r.scene.ngeom += 1

img = r.render()
plt.imsave(FU / 'pwpick_render.png', img)
print('rendered ->', FU / 'pwpick_render.png')
