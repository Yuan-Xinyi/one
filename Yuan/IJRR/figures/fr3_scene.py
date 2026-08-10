"""MuJoCo scene with the real FR3 meshes, for the figure scripts.

Not a figure itself: a helper the figure files import so they can show the
actual robot instead of a stick figure. It builds one MJCF holding N *copies* of
the same scene (an FR3 on a pedestal, a wall with a doorway, a hinged leaf with
a handle), laid out side by side. Every copy owns a geom group, so a copy can be
rendered alone by enabling group 0 (shared floor) plus its own group.

The kinematic chain is generated from the same constants as
``one.robots.manipulators.franka.fr3_pen.batched_fr3_kin``: joint ``i`` sits in a
body placed at ``zero_tfs[i]`` and rotates about its local z, so
``d.qpos[joint i] = q[i]`` reproduces ``BatchedFR3Kinematics`` exactly (checked
to 1e-16 on the TCP).

Meshes come from the repo: ``one/robots/manipulators/franka/fr3/meshes/visual``
(link0..link7) and ``one/robots/end_effectors/fr3_gripper/meshes`` (hand,
finger). Rendering needs a GL backend; set ``MUJOCO_GL=osmesa`` if there is no
display (slower, ~20 fps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mujoco

_ONE = Path(__file__).resolve().parents[3] / 'one'
MESH_ARM = _ONE / 'robots/manipulators/franka/fr3/meshes/visual'
MESH_HAND = _ONE / 'robots/end_effectors/fr3_gripper/meshes'

# (rotx [rad], translation) of the fixed transform in front of joint i;
# mirrors BatchedFR3Kinematics._make_zero_tfs.
ZERO_TFS = [(0.0, (0.0, 0.0, 0.333)),
            (-math.pi / 2, (0.0, 0.0, 0.0)),
            (math.pi / 2, (0.0, -0.316, 0.0)),
            (math.pi / 2, (0.0825, 0.0, 0.0)),
            (-math.pi / 2, (-0.0825, 0.384, 0.0)),
            (math.pi / 2, (0.0, 0.0, 0.0)),
            (math.pi / 2, (0.088, 0.0, 0.0))]

WOOD = '0.72 0.55 0.35'
WALL_RGBA = '0.80 0.81 0.84 0.16'   # translucent so it never hides the arm
DARK = '0.18 0.18 0.20 1'


FR3_WHITE = np.array([0.94, 0.94, 0.95])


def _rgba(color: str | None, alpha: float = 1.0, tint: float = 0.0) -> str:
    """FR3 white by default; ``tint`` blends a matplotlib hex colour into it."""
    v = FR3_WHITE
    if color is not None and tint > 0.0:
        c = np.array([int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        v = (1.0 - tint) * v + tint * c
    return f'{v[0]:.3f} {v[1]:.3f} {v[2]:.3f} {alpha:.3f}'


def _arm_xml(k: int, rgba: str, group: int, jaw: float = 0.026,
             depth: int = 0) -> str:
    rx, pos = ZERO_TFS[depth]
    if depth < 6:
        inner = _arm_xml(k, rgba, group, jaw, depth + 1)
    else:
        inner = f"""
          <body name="r{k}_hand" pos="0 0 0.107" euler="0 0 {-math.pi / 4}">
            <geom type="mesh" mesh="hand" rgba="{rgba}" group="{group}"/>
            <body name="r{k}_lfinger" pos="0 {0.5 * jaw} 0.0584">
              <geom type="mesh" mesh="finger" rgba="{DARK}" group="{group}"/>
            </body>
            <body name="r{k}_rfinger" pos="0 {-0.5 * jaw} 0.0584" euler="0 0 {math.pi}">
              <geom type="mesh" mesh="finger" rgba="{DARK}" group="{group}"/>
            </body>
            <site name="r{k}_tcp" pos="0 0 0.1034" size="0.004" group="4"/>
          </body>"""
    return f"""
        <body name="r{k}_link{depth + 1}" pos="{pos[0]} {pos[1]} {pos[2]}" euler="{rx} 0 0">
          <joint name="r{k}_j{depth + 1}" type="hinge" axis="0 0 1" range="-3.2 3.2"/>
          <geom type="mesh" mesh="link{depth + 1}" rgba="{rgba}" group="{group}"/>{inner}
        </body>"""


def _copy_xml(k: int, spec) -> str:
    """One copy: arm (+ leaf), and the static scenery when ``spec.scenery``.

    Several copies can share a group and an offset: that is how a whole
    trajectory is drawn as one static overlay, one copy per sampled pose, with
    the alpha ramp in ``spec.alpha``.
    """
    door, base, offset, group = spec.door, spec.base, spec.offset, spec.group
    rgba = _rgba(spec.color, spec.alpha, spec.tint)
    h = door.hinge + offset
    bx, by, bz = base.t + offset
    yaw_leaf = math.radians(door.leaf_dir_deg)
    hw, hh = 0.5 * door.width, 0.5 * door.height
    wall_len = 0.60
    # wall slabs left and right of the doorway, in the closed-leaf plane
    walls = ''
    for tag, s0 in (('a', -wall_len), ('b', door.width)) if spec.scenery else ():
        c = h + (s0 + 0.5 * wall_len) * door.d0 + np.array([0.0, 0.0, hh])
        walls += (f'<geom name="w{k}{tag}" type="box" pos="{c[0]} {c[1]} {c[2]}" '
                  f'euler="0 0 {yaw_leaf}" size="{0.5 * wall_len} 0.04 {hh}" '
                  f'rgba="{WALL_RGBA}" group="{group}"/>\n    ')
    a = spec.alpha
    scenery = '' if not spec.scenery else f"""
    <geom name="ped{k}" type="box" pos="{bx} {by} {0.5 * bz}"
          size="0.115 0.115 {0.5 * bz}" rgba="0.30 0.31 0.34 1" group="{group}"/>
    {walls}{_frame_xml(k, door, h, group)}"""
    return f"""
    <body name="r{k}_base" pos="{bx} {by} {bz}" euler="0 0 {math.radians(base.yaw_deg)}">
      <geom type="mesh" mesh="link0" rgba="{rgba}" group="{group}"/>{_arm_xml(k, rgba, group, door.jaw_width)}
    </body>{scenery}
    <body name="r{k}_leaf" pos="{h[0]} {h[1]} {door.z_bottom}" euler="0 0 {yaw_leaf}">
      <joint name="r{k}_door" type="hinge" axis="0 0 1" range="-3.2 3.2"/>
      <geom type="box" pos="{hw} 0 {hh}" size="{hw} 0.018 {hh}"
            rgba="{WOOD} {a:.3f}" group="{group}"/>
      <geom type="cylinder" pos="{door.handle_r} {-door.handle_offset} {door.handle_z}"
            size="{door.handle_radius} 0.10" rgba="0.55 0.56 0.60 {a:.3f}" group="{group}"/>
      <geom type="cylinder" fromto="{door.handle_r} -0.018 {door.handle_z}
            {door.handle_r} {-door.handle_offset} {door.handle_z}"
            size="0.009" rgba="0.55 0.56 0.60 {a:.3f}" group="{group}"/>
    </body>"""


def _frame_xml(k: int, door, h: np.ndarray, group: int) -> str:
    """Slim door frame: a post on each side of the opening plus a lintel."""
    yaw = math.radians(door.leaf_dir_deg)
    hh = 0.5 * door.height
    out = ''
    for tag, s0 in (('l', -0.03), ('r', door.width + 0.03)):
        c = h + s0 * door.d0 + np.array([0.0, 0.0, hh])
        out += (f'\n    <geom name="f{k}{tag}" type="box" pos="{c[0]} {c[1]} {c[2]}" '
                f'euler="0 0 {yaw}" size="0.03 0.06 {hh}" rgba="0.55 0.52 0.48 1" '
                f'group="{group}"/>')
    c = h + 0.5 * door.width * door.d0 + np.array([0.0, 0.0, door.height + 0.03])
    out += (f'\n    <geom name="f{k}t" type="box" pos="{c[0]} {c[1]} {c[2]}" '
            f'euler="0 0 {yaw}" size="{0.5 * door.width + 0.06} 0.06 0.03" '
            f'rgba="0.55 0.52 0.48 1" group="{group}"/>')
    return out


@dataclass
class CopySpec:
    """One drawable copy of the scene."""
    door: object
    base: object
    offset: np.ndarray
    group: int
    alpha: float = 1.0
    color: str | None = None
    tint: float = 0.0
    scenery: bool = True        # draw pedestal, wall and door frame with it


class DoorScene:
    """Copies of the door scene in one model, driven by joint values.

    ``from_variants`` lays the variants out side by side, one copy each. With
    ``ghosts > 1`` every variant gets ``ghosts`` copies at the *same* place and
    in the same geom group, with alpha ramping up: set each of them to a
    different pose of the same rollout and one render is the whole trajectory as
    a static overlay.
    """

    @classmethod
    def from_variants(cls, variants, ghosts: int = 1, alpha_min: float = 0.16,
                      tint: float = 0.0, **kw) -> 'DoorScene':
        specs, offsets = [], []
        for k, v in enumerate(variants):
            off = np.array([0.0, k * kw.get('spacing', 2.6), 0.0])
            offsets.append(off)
            for j in range(ghosts):
                a = 1.0 if ghosts == 1 else (
                    alpha_min + (1.0 - alpha_min) * (j / (ghosts - 1)) ** 2.4)
                specs.append(CopySpec(door=v.door, base=v.base, offset=off,
                                      group=1 + k, alpha=a, color=v.color,
                                      tint=tint, scenery=(j == ghosts - 1)))
        scene = cls(specs, n_panels=len(variants), **kw)
        scene.offsets = offsets
        scene.ghosts = ghosts
        return scene

    def __init__(self, specs, n_panels: int | None = None, spacing: float = 2.6,
                 offwidth: int = 1920, offheight: int = 1200, fovy: float = 22.0):
        self.specs = list(specs)
        self.n = len(self.specs)
        self.n_panels = n_panels if n_panels is not None else self.n
        self.ghosts = 1
        self.spacing = spacing
        self.offsets = [sp.offset for sp in self.specs]
        meshes = '\n    '.join(
            f'<mesh name="link{i}" file="{MESH_ARM}/link{i}.stl"/>' for i in range(8))
        meshes += (f'\n    <mesh name="hand" file="{MESH_HAND}/hand.stl"/>'
                   f'\n    <mesh name="finger" file="{MESH_HAND}/finger.stl"/>')
        copies = '\n'.join(_copy_xml(k, sp) for k, sp in enumerate(self.specs))
        xml = f"""<mujoco model="door_scene">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.45 0.45 0.45" specular="0.08 0.08 0.08"/>
    <global offwidth="{offwidth}" offheight="{offheight}" fovy="{fovy}"
            azimuth="-90" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.05"/>
  </visual>
  <asset>
    {meshes}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98"
             rgb2="0.82 0.86 0.92" width="256" height="256"/>
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
        self.xml = xml
        self.fovy = fovy
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._qadr = [[self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'r{k}_j{i + 1}')]
            for i in range(7)] for k in range(self.n)]
        self._dadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'r{k}_door')]
            for k in range(self.n)]
        self._renderer = None
        self._opts = []
        for k in range(self.n_panels):
            o = mujoco.MjvOption()
            o.geomgroup[:] = 0
            o.geomgroup[0] = 1
            o.geomgroup[1 + k] = 1
            self._opts.append(o)
        self.opt_all = mujoco.MjvOption()

    # -- state -------------------------------------------------------------- #
    def set_state(self, k: int, q: np.ndarray, door_angle: float, swing: float):
        for i in range(7):
            self.data.qpos[self._qadr[k][i]] = float(q[i])
        self.data.qpos[self._dadr[k]] = float(swing * door_angle)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    # -- rendering ---------------------------------------------------------- #
    def renderer(self, width: int, height: int) -> mujoco.Renderer:
        if (self._renderer is None or self._renderer.width != width
                or self._renderer.height != height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width,
                                             max_geom=20000)
        return self._renderer

    def camera(self, k: int, distance: float, azimuth: float, elevation: float,
               lookat: np.ndarray) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(lookat) + self.offsets[k]
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def render(self, k: int, cam, width: int = 900, height: int = 760,
               traces: list | None = None) -> np.ndarray:
        """RGB image of copy ``k`` alone. ``traces`` are (points, rgba) polylines."""
        r = self.renderer(width, height)
        r.update_scene(self.data, camera=cam, scene_option=self._opts[k])
        if traces:
            for pts, rgba in traces:
                add_polyline(r.scene, np.asarray(pts) + self.offsets[k], rgba)
        return r.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def add_polyline(scene, pts: np.ndarray, rgba, width: float = 0.006):
    """Append a polyline to an mjvScene as thin capsules."""
    if len(pts) < 2:
        return
    rgba = np.asarray(rgba, dtype=np.float32)
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(a, dtype=np.float64),
                             np.asarray(b, dtype=np.float64))
        scene.ngeom += 1
