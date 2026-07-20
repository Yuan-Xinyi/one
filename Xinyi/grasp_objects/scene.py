"""Scene definition for :mod:`grasp_cube`.

Keep physical layout, robot mounting and scene objects in this file. Objects
returned by ``build_static_objects`` are both rendered and included as static
obstacles by the picking script, so add tables, guards, boxes and other fixed
workspace geometry there.
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for path in (_PROJECT_ROOT, _THIS):
    if path not in sys.path:
        sys.path.insert(0, path)

import one.collider.mj_collider as ocm
import one.scene.scene_object_primitive as ossop
import one.utils.constant as ouc
import one.utils.math as oum
import one.viewer.world as ovw
from one.robots.end_effectors.xhand.xhand_right_withcc import XHandRight
from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7
from planning_utils import TABLE_TOP_Z


# Table geometry. ``TABLE_HEIGHT_OFFSET`` is the measured correction applied to
# the physical tabletop; objects resting on it must use PLANNING_TABLE_TOP_Z.
TABLE_ORIGIN = np.array([0.75, -0.25, 0.0], dtype=np.float32)
TABLE_X, TABLE_Y = 0.9, 1.6
TABLE_TOP_THICK, TABLE_LEG = 0.04, 0.05
TABLE_RGB = (0.55, 0.42, 0.30)
TABLE_HEIGHT_OFFSET = float(os.environ.get('TABLE_HEIGHT_OFFSET', 0.016))
PLANNING_TABLE_TOP_Z = TABLE_TOP_Z + TABLE_HEIGHT_OFFSET
# 5 mm is too large for the compact XHand: its tightly packed finger/palm links
# sit 5-9 mm apart during a pinch, so a 5 mm margin band registers them as phantom
# self-collisions (is_collided counts any contact within margin, and auto_acm only
# exempts pairs that already overlap at the HOME rest pose, not ones that merely
# come within-margin while grasping). That rejected EVERY grasp config (0 solvable).
# 2 mm keeps a real safety clearance to the table/object while clearing the phantom
# self-contacts (verified: grasp_tool goes 0 -> 4+ reachable grasps).
COLLISION_MARGIN = float(os.environ.get('COLLISION_MARGIN', 0.002))

# The arm base sits on an 8 mm plate. Raising the corrected tabletop does not
# move the robot base; it only changes their relative height.
BASE_PLATE_THICKNESS = float(os.environ.get('BASE_PLATE_THICKNESS', 0.008))
ROBOT_BASE_POS = np.array([0.30, -0.25,
                           TABLE_TOP_Z + BASE_PLATE_THICKNESS], dtype=np.float32)
TABLE_Z_BASE = PLANNING_TABLE_TOP_Z - float(ROBOT_BASE_POS[2])
FLANGE_Z = 0.0
MOUNT_RPY = 4.71239                  # XHand yaw at the flange (270 deg)

# Rear safety wall. The cube is in the robot's +X workspace, so -X is behind the
# robot. ``REAR_WALL_CLEARANCE`` is measured from the robot-base origin to the
# wall's nearest face. Override these values to match the physical installation.
REAR_WALL_CLEARANCE = float(os.environ.get('REAR_WALL_CLEARANCE', 0.20))
REAR_WALL_THICKNESS = float(os.environ.get('REAR_WALL_THICKNESS', 0.05))
REAR_WALL_WIDTH = float(os.environ.get('REAR_WALL_WIDTH', 1.60))
REAR_WALL_HEIGHT = float(os.environ.get('REAR_WALL_HEIGHT', 1.80))
REAR_WALL_RGB = (0.45, 0.48, 0.52)

# Target cube geometry and the fallback pose used without camera perception.
CUBE_SIZE = 0.06
CUBE_FORWARD = 0.35
CUBE_POS = np.array([ROBOT_BASE_POS[0] + CUBE_FORWARD, ROBOT_BASE_POS[1],
                     PLANNING_TABLE_TOP_Z + CUBE_SIZE / 2], dtype=np.float32)
CUBE_ROT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2)
CUBE_RGB = (0.85, 0.55, 0.20)

# Collision-free arm pose used only by this file's scene viewer.
VIEW_ARM_DEG = np.array([-16.9, -34.8, 18.8, 20.5, 86.9, 12.0, -79.8],
                        dtype=np.float32)


def build_table():
    """Build the tabletop and four legs as static collision objects."""
    ox, oy = float(TABLE_ORIGIN[0]), float(TABLE_ORIGIN[1])
    leg_h = PLANNING_TABLE_TOP_Z - TABLE_TOP_THICK
    parts = [ossop.box(
        pos=(ox, oy, PLANNING_TABLE_TOP_Z - TABLE_TOP_THICK / 2),
        xyz_lengths=(TABLE_X, TABLE_Y, TABLE_TOP_THICK),
        rgb=TABLE_RGB, collision_type=ouc.CollisionType.AABB)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(ossop.box(
                pos=(ox + sx * (TABLE_X / 2 - TABLE_LEG / 2),
                     oy + sy * (TABLE_Y / 2 - TABLE_LEG / 2), leg_h / 2),
                xyz_lengths=(TABLE_LEG, TABLE_LEG, leg_h), rgb=TABLE_RGB,
                collision_type=ouc.CollisionType.AABB))
    return parts


def build_rear_wall():
    """Build the fixed collision wall behind the robot (-X direction)."""
    wall_x = (float(ROBOT_BASE_POS[0]) - REAR_WALL_CLEARANCE
              - REAR_WALL_THICKNESS / 2)
    return ossop.box(
        pos=(wall_x, float(ROBOT_BASE_POS[1]), REAR_WALL_HEIGHT / 2),
        xyz_lengths=(REAR_WALL_THICKNESS, REAR_WALL_WIDTH, REAR_WALL_HEIGHT),
        rgb=REAR_WALL_RGB, collision_type=ouc.CollisionType.AABB)


def build_static_objects():
    """Return all fixed scene objects used by rendering and collision checks.

    Add new fixed workspace objects to this list. Use an appropriate
    ``collision_type`` so motion planning also avoids them.
    """
    statics = [ossop.plane(pos=(0, 0, 0.0))]
    statics.extend(build_table())
    statics.append(build_rear_wall())

    # Add fixed scene objects here, for example:
    # statics.append(ossop.box(
    #     pos=(0.2, -0.8, 1.0), xyz_lengths=(0.2, 0.1, 0.2),
    #     rgb=(0.4, 0.4, 0.4), collision_type=ouc.CollisionType.AABB))
    return statics


def build_robot():
    """Build the xArm7 and mount the right XHand at the calibrated flange pose."""
    robot = XArm7(pos=ROBOT_BASE_POS)
    robot.left_hand = XHandRight()
    mount_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, FLANGE_Z], dtype=np.float32),
        rotmat=oum.rotmat_from_axangle(ouc.StandardAxis.Z, MOUNT_RPY))
    robot.mount(robot.left_hand, robot.runtime_lnks[-1], mount_tf, update=True)
    return robot


def build_scene(cube_pos=None, cube_rot=None):
    """Build and return ``(robot, statics, cube)`` for the picking demo."""
    if cube_pos is None:
        cube_pos = CUBE_POS
    if cube_rot is None:
        cube_rot = CUBE_ROT
    cube = ossop.box(
        pos=np.asarray(cube_pos, dtype=np.float32),
        xyz_lengths=(CUBE_SIZE,) * 3,
        rotmat=np.asarray(cube_rot, dtype=np.float32),
        rgb=CUBE_RGB, collision_type=ouc.CollisionType.MESH,
        is_floating=True)
    return build_robot(), build_static_objects(), cube


def build_collider(robot, statics, target=None):
    """Build the planning collider from the exact static objects in the scene.

    Every object in ``statics`` is included automatically. When ``target`` is
    provided, a fixed oriented proxy is also included for free-space planning.
    Omit it for grasp descent/lift, where hand-to-target contact is intentional.
    """
    collider = ocm.MJCollider()
    for obj in [robot] + list(statics):
        collider.append(obj)
    if target is not None:
        # The procedural cube's MESH shape has no backing mesh file, which
        # MuJoCo cannot register as an asset. An equal-size oriented box is exact
        # for this cube and keeps its detected yaw without AABB over-expansion.
        target_obstacle = ossop.box(
            pos=target.pos, rotmat=target.rotmat,
            xyz_lengths=(CUBE_SIZE,) * 3,
            collision_type=ouc.CollisionType.OBB)
        target_obstacle.name = 'target_transit_obstacle'
        target_obstacle.collision_group = ouc.CollisionGroup.STATIC
        collider.append(target_obstacle)
    collider.actors = [robot]
    collider.compile(margin=COLLISION_MARGIN, auto_acm=True)
    return collider


def main():
    """Display the standalone scene without planning or hardware connection."""
    base = ovw.World(cam_pos=(1.6, 0.4, 1.6),
                     cam_lookat_pos=(0.45, -0.1, 0.95))
    robot, statics, cube = build_scene()

    qs = robot.qs.astype(np.float64).copy()
    qs[robot.chain('main').active_jnt_ids] = np.deg2rad(VIEW_ARM_DEG)
    robot.fk(qs=qs)

    ossop.frame().attach_to(base.scene)
    for obj in [robot] + statics + [cube]:
        obj.attach_to(base.scene)
    base.run()


if __name__ == '__main__':
    main()
