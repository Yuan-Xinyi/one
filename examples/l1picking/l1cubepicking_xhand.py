"""Cube-picking demo (simplified): an xArm7 on a tabletop, fitted with a 12-DOF
XHand, picks a single 6 cm cube off the table and lifts it. No bins, no pile, no
physics -- just one cube and one top-down pick.

xArm7 <-> XHand connection (mirrors the fr3_xhand URDF scheme): the hand is
mounted on link7 (the flange) by  T(0, 0, FLANGE_Z) @ Rz(MOUNT_RPY)  -- a small
push-out along the tool axis plus a 270 deg yaw that aligns the hand with the
arm. Change FLANGE_Z / MOUNT_RPY to retune the attachment.

Keys:  F = step one frame   G = play/pause   R = replay
Headless (plan the pick, no window):  ONE_HEADLESS=1
"""
import os
import sys
import builtins

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for p in (_PROJECT_ROOT, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import one.utils.constant as ouc                               # noqa: E402
import one.utils.math as oum                                   # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.collider.mj_collider as ocm                         # noqa: E402
import one.motion.probabilistic.rrt as ompr                    # noqa: E402
from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7     # noqa: E402
from one.robots.end_effectors.xhand.xhand_right import XHandRight  # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
from l1picking import (TABLE_TOP_Z,                            # noqa: E402
                       chain_planning_context, plan_segment)


# =============================== configuration ===============================
CHAIN = 'main'                       # xArm7 arm chain (7-DOF, numerical IK)

# Table (rotated 90 deg / enlarged vs l1picking's): long axis along y. Placed so
# the arm (base x = 0.30) sits on the table's -x long edge.
TABLE_ORIGIN = np.array([0.75, -0.25, 0.0], dtype=np.float32)
TABLE_X, TABLE_Y = 0.9, 1.6          # tabletop extents (short x, long y)
TABLE_TOP_THICK, TABLE_LEG = 0.04, 0.05
TABLE_RGB = (0.55, 0.42, 0.30)

# xArm7 base, and the XHand-on-flange connection.
ROBOT_BASE_POS = np.array([0.30, -0.25, TABLE_TOP_Z], dtype=np.float32)
FLANGE_Z = 0.0                       # push-out along link7 +Z, metres
MOUNT_RPY = 4.71239                  # hand yaw about Z at the flange (270 deg)

# The cube: 6 cm, directly in front of the arm (a fixed distance ahead in +x).
CUBE_SIZE = 0.06
CUBE_FORWARD = 0.35
CUBE_POS = np.array([ROBOT_BASE_POS[0] + CUBE_FORWARD, ROBOT_BASE_POS[1],
                     TABLE_TOP_Z + CUBE_SIZE / 2], dtype=np.float32)
CUBE_ROT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2)   # 90 deg / z
CUBE_RGB = (0.85, 0.55, 0.20)

UP = np.array([0.0, 0.0, 0.15], dtype=np.float32)   # lift after grasp
APPROACH_H = 0.12                    # straight-down approach height over the cube
GRASP_PRIMITIVE = 'power'            # close ALL five fingers around the cube


# ============================== two IK helpers ==============================
# These two are the only sub-functions: each is called repeatedly inside the
# pick search below, so inlining them would just duplicate code.
def solve_ik(robot, ctx, pos, rot, tcp, ref, collision_free=False):
    """The IK solution for ``tcp`` at (pos, rot) closest to ``ref`` in joint
    space (xArm7 'main' chain). With ``collision_free`` it skips colliding
    solutions. Returns a full qs vector, or None if unreachable."""
    chain = robot.chain(CHAIN)
    ref_active = chain.extract_active_qs(ref)
    best, best_d = None, None
    for s in robot.ik(pos, rot, chain=CHAIN, tcp=tcp,
                      ref_qs=ref_active, max_solutions=8):
        if collision_free and not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(chain.extract_active_qs(s) - ref_active))
        if best_d is None or d < best_d:
            best, best_d = s.astype(np.float64), d
    return best


def cartesian_path(robot, ctx, tcp, start_q, p0, p1, rot, nstep=12):
    """A straight Cartesian move of ``tcp`` from world ``p0`` to ``p1`` at the
    fixed orientation ``rot``: solve IK at each step, seeded from the previous
    config so the arm stays on one branch. Returns the per-step config list, or
    None if any step has no IK or the densified path hits the statics."""
    prev, path = start_q, []
    for t in np.linspace(0, 1, nstep):
        q = solve_ik(robot, ctx, p0 * (1 - t) + p1 * t, rot, tcp, prev)
        if q is None:
            return None
        path.append(q)
        prev = q
    for a, b in zip(path[:-1], path[1:]):           # collision-check the line
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(1.5))))
        for t in np.linspace(0, 1, n):
            if not ctx.is_state_valid(a + (b - a) * t):
                return None
    return path


def power_grasp_amount(hand, cube, nstep=26):
    """Find the power closure that makes the fingers just WRAP the cube without
    piercing it: close fully (amount = 1.0, fingers dig into the cube), then open
    the fingers step by step and stop at the first amount where no fingertip
    vertex is still inside the cube -- i.e. the fingers in a power pose resting on
    the cube surface. ``hand`` must already be posed at the grasp (its base in
    place). Returns that amount in [0, 1]."""
    to_cube = np.linalg.inv(cube.wd_tf).astype(np.float64)   # world -> cube frame
    half = CUBE_SIZE / 2.0
    fingers = [ln for ln in hand.structure.lnk_map if ln != 'palm']
    for amount in np.linspace(1.0, 0.0, nstep):
        hand.power(float(amount))
        pierces = False
        for ln in fingers:
            loc = hand._world_vs(ln) @ to_cube[:3, :3].T + to_cube[:3, 3]
            if np.any(np.all(np.abs(loc) <= half, axis=1)):   # a vertex inside
                pierces = True
                break
        if not pierces:
            return float(amount)
    return 0.0


# ================================== the demo ==================================
def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    base = ovw.World(cam_pos=(1.6, 0.4, 1.6), cam_lookat_pos=(0.30, 0.10, 0.95))
    builtins.base = base

    # ---- scene: ground + table (top + 4 legs) + the cube ----
    ox, oy = float(TABLE_ORIGIN[0]), float(TABLE_ORIGIN[1])
    leg_h = TABLE_TOP_Z - TABLE_TOP_THICK
    statics = [ossop.plane(pos=(0, 0, 0.0)),
               ossop.box(pos=(ox, oy, TABLE_TOP_Z - TABLE_TOP_THICK / 2),
                         xyz_lengths=(TABLE_X, TABLE_Y, TABLE_TOP_THICK),
                         rgb=TABLE_RGB, collision_type=ouc.CollisionType.AABB)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            statics.append(ossop.box(
                pos=(ox + sx * (TABLE_X / 2 - TABLE_LEG / 2),
                     oy + sy * (TABLE_Y / 2 - TABLE_LEG / 2), leg_h / 2),
                xyz_lengths=(TABLE_LEG, TABLE_LEG, leg_h), rgb=TABLE_RGB,
                collision_type=ouc.CollisionType.AABB))
    cube = ossop.box(pos=CUBE_POS, xyz_lengths=(CUBE_SIZE,) * 3, rotmat=CUBE_ROT,
                     rgb=CUBE_RGB, collision_type=ouc.CollisionType.MESH,
                     is_free=True)

    # ---- robot: xArm7 with the XHand mounted on the flange (link7) ----
    robot = XArm7(pos=ROBOT_BASE_POS)
    robot.left_hand = XHandRight()
    mount_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, FLANGE_Z], dtype=np.float32),
        rotmat=oum.rotmat_from_axangle(ouc.StandardAxis.Z, MOUNT_RPY))
    robot.mount(robot.left_hand, robot.runtime_lnks[-1], mount_tf, update=True)

    ossop.frame().attach_to(base.scene)
    for e in [robot] + statics + [cube]:
        e.attach_to(base.scene)

    # ---- collider + planner (the cube is grasped, so it is NOT an obstacle) ----
    mjc = ocm.MJCollider()
    for e in [robot] + statics:
        mjc.append(e)
    mjc.actors = [robot]
    mjc.compile(margin=0.0, auto_acm=True)
    ctx = chain_planning_context(robot, mjc, CHAIN)
    planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36,
                                     goal_bias=0.3)

    # ---- top-down power grasp: drop the PALM straight down onto the cube ----
    # We aim the hand's palm centre (the power_center tcp) at the cube and try a
    # few yaws about the vertical, so the five fingers wrap the cube from around
    # its centre (not a fingertip pinch). The cube is excluded from the collider,
    # so the fingers may enclose it -- only arm/hand vs table is checked. The
    # pre->grasp descent and grasp->lift are straight vertical Cartesian moves.
    R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    palm_tcp = robot.left_hand.tcp('power_center')
    home = robot.qs.astype(np.float64).copy()
    pre_pos = CUBE_POS + np.array([0.0, 0.0, APPROACH_H], dtype=np.float32)
    traj = grasp_idx = grasp_amount = None
    for yaw in np.linspace(0.0, 2 * np.pi, 12, endpoint=False):
        rot = oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw) @ R_down
        pre_q = solve_ik(robot, ctx, pre_pos, rot, palm_tcp, home,
                         collision_free=True)
        if pre_q is None:
            continue
        descend = cartesian_path(robot, ctx, palm_tcp, pre_q, pre_pos,
                                 CUBE_POS, rot)
        if descend is None:
            continue
        retreat = cartesian_path(robot, ctx, palm_tcp, descend[-1],
                                 CUBE_POS, CUBE_POS + UP, rot)
        if retreat is None:
            continue
        # close until the fingers pierce the cube, then open back to first touch
        # -> the power pose that just wraps it.
        robot.fk(qs=descend[-1])
        grasp_amount = power_grasp_amount(robot.left_hand, cube)
        traj = plan_segment(planner, home, pre_q)   # RRT:  home -> pre
        traj += descend[1:]                          # straight down onto the cube
        grasp_idx = len(traj) - 1                    # the hand closes at this step
        traj += retreat[1:]                          # straight up:  lift
        break
    if traj is None:
        raise RuntimeError("no feasible top-down grasp found for the cube")
    print(f"cube at ({CUBE_POS[0]:.3f}, {CUBE_POS[1]:.3f}); "
          f"pick {len(traj)} waypoints (grasp@{grasp_idx}); "
          f"power amount {grasp_amount:.2f}")

    if headless:
        return

    # ---- interactive playback: step through traj, close the hand at grasp_idx --
    import pyglet.window.key as key
    st = {"i": 0, "held": False, "playing": False}

    def reset():
        if st["held"]:
            robot.left_hand.release(cube)
        robot.left_hand.open_hand()
        robot.fk(qs=traj[0])
        cube.set_pos_rotmat(pos=CUBE_POS, rotmat=CUBE_ROT)
        st["i"], st["held"] = 0, False
        base.scene.dirty = True

    def step():
        if st["i"] >= len(traj):
            st["playing"] = False
            return
        robot.fk(qs=traj[st["i"]])
        if st["i"] == grasp_idx and not st["held"]:
            robot.left_hand.grasp(cube, primitive=GRASP_PRIMITIVE,
                                  amount=grasp_amount)
            st["held"] = True
        st["i"] += 1
        base.scene.dirty = True

    def tick(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(key.R):
            reset(); return
        if im.is_key_pressed_edge(key.G):
            st["playing"] = not st["playing"]
        if im.is_key_pressed_edge(key.F):
            st["playing"] = False; step()
        if st["playing"]:
            step()

    reset()
    print("F: step   G: play/pause   R: replay")
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
