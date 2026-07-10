"""Grasp-pose search for an xArm7 + XHand: plan PINCH grasps on the cube with the
shared antipodal grasp planner (one.grasp.antipodal), then find one that is also
reachable and collision-free for the arm.

Grasp planning is NOT hand-rolled here: the XHand is presented to the parallel-jaw
``antipodal`` planner via ``hand.spawn_jaw('pinch')`` -- the exact same path
o6cylstlplanning.py uses for the O6 hand. ``antipodal`` samples antipodal contact
pairs on the cube, aligns the pinch opposition axis, and rejects hand-vs-cube
collisions, returning grasp poses (grasp-center frame) + a jaw width per grasp.

This script then layers the ARM on top of those grasps:
  * map each grasp pose into world (cube.wd_tf @ pose),
  * solve arm IK for the grasp-center tcp at the pre-grasp and grasp poses,
  * keep grasps whose approach / descend / lift are collision-free vs table+self.
The pinch jaw width sets how far the fingers close (hand.set_jaw_width).

NOTE: antipodal is a PARALLEL-JAW (opposition) planner, so only opposition
primitives apply. 'pinch' is the validated default; 'power' is a whole-hand
envelope (not a jaw) and is rejected by spawn_jaw.

Keys: F step  G play/pause  R replay  C collision spheres on/off  ENTER run on robot
Headless (plan only): ONE_HEADLESS=1

Cube pose source (instead of the hardcoded CUBE_POS/CUBE_ROT), by priority:
  * ONE_FP=1     -> FoundationPose (PREFERRED, full 6D, most accurate). First run,
    in env_isaaclab: `python RealExperiments/foundationpose_then_play.py --no_play`
    to write camera_T_cube to /tmp/foundationpose_cube_pose.npy; this script maps
    it into the base frame with the D435 extrinsic. Override path via ONE_FP_POSE.
  * ONE_CAMERA=1 -> in-house point-cloud clustering (one/camera/RS435/detect_cube).
    Yaw only, lower accuracy; CUBE_RGB="r,g,b" (0-1) biases the colour match.
The robot base frame == sim-world axes with origin at ROBOT_BASE_POS, so the sim
cube is placed at ROBOT_BASE_POS + p_base. Combine with ONE_REAL=1 to close the
loop: see the cube -> plan -> grasp it.

Real hardware (opt-in: ONE_REAL=1): on startup the arm's CURRENT joints are read
and used as the planning start / IK seed (instead of HOME_DEG), and ENTER streams
the selected candidate's pick to the real xArm7 + XHand. The grasp itself is
torque-feedback closed: the fingers close gradually and each freezes the instant
it presses the object (joint torque > CONTACT_TORQUE), so contact -- not just the
planned pose -- decides when to lift. IP / port / speeds / torque threshold are
the ONE_ARM_IP / ONE_HAND_PORT env vars and the REAL_* / *_TORQUE constants below.
Driver code: one/control (xarm7.XArm7X, xhand_x.XHandX).
"""
import os
import sys
import time
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
import one.robots.base.tcp as orbt                             # noqa: E402
from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7     # noqa: E402
from one.robots.end_effectors.xhand.xhand_right_withcc import XHandRight  # noqa: E402
from one.grasp.antipodal import antipodal                      # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
from l1picking import (TABLE_TOP_Z,                            # noqa: E402
                       chain_planning_context, plan_segment)


# =============================== configuration ===============================
CHAIN = 'main'                       # xArm7 arm chain (7-DOF, numerical IK)

TABLE_ORIGIN = np.array([0.75, -0.25, 0.0], dtype=np.float32)
TABLE_X, TABLE_Y = 0.9, 1.6
TABLE_TOP_THICK, TABLE_LEG = 0.04, 0.05
TABLE_RGB = (0.55, 0.42, 0.30)

ROBOT_BASE_POS = np.array([0.30, -0.25, TABLE_TOP_Z], dtype=np.float32)
FLANGE_Z = 0.0
MOUNT_RPY = 4.71239                  # hand yaw about Z at the flange (270 deg)

# xArm7 home (J1..J7, degrees). The all-zeros default has the arm/hand resting
# in collision with the table, so RRT from it never connects (it burns all
# max_iters, ~28 s, freezing the viewer on first play). This reachable, clear-of-
# table config is the planning start AND the IK seed for the grasp search.
HOME_DEG = np.array([-16.9, -34.8, 18.8, 20.5, 86.9, 12.0, -79.8],
                    dtype=np.float32)

CUBE_SIZE = 0.06
CUBE_FORWARD = 0.35
CUBE_POS = np.array([ROBOT_BASE_POS[0] + CUBE_FORWARD, ROBOT_BASE_POS[1],
                     TABLE_TOP_Z + CUBE_SIZE / 2], dtype=np.float32)
CUBE_ROT = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.pi / 2)
CUBE_RGB = (0.85, 0.55, 0.20)

UP = np.array([0.0, 0.0, 0.15], dtype=np.float32)   # lift after grasp
GRASP_PRIMITIVE = os.environ.get('GRASP_PRIMITIVE', 'pinch')   # opposition only

# antipodal grasp-planning parameters (mirrors o6cylstlplanning.py's PLAN_KW):
# surface sampling density, contact-normal opposition tolerance, roll resolution,
# how many collision-free central grasps to keep, and extra jaw clearance per
# grasp. Off-centre samples are removed before expensive hand collision checks.
# density 0.0006 matches test_xhand_cube_grasp.py: a 6 cm cube has few large faces,
# so sample densely -- at the old 0.0015 only ~14 surface points were drawn and the
# best-seated grasp swung wildly run to run (1-7 mm); 0.0006 settles it near ~1 mm.
PLAN_KW = dict(density=0.0006, normal_tol_deg=25, roll_step_deg=30,
               max_grasps=120, clearance=0.003)

# Approach-tilt preference. The XHand pinch seats both pads LEVEL on two vertical
# faces only when the hand is TILTED, not top-down: its thumb and index pads are
# offset ~31 mm along the approach axis (the thumb reaches ~3 cm further), so a
# straight top-down pinch lands them at different heights (screening: top-down best
# seating 4.8 mm vs 0.9 mm at a tilt). A downward tilt also clears the table and is
# more arm-reachable than a near-horizontal side approach. So prefer grasps whose
# approach points ~GRASP_TILT_DEG below horizontal, ranked JOINTLY with pad seating.
# Pad SEATING is the primary quality (and already disfavours top-down, which seats
# worst); tilt is a GENTLE rail so it never overrides a real seating advantage --
# 0.01 m/rad means a 20 deg tilt gap costs only ~3.5 mm of seating, enough to break
# near-ties toward the reachable/table-clearing band but not to pick a worse pinch.
GRASP_TILT_DEG = float(os.environ.get('GRASP_TILT_DEG', 40.0))   # below horizontal
GRASP_TILT_W = float(os.environ.get('GRASP_TILT_W', 0.01))       # seat-m per rad err

# Centroid offset: the line joining the two contact points should pass CLOSE to the
# object's centre of mass, else the grip has a moment arm and the object twists out
# (unstable). Reject clearly off-centre contact lines before doing expensive IK,
# then use the remaining perpendicular offset as a soft ranking term.
GRASP_CENTER_W = float(os.environ.get('GRASP_CENTER_W', 1.0))    # cost-m per offset-m
GRASP_CENTER_MAX = float(os.environ.get('GRASP_CENTER_MAX', 0.012))  # hard limit (m)

# Antipodal surface sampling is random and the reachable band is narrow, so the
# best reachable grasp varies run to run. Re-plan up to PLAN_ATTEMPTS times, KEEP
# the lowest-COST result (cost = seat + centroid offset + tilt, so this optimises
# the whole objective, not just seating), stop early once cost <= COST_OK. Makes the
# picked grasp reliably good, not luck-dependent. COST_OK ~ seat 3mm + offset 10mm.
PLAN_ATTEMPTS = int(os.environ.get('GRASP_ATTEMPTS', 4))
COST_OK = float(os.environ.get('GRASP_COST_OK', 0.015))         # m, good-enough cost

# ----------------------------- real hardware (opt-in) -----------------------------
# Enabled only when ONE_REAL=1, so the plain run stays pure-simulation. When on,
# the arm's measured joints seed planning and ENTER replays the pick on the robot.
REAL_ROBOT = bool(os.environ.get("ONE_REAL"))
REAL_ARM_IP = os.environ.get("ONE_ARM_IP", "192.168.1.205")
REAL_HAND_PORT = os.environ.get("ONE_HAND_PORT", "/dev/ttyUSB0")
ARM_MAX_JNTVEL = np.deg2rad(25.0)   # per-joint speed cap for real moves (rad/s)
ARM_CTRL_FREQ = 100.0               # servo-stream rate for the real arm (Hz)
HAND_SPEED = 0.6                    # finger slew speed for real open/close (rad/s)

# Tactile (torque-feedback) grasp: instead of snapping to the planned grasp pose,
# the fingers close gradually and each one FREEZES the moment its joint torque
# crosses CONTACT_TORQUE -- i.e. it stops as soon as it presses the object. The
# close ends once every REQUIRED finger (by grasp type) has made contact.
#   ids per finger = URDF/hardware order thumb0-2, index0-2, middle0-1, ring0-1,
#   pinky0-1. Contact is read on the FLEXION joints only (the swing/abduction
#   joint0 of thumb/index carries preshape torque and would false-trigger).
HAND_FINGER_IDS = {'thumb': (0, 1, 2), 'index': (3, 4, 5),
                   'middle': (6, 7), 'ring': (8, 9), 'pinky': (10, 11)}
HAND_CONTACT_IDS = {'thumb': (1, 2), 'index': (4, 5),
                    'middle': (6, 7), 'ring': (8, 9), 'pinky': (10, 11)}
# Which fingers MUST press the object for the tactile close to be considered
# secured (per opposition primitive; the antipodal pinch opposes thumb<->index).
REQUIRED_FINGERS = {'pinch': ('thumb', 'index'),
                    'tripod': ('thumb', 'index', 'middle')}
CONTACT_TORQUE = 150.0   # 50% of driver tor_max=300; tune cautiously on hardware
HAND_CLOSE_SPEED = 0.35  # finger slew speed while closing to contact (rad/s)
HAND_CTRL_FREQ = 50.0    # feedback-close loop rate (Hz; each cycle is a read move)
# The planned grasp pose is only sim's contact ESTIMATE -- on hardware the finger
# may still be shy of the object there. So the required fingers are allowed to curl
# this much PAST the planned pose (rad, clipped to the joint limit) and torque
# feedback stops them at the real contact. Raise if fingers stall short of objects.
HAND_CLOSE_MARGIN = 0.6


# ============================== arm IK helpers ==============================
def solve_ik(robot, ctx, pos, rot, tcp, ref, collision_free=False,
             max_solutions=8):
    """IK for ``tcp`` at (pos, rot) closest to ``ref``; with ``collision_free``
    skip arm/hand-vs-table-colliding solutions. Full qs or None.

    ``max_solutions`` caps how many converged selik branches are collected before
    the closest-to-``ref`` is picked. A continuation step (seeded by ``ref``, which
    selik sorts its seeds against) only needs the first/closest branch, so pass 1
    there -- collecting all 8 reruns the Jacobian solver up to 8x for no gain."""
    chain = robot.chain(CHAIN)
    ref_active = chain.extract_active_qs(ref)
    best, best_d = None, None
    for s in robot.ik(pos, rot, chain=CHAIN, tcp=tcp,
                      ref_qs=ref_active, max_solutions=max_solutions):
        if collision_free and not ctx.is_state_valid(s.astype(np.float64)):
            continue
        d = float(np.linalg.norm(chain.extract_active_qs(s) - ref_active))
        if best_d is None or d < best_d:
            best, best_d = s.astype(np.float64), d
    return best


def cartesian_path(robot, ctx, tcp, start_q, p0, p1, rot, nstep=12):
    """Straight Cartesian move of ``tcp`` from p0 to p1 at orientation ``rot``,
    seeded continuously. None if any step has no IK or the densified path collides
    (arm/open hand vs table)."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    prev, path = start_q, []
    for t in np.linspace(0, 1, nstep):
        q = solve_ik(robot, ctx, p0 * (1 - t) + p1 * t, rot, tcp, prev,
                     max_solutions=1)   # continuation: closest branch only
        if q is None:
            return None
        path.append(q)
        prev = q
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(1.5))))
        for t in np.linspace(0, 1, n):
            if not ctx.is_state_valid(a + (b - a) * t):
                return None
    return path


# ================================ real hardware ================================
def connect_real_robot():
    """Connect to the real xArm7 + XHand (opt-in via ONE_REAL=1). Returns
    ``(arm, hand)``; ``(None, None)`` when disabled or the connection fails, so
    the caller transparently falls back to a simulation-only run."""
    if not REAL_ROBOT:
        return None, None
    try:
        from one.control.manipulators.xarm7.xarm7 import XArm7X
        from one.control.end_effector.xhand.xhand_x import XHandX
        arm = XArm7X(ip=REAL_ARM_IP)
        hand = XHandX(port=REAL_HAND_PORT)
        print(f"[real] connected: arm {REAL_ARM_IP}, hand {REAL_HAND_PORT}")
        return arm, hand
    except Exception as e:
        print(f"[real] connection failed ({type(e).__name__}: {e}); "
              f"continuing in simulation only")
        return None, None


def sim_to_real_hand(qs):
    """Sim hand qs -> real XHand 12-finger command (radians). The URDF joint
    order (thumb0-2, index0-2, middle0-1, ring0-1, pinky0-1) is exactly the
    hardware finger-id order, so the mapping is the identity on the first 12."""
    return np.asarray(qs, dtype=float)[:12]


def tactile_close(hand_x, start12, target12, required,
                  torque_thresh=CONTACT_TORQUE, speed=HAND_CLOSE_SPEED,
                  freq=HAND_CTRL_FREQ):
    """Close the fingers from ``start12`` toward ``target12``, but stop each finger
    the instant it presses the object: every cycle reads the 12 FingerStates and
    freezes a finger once its flexion-joint torque crosses ``torque_thresh``. The
    close ends when every finger in ``required`` (sim names) has contacted, or all
    fingers reach their planned target. Returns ``{finger: contacted_bool}``.

    Torque units are raw hardware counts, so ``torque_thresh`` must be tuned on the
    real hand -- the live per-cycle print of the required fingers' torque is there
    to calibrate it."""
    start = np.asarray(start12, dtype=float).copy()
    target = np.asarray(target12, dtype=float).copy()
    q = start.copy()
    frozen = np.zeros(12, dtype=bool)
    contacted = {f: False for f in HAND_FINGER_IDS}
    step = max(speed / freq, 1e-9)
    dt = 1.0 / freq
    max_iter = int(np.ceil(float(np.max(np.abs(target - start))) / step)) + 5
    next_t = time.perf_counter()
    last_print = 0.0
    for _ in range(max_iter):
        # advance only un-frozen joints one slew step toward the target
        adv = np.where(frozen, 0.0, np.clip(target - q, -step, step))
        q = q + adv
        states = hand_x.move(q, read=True)
        if states is not None:
            torq = np.array([abs(float(s.torque)) for s in states])
            for f, ids in HAND_CONTACT_IDS.items():
                if not contacted[f] and float(np.max(torq[list(ids)])) >= torque_thresh:
                    contacted[f] = True
                    for i in HAND_FINGER_IDS[f]:
                        frozen[i] = True            # hold this finger; stop pressing harder
                    print(f"[real]   contact: {f} (torque "
                          f"{float(np.max(torq[list(ids)])):.0f})")
            now = time.perf_counter()
            if now - last_print > 0.3:               # live torques to tune the threshold
                rd = {f: int(np.max(torq[list(HAND_CONTACT_IDS[f])])) for f in required}
                print(f"[real]   closing... required torque {rd}")
                last_print = now
            if all(contacted[f] for f in required):
                break
        if np.all(frozen | (np.abs(target - q) <= step)):
            break                                    # everyone frozen or at target
        next_t += dt
        sleep_t = next_t - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            next_t = time.perf_counter()
    return contacted


# ============================== grasp planning ==============================
def pad_seating(jaw, pose_local, jw):
    """Robust contact-patch distance from the pinch pads to the cube surface.

    The cube is axis-aligned at the origin in ``pose_local``. For each pad, score
    the mean of its nearest 10 percent of mesh vertices instead of the single
    nearest vertex. This prevents one pad corner grazing a cube edge from looking
    like a well-seated contact. Absolute signed distance is used so small mesh
    penetration is penalised rather than rewarded.

    antipodal scores a grasp by contact-normal alignment + jaw centering, NOT by
    whether the pads actually reach the object. On a symmetric cube EVERY opposing
    face-pair ties on that score, so the score can't tell a flat, well-seated pinch
    from one cocked over an edge. This metric supplies that missing distinction."""
    half = CUBE_SIZE / 2
    jaw.grip_at(pose_local[:3, 3], pose_local[:3, :3], jw)
    pads = [jaw._spec.thumb_pad] + list(jaw._spec.opp_pads)
    total = 0.0
    for link in pads:
        v = jaw._world_vs(link)                     # pad vertices (cube-local frame)
        a = np.abs(v) - half                        # signed dist to the cube box
        signed_d = (np.linalg.norm(np.maximum(a, 0.0), axis=1)
                    + np.minimum(a.max(1), 0.0))
        surface_d = np.abs(signed_d)
        n_patch = min(len(surface_d), max(3, int(np.ceil(0.1 * len(surface_d)))))
        nearest = np.partition(surface_d, n_patch - 1)[:n_patch]
        total += float(nearest.mean())
    return total


def grasp_centerline_offset(jaw, pose_local, jw):
    """Perpendicular distance from the cube centroid to the pinch contact line."""
    midpoint = np.asarray(pose_local[:3, 3], dtype=np.float64)
    open_dir = np.asarray(jaw.open_dir_at(jw), dtype=np.float64)
    axis = np.asarray(pose_local[:3, :3], dtype=np.float64) @ open_dir
    axis /= np.linalg.norm(axis) + oum.eps
    perpendicular = midpoint - axis * np.dot(midpoint, axis)
    return float(np.linalg.norm(perpendicular))


def centered_grasp_mask(jaw, poses_local, jaw_widths):
    """Vectorised pre-collision filter for contact lines near the cube centre."""
    poses = np.asarray(poses_local, dtype=np.float64)
    open_dirs = np.asarray(jaw.open_dir_at(jaw_widths), dtype=np.float64)
    axes = np.einsum('nij,nj->ni', poses[:, :3, :3], open_dirs)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + oum.eps
    midpoints = poses[:, :3, 3]
    parallel = np.sum(midpoints * axes, axis=1, keepdims=True) * axes
    return np.linalg.norm(midpoints - parallel, axis=1) <= GRASP_CENTER_MAX


def plan_grasps(robot, ctx, hand, cube, home):
    """Plan reachable, collision-free pinch grasps on the cube.

    Grasp generation is delegated to the shared antipodal planner: the XHand is
    bound as a parallel jaw via ``spawn_jaw`` and ``antipodal`` returns grasp
    poses (grasp-center frame, cube-LOCAL) + jaw widths, already filtered for
    hand-vs-cube collision. Here we only add the ARM: map each pose to world,
    solve IK for the grasp-center tcp, and keep grasps whose pre-grasp / descend /
    lift are collision-free vs the table + self.

    Candidates are ranked by a JOINT cost of real PAD SEATING (best contact first)
    and APPROACH TILT (prefer ~GRASP_TILT_DEG below horizontal). Antipodal's own
    score is degenerate on a cube (all opposing-face grasps tie), so it can't tell
    a flat, level pinch from an edge-cocked one; seating can. Tilt is added because
    the XHand pinch only seats level when tilted (thumb/index pad offset ~31 mm
    along approach), and a downward tilt clears the table and is more reachable than
    side-on. Cost = seat_gap + GRASP_TILT_W * |elev - GRASP_TILT_DEG| (rad).

    Returns ``(candidates, stats)`` where each candidate is
    ``(seat, jaw_width, pre_q, descend, retreat, grasp_qs)`` -- ``seat`` is the
    pad-seating gap (m, lower is better), ``grasp_qs`` the full 12-dof finger pose
    at the planned closure, best (jointly) first."""
    jaw = hand.spawn_jaw(GRASP_PRIMITIVE)          # immutable parallel-jaw clone
    grasps = antipodal(
        jaw, cube, **PLAN_KW,
        candidate_filter=lambda poses, widths: centered_grasp_mask(
            jaw, poses, widths))                    # cube-local, central first
    print(f"antipodal: {len(grasps)} collision-free '{GRASP_PRIMITIVE}' grasps "
          f"(jaw range {np.round(np.array(jaw.jaw_range) * 1000, 1)} mm); "
          f"tilt target {GRASP_TILT_DEG:.0f} deg below horizontal")
    tilt_target = np.radians(GRASP_TILT_DEG)

    candidates = []
    diag = []                                       # (cost, seat, elev_deg) for logging
    stats = dict(off_center=0, ik=0, descend=0, retreat=0, ok=0)
    for pose, pre_pose, jw, score in grasps:
        wpose = cube.wd_tf @ pose                   # grasp-center pose in world
        wpre = cube.wd_tf @ pre_pose               # pre-grasp (retreated) pose
        rot = wpose[:3, :3]
        # approach elevation in the WORLD frame (gravity-relative, so correct even
        # for a tilted detected cube): +ve = pointing below horizontal (downward).
        elev = float(-np.arcsin(np.clip(wpose[2, 2], -1.0, 1.0)))
        centroid_off = grasp_centerline_offset(jaw, pose, jw)
        if centroid_off > GRASP_CENTER_MAX:
            stats['off_center'] += 1
            continue
        # grasp-center tcp on the ROBOT's mounted hand (loc_tf is closure-
        # dependent; identical geometry on the jaw clone and the real hand).
        grasp_tcp = orbt.TCP(hand.runtime_root_lnk,
                             jaw._grasp_center_loc_tf(jw))
        hand.open_hand()                            # approach checked open-handed
        pre_q = solve_ik(robot, ctx, wpre[:3, 3].astype(np.float32), rot,
                         grasp_tcp, home, collision_free=True)
        if pre_q is None:
            stats['ik'] += 1; continue
        descend = cartesian_path(robot, ctx, grasp_tcp, pre_q,
                                 wpre[:3, 3], wpose[:3, 3], rot)
        if descend is None:
            stats['descend'] += 1; continue
        retreat = cartesian_path(robot, ctx, grasp_tcp, descend[-1],
                                 wpose[:3, 3], wpose[:3, 3] + UP, rot)
        if retreat is None:
            stats['retreat'] += 1; continue
        seat = pad_seating(jaw, pose, jw)           # real pad-to-cube gap (m)
        grasp_qs = np.asarray(jaw.qs, dtype=float).copy()
        cost = (seat + GRASP_TILT_W * abs(elev - tilt_target)
                + GRASP_CENTER_W * centroid_off)    # joint rank
        stats['ok'] += 1
        candidates.append((cost, float(seat), float(jw), pre_q, descend, retreat,
                           grasp_qs))
        diag.append((cost, seat, np.degrees(elev), centroid_off))
    candidates.sort(key=lambda c: c[0])             # best joint cost first
    if os.environ.get("DEBUG") and diag:
        diag.sort(key=lambda d: d[0])
        print("  reachable grasps (best-first): seat_mm / elev_deg / centroid_off_mm")
        for cost, seat, elevd, coff in diag[:8]:
            print(f"    seat {seat * 1000:5.1f} mm   elev {elevd:+5.1f} deg   "
                  f"centroid_off {coff * 1000:5.1f} mm")
    best_cost = candidates[0][0] if candidates else float("inf")
    # drop the sort-key, keep the (seat, jw, pre_q, descend, retreat, grasp_qs) shape
    candidates = [c[1:] for c in candidates]
    return candidates, stats, best_cost


# ============================== cube pose source ==============================
# FoundationPose writes camera_T_cube here (RealExperiments/foundationpose_then_play.py
# --no_play, run in env_isaaclab). Overridable with ONE_FP_POSE.
FP_POSE_NPY = os.environ.get("ONE_FP_POSE", "/tmp/foundationpose_cube_pose.npy")


def _cube_pose_from_fp(npy_path):
    """Map a FoundationPose ``camera_T_cube`` (4x4, camera frame) into the sim
    world. Applies the fresh D435 eye-to-hand extrinsic (``base_T_cube =
    T_base_cam @ camera_T_cube``), then base->world (``world = ROBOT_BASE_POS +
    p_base``). Returns (world_pos, world_rot). FoundationPose gives the cube's
    FULL 6D orientation -- far more accurate than the point-cloud clustering, which
    is why this is the preferred source."""
    from one.camera.RS435.detect_cube import load_extrinsics
    cam_T_cube = np.load(npy_path).astype(np.float64)
    if cam_T_cube.shape != (4, 4):
        raise ValueError(f"[fp] {npy_path}: expected a 4x4 camera_T_cube, got "
                         f"{cam_T_cube.shape}")
    yaml_path = os.environ.get("ONE_CAM_YAML")
    T_base_cam, _ = (load_extrinsics(yaml_path) if yaml_path else load_extrinsics())
    base_T_cube = T_base_cam @ cam_T_cube
    base_pos = base_T_cube[:3, 3]
    world_pos = (ROBOT_BASE_POS + base_pos).astype(np.float32)
    world_rot = base_T_cube[:3, :3].astype(np.float32)
    age = time.time() - os.path.getmtime(npy_path)
    print(f"[fp] camera_T_cube <- {npy_path} ({age:.0f}s old)")
    print(f"[fp] cube @ base [{base_pos[0]:+.3f} {base_pos[1]:+.3f} "
          f"{base_pos[2]:+.3f}] m  ->  world [{world_pos[0]:+.3f} "
          f"{world_pos[1]:+.3f} {world_pos[2]:+.3f}]")
    if age > 120:
        print("[fp] WARNING: pose file is >2 min old -- re-run "
              "foundationpose_then_play.py --no_play if the cube moved.")
    return world_pos, world_rot


def resolve_cube_pose():
    """The cube pose (world pos, rotmat) to grasp. Three sources, by priority:

    * ONE_FP=1     -> FoundationPose (PREFERRED): read camera_T_cube from
      ONE_FP_POSE (default /tmp/foundationpose_cube_pose.npy, written by
      RealExperiments/foundationpose_then_play.py --no_play in env_isaaclab) and
      map it through the D435 extrinsic. Full 6D pose, highest accuracy.
    * ONE_CAMERA=1 -> in-house point-cloud clustering (detect_cube). Yaw only,
      lower accuracy; fallback when FoundationPose isn't available.
    * neither      -> the hardcoded CUBE_POS / CUBE_ROT (pure sim).

    Detections are BASE-frame; the robot base frame shares the sim-world axes with
    its origin at ROBOT_BASE_POS, so ``world = ROBOT_BASE_POS + p_base`` (base
    table top z=0 -> world z=TABLE_TOP_Z). Raises rather than silently grasping the
    stale hardcoded pose if the requested source yields nothing."""
    if os.environ.get("ONE_FP"):
        if not os.path.exists(FP_POSE_NPY):
            raise RuntimeError(
                f"[fp] pose file not found: {FP_POSE_NPY}. Run (env_isaaclab): "
                "python RealExperiments/foundationpose_then_play.py --no_play")
        return _cube_pose_from_fp(FP_POSE_NPY)
    if not os.environ.get("ONE_CAMERA"):
        return CUBE_POS.copy(), CUBE_ROT.copy()
    from one.camera.RS435.detect_cube import (load_extrinsics, capture_base_cloud,
                                              detect_cube_base)
    yaml_path = os.environ.get("ONE_CAM_YAML")
    T_base_cam, _ = (load_extrinsics(yaml_path) if yaml_path else load_extrinsics())
    print("[camera] capturing D435 cloud to locate the cube ...")
    pts, cols = capture_base_cloud(T_base_cam)
    target = os.environ.get("CUBE_RGB")
    target_rgb = np.array([float(v) for v in target.split(",")]) if target else None
    res = detect_cube_base(pts, cols, cube_size=CUBE_SIZE, target_rgb=target_rgb)
    if res is None:
        raise RuntimeError("[camera] no cube detected on the table (ONE_CAMERA=1); "
                           "check lighting / the workspace box in detect_cube.py")
    center_base, yaw, info = res
    world_pos = (ROBOT_BASE_POS + center_base).astype(np.float32)
    world_rot = oum.rotmat_from_axangle(ouc.StandardAxis.Z, float(yaw))
    print(f"[camera] cube @ base [{center_base[0]:+.3f} {center_base[1]:+.3f} "
          f"{center_base[2]:+.3f}] m  yaw {np.degrees(yaw):+.1f} deg  "
          f"(table_z {info['table_z']:+.3f}, {info['n']} pts)  ->  world "
          f"[{world_pos[0]:+.3f} {world_pos[1]:+.3f} {world_pos[2]:+.3f}]")
    return world_pos, world_rot


# ================================== the demo ==================================
def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    base = ovw.World(cam_pos=(1.6, 0.4, 1.6), cam_lookat_pos=(0.45, -0.1, 0.95))
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
    cube_pos, cube_rot = resolve_cube_pose()     # hardcoded, or D435-detected
    cube = ossop.box(pos=cube_pos, xyz_lengths=(CUBE_SIZE,) * 3, rotmat=cube_rot,
                     rgb=CUBE_RGB, collision_type=ouc.CollisionType.MESH,
                     is_floating=True)

    # ---- robot: xArm7 with the XHand on the flange ----
    robot = XArm7(pos=ROBOT_BASE_POS)
    robot.left_hand = XHandRight()
    mount_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, FLANGE_Z], dtype=np.float32),
        rotmat=oum.rotmat_from_axangle(ouc.StandardAxis.Z, MOUNT_RPY))
    robot.mount(robot.left_hand, robot.runtime_lnks[-1], mount_tf, update=True)
    hand = robot.left_hand

    # ---- real robot (opt-in): its CURRENT joints become the planning start ----
    arm_x, hand_x = connect_real_robot()

    # Drive the arm to the planning start before the collider/ctx are built, so
    # the ACM and the `home` used everywhere below reflect this pose. The start is
    # the real arm's measured joints when connected, else the canonical
    # collision-free HOME. (HOME_DEG is known clear of the table; an arbitrary
    # measured pose may not be -- guarded with a warning once ctx exists.)
    qs_home = robot.qs.astype(np.float64).copy()
    if arm_x is not None:
        q_start = arm_x.get_jnt_values().astype(np.float64)
        print(f"[real] arm start (deg): {np.round(np.degrees(q_start), 1)}")
    else:
        q_start = np.deg2rad(HOME_DEG).astype(np.float64)
    qs_home[robot.chain(CHAIN).active_jnt_ids] = q_start
    robot.fk(qs=qs_home)

    ossop.frame().attach_to(base.scene)
    for e in [robot] + statics + [cube]:
        e.attach_to(base.scene)

    # ---- collider + planner: arm + open hand vs table (approach avoidance) ----
    mjc = ocm.MJCollider()
    for e in [robot] + statics:
        mjc.append(e)
    mjc.actors = [robot]
    mjc.compile(margin=0.0, auto_acm=True)
    ctx = chain_planning_context(robot, mjc, CHAIN)
    planner = ompr.RRTConnectPlanner(pln_ctx=ctx, extend_step_size=np.pi / 36,
                                     goal_bias=0.3)
    if arm_x is not None and not ctx.is_state_valid(robot.qs.astype(np.float64)):
        print("[real] WARNING: the arm's current pose collides in sim; RRT from "
              "it may fail. Jog the real arm clear of the table and rerun.")

    # ---- plan grasps (antipodal) + reachable arm motions ----
    # Re-plan (fresh random sampling) up to PLAN_ATTEMPTS, keeping the lowest-cost
    # result; stop early once within COST_OK. Guards against an unlucky sample where
    # no good grasp lands in the reachable band.
    home = robot.qs.astype(np.float64).copy()
    candidates, stats, best_cost = plan_grasps(robot, ctx, hand, cube, home)
    for attempt in range(2, PLAN_ATTEMPTS + 1):
        if candidates and best_cost <= COST_OK:
            break                                    # already good enough
        more, stats2, more_cost = plan_grasps(robot, ctx, hand, cube, home)
        if more and more_cost < best_cost:
            candidates, stats, best_cost = more, stats2, more_cost   # keep best cost
        print(f"  re-plan {attempt}/{PLAN_ATTEMPTS}: best cost {best_cost * 1000:.1f} mm")
    if os.environ.get("DEBUG"):
        print(f"  reject stats: {stats}")
    if not candidates:
        raise RuntimeError(f"no collision-free '{GRASP_PRIMITIVE}' grasp found")
    seat, jw, pre_q, descend, retreat, grasp_qs = candidates[0]
    traj = plan_segment(planner, home, pre_q)
    traj += descend[1:]
    grasp_idx = len(traj) - 1
    traj += retreat[1:]
    print(f"'{GRASP_PRIMITIVE}': {len(candidates)} reachable grasps; "
          f"best = pad gap {seat * 1000:.1f} mm / jaw {jw * 1000:.1f} mm; "
          f"pick {len(traj)} waypoints (grasp@{grasp_idx})")

    if headless:
        return

    # ---- browse candidates ONE BY ONE ----
    # N / B : next / prev candidate (freeze at its grasp pose, best-first)
    # G / F : play / step the SELECTED candidate's pick;  R: reset it
    # C     : toggle the hand's collision spheres
    # ENTER : run the SELECTED candidate's pick on the real robot (ONE_REAL=1)
    import pyglet.window.key as key
    st = {"sel": 0, "i": 0, "held": False, "playing": False,
          "spheres": False, "traj": None, "gidx": 0}
    hand.open_hand()
    open_hand_qs = np.asarray(hand.qs, dtype=float).copy()   # real-hand "open" target
    hand_lo = np.asarray(hand._compiled.jlmt_low_by_idx, dtype=float)   # finger limits
    hand_hi = np.asarray(hand._compiled.jlmt_high_by_idx, dtype=float)  # (over-close clip)

    def set_spheres(on):
        if on:
            hand.show_collision_spheres(base.scene, alpha=0.35)
        else:
            hand.hide_collision_spheres()
        base.scene.dirty = True

    def show(sel):
        """Freeze at candidate ``sel``'s grasp pose (hand closed on the cube)."""
        st["sel"] = sel % len(candidates)
        seat, jw, pre_q, descend, retreat, gqs = candidates[st["sel"]]
        if st["held"]:
            hand.unmount(cube); st["held"] = False
        robot.fk(qs=descend[-1])
        hand.fk(qs=gqs)
        cube.set_pos_rotmat(pos=cube_pos, rotmat=cube_rot)
        st["i"], st["playing"], st["traj"] = 0, False, None
        if st["spheres"]:
            set_spheres(True)
        print(f"candidate {st['sel'] + 1}/{len(candidates)}: pad gap "
              f"{seat * 1000:.1f} mm, jaw {jw * 1000:.1f} mm")
        base.scene.dirty = True

    def ensure_traj():
        if st["traj"] is None:
            _, _, pre_q, descend, retreat, _ = candidates[st["sel"]]
            t = plan_segment(planner, home, pre_q) + descend[1:]
            st["gidx"] = len(t) - 1
            st["traj"] = t + retreat[1:]
        return st["traj"]

    def reset():
        if st["held"]:
            hand.unmount(cube); st["held"] = False
        hand.open_hand()
        robot.fk(qs=home)
        cube.set_pos_rotmat(pos=cube_pos, rotmat=cube_rot)
        st["i"], st["playing"] = 0, False
        base.scene.dirty = True

    def step():
        traj = ensure_traj()
        gqs = candidates[st["sel"]][5]
        if st["i"] >= len(traj):
            st["playing"] = False
            return
        robot.fk(qs=traj[st["i"]])
        if st["i"] == st["gidx"] and not st["held"]:
            hand.fk(qs=gqs)
            loc = np.linalg.inv(hand.runtime_root_lnk.tf) @ cube.tf
            hand.mount(cube, hand.runtime_root_lnk, loc)
            st["held"] = True
        if st["spheres"]:
            set_spheres(True)
        st["i"] += 1
        base.scene.dirty = True

    def execute_real():
        """Stream the SELECTED candidate's planned pick to the real robot: open
        the hand, sync the arm to the path start, approach+descend, close the
        grasp at the grasp waypoint, then lift. No-op (with a note) when the
        hardware isn't connected."""
        if arm_x is None:
            print("[real] not connected -- set ONE_REAL=1 (and check the IP / "
                  "port) to run on hardware")
            return
        sel = st["sel"]
        traj = ensure_traj()
        gidx = st["gidx"]
        gqs = candidates[sel][5]
        chain = robot.chain(CHAIN)
        arm_path = [chain.extract_active_qs(np.asarray(q, np.float64)) for q in traj]
        print(f"[real] candidate {sel + 1}/{len(candidates)}: streaming "
              f"{len(arm_path)} waypoints (grasp@{gidx}) ...")
        try:
            if hand_x is not None:
                hand_x.move_to(sim_to_real_hand(open_hand_qs), speed=HAND_SPEED)
            arm_x.move_j(arm_path[0], speed=ARM_MAX_JNTVEL, wait=True)  # sync start
            arm_x.stream_jnt_path(arm_path[:gidx + 1], control_freq=ARM_CTRL_FREQ,
                                  max_jntvel=ARM_MAX_JNTVEL)             # descend
            if hand_x is not None:
                # tactile grasp in TWO phases so the thumb reliably OPPOSES first.
                # The thumb's opposition is a preshape SWING (thumb_joint0), not a
                # flexion -- but open_hand() zeros every joint, so the thumb starts
                # flat. If that swing runs inside the feedback close (which reads
                # torque on the flexion joints), the swing's own load false-triggers
                # "contact" and freezes the thumb half-swung -- the thumb never
                # closes. So:
                #   1) PRESHAPE (plain position move, no feedback): drive every
                #      non-flexion joint to the planned pose -- crucially the thumb
                #      swing into opposition -- with only the flexion (contact)
                #      joints still open.
                #   2) tactile close: curl ONLY the flexion joints until each
                #      required finger presses. They may curl PAST the planned pose
                #      (sim's contact estimate) so a real gap still closes; torque
                #      stops them.
                required = REQUIRED_FINGERS[GRASP_PRIMITIVE]
                open12 = sim_to_real_hand(open_hand_qs)
                target12 = sim_to_real_hand(gqs).copy()
                contact_ids = sorted({i for f in required
                                      for i in HAND_CONTACT_IDS[f]})
                preshape12 = target12.copy()
                for i in contact_ids:
                    preshape12[i] = open12[i]        # open ONLY the flexion joints
                for f in required:
                    for i in HAND_CONTACT_IDS[f]:
                        if target12[i] >= open12[i]:           # closing = curl up
                            target12[i] = min(target12[i] + HAND_CLOSE_MARGIN, hand_hi[i])
                        else:
                            target12[i] = max(target12[i] - HAND_CLOSE_MARGIN, hand_lo[i])
                # 1) swing the thumb into opposition + preshape the rest (blocks)
                hand_x.move_to(preshape12, speed=HAND_SPEED)
                # 2) force-close the flexion joints from the seated preshape
                contacted = tactile_close(hand_x, preshape12, target12, required)
                miss = [f for f in required if not contacted[f]]
                if miss:
                    print(f"[real] WARNING: no contact on {miss}; closed to the "
                          f"planned pose anyway -- grasp may be loose")
                else:
                    print(f"[real] grasp secured (contact on {list(required)})")
            arm_x.stream_jnt_path(arm_path[gidx:], control_freq=ARM_CTRL_FREQ,
                                  max_jntvel=ARM_MAX_JNTVEL)             # lift
            print("[real] done.")
        except Exception as e:
            print(f"[real] execution error: {type(e).__name__}: {e}")
            try:
                arm_x.clean_error()
            except Exception:
                pass

    def tick(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(key.ENTER):
            execute_real(); return
        if im.is_key_pressed_edge(key.N):
            show(st["sel"] + 1); return
        if im.is_key_pressed_edge(key.B):
            show(st["sel"] - 1); return
        if im.is_key_pressed_edge(key.R):
            reset(); return
        if im.is_key_pressed_edge(key.C):
            st["spheres"] = not st["spheres"]; set_spheres(st["spheres"])
        if im.is_key_pressed_edge(key.G):
            st["playing"] = not st["playing"]
        if im.is_key_pressed_edge(key.F):
            st["playing"] = False; step()
        if st["playing"]:
            step()

    show(0)
    print(f"{len(candidates)} candidates.  N/B: next/prev candidate   "
          f"G: play   F: step   R: reset   C: collision spheres" +
          ("   ENTER: run on robot" if arm_x is not None else ""))
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
