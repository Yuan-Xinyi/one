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

Keys: F step  G play/pause  R replay  C collision spheres on/off
      ENTER preview real path / confirm execution
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
and used as the planning start / IK seed (instead of HOME_DEG). First ENTER
previews a freshly planned path; second ENTER streams that path to the real
xArm7 + XHand. The grasp itself is
torque-feedback closed: the fingers close gradually and each freezes after
baseline-corrected torque, minimum travel and position lag confirm contact, so
contact -- not just the planned pose -- decides when to lift. IP / port / speeds are
the ONE_ARM_IP / ONE_HAND_PORT env vars and the REAL_* / *_TORQUE constants below.
Driver code: one/control (xarm7.XArm7X, xhand_x.XHandX).

Set ONE_DIAG_GRASP=1 together with ONE_REAL=1 for a safe calibration run: ENTER
moves to the grasp and preshapes the hand, then stops before tactile close/lift,
reads all real joints back into simulation and reports index-to-table clearance.
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
import one.motion.probabilistic.rrt as ompr                    # noqa: E402
import one.robots.base.tcp as orbt                             # noqa: E402
from one.grasp.antipodal import antipodal                      # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
from planning_utils import chain_planning_context, plan_segment  # noqa: E402
from scene import (                                            # noqa: E402
    CUBE_POS, CUBE_ROT, CUBE_SIZE, PLANNING_TABLE_TOP_Z,
    ROBOT_BASE_POS, TABLE_Z_BASE, build_collider, build_scene)


# =============================== configuration ===============================
CHAIN = 'main'                       # xArm7 arm chain (7-DOF, numerical IK)

# xArm7 home (J1..J7, degrees). The all-zeros default has the arm/hand resting
# in collision with the table, so RRT from it never connects (it burns all
# max_iters, ~28 s, freezing the viewer on first play). This reachable, clear-of-
# table config is the planning start AND the IK seed for the grasp search.
HOME_DEG = np.array([-16.9, -34.8, 18.8, 20.5, 86.9, 12.0, -79.8],
                    dtype=np.float32)
# Keep the base-yaw joint in the forward hemisphere. The rear workspace contains
# unmodelled equipment/cables, so collision geometry alone is not sufficient.
ARM_J1_SAFE_MIN_DEG = float(os.environ.get('ARM_J1_SAFE_MIN_DEG', -90.0))
ARM_J1_SAFE_MAX_DEG = float(os.environ.get('ARM_J1_SAFE_MAX_DEG', 90.0))

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
               max_grasps=48, clearance=0.003)

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
GRASP_TILT_FILTER_DEG = float(os.environ.get('GRASP_TILT_FILTER_DEG', 22.0))

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
PREGRASP_IK_SOLUTIONS = int(os.environ.get('PREGRASP_IK_SOLUTIONS', 1))
REACHABLE_GRASP_TARGET = int(os.environ.get('REACHABLE_GRASP_TARGET', 4))

# ----------------------------- real hardware (opt-in) -----------------------------
# Enabled only when ONE_REAL=1, so the plain run stays pure-simulation. When on,
# the arm's measured joints seed planning and ENTER replays the pick on the robot.
REAL_ROBOT = bool(os.environ.get("ONE_REAL"))
REAL_ARM_IP = os.environ.get("ONE_ARM_IP", "192.168.1.205")
REAL_HAND_PORT = os.environ.get("ONE_HAND_PORT", "/dev/ttyUSB0")
REAL_DIAG_GRASP = bool(os.environ.get('ONE_DIAG_GRASP'))
REAL_DIAG_THUMB_CLEARANCE_MM = float(
    os.environ.get('ONE_DIAG_THUMB_CLEARANCE_MM', 5.0))
REAL_PREVIEW_START_TOL_DEG = float(
    os.environ.get('REAL_PREVIEW_START_TOL_DEG', 0.5))
ARM_MAX_JNTVEL = np.deg2rad(25.0)   # per-joint speed cap for real moves (rad/s)
ARM_CTRL_FREQ = 100.0               # servo-stream rate for the real arm (Hz)
HAND_SPEED = 0.6                    # finger slew speed for real open/close (rad/s)

# Tactile (torque-feedback) grasp: instead of snapping to the planned grasp pose,
# the fingers close gradually and each one freezes after baseline-corrected torque
# crosses CONTACT_TORQUE with enough joint travel and tracking lag. The close
# ends once every REQUIRED finger (by grasp type) has made confirmed contact.
#   ids per finger = URDF/hardware order thumb0-2, index0-2, middle0-1, ring0-1,
#   pinky0-1. Contact is read on the PROXIMAL flexion joint only:
#   - joint0 (thumb/index swing/abduction) carries preshape torque -> false-trigger.
#   - the thumb DISTAL joint2 does not track its own command on the real hand
#     (measured flex2 ~= 0.18 * flex1 regardless of the flex2 target -- it is
#     mechanically coupled / under-actuated). Judged for contact it registers a
#     huge fake lag + strain torque at ~14 deg and freezes the thumb BEFORE it
#     reaches the object. So the thumb contacts on joint1 (which tracks) alone;
#     joint2 still curls (coupled) and is still frozen with the finger on contact.
HAND_FINGER_IDS = {'thumb': (0, 1, 2), 'index': (3, 4, 5),
                   'middle': (6, 7), 'ring': (8, 9), 'pinky': (10, 11)}
HAND_CONTACT_IDS = {'thumb': (1,), 'index': (4, 5),
                    'middle': (6, 7), 'ring': (8, 9), 'pinky': (10, 11)}
# Which fingers MUST press the object for the tactile close to be considered
# secured (per opposition primitive; the antipodal pinch opposes thumb<->index).
REQUIRED_FINGERS = {'pinch': ('thumb', 'index'),
                    'tripod': ('thumb', 'index', 'middle')}
CONTACT_TORQUE = 200.0   # 67% of driver tor_max=300; tune cautiously on hardware
HAND_CLOSE_SPEED = 0.35  # finger slew speed while closing to contact (rad/s)
HAND_CTRL_FREQ = 50.0    # feedback-close loop rate (Hz; each cycle is a read move)
CONTACT_MIN_TRAVEL = float(os.environ.get('CONTACT_MIN_TRAVEL', 0.20))
CONTACT_POSITION_ERROR = float(os.environ.get('CONTACT_POSITION_ERROR', 0.04))
CONTACT_CONFIRM_CYCLES = int(os.environ.get('CONTACT_CONFIRM_CYCLES', 2))
CONTACT_BASELINE_SAMPLES = int(os.environ.get('CONTACT_BASELINE_SAMPLES', 3))
# Contact is judged by the torque RISE above each joint's free-closing level, not
# an absolute threshold. A constant preload / a zero-fallback baseline / a joint
# stiff from the first cycle then no longer reads as contact (the thumb froze at
# 14.8 deg because its ~200 resting torque, minus a failed zero baseline, cleared
# CONTACT_TORQUE immediately); only a genuine climb while pressing the object does.
CONTACT_TORQUE_RISE = float(os.environ.get('CONTACT_TORQUE_RISE', 120.0))
CONTACT_WARMUP_CYCLES = int(os.environ.get('CONTACT_WARMUP_CYCLES', 3))
HAND_REPLY_SETTLE = float(os.environ.get('HAND_REPLY_SETTLE', 0.10))
# Required pinch fingers target their full closing-side joint limits. Torque
# feedback is the only normal stop before that limit, so missed contact feedback
# can drive a finger all the way closed.


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
    """Close fingers from ``start12`` toward ``target12`` and stop on confirmed
    object contact. Every cycle reads the 12 FingerStates; the close ends when all
    ``required`` fingers have contacted or all joints reach their targets. Returns
    ``{finger: contacted_bool}``.

    Contact uses the change from a preshape torque baseline, a minimum closing
    travel and consecutive threshold hits. This avoids freezing the thumb on its
    normal position-hold preload before it has visibly curled.

    Torque units are raw hardware counts, so ``torque_thresh`` must be tuned on
    the real hand. Live logs report the baseline-corrected value."""
    start = np.asarray(start12, dtype=float).copy()
    target = np.asarray(target12, dtype=float).copy()
    q = start.copy()
    frozen = np.zeros(12, dtype=bool)
    contacted = {f: False for f in HAND_FINGER_IDS}
    hit_counts = {f: 0 for f in required}
    step = max(speed / freq, 1e-9)
    dt = 1.0 / freq
    max_iter = int(np.ceil(float(np.max(np.abs(target - start))) / step)) + 5
    next_t = time.perf_counter()
    last_print = 0.0
    last_states = None
    cycle = 0                              # closing-loop cycle counter (warmup)
    torque_ref = np.full(12, np.inf)       # per-joint lowest free-closing torque

    # move_to() ends with an unread fire-and-forget reply. Let that full frame
    # arrive before flushing; an immediate flush can catch it halfway and leave a
    # trailing partial frame, which caused the observed CRC mismatch and a false
    # zero torque baseline. Then collect several valid preload samples.
    time.sleep(HAND_REPLY_SETTLE)
    if getattr(hand_x, 'ser', None) is not None:
        hand_x.ser.reset_input_buffer()
    baseline_samples = []
    baseline_position_samples = []
    required_samples = max(1, CONTACT_BASELINE_SAMPLES)
    max_attempts = required_samples + 5
    contact_ids = sorted({i for f in required for i in HAND_CONTACT_IDS[f]})
    for attempt in range(1, max_attempts + 1):
        states = hand_x.move(start, read=True)
        row = ([float(s.torque) for s in states[:12]]
               if states is not None and len(states) >= 12 else None)
        # An all-zero torque row on the contact joints is almost always a CRC-
        # dropped frame (a real XHand at rest still reports non-zero preload); it
        # must be rejected, else the finger's resting bias reads as contact -- the
        # exact zero-baseline failure that froze the thumb early.
        if row is not None and any(abs(row[i]) > 1e-6 for i in contact_ids):
            baseline_samples.append(row)
            baseline_position_samples.append(
                [float(s.position) for s in states[:12]])
            last_states = states
            if len(baseline_samples) >= required_samples:
                break
        else:
            print(f'[real]   torque baseline read {attempt}/{max_attempts} '
                  'failed or all-zero; resynchronizing serial input')
            time.sleep(HAND_REPLY_SETTLE)
            if getattr(hand_x, 'ser', None) is not None:
                hand_x.ser.reset_input_buffer()
        time.sleep(dt)
    if len(baseline_samples) < required_samples:
        raise RuntimeError(
            f'failed to read a valid XHand torque baseline '
            f'({len(baseline_samples)}/{required_samples} samples); '
            'aborting before tactile close')
    # FingerState.torque is a uint16 wire value, so later subtraction is
    # performed modulo 2**16.
    torque_bias = np.median(np.asarray(baseline_samples), axis=0)
    print('[real]   torque baseline thumb/index: '
          f'{np.rint(torque_bias[:6]).astype(int).tolist()}')
    if baseline_position_samples:
        baseline_position = np.median(
            np.asarray(baseline_position_samples), axis=0)
        print('[real]   thumb preshape rad: cmd '
              f'{np.round(start[:3], 3).tolist()} / measured '
              f'{np.round(baseline_position[:3], 3).tolist()}')

    for _ in range(max_iter):
        # advance only un-frozen joints one slew step toward the target
        adv = np.where(frozen, 0.0, np.clip(target - q, -step, step))
        q = q + adv
        states = hand_x.move(q, read=True)
        if states is not None:
            last_states = states
            raw_torque = np.array([float(s.torque) for s in states[:12]])
            measured_q = np.array([float(s.position) for s in states[:12]])
            signed_delta = ((raw_torque - torque_bias + 32768.0) % 65536.0
                            - 32768.0)
            torq = np.abs(signed_delta)
            cycle += 1
            # Track each still-moving joint's lowest torque so far (after a short
            # warmup); contact is judged by the RISE above that free-closing level,
            # so a constant preload or a joint stiff from the start does not trip.
            if cycle > CONTACT_WARMUP_CYCLES:
                torque_ref = np.where(~frozen, np.minimum(torque_ref, torq),
                                      torque_ref)
            for f in required:
                ids = HAND_CONTACT_IDS[f]
                travel = float(np.max(np.abs(q[list(ids)] - start[list(ids)])))
                ref = np.where(np.isfinite(torque_ref[list(ids)]),
                               torque_ref[list(ids)], 0.0)
                rise = float(np.max(torq[list(ids)] - ref))
                pos_error = float(np.max(np.abs(
                    q[list(ids)] - measured_q[list(ids)])))
                if (travel >= CONTACT_MIN_TRAVEL
                        and rise >= CONTACT_TORQUE_RISE
                        and pos_error >= CONTACT_POSITION_ERROR):
                    hit_counts[f] += 1
                else:
                    hit_counts[f] = 0
                if not contacted[f] and hit_counts[f] >= CONTACT_CONFIRM_CYCLES:
                    contacted[f] = True
                    for i in HAND_FINGER_IDS[f]:
                        frozen[i] = True            # hold this finger; stop pressing harder
                    print(f"[real]   contact: {f} (torque rise {rise:.0f}, "
                          f"travel {np.degrees(travel):.1f} deg, lag "
                          f"{np.degrees(pos_error):.1f} deg)")
            now = time.perf_counter()
            if now - last_print > 0.3:               # live values to tune the rise threshold
                rd = {
                    f: (int(np.max(torq[list(HAND_CONTACT_IDS[f])]
                                   - np.where(np.isfinite(torque_ref[list(HAND_CONTACT_IDS[f])]),
                                              torque_ref[list(HAND_CONTACT_IDS[f])], 0.0))),
                        round(float(np.degrees(np.max(np.abs(
                            q[list(HAND_CONTACT_IDS[f])]
                            - start[list(HAND_CONTACT_IDS[f])])))), 1),
                        round(float(np.degrees(np.max(np.abs(
                            q[list(HAND_CONTACT_IDS[f])]
                            - measured_q[list(HAND_CONTACT_IDS[f])])))), 1))
                    for f in required}
                print('[real]   closing... torque_rise / travel_deg / lag_deg '
                      f'{rd}')
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
    if last_states is not None:
        measured = np.array([float(s.position) for s in last_states[:12]])
        print('[real]   thumb close rad [swing, flex1, flex2]: start '
              f'{np.round(start[:3], 3).tolist()} -> cmd '
              f'{np.round(q[:3], 3).tolist()} / measured '
              f'{np.round(measured[:3], 3).tolist()} -> limit target '
              f'{np.round(target[:3], 3).tolist()}')
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


def planning_grasp_mask(jaw, cube, poses_local, jaw_widths):
    """Cheap centre/tilt filter applied before hand FK and mesh collision."""
    poses = np.asarray(poses_local, dtype=np.float64)
    open_dirs = np.asarray(jaw.open_dir_at(jaw_widths), dtype=np.float64)
    axes = np.einsum('nij,nj->ni', poses[:, :3, :3], open_dirs)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + oum.eps
    midpoints = poses[:, :3, 3]
    parallel = np.sum(midpoints * axes, axis=1, keepdims=True) * axes
    centered = np.linalg.norm(midpoints - parallel, axis=1) <= GRASP_CENTER_MAX

    cube_rot = np.asarray(cube.wd_tf[:3, :3], dtype=np.float64)
    world_rots = np.einsum('ij,njk->nik', cube_rot, poses[:, :3, :3])
    elevations = -np.arcsin(np.clip(world_rots[:, 2, 2], -1.0, 1.0))
    tilt_ok = (np.abs(elevations - np.radians(GRASP_TILT_DEG))
               <= np.radians(GRASP_TILT_FILTER_DEG))
    return centered & tilt_ok


def plan_grasps(robot, grasp_ctx, transit_ctx, hand, cube, home):
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
        candidate_filter=lambda poses, widths: planning_grasp_mask(
            jaw, cube, poses, widths))              # cheap filter before FK/collision
    print(f"antipodal: {len(grasps)} collision-free '{GRASP_PRIMITIVE}' grasps "
          f"(jaw range {np.round(np.array(jaw.jaw_range) * 1000, 1)} mm); "
          f"tilt target {GRASP_TILT_DEG:.0f} deg below horizontal")
    tilt_target = np.radians(GRASP_TILT_DEG)
    hand.open_hand()
    open_hand_qs = np.asarray(hand.qs, dtype=float).copy()

    def set_collision_hand(qs):
        """Keep the mounted hand's MuJoCo joints in sync with its planned pose."""
        hand.fk(qs=np.asarray(qs, dtype=float))
        for planning_ctx in (grasp_ctx, transit_ctx):
            planning_ctx.collider.set_mecba_qpos(hand, hand.qs)
            planning_ctx.clear_cache()

    candidates = []
    diag = []                                       # (cost, seat, elev_deg) for logging
    stats = dict(off_center=0, ik=0, descend=0, closed_collision=0,
                 retreat=0, ok=0)
    # Antipodal's score ties on a cube. Rank cheaply by centroid/tilt first so
    # expensive arm IK starts with task-relevant candidates, then stop once there
    # are enough reachable grasps for selection and fallback.
    ranked_grasps = []
    for grasp in grasps:
        pose, _pre_pose, jw, _score = grasp
        wpose = cube.wd_tf @ pose
        elev = float(-np.arcsin(np.clip(wpose[2, 2], -1.0, 1.0)))
        centroid_off = grasp_centerline_offset(jaw, pose, jw)
        preliminary_cost = (GRASP_TILT_W * abs(elev - tilt_target)
                            + GRASP_CENTER_W * centroid_off)
        ranked_grasps.append((preliminary_cost, grasp))
    ranked_grasps.sort(key=lambda item: item[0])

    for _preliminary_cost, (pose, pre_pose, jw, score) in ranked_grasps:
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
        set_collision_hand(open_hand_qs)            # approach checked open-handed
        pre_q = solve_ik(robot, transit_ctx, wpre[:3, 3].astype(np.float32), rot,
                         grasp_tcp, home, collision_free=True,
                         max_solutions=PREGRASP_IK_SOLUTIONS)
        if pre_q is None:
            stats['ik'] += 1; continue
        descend = cartesian_path(robot, grasp_ctx, grasp_tcp, pre_q,
                                 wpre[:3, 3], wpose[:3, 3], rot)
        if descend is None:
            stats['descend'] += 1; continue
        seat = pad_seating(jaw, pose, jw)           # also sets the jaw closure
        grasp_qs = np.asarray(jaw.qs, dtype=float).copy()
        # The hand is a mounted child, not an arm-planning actor. Explicitly push
        # its closed joints into MuJoCo before checking the grasp and lift; merely
        # calling hand.fk() does not update the collider's child qpos.
        set_collision_hand(grasp_qs)
        if not grasp_ctx.is_state_valid(descend[-1]):
            stats['closed_collision'] += 1
            set_collision_hand(open_hand_qs)
            continue
        # The nominal planned closure is collision-checked for lift. Real tactile
        # execution may close farther until torque contact or a joint limit.
        retreat = cartesian_path(robot, grasp_ctx, grasp_tcp, descend[-1],
                                 wpose[:3, 3], wpose[:3, 3] + UP, rot)
        if retreat is None:
            stats['retreat'] += 1
            set_collision_hand(open_hand_qs)
            continue
        set_collision_hand(open_hand_qs)
        cost = (seat + GRASP_TILT_W * abs(elev - tilt_target)
                + GRASP_CENTER_W * centroid_off)    # joint rank
        stats['ok'] += 1
        candidates.append((cost, float(seat), float(jw), pre_q, descend, retreat,
                           grasp_qs))
        diag.append((cost, seat, np.degrees(elev), centroid_off))
        if len(candidates) >= REACHABLE_GRASP_TARGET:
            break
    candidates.sort(key=lambda c: c[0])             # best joint cost first
    set_collision_hand(open_hand_qs)
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
FP_SNAP_TO_TABLE = os.environ.get('ONE_FP_SNAP_Z', '1') != '0'
FP_SNAP_UPRIGHT = os.environ.get('ONE_FP_UPRIGHT', '1') != '0'


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
    base_pos = base_T_cube[:3, 3].copy()
    base_rot = base_T_cube[:3, :3].copy()
    raw_z = float(base_pos[2])
    if FP_SNAP_UPRIGHT:
        # A cube resting face-down on a flat table cannot carry arbitrary
        # FoundationPose roll/pitch. Cube symmetry makes those components prone
        # to frame flips/noise, and retaining them while fixing centre height can
        # put a rotated corner through the table. Preserve only tabletop yaw.
        yaw = float(np.arctan2(base_rot[1, 0], base_rot[0, 0]))
        base_rot = oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw).astype(np.float64)
    if FP_SNAP_TO_TABLE:
        # This demo picks a known 60 mm cube resting on the corrected tabletop.
        # Keep x/y and full orientation, but enforce non-penetrating contact with
        # that surface in the robot-base frame.
        # Support radius of a rotated box along base Z. With upright projection
        # this is exactly 30 mm; the general expression also prevents penetration
        # if ONE_FP_UPRIGHT=0 while table snapping remains enabled.
        support_z = 0.5 * CUBE_SIZE * float(np.sum(np.abs(base_rot[2, :])))
        base_pos[2] = TABLE_Z_BASE + support_z
    world_pos = (ROBOT_BASE_POS + base_pos).astype(np.float32)
    world_rot = base_rot.astype(np.float32)
    age = time.time() - os.path.getmtime(npy_path)
    print(f"[fp] camera_T_cube <- {npy_path} ({age:.0f}s old)")
    print(f"[fp] cube @ base [{base_pos[0]:+.3f} {base_pos[1]:+.3f} "
          f"{base_pos[2]:+.3f}] m  ->  world [{world_pos[0]:+.3f} "
          f"{world_pos[1]:+.3f} {world_pos[2]:+.3f}]")
    if FP_SNAP_TO_TABLE and abs(raw_z - base_pos[2]) > 0.001:
        print(f"[fp] table constraint: center z {raw_z:+.3f} -> "
              f"{base_pos[2]:+.3f} m (set ONE_FP_SNAP_Z=0 to disable)")
    if FP_SNAP_UPRIGHT:
        print("[fp] tabletop constraint: roll/pitch -> 0 "
              "(set ONE_FP_UPRIGHT=0 to disable)")
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
    its origin at ROBOT_BASE_POS, so ``world = ROBOT_BASE_POS + p_base``. Because
    the base sits on an 8 mm plate, table top is z=-0.008 in base coordinates and
    maps to the corrected tabletop height in world. Raises rather than silently
    grasping the stale hardcoded pose if the requested source yields nothing."""
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

    # Scene geometry lives in scene.py. Anything added to its static-object list
    # is rendered and included in collision planning.
    cube_pos, cube_rot = resolve_cube_pose()     # hardcoded, or D435-detected
    robot, statics, cube = build_scene(cube_pos, cube_rot)
    hand = robot.left_hand

    # ---- real robot (opt-in): its CURRENT joints become the planning start ----
    arm_x, hand_x = connect_real_robot()

    # Set the simulated arm to the planning start before collision contexts are
    # built, so the ACM and the `home` used below reflect this pose. The start is
    # the real arm's measured joints when connected, else the canonical
    # collision-free HOME. (HOME_DEG is known clear of the table; an arbitrary
    # measured pose may not be -- guarded once the contexts exist.)
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

    # Two collision views share the exact rendered scene. Free-space transit also
    # treats the cube as an obstacle; grasp motion omits it to allow pad contact.
    grasp_mjc = build_collider(robot, statics)
    transit_mjc = build_collider(robot, statics, target=cube)
    j1_safe = np.radians([ARM_J1_SAFE_MIN_DEG, ARM_J1_SAFE_MAX_DEG])
    grasp_ctx = chain_planning_context(
        robot, grasp_mjc, CHAIN,
        joint_limit_overrides={'joint1': tuple(j1_safe)})
    transit_ctx = chain_planning_context(
        robot, transit_mjc, CHAIN,
        joint_limit_overrides={'joint1': tuple(j1_safe)})
    print(f"[plan] joint1 forward-only range: [{ARM_J1_SAFE_MIN_DEG:+.1f}, "
          f"{ARM_J1_SAFE_MAX_DEG:+.1f}] deg")
    planner = ompr.RRTConnectPlanner(pln_ctx=transit_ctx,
                                     extend_step_size=np.pi / 36,
                                     goal_bias=0.3)
    if (arm_x is not None
            and not transit_ctx.is_state_valid(robot.qs.astype(np.float64))):
        raise RuntimeError(
            "[real] current arm pose collides with the scene/cube or is outside "
            "the forward-only "
            f"joint1 range [{ARM_J1_SAFE_MIN_DEG}, {ARM_J1_SAFE_MAX_DEG}] deg. "
            "Jog it into the safe front region and rerun.")

    # ---- plan grasps (antipodal) + reachable arm motions ----
    # Re-plan (fresh random sampling) up to PLAN_ATTEMPTS, keeping the lowest-cost
    # result; stop early once within COST_OK. Guards against an unlucky sample where
    # no good grasp lands in the reachable band.
    home = robot.qs.astype(np.float64).copy()
    candidates, stats, best_cost = plan_grasps(
        robot, grasp_ctx, transit_ctx, hand, cube, home)
    for attempt in range(2, PLAN_ATTEMPTS + 1):
        if candidates and best_cost <= COST_OK:
            break                                    # already good enough
        more, stats2, more_cost = plan_grasps(
            robot, grasp_ctx, transit_ctx, hand, cube, home)
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
    # ENTER : first preview, then confirm the real pick (ONE_REAL=1)
    import pyglet.window.key as key
    st = {"sel": 0, "i": 0, "held": False, "playing": False,
          "spheres": False, "traj": None, "gidx": 0,
          "real_preview": None}
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
        st["real_preview"] = None
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
        st["real_preview"] = None
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
        """First ENTER previews a freshly planned real path; second executes it."""
        if arm_x is None:
            print("[real] not connected -- set ONE_REAL=1 (and check the IP / "
                  "port) to run on hardware")
            return
        if (st["real_preview"] is not None
                and st["i"] < len(st["traj"])):
            print('[real] simulation preview is not finished; let it play to '
                  'the end, then press ENTER again')
            return
        sel = st["sel"]
        chain = robot.chain(CHAIN)
        try:
            # The viewer may currently show a closed candidate. Free-space
            # planning and real execution start open-handed, so synchronize both
            # collision models before validating the freshly measured arm pose.
            hand.fk(qs=open_hand_qs)
            for planning_ctx in (grasp_ctx, transit_ctx):
                planning_ctx.collider.set_mecba_qpos(hand, hand.qs)
                planning_ctx.clear_cache()

            current_arm = np.asarray(arm_x.get_jnt_values(), dtype=np.float64)
            current_qs = home.copy()
            current_qs[chain.active_jnt_ids] = current_arm
            if not transit_ctx.is_state_valid(current_qs):
                print('[real] ERROR: current pose with the open hand collides with '
                      'the wall/table/cube or is outside the safe joint range; '
                      'not moving. Jog to a safe pose and retry.')
                return

            preview = st["real_preview"]
            if preview is None:
                _seat, _jw, pre_q, descend, retreat, gqs = candidates[sel]
                print(f"[real] candidate {sel + 1}/{len(candidates)}: planning "
                      "simulation preview from current joints ...")
                pre_q = np.asarray(pre_q, dtype=np.float64)
                if transit_ctx.states_equal(current_qs, pre_q):
                    approach = [pre_q]
                    print('[real] current pose already matches the pre-grasp')
                else:
                    approach = plan_segment(planner, current_qs, pre_q)
                    print(f'[real] collision-free current -> pre-grasp: '
                          f'{len(approach)} waypoints')

                # Build the path from the current measured state, never via the
                # old startup/home waypoint.
                real_traj = approach + list(descend[1:])
                gidx = len(real_traj) - 1
                real_traj += list(retreat[1:])
                arm_path = [
                    chain.extract_active_qs(np.asarray(q, np.float64))
                    for q in real_traj]
                st["real_preview"] = {
                    "sel": sel,
                    "start_qs": current_qs.copy(),
                    "arm_path": arm_path,
                    "gidx": gidx,
                    "gqs": np.asarray(gqs, dtype=np.float64).copy(),
                }

                # Play exactly this freshly planned real path in simulation.
                if st["held"]:
                    hand.unmount(cube)
                    st["held"] = False
                cube.set_pos_rotmat(pos=cube_pos, rotmat=cube_rot)
                robot.fk(qs=current_qs)
                hand.fk(qs=open_hand_qs)
                st["traj"] = real_traj
                st["gidx"] = gidx
                st["i"] = 0
                st["playing"] = True
                base.scene.dirty = True
                print(f'[real] simulation preview: {len(arm_path)} waypoints '
                      f'(grasp@{gidx}); wait for playback, then press ENTER again '
                      'to execute')
                return

            if preview["sel"] != sel:
                st["real_preview"] = None
                print('[real] candidate changed; preview cleared. Press ENTER to '
                      'plan the new candidate')
                return

            start_qs = preview["start_qs"]
            start_arm = chain.extract_active_qs(start_qs)
            start_err_deg = float(np.max(np.abs(np.degrees(
                current_arm - start_arm))))
            if (start_err_deg > REAL_PREVIEW_START_TOL_DEG
                    or not transit_ctx.is_motion_valid(current_qs, start_qs)):
                st["real_preview"] = None
                print(f'[real] current joints changed after preview '
                      f'(max {start_err_deg:.2f} deg); execution cancelled. '
                      'Press ENTER to replan and preview again')
                return

            arm_path = preview["arm_path"]
            gidx = preview["gidx"]
            gqs = preview["gqs"]
            st["real_preview"] = None
            print(f'[real] preview confirmed: executing {len(arm_path)} waypoints '
                  f'(grasp@{gidx}) ...')

            if hand_x is not None:
                hand_x.move_to(sim_to_real_hand(open_hand_qs), speed=HAND_SPEED)
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
                            target12[i] = hand_hi[i]
                        else:
                            target12[i] = hand_lo[i]
                # 1) swing the thumb into opposition + preshape the rest (blocks)
                hand_x.move_to(preshape12, speed=HAND_SPEED)
                if REAL_DIAG_GRASP:
                    time.sleep(0.3)  # let the final position command settle
                    # move_to() streams fire-and-forget packets, so its final
                    # unread reply can leave RX framing between packets. Flush it
                    # before the first parsed read and retry transient serial
                    # timeouts/partial frames without changing the safe preshape.
                    if getattr(hand_x, 'ser', None) is not None:
                        hand_x.ser.reset_input_buffer()
                    preshape_states = None
                    for read_attempt in range(1, 6):
                        states = hand_x.move(preshape12, read=True)
                        if states is not None and len(states) >= 12:
                            preshape_states = states
                            break
                        print(f'[diag] XHand state read {read_attempt}/5 failed; retrying')
                        if getattr(hand_x, 'ser', None) is not None:
                            hand_x.ser.reset_input_buffer()
                        time.sleep(0.1)
                    if not preshape_states or len(preshape_states) < 12:
                        print('[diag] ERROR: failed to read 12 XHand joint states; '
                              'stopping before close. Check serial replies/timeout.')
                        return

                    measured_arm = np.asarray(arm_x.get_jnt_values(), dtype=float)
                    measured_hand = np.array(
                        [float(s.position) for s in preshape_states[:12]], dtype=float)

                    def fingertip_clearances(arm_active, hand_qs):
                        full_qs = np.asarray(robot.qs, dtype=float).copy()
                        full_qs[chain.active_jnt_ids] = arm_active
                        robot.fk(qs=full_qs)
                        hand.fk(qs=hand_qs)
                        links = {'thumb': 'thumb_rota_link2',
                                 'index': 'index_rota_link2'}
                        return {
                            name: float(hand._world_vs(link)[:, 2].min()
                                        - PLANNING_TABLE_TOP_Z)
                            for name, link in links.items()
                        }

                    planned_arm = np.asarray(arm_path[gidx], dtype=float)
                    planned_clear = fingertip_clearances(planned_arm, preshape12)
                    arm_only_clear = fingertip_clearances(measured_arm, preshape12)
                    measured_clear = fingertip_clearances(measured_arm, measured_hand)
                    arm_err = np.degrees(measured_arm - planned_arm)
                    hand_err = np.degrees(measured_hand - preshape12)
                    print('[diag] stopped at grasp pose BEFORE tactile close/lift')
                    for finger in ('thumb', 'index'):
                        print(f'[diag] {finger}-table clearance: planned '
                              f'{planned_clear[finger] * 1000:+.1f} mm')
                        print(f'[diag] {finger}-table clearance: measured arm + '
                              f'planned hand {arm_only_clear[finger] * 1000:+.1f} mm')
                        print(f'[diag] {finger}-table clearance: measured arm + '
                              f'measured hand {measured_clear[finger] * 1000:+.1f} mm')
                    thumb_sim_mm = measured_clear['thumb'] * 1000
                    print(f'[diag] thumb clearance comparison: simulation-backprojected '
                          f'{thumb_sim_mm:+.1f} mm vs measured '
                          f'{REAL_DIAG_THUMB_CLEARANCE_MM:+.1f} mm; sim-real error '
                          f'{thumb_sim_mm - REAL_DIAG_THUMB_CLEARANCE_MM:+.1f} mm')
                    print(f'[diag] arm tracking error deg: '
                          f'{np.round(arm_err, 3).tolist()}')
                    print(f'[diag] hand tracking/zero error deg: '
                          f'{np.round(hand_err, 3).tolist()}')
                    print('[diag] viewer now shows the measured-joint back-projection; '
                          'robot remains at the low pose -- jog/reset it manually')
                    base.scene.dirty = True
                    return
                # 2) force-close the flexion joints from the seated preshape
                contacted = tactile_close(hand_x, preshape12, target12, required)
                miss = [f for f in required if not contacted[f]]
                if miss:
                    print(f"[real] WARNING: no contact on {miss}; closed to the "
                          f"joint-limit target -- grasp may be loose")
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
          (("   ENTER: preview / diagnostic pose" if REAL_DIAG_GRASP else
            "   ENTER: preview / confirm execution")
           if arm_x is not None else ""))
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
