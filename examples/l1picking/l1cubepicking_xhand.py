"""Grasp-pose search for an xArm7 + XHand: given an object, for each of the three
hand grasps (pinch / tripod / power) find a reachable, COLLISION-FREE grasp whose
opposing fingers actually touch the object.

Quality judgement is deliberately simple (no caging / force-closure scoring):
  * GOOD grasp  = the grasp's opposing fingers REALLY touch the object, AND the
    hand + arm do not collide with anything else (table / self).
  * fingers that can't reach the object just don't count -- a grasp whose thumb
    never touches is not a grasp.

Collision uses the XHand's sphere model (xhand_right_withcc): per-link spheres
give fast sphere-vs-object (cube box) and sphere-vs-table tests, plus the hand's
own thumb-tip self-collision check.

Candidates are sampled from SEVERAL approach directions (top-down + the four
sides), so e.g. pinch can oppose the cube from the side instead of only from
straight above.

Switch grasp:  GRASP_PRIMITIVE=pinch|tripod|power  (env var or below).
Keys: F step  G play/pause  R replay  C collision spheres on/off
Headless (plan only): ONE_HEADLESS=1
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
import one.robots.base.tcp as orbt                             # noqa: E402
from one.robots.manipulators.xarm.xarm7.xarm7 import XArm7     # noqa: E402
from one.robots.end_effectors.xhand.xhand_right_withcc import XHandRight  # noqa: E402
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
APPROACH_H = 0.12                    # straight-line approach distance to the cube
N_ROLL = 6                           # rolls about each approach axis
SPHERE_MARGIN = 0.002                # sphere surface within this of cube = touch
GRASP_PRIMITIVE = os.environ.get('GRASP_PRIMITIVE', 'pinch')   # pinch/tripod/power

# Approach directions: the unit vector the hand travels ALONG toward the cube
# (i.e. the grasp-frame +z). A straight-down approach holds the palm horizontal,
# so the open thumb hangs straight down and dips under the table before it can
# close; letting the hand LEAN (tilt the approach axis off vertical) raises the
# thumb side and keeps the open hand clear of the table. We sample straight-down
# plus several tilts leaning toward a ring of azimuths. (Pure sideways approaches
# are unreachable for this arm, so they're not included.)
TILT_ANGLES = (0.0, np.deg2rad(20), np.deg2rad(40), np.deg2rad(60))
N_AZIMUTH = 4                                     # azimuths each non-zero tilt leans toward


def _approach_dirs():
    """Straight-down (-z) plus tilted variants: each tilt leans the approach axis
    off vertical toward N_AZIMUTH evenly-spaced compass directions."""
    dirs = {'top': (0.0, 0.0, -1.0)}
    for tilt in TILT_ANGLES:
        if tilt == 0.0:
            continue
        for k in range(N_AZIMUTH):
            az = 2 * np.pi * k / N_AZIMUTH
            d = (np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az),
                 -np.cos(tilt))
            dirs[f't{int(round(np.degrees(tilt)))}_a{int(round(np.degrees(az)))}'] = d
    return dirs


APPROACH_DIRS = _approach_dirs()

# Per-grasp setup: palm tcp aimed at the cube, hand-local offset into the
# opposition region, the fingers that MUST touch, and the min total touching
# fingers for a valid grasp.
GRASP_PARAMS = {
    'pinch':  dict(tcp='pinch_center', offset=(0.0, 0.0, 0.0),
                   required=('thumb', 'index'), min_total=2),
    'tripod': dict(tcp='pinch_center', offset=(0.0, 0.0, 0.0),
                   required=('thumb', 'index', 'middle'), min_total=3),
    'power':  dict(tcp='power_center', offset=(0.03, 0.0, 0.0),
                   required=('thumb',), min_total=2),
}

FINGER_JOINTS = {
    'thumb':  ('thumb_joint0', 'thumb_joint1', 'thumb_joint2'),
    'index':  ('index_joint1', 'index_joint2'),
    'middle': ('middle_joint0', 'middle_joint1'),
    'ring':   ('ring_joint0', 'ring_joint1'),
    'pinky':  ('pinky_joint0', 'pinky_joint1'),
}
FINGER_LINKS = {
    'thumb':  ('thumb_bend_link', 'thumb_rota_link1', 'thumb_rota_link2'),
    'index':  ('index_bend_link', 'index_rota_link1', 'index_rota_link2'),
    'middle': ('mid_link1', 'mid_link2'),
    'ring':   ('ring_link1', 'ring_link2'),
    'pinky':  ('pinky_link1', 'pinky_link2'),
}


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


def approach_rot(a, roll):
    """Grasp-frame rotation whose +z is the approach direction ``a`` (unit, hand
    travels toward the cube along it), rolled by ``roll`` about that axis."""
    a = np.asarray(a, np.float64); a = a / np.linalg.norm(a)
    ref = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0., 1, 0])
    x = np.cross(ref, a); x /= np.linalg.norm(x)
    y = np.cross(a, x)
    R0 = np.stack([x, y, a], axis=1)
    return (oum.rotmat_from_axangle(a.astype(np.float32), float(roll)) @ R0)


# ============================== sphere collision ==============================
def sphere_finger_map(hand):
    """Which finger each collision sphere belongs to (None = palm/other)."""
    chk = hand.collision_checker
    link_of = [chk.link_order[int(i)] for i in np.asarray(chk.sphere_link_indices)]
    finger_of = {ln: f for f, lns in FINGER_LINKS.items() for ln in lns}
    return np.array([finger_of.get(ln) for ln in link_of], dtype=object)


def sphere_link_map(hand):
    """Which link each collision sphere belongs to (parallel to sphere_finger_map,
    but keeps the link name so pad-tip spheres can be singled out)."""
    chk = hand.collision_checker
    return np.array([chk.link_order[int(i)] for i in np.asarray(chk.sphere_link_indices)],
                    dtype=object)


def required_pad_links(spec):
    """{finger: pad_link} from a grasp spec's 'pads' = (thumb_tip, [opposing_tips]);
    empty for pads=None (e.g. power -> no designated tips, falls back to any-sphere)."""
    pads = spec.get('pads')
    if not pads:
        return {}
    finger_of = {ln: f for f, lns in FINGER_LINKS.items() for ln in lns}
    thumb_link, opp = pads
    out = {'thumb': thumb_link}
    for ln in opp:
        out[finger_of[ln]] = ln
    return out


def pad_touches(hand, cube, sphere_link, pad_links, margin=SPHERE_MARGIN):
    """True if any of the given pad-tip links' spheres touch the cube."""
    centers, radii = hand_spheres(hand)
    touch = sphere_box_gap(centers, radii, cube.wd_tf, CUBE_SIZE / 2) < margin
    pad_set = set(pad_links)
    mask = np.array([ln in pad_set for ln in sphere_link], dtype=bool)
    return bool(np.any(touch & mask))

def hand_spheres(hand):
    """World-frame collision-sphere centres (n,3) and radii (n,) at the hand's
    current pose (light: just the FK, no self-collision pass)."""
    chk = hand.collision_checker
    local = np.asarray(chk.update(np.asarray(hand.qs, dtype=float)))
    root = hand.runtime_root_lnk.tf
    return local @ root[:3, :3].T + root[:3, 3], np.asarray(chk.sphere_radii)


def sphere_box_gap(centers, radii, box_tf, half):
    """Surface gap of each sphere to an axis-aligned box of half-extent ``half``
    at pose ``box_tf`` -- negative means the sphere intersects the box."""
    inv = np.linalg.inv(box_tf).astype(np.float64)
    loc = centers @ inv[:3, :3].T + inv[:3, 3]
    q = np.abs(loc) - half
    d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, 1), 0.0)
    return d - radii


def cube_touches(hand, cube, sphere_finger, margin=SPHERE_MARGIN):
    """Which fingers' spheres touch the cube. Returns {finger: n_touching_spheres}."""
    centers, radii = hand_spheres(hand)
    touch = sphere_box_gap(centers, radii, cube.wd_tf, CUBE_SIZE / 2) < margin
    out = {f: 0 for f in FINGER_JOINTS}
    for f in FINGER_JOINTS:
        out[f] = int(np.sum(touch & (sphere_finger == f)))
    return out


def finger_below_table(hand, sphere_finger, finger):
    """True if any of ``finger``'s spheres dips below the tabletop."""
    centers, radii = hand_spheres(hand)
    m = sphere_finger == finger
    return bool(np.any(centers[m][:, 2] - radii[m] < TABLE_TOP_Z))


def hand_env_collision(hand, sphere_finger):
    """The CLOSED hand colliding with the environment: any sphere below the
    tabletop, or the hand's own thumb-tip self-collision."""
    centers, radii, self_hit = hand.collision_sphere_world()
    centers, radii = np.asarray(centers), np.asarray(radii)
    if np.any(centers[:, 2] - radii < TABLE_TOP_Z):
        return True
    return bool(np.asarray(self_hit).any())


def wrap_grasp_qs(hand, cube, sphere_finger, nstep=16):
    """Single-basis eigengrasp close of GRASP_PRIMITIVE on the cube: apply the
    fixed preshape (full, not amplitude-scaled), then drive EVERY closing finger
    together along the one 'closing' direction by a shared amplitude a in [0,1].
    Each finger freezes at the amplitude where its contact target first touches
    the cube -- the pad tip for pad-defined grasps, any sphere otherwise -- or one
    step before it would dip below the table. The shared sweep stops once all
    required pads have touched. Returns the full finger qs."""
    spec = hand._GRASP_TABLE[GRASP_PRIMITIVE]
    preshape, closing = spec['preshape'], spec['closing']
    pad_of = required_pad_links(spec)              # {finger: pad_link} or {}
    sphere_link = sphere_link_map(hand)

    qidx = {j.name: i for i, j in enumerate(hand.structure.jnts)}
    qs = np.zeros(hand._compiled.n_jnts, dtype=np.float32)
    for j, v in preshape.items():                  # preshape applied in full
        qs[qidx[j]] = v

    active = {f: [j for j in js if j in closing]    # fingers this grasp drives
              for f, js in FINGER_JOINTS.items()}
    active = {f: cj for f, cj in active.items() if cj}
    frozen = {f: None for f in active}             # amplitude each finger stopped at
    prev_a = {f: 0.0 for f in active}              # last collision-free amplitude
    reason = {f: None for f in active}             # why each finger froze (diagnostic)

    def contacts(f):
        if f in pad_of:                            # pad-tip spheres only
            return pad_touches(hand, cube, sphere_link, (pad_of[f],))
        return cube_touches(hand, cube, sphere_finger)[f] > 0   # any sphere

    for a in np.linspace(0.0, 1.0, nstep):
        for f, cj in active.items():               # advance only un-frozen fingers
            af = frozen[f] if frozen[f] is not None else a
            for j in cj:
                qs[qidx[j]] = closing[j] * af
        hand.fk(qs=qs)
        for f in active:
            if frozen[f] is not None:
                continue
            if finger_below_table(hand, sphere_finger, f):
                frozen[f] = prev_a[f]; reason[f] = 'below_table'   # one step before the table
            elif contacts(f):
                frozen[f] = a; reason[f] = 'contact'               # just touching -> freeze here
            else:
                prev_a[f] = a
        if pad_of and all(frozen[f] is not None for f in pad_of):
            break                                  # all required pads touched
        if all(frozen[f] is not None for f in active):
            break

    if os.environ.get("DEBUG"):                    # diagnostic: amplitude + reason per finger
        print("  freeze:", {f: (round((frozen[f] or 0.0), 2), reason[f]) for f in active})

    for f, cj in active.items():                   # settle to frozen amplitudes
        af = frozen[f] if frozen[f] is not None else 1.0
        for j in cj:
            qs[qidx[j]] = closing[j] * af
    hand.fk(qs=qs)
    return qs


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
    cube = ossop.box(pos=CUBE_POS, xyz_lengths=(CUBE_SIZE,) * 3, rotmat=CUBE_ROT,
                     rgb=CUBE_RGB, collision_type=ouc.CollisionType.MESH,
                     is_free=True)

    # ---- robot: xArm7 with the XHand on the flange ----
    robot = XArm7(pos=ROBOT_BASE_POS)
    robot.left_hand = XHandRight()
    mount_tf = oum.tf_from_pos_rotmat(
        pos=np.array([0.0, 0.0, FLANGE_Z], dtype=np.float32),
        rotmat=oum.rotmat_from_axangle(ouc.StandardAxis.Z, MOUNT_RPY))
    robot.mount(robot.left_hand, robot.runtime_lnks[-1], mount_tf, update=True)
    hand = robot.left_hand
    sphere_finger = sphere_finger_map(hand)

    # Drive the arm to the collision-free HOME before the collider/ctx are built,
    # so the ACM and the `home` used everywhere below reflect this valid pose.
    qs_home = robot.qs.astype(np.float64).copy()
    qs_home[robot.chain(CHAIN).active_jnt_ids] = np.deg2rad(HOME_DEG)
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

    # ---- search GRASP_PRIMITIVE over approach dirs x rolls ----
    gp = GRASP_PARAMS[GRASP_PRIMITIVE]
    grasp_loc = hand.tcp(gp['tcp']).loc_tf.copy()
    grasp_loc[:3, 3] = grasp_loc[:3, 3] + np.asarray(gp['offset'], np.float32)
    palm_tcp = orbt.TCP(hand.runtime_root_lnk, grasp_loc)
    home = robot.qs.astype(np.float64).copy()
    candidates = []
    stats = dict(ik=0, descend=0, required=0, few=0, env=0, retreat=0, ok=0)
    for dname, a in APPROACH_DIRS.items():
        a = np.asarray(a, np.float64)
        pre_pos = (CUBE_POS - a * APPROACH_H).astype(np.float32)
        for roll in np.linspace(0.0, 2 * np.pi, N_ROLL, endpoint=False):
            hand.open_hand()
            rot = approach_rot(a, roll)
            pre_q = solve_ik(robot, ctx, pre_pos, rot, palm_tcp, home,
                             collision_free=True)
            if pre_q is None:
                stats['ik'] += 1; continue
            descend = cartesian_path(robot, ctx, palm_tcp, pre_q, pre_pos,
                                     CUBE_POS, rot)
            if descend is None:
                stats['descend'] += 1; continue
            robot.fk(qs=descend[-1])
            gqs = wrap_grasp_qs(hand, cube, sphere_finger)
            touch = cube_touches(hand, cube, sphere_finger)
            touched = [f for f in FINGER_JOINTS if touch[f] > 0]
            if not all(touch[f] > 0 for f in gp['required']):
                stats['required'] += 1; continue   # opposing fingers don't touch
            if len(touched) < gp['min_total']:
                stats['few'] += 1; continue         # too few fingers touch
            if hand_env_collision(hand, sphere_finger):
                stats['env'] += 1; continue         # closed hand hits table/self
            retreat = cartesian_path(robot, ctx, palm_tcp, descend[-1],
                                     CUBE_POS, CUBE_POS + UP, rot)
            if retreat is None:
                stats['retreat'] += 1; continue
            stats['ok'] += 1
            n_spheres = sum(touch.values())
            candidates.append((len(touched), n_spheres, pre_q, descend, retreat,
                               gqs, dname, float(roll)))
    if os.environ.get("DEBUG"):
        print(f"  reject stats: {stats}")
    if not candidates:
        raise RuntimeError(f"no collision-free '{GRASP_PRIMITIVE}' grasp found")
    # prefer more touching fingers, then more contact spheres (firmer grip)
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    nf, ns, pre_q, descend, retreat, grasp_qs, dname, roll = candidates[0]
    traj = plan_segment(planner, home, pre_q)
    traj += descend[1:]
    grasp_idx = len(traj) - 1
    traj += retreat[1:]
    print(f"'{GRASP_PRIMITIVE}': {len(candidates)} collision-free grasps; "
          f"best = {nf} fingers / {ns} contact spheres via '{dname}' approach "
          f"(roll {np.degrees(roll):.0f} deg); pick {len(traj)} waypoints "
          f"(grasp@{grasp_idx})")

    if headless:
        return

    # ---- browse candidates ONE BY ONE ----
    # N / B : next / prev candidate (freeze at its grasp pose, best-first)
    # G / F : play / step the SELECTED candidate's pick;  R: reset it
    # C     : toggle the hand's collision spheres
    import pyglet.window.key as key
    st = {"sel": 0, "i": 0, "held": False, "playing": False,
          "spheres": False, "traj": None, "gidx": 0}

    def set_spheres(on):
        if on:
            hand.show_collision_spheres(base.scene, alpha=0.35)
        else:
            hand.hide_collision_spheres()
        base.scene.dirty = True

    def show(sel):
        """Freeze at candidate ``sel``'s grasp pose (hand closed on the cube)."""
        st["sel"] = sel % len(candidates)
        nf, ns, pre_q, descend, retreat, gqs, dname, roll = candidates[st["sel"]]
        if st["held"]:
            hand.unmount(cube); st["held"] = False
        robot.fk(qs=descend[-1])
        hand.fk(qs=gqs)
        cube.set_pos_rotmat(pos=CUBE_POS, rotmat=CUBE_ROT)
        st["i"], st["playing"], st["traj"] = 0, False, None
        if st["spheres"]:
            set_spheres(True)
        print(f"candidate {st['sel'] + 1}/{len(candidates)}: {nf} fingers / "
              f"{ns} spheres, '{dname}' approach, roll {np.degrees(roll):.0f} deg")
        base.scene.dirty = True

    def ensure_traj():
        if st["traj"] is None:
            _, _, pre_q, descend, retreat, _, _, _ = candidates[st["sel"]]
            t = plan_segment(planner, home, pre_q) + descend[1:]
            st["gidx"] = len(t) - 1
            st["traj"] = t + retreat[1:]
        return st["traj"]

    def reset():
        if st["held"]:
            hand.unmount(cube); st["held"] = False
        hand.open_hand()
        robot.fk(qs=home)
        cube.set_pos_rotmat(pos=CUBE_POS, rotmat=CUBE_ROT)
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

    def tick(dt):
        im = base.input_manager
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
          f"G: play   F: step   R: reset   C: collision spheres")
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
