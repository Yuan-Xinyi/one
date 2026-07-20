"""Grasp planning for an arbitrary mesh object (default ``textured_mesh.obj``)
with the XHand pinch, on the xArm7.

Same planner and ranking as :mod:`grasp_cube` (antipodal generation -> pad seating
+ approach-tilt + centroid-proximity + arm reachability), generalised from the
cube box to a triangle mesh:

  * pad seating   -> nearest-surface distance via a KD-tree over the mesh vertices
  * centroid line -> perpendicular distance from the contact line to the mesh
                     centre of mass (not the origin)
  * centre cap    -> relaxed (a legitimate handle grasp on a big tool is offset
                     from the COM), still soft-ranked

The full flow mirrors :mod:`grasp_cube`: FoundationPose locates the object -> plan
-> browse -> real execution (ENTER previews, ENTER again streams the pick with a
torque-feedback tactile close). Low-level arm/hand helpers and all tuning constants
are reused from :mod:`grasp_cube`.

Object pose source (resolve_tool_pose):
  * ONE_FP=1 -> FoundationPose real pose (camera_T_object from ONE_FP_POSE, default
    /tmp/foundationpose_tool_pose.npy, mapped via the D435 extrinsic). No fake stand.
  * else     -> a hardcoded upright pose lifted onto a stand (pure-sim demo; this
    object's graspable feature is too low to reach flat on the table).

Run:
    conda activate one
    python grasp_tool.py                       # pure sim: plan + browse
    ONE_HEADLESS=1 python grasp_tool.py         # plan only (no window)
    ONE_FP=1 python grasp_tool.py               # FoundationPose real pose, sim preview
    ONE_FP=1 ONE_REAL=1 python grasp_tool.py    # + real xArm7 + XHand (ENTER to run)
    TOOL_MESH=/path/to/other.obj python grasp_tool.py
Keys: N/B next/prev   G/F play/step   R reset   C spheres   ENTER preview/execute(real)
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import trimesh

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for _p in (_PROJECT_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import one.utils.constant as ouc                               # noqa: E402
import one.utils.math as oum                                   # noqa: E402
import one.scene.scene_object as osso                          # noqa: E402
import one.scene.scene_object_primitive as ossop              # noqa: E402
import one.collider.mj_collider as ocm                         # noqa: E402
import one.motion.probabilistic.rrt as ompr                    # noqa: E402
import one.robots.base.tcp as orbt                             # noqa: E402
import one.viewer.world as ovw                                 # noqa: E402
from one.grasp.antipodal import antipodal                      # noqa: E402

from planning_utils import chain_planning_context, plan_segment  # noqa: E402
from scene import (build_robot, build_static_objects,          # noqa: E402
                   ROBOT_BASE_POS, PLANNING_TABLE_TOP_Z, COLLISION_MARGIN)
import grasp_cube as gc                                        # noqa: E402


# =============================== configuration ===============================
TOOL_MESH = os.environ.get('TOOL_MESH', os.path.join(_THIS, 'textured_mesh.obj'))
TOOL_FORWARD = float(os.environ.get('TOOL_FORWARD', 0.35))   # +x from robot base
TOOL_YAW_DEG = float(os.environ.get('TOOL_YAW_DEG', 0.0))    # about world z
TOOL_RGB = (0.72, 0.60, 0.32)
# This object's graspable feature sits only 2-7 cm above its base, so resting it
# flat on the table leaves NO room for the (large) XHand -- every grasp hits the
# table (verified: 0 reachable on the table vs 50-60 reachable when raised ~15 cm).
# So stand it on a pedestal of this height; a matching support box is added to the
# scene as a real collision obstacle. Set TOOL_LIFT=0 to place it flat on the table.
TOOL_LIFT = float(os.environ.get('TOOL_LIFT', 0.15))
SUPPORT_RGB = (0.50, 0.50, 0.55)

# Centroid-proximity cap: how far the contact line may sit from the object's COM.
# Default is effectively OFF (1 m) so ANY part of the tool is eligible -- grab
# wherever is reachable, not only near the centre. The COM proximity is still a
# SOFT ranking term (gc.GRASP_CENTER_W), so a centred/stable grasp is preferred
# when several are reachable, but an off-centre one (e.g. a handle) is still taken
# if it is the only reachable grasp. Set GRASP_CENTER_MAX to a small value to
# re-restrict to near-centre grasps.
CENTER_MAX = float(os.environ.get('GRASP_CENTER_MAX', 1.0))
# Approach-tilt is a cube-tuned HARD prune (40 +/- 22 deg) that rejects most of an
# arbitrary object's REACHABLE grasps (they sit at other tilts). Keep it only as a
# wide gate here (the tilt PREFERENCE still ranks softly via gc.GRASP_TILT_W).
TILT_FILTER_DEG = float(os.environ.get('TOOL_TILT_FILTER_DEG', 85.0))

# The raw scan is ~60k faces; antipodal's surface ray-cast and per-grasp
# gripper-vs-mesh collision are both O(faces), so decimate the planning/collision
# mesh hard (3k keeps a 60 mm graspable feature). Rendering uses the same mesh.
TOOL_MAX_FACES = int(os.environ.get('TOOL_MAX_FACES', 3000))

# Sample the whole surface so grasps are found on EVERY part of the tool (handle,
# neck, body, tip), not just a few spots. Denser + more grasps = better coverage
# of "grab anywhere", at some planning cost.
PLAN_KW = dict(density=0.005, normal_tol_deg=25, roll_step_deg=45,
               max_grasps=80, clearance=0.003)
# Re-sampling a big mesh is expensive; default to a single pass.
ATTEMPTS = int(os.environ.get('GRASP_ATTEMPTS', 1))

# FoundationPose writes camera_T_object here (same convention as grasp_cube's
# ONE_FP, but for THIS object's model). Overridable with ONE_FP_POSE.
FP_POSE_NPY = os.environ.get('ONE_FP_POSE', '/tmp/foundationpose_tool_pose.npy')


# ============================== mesh target ==============================
def _tool_stl_path():
    """Return a decimated STL for the target (the ``one`` loader reads STL, and STL
    registers as a MuJoCo collision asset). Built once and cached next to the source
    mesh; keeps at most ``TOOL_MAX_FACES`` faces so collision checks stay fast."""
    stl = os.path.splitext(TOOL_MESH)[0] + f'_plan{TOOL_MAX_FACES}.stl'
    if (not os.path.exists(stl)
            or os.path.getmtime(stl) < os.path.getmtime(TOOL_MESH)):
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(TOOL_MESH)
        if len(m.triangles) > TOOL_MAX_FACES:
            m = m.simplify_quadric_decimation(TOOL_MAX_FACES)
        m.remove_degenerate_triangles()
        m.remove_duplicated_vertices()
        m.compute_vertex_normals()      # STL writer needs normals
        m.compute_triangle_normals()
        o3d.io.write_triangle_mesh(stl, m)
        print(f"[tool] decimated {os.path.basename(TOOL_MESH)} -> "
              f"{len(m.triangles)} faces ({os.path.basename(stl)})")
    return stl


_TREE = None      # KD-tree over the object-LOCAL mesh vertices (seating queries)
_COM = None       # object-local centre of mass (centroid-proximity ranking)
_Z_BOTTOM = None  # object-local min z (to rest the mesh on the table)
_BOUNDS = None    # object-local (min, max) bbox (for the support pedestal)
_VERTS = None     # object-local vertices (for snap-to-table)

# FoundationPose depth noise can place the object slightly INTO the table. Snap it
# to REST on the tabletop (corrects that depth error). The object resting on the
# table is then excluded from collision (build_collider) so the resting contact
# does not invalidate every arm config -- that persistent tool-vs-table collision
# was the bug that hid ALL real grasps. Set ONE_FP_SNAP_Z=0 to disable snapping.
FP_SNAP_TO_TABLE = os.environ.get('ONE_FP_SNAP_Z', '1') != '0'
TABLE_CLEARANCE = float(os.environ.get('TABLE_CLEARANCE', 0.001))


def _mesh_ref():
    global _TREE, _COM, _Z_BOTTOM, _BOUNDS, _VERTS
    if _TREE is None:
        m = trimesh.load(_tool_stl_path(), force='mesh')
        verts = np.asarray(m.vertices, dtype=np.float64)
        _VERTS = verts
        _TREE = cKDTree(verts)
        # center_mass is NaN for a non-watertight scan; fall back to the vertex
        # centroid (a fine stability reference for the centroid-proximity term).
        com = np.asarray(m.center_mass, dtype=np.float64)
        _COM = com if np.all(np.isfinite(com)) else verts.mean(axis=0)
        _Z_BOTTOM = float(verts[:, 2].min())
        _BOUNDS = (verts.min(axis=0), verts.max(axis=0))
    return _TREE, _COM, _Z_BOTTOM


def _snap_to_table(pos, rot):
    """Lift ``pos`` so the object's lowest vertex clears the tabletop by
    TABLE_CLEARANCE (never lower it). Fixes the FP-into-table persistent collision."""
    _mesh_ref()
    world_z_min = float((_VERTS @ np.asarray(rot, np.float64).T)[:, 2].min() + pos[2])
    target = PLANNING_TABLE_TOP_Z + TABLE_CLEARANCE
    if world_z_min < target:
        pos = np.asarray(pos, np.float32).copy()
        pos[2] += np.float32(target - world_z_min)
    return pos


def build_support(tool_pos, tool_rot):
    """A pedestal box under the raised object: top at the object's base, standing
    on the table. Real collision obstacle so the arm avoids it. None if TOOL_LIFT=0."""
    if TOOL_LIFT <= 1e-6:
        return None
    _mesh_ref()
    # A THIN central post, not a full-footprint pedestal: the graspable feature is
    # near the object's base, so a wide stand under it re-blocks the hand exactly
    # like the table did. A narrow post under the centroid holds the object while
    # leaving the feature's sides clear for the pinch.
    foot = float(os.environ.get('SUPPORT_FOOT', 0.05))
    top_z = float(tool_pos[2]) + _Z_BOTTOM          # object base height
    return ossop.box(
        pos=(float(tool_pos[0]), float(tool_pos[1]),
             PLANNING_TABLE_TOP_Z + 0.5 * (top_z - PLANNING_TABLE_TOP_Z)),
        xyz_lengths=(foot, foot, max(1e-3, top_z - PLANNING_TABLE_TOP_Z)),
        rgb=SUPPORT_RGB, collision_type=ouc.CollisionType.AABB)


def tool_pose():
    """World (pos, rotmat) placing the mesh upright, standing on a TOOL_LIFT stand
    (or flat on the table when TOOL_LIFT=0)."""
    _, _, z_bottom = _mesh_ref()
    pos = np.array([ROBOT_BASE_POS[0] + TOOL_FORWARD, ROBOT_BASE_POS[1],
                    PLANNING_TABLE_TOP_Z - z_bottom + TOOL_LIFT], dtype=np.float32)
    rot = oum.rotmat_from_axangle(ouc.StandardAxis.Z, np.radians(TOOL_YAW_DEG))
    return pos, rot


def build_tool(pos, rot):
    obj = osso.SceneObject.from_file(
        _tool_stl_path(), collision_type=ouc.CollisionType.MESH,
        is_floating=True, rgb=TOOL_RGB)
    obj.set_pos_rotmat(pos=np.asarray(pos, np.float32),
                       rotmat=np.asarray(rot, np.float32))
    return obj


def _tool_pose_from_fp(npy_path):
    """Map a FoundationPose ``camera_T_object`` (4x4, camera frame) into the sim
    world, same as grasp_cube's ONE_FP path: ``base_T_obj = T_base_cam @
    camera_T_obj``, then ``world = ROBOT_BASE_POS + p_base``. Full 6D pose, no
    snapping (the object can rest in any real orientation)."""
    from one.camera.RS435.detect_cube import load_extrinsics
    cam_T = np.load(npy_path).astype(np.float64)
    if cam_T.shape != (4, 4):
        raise ValueError(f"[fp] {npy_path}: expected a 4x4 camera_T_object, "
                         f"got {cam_T.shape}")
    yaml_path = os.environ.get('ONE_CAM_YAML')
    T_base_cam, _ = (load_extrinsics(yaml_path) if yaml_path else load_extrinsics())
    base_T = T_base_cam @ cam_T
    world_pos = (ROBOT_BASE_POS + base_T[:3, 3]).astype(np.float32)
    world_rot = base_T[:3, :3].astype(np.float32)
    age = time.time() - os.path.getmtime(npy_path)
    print(f"[fp] camera_T_object <- {npy_path} ({age:.0f}s old)  ->  world "
          f"[{world_pos[0]:+.3f} {world_pos[1]:+.3f} {world_pos[2]:+.3f}]")
    if age > 120:
        print("[fp] WARNING: pose file is >2 min old -- re-detect if the object moved.")
    return world_pos, world_rot


def resolve_tool_pose():
    """Object pose (world_pos, world_rot, add_stand) to grasp, by priority:

    * ONE_FP=1 -> FoundationPose real pose (full 6D, from ONE_FP_POSE), no
      artificial stand -- the real object is held however it actually is.
    * else     -> the hardcoded upright pose lifted onto a stand (pure-sim demo;
      this object's graspable feature is too low to reach flat on the table).
    """
    if os.environ.get('ONE_FP'):
        if not os.path.exists(FP_POSE_NPY):
            raise RuntimeError(
                f"[fp] pose file not found: {FP_POSE_NPY}. Detect this object with "
                "FoundationPose first (writes camera_T_object), or set ONE_FP_POSE.")
        pos, rot = _tool_pose_from_fp(FP_POSE_NPY)
        if FP_SNAP_TO_TABLE:
            snapped = _snap_to_table(pos, rot)
            if float(snapped[2] - pos[2]) > 1e-4:
                print(f"[fp] snap-to-table: lifted {float(snapped[2]-pos[2])*1000:.1f} mm "
                      "so the object rests on the tabletop (was penetrating it)")
            pos = snapped
        return pos, rot, False
    pos, rot = tool_pose()
    return pos, rot, True


# ======================= mesh-general grasp metrics =======================
def pad_seating(jaw, pose_local, jw):
    """Mean nearest-surface distance from the pinch pads to the mesh (object-local
    frame), averaged over each pad's nearest 10% of vertices -- the mesh analogue
    of the cube box-SDF seating in grasp_cube.pad_seating."""
    tree, _, _ = _mesh_ref()
    jaw.grip_at(pose_local[:3, 3], pose_local[:3, :3], jw)
    total = 0.0
    for link in [jaw._spec.thumb_pad] + list(jaw._spec.opp_pads):
        v = jaw._world_vs(link)                     # pad vertices (object-local)
        d, _ = tree.query(v)
        n = min(len(d), max(3, int(np.ceil(0.1 * len(d)))))
        total += float(np.partition(d, n - 1)[:n].mean())
    return total


def centerline_offset(jaw, pose_local, jw):
    """Perpendicular distance from the mesh centre of mass to the pinch contact
    line (through the contact midpoint along the opposition axis)."""
    _, com, _ = _mesh_ref()
    midpoint = np.asarray(pose_local[:3, 3], dtype=np.float64)
    open_dir = np.asarray(jaw.open_dir_at(jw), dtype=np.float64)
    axis = np.asarray(pose_local[:3, :3], dtype=np.float64) @ open_dir
    axis /= np.linalg.norm(axis) + oum.eps
    d = midpoint - com
    return float(np.linalg.norm(d - axis * np.dot(d, axis)))


def planning_grasp_mask(jaw, obj, poses_local, jaw_widths):
    """Cheap centre/tilt filter (COM-relative) before hand FK and mesh collision."""
    _, com, _ = _mesh_ref()
    poses = np.asarray(poses_local, dtype=np.float64)
    open_dirs = np.asarray(jaw.open_dir_at(jaw_widths), dtype=np.float64)
    axes = np.einsum('nij,nj->ni', poses[:, :3, :3], open_dirs)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + oum.eps
    d = poses[:, :3, 3] - com
    parallel = np.sum(d * axes, axis=1, keepdims=True) * axes
    centered = np.linalg.norm(d - parallel, axis=1) <= CENTER_MAX

    obj_rot = np.asarray(obj.wd_tf[:3, :3], dtype=np.float64)
    world_rots = np.einsum('ij,njk->nik', obj_rot, poses[:, :3, :3])
    elevations = -np.arcsin(np.clip(world_rots[:, 2, 2], -1.0, 1.0))
    tilt_ok = (np.abs(elevations - np.radians(gc.GRASP_TILT_DEG))
               <= np.radians(TILT_FILTER_DEG))
    return centered & tilt_ok


# ============================== grasp planning ==============================
def plan_grasps(robot, grasp_ctx, transit_ctx, hand, obj, home):
    """Plan reachable, collision-free pinch grasps on the mesh object. Mirrors
    grasp_cube.plan_grasps with the mesh-general metrics above and a relaxed
    centre cap. Returns ``(candidates, stats, best_cost)``; each candidate is
    ``(seat, jaw_width, pre_q, descend, retreat, grasp_qs)``, best cost first."""
    jaw = hand.spawn_jaw(gc.GRASP_PRIMITIVE)
    grasps = antipodal(
        jaw, obj, **PLAN_KW,
        candidate_filter=lambda poses, widths: planning_grasp_mask(
            jaw, obj, poses, widths))
    print(f"antipodal: {len(grasps)} collision-free '{gc.GRASP_PRIMITIVE}' grasps "
          f"(jaw range {np.round(np.array(jaw.jaw_range) * 1000, 1)} mm); "
          f"tilt target {gc.GRASP_TILT_DEG:.0f} deg below horizontal")
    tilt_target = np.radians(gc.GRASP_TILT_DEG)
    # Approach with the pinch PRESHAPE (thumb+index positioned, middle/ring/pinky
    # tucked away), NOT the fully splayed open hand -- for a fingertip pinch the
    # idle fingers should be out of the way, and the splayed hand collides with the
    # object where the preshape does not (this alone recovered ~17 reachable grasps).
    hand.pinch(0.0)
    open_hand_qs = np.asarray(hand.qs, dtype=float).copy()

    def set_collision_hand(qs):
        hand.fk(qs=np.asarray(qs, dtype=float))
        for planning_ctx in (grasp_ctx, transit_ctx):
            planning_ctx.collider.set_mecba_qpos(hand, hand.qs)
            planning_ctx.clear_cache()

    # cheap pre-ranking by centre/tilt so expensive IK starts on the best poses
    ranked = []
    for grasp in grasps:
        pose, _pre, jw, _score = grasp
        wpose = obj.wd_tf @ pose
        elev = float(-np.arcsin(np.clip(wpose[2, 2], -1.0, 1.0)))
        pre_cost = (gc.GRASP_TILT_W * abs(elev - tilt_target)
                    + gc.GRASP_CENTER_W * centerline_offset(jaw, pose, jw))
        ranked.append((pre_cost, grasp))
    ranked.sort(key=lambda item: item[0])

    candidates, diag = [], []
    stats = dict(off_center=0, ik=0, descend=0, closed_collision=0,
                 retreat=0, ok=0)
    for _pre_cost, (pose, pre_pose, jw, score) in ranked:
        wpose = obj.wd_tf @ pose
        wpre = obj.wd_tf @ pre_pose
        rot = wpose[:3, :3]
        elev = float(-np.arcsin(np.clip(wpose[2, 2], -1.0, 1.0)))
        centroid_off = centerline_offset(jaw, pose, jw)
        if centroid_off > CENTER_MAX:
            stats['off_center'] += 1
            continue
        grasp_tcp = orbt.TCP(hand.runtime_root_lnk,
                             jaw._grasp_center_loc_tf(jw))
        set_collision_hand(open_hand_qs)
        pre_q = gc.solve_ik(robot, transit_ctx, wpre[:3, 3].astype(np.float32),
                            rot, grasp_tcp, home, collision_free=True,
                            max_solutions=gc.PREGRASP_IK_SOLUTIONS)
        if pre_q is None:
            stats['ik'] += 1; continue
        descend = gc.cartesian_path(robot, grasp_ctx, grasp_tcp, pre_q,
                                    wpre[:3, 3], wpose[:3, 3], rot)
        if descend is None:
            stats['descend'] += 1; continue
        seat = pad_seating(jaw, pose, jw)           # also sets the jaw closure
        grasp_qs = np.asarray(jaw.qs, dtype=float).copy()
        set_collision_hand(grasp_qs)
        if not grasp_ctx.is_state_valid(descend[-1]):
            stats['closed_collision'] += 1
            set_collision_hand(open_hand_qs)
            continue
        retreat = gc.cartesian_path(robot, grasp_ctx, grasp_tcp, descend[-1],
                                    wpose[:3, 3], wpose[:3, 3] + gc.UP, rot)
        if retreat is None:
            stats['retreat'] += 1
            set_collision_hand(open_hand_qs)
            continue
        set_collision_hand(open_hand_qs)
        cost = (seat + gc.GRASP_TILT_W * abs(elev - tilt_target)
                + gc.GRASP_CENTER_W * centroid_off)
        stats['ok'] += 1
        candidates.append((cost, float(seat), float(jw), pre_q, descend, retreat,
                           grasp_qs))
        diag.append((cost, seat, np.degrees(elev), centroid_off))
        if len(candidates) >= gc.REACHABLE_GRASP_TARGET:
            break
    candidates.sort(key=lambda c: c[0])
    set_collision_hand(open_hand_qs)
    if os.environ.get("DEBUG") and diag:
        diag.sort(key=lambda d: d[0])
        print("  reachable grasps (best-first): seat_mm / elev_deg / centroid_off_mm")
        for cost, seat, elevd, coff in diag[:8]:
            print(f"    seat {seat * 1000:5.1f} mm   elev {elevd:+5.1f} deg   "
                  f"centroid_off {coff * 1000:5.1f} mm")
    best_cost = candidates[0][0] if candidates else float("inf")
    candidates = [c[1:] for c in candidates]
    return candidates, stats, best_cost


# ================================== the demo ==================================
def main():
    headless = bool(os.environ.get("ONE_HEADLESS"))
    base = ovw.World(cam_pos=(1.6, 0.4, 1.6), cam_lookat_pos=(0.45, -0.1, 0.95))

    tool_pos, tool_rot, add_stand = resolve_tool_pose()   # FP real pose, or hardcoded
    robot = build_robot()
    statics = build_static_objects()
    if add_stand:                                  # pure-sim: raise low feature clear
        support = build_support(tool_pos, tool_rot)
        if support is not None:
            statics.append(support)
    tool = build_tool(tool_pos, tool_rot)
    hand = robot.left_hand

    # real robot (opt-in ONE_REAL): its CURRENT joints become the planning start.
    arm_x, hand_x = gc.connect_real_robot()
    qs_home = robot.qs.astype(np.float64).copy()
    if arm_x is not None:
        q_start = arm_x.get_jnt_values().astype(np.float64)
        print(f"[real] arm start (deg): {np.round(np.degrees(q_start), 1)}")
    else:
        q_start = np.deg2rad(gc.HOME_DEG).astype(np.float64)
    qs_home[robot.chain(gc.CHAIN).active_jnt_ids] = q_start
    robot.fk(qs=qs_home)

    ossop.frame().attach_to(base.scene)
    for e in [robot] + statics + [tool]:
        e.attach_to(base.scene)

    # grasp view omits the tool (pad contact intended); transit view includes it
    grasp_mjc = ocm.MJCollider()
    for e in [robot] + statics:
        grasp_mjc.append(e)
    grasp_mjc.actors = [robot]
    grasp_mjc.compile(margin=COLLISION_MARGIN, auto_acm=True)
    transit_mjc = ocm.MJCollider()
    for e in [robot] + statics + [tool]:
        transit_mjc.append(e)
    transit_mjc.actors = [robot]
    # The object rests ON the table (it was snapped down to seat there). That
    # resting contact is NOT an obstacle the arm must avoid -- exclude it, else it
    # is a PERSISTENT collision that invalidates EVERY arm config (0 grasps). The
    # arm still avoids the object itself (tool-vs-robot pairs stay active).
    tool_static_excl = [(tool, s) for s in statics]
    transit_mjc.compile(margin=COLLISION_MARGIN, auto_acm=True,
                        extra_excludes=tool_static_excl)

    j1_safe = tuple(np.radians([gc.ARM_J1_SAFE_MIN_DEG, gc.ARM_J1_SAFE_MAX_DEG]))
    grasp_ctx = chain_planning_context(robot, grasp_mjc, gc.CHAIN,
                                       joint_limit_overrides={'joint1': j1_safe})
    transit_ctx = chain_planning_context(robot, transit_mjc, gc.CHAIN,
                                         joint_limit_overrides={'joint1': j1_safe})
    planner = ompr.RRTConnectPlanner(pln_ctx=transit_ctx,
                                     extend_step_size=np.pi / 36, goal_bias=0.3)
    if (arm_x is not None
            and not transit_ctx.is_state_valid(robot.qs.astype(np.float64))):
        raise RuntimeError(
            "[real] current arm pose collides with the scene/object or is outside "
            f"the safe joint1 range [{gc.ARM_J1_SAFE_MIN_DEG}, "
            f"{gc.ARM_J1_SAFE_MAX_DEG}] deg. Jog it clear and rerun.")

    home = robot.qs.astype(np.float64).copy()
    candidates, stats, best_cost = plan_grasps(
        robot, grasp_ctx, transit_ctx, hand, tool, home)
    for attempt in range(2, ATTEMPTS + 1):
        if candidates and best_cost <= gc.COST_OK:
            break
        more, stats2, more_cost = plan_grasps(
            robot, grasp_ctx, transit_ctx, hand, tool, home)
        if more and more_cost < best_cost:
            candidates, stats, best_cost = more, stats2, more_cost
        print(f"  re-plan {attempt}/{ATTEMPTS}: best cost {best_cost * 1000:.1f} mm")
    print(f"  reject stats: {stats}   (off_center/ik/descend/closed_collision/"
          f"retreat/ok)")
    if not candidates:
        print(f"[plan] no ARM-REACHABLE '{gc.GRASP_PRIMITIVE}' grasp on "
              f"{os.path.basename(TOOL_MESH)} at this pose: antipodal found grasp "
              "poses but every one failed arm IK / collision (see reject stats). "
              "The object may be out of reach or its graspable feature blocked at "
              "this detected pose.")
        if headless:
            return
        print("[plan] opening the viewer to inspect the detected object pose "
              "(robot at home; no grasp to play).")
        base.run()
        return
    # Pick the best candidate whose free-space transit (home -> pre-grasp) also
    # plans -- RRT can fail for a hard pre-grasp even when the grasp is valid.
    traj = grasp_idx = None
    for i, cand in enumerate(candidates):
        try:
            seg = plan_segment(planner, home, cand[2], max_iters=5000)
        except RuntimeError:
            continue
        traj = seg + cand[3][1:]
        grasp_idx = len(traj) - 1
        traj += cand[4][1:]
        candidates.insert(0, candidates.pop(i))     # open the viewer on this one
        break
    if traj is None:
        print(f"'{gc.GRASP_PRIMITIVE}' on {os.path.basename(TOOL_MESH)}: "
              f"{len(candidates)} reachable grasps, but none has a collision-free "
              f"transit path from home (RRT failed); showing grasp poses only.")
    else:
        seat, jw = candidates[0][0], candidates[0][1]
        print(f"'{gc.GRASP_PRIMITIVE}' on {os.path.basename(TOOL_MESH)}: "
              f"{len(candidates)} reachable grasps; best = pad gap {seat * 1000:.1f} mm "
              f"/ jaw {jw * 1000:.1f} mm; pick {len(traj)} waypoints (grasp@{grasp_idx})")
    if headless:
        return

    # ---- browse candidates one by one ----
    #  N/B next/prev  G/F play/step  R reset  C spheres  ENTER preview/execute(real)
    import pyglet.window.key as key
    st = {"sel": 0, "i": 0, "held": False, "playing": False,
          "spheres": False, "traj": None, "gidx": 0, "real_preview": None}
    hand.open_hand()
    open_hand_qs = np.asarray(hand.qs, dtype=float).copy()
    hand_lo = np.asarray(hand._compiled.jlmt_low_by_idx, dtype=float)
    hand_hi = np.asarray(hand._compiled.jlmt_high_by_idx, dtype=float)

    def set_spheres(on):
        (hand.show_collision_spheres(base.scene, alpha=0.35) if on
         else hand.hide_collision_spheres())
        base.scene.dirty = True

    def show(sel):
        st["real_preview"] = None
        st["sel"] = sel % len(candidates)
        _seat, _jw, _pre, descend, _retreat, gqs = candidates[st["sel"]]
        if st["held"]:
            hand.unmount(tool); st["held"] = False
        robot.fk(qs=descend[-1])
        hand.fk(qs=gqs)
        tool.set_pos_rotmat(pos=tool_pos, rotmat=tool_rot)
        st["i"], st["playing"], st["traj"] = 0, False, None
        if st["spheres"]:
            set_spheres(True)
        print(f"candidate {st['sel'] + 1}/{len(candidates)}: pad gap "
              f"{_seat * 1000:.1f} mm, jaw {_jw * 1000:.1f} mm")
        base.scene.dirty = True

    def ensure_traj():
        if st["traj"] is None:
            _, _, pre_q, descend, retreat, _ = candidates[st["sel"]]
            try:
                t = plan_segment(planner, home, pre_q, max_iters=5000) + descend[1:]
            except RuntimeError:
                print("  (no transit path for this candidate; try N for another)")
                st["traj"] = []
                return st["traj"]
            st["gidx"] = len(t) - 1
            st["traj"] = t + retreat[1:]
        return st["traj"]

    def reset():
        st["real_preview"] = None
        if st["held"]:
            hand.unmount(tool); st["held"] = False
        hand.open_hand()
        robot.fk(qs=home)
        tool.set_pos_rotmat(pos=tool_pos, rotmat=tool_rot)
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
            loc = np.linalg.inv(hand.runtime_root_lnk.tf) @ tool.tf
            hand.mount(tool, hand.runtime_root_lnk, loc)
            st["held"] = True
        if st["spheres"]:
            set_spheres(True)
        st["i"] += 1
        base.scene.dirty = True

    def execute_real():
        """First ENTER previews a freshly planned real path from the current
        measured joints; second ENTER streams it to the arm + XHand with a
        torque-feedback tactile close. Mirrors grasp_cube.execute_real."""
        if arm_x is None:
            print("[real] not connected -- set ONE_REAL=1 (and check the IP / port)")
            return
        if st["real_preview"] is not None and st["i"] < len(st["traj"] or []):
            print("[real] preview not finished; let it play to the end, then ENTER")
            return
        sel = st["sel"]
        chain = robot.chain(gc.CHAIN)
        try:
            hand.fk(qs=open_hand_qs)
            for pc in (grasp_ctx, transit_ctx):
                pc.collider.set_mecba_qpos(hand, hand.qs); pc.clear_cache()
            current_arm = np.asarray(arm_x.get_jnt_values(), dtype=np.float64)
            current_qs = home.copy(); current_qs[chain.active_jnt_ids] = current_arm
            if not transit_ctx.is_state_valid(current_qs):
                print("[real] ERROR: current open-hand pose collides / out of safe "
                      "range; not moving. Jog clear and retry."); return

            preview = st["real_preview"]
            if preview is None:
                _seat, _jw, pre_q, descend, retreat, gqs = candidates[sel]
                pre_q = np.asarray(pre_q, dtype=np.float64)
                if transit_ctx.states_equal(current_qs, pre_q):
                    approach = [pre_q]
                else:
                    try:
                        approach = plan_segment(planner, current_qs, pre_q, max_iters=8000)
                    except RuntimeError:
                        print("[real] no collision-free path current -> pre-grasp; "
                              "try another candidate (N)"); return
                    print(f"[real] current -> pre-grasp: {len(approach)} waypoints")
                real_traj = approach + list(descend[1:])
                gidx = len(real_traj) - 1
                real_traj += list(retreat[1:])
                arm_path = [chain.extract_active_qs(np.asarray(q, np.float64))
                            for q in real_traj]
                st["real_preview"] = {"sel": sel, "start_qs": current_qs.copy(),
                                      "arm_path": arm_path, "gidx": gidx,
                                      "gqs": np.asarray(gqs, dtype=np.float64).copy()}
                if st["held"]:
                    hand.unmount(tool); st["held"] = False
                tool.set_pos_rotmat(pos=tool_pos, rotmat=tool_rot)
                robot.fk(qs=current_qs); hand.fk(qs=open_hand_qs)
                st["traj"], st["gidx"], st["i"], st["playing"] = real_traj, gidx, 0, True
                base.scene.dirty = True
                print(f"[real] simulation preview: {len(arm_path)} waypoints "
                      f"(grasp@{gidx}); watch it, then ENTER again to execute")
                return

            if preview["sel"] != sel:
                st["real_preview"] = None
                print("[real] candidate changed; preview cleared. ENTER to replan"); return
            start_arm = chain.extract_active_qs(preview["start_qs"])
            start_err = float(np.max(np.abs(np.degrees(current_arm - start_arm))))
            if (start_err > gc.REAL_PREVIEW_START_TOL_DEG
                    or not transit_ctx.is_motion_valid(current_qs, preview["start_qs"])):
                st["real_preview"] = None
                print(f"[real] joints moved since preview (max {start_err:.2f} deg); "
                      "cancelled. ENTER to replan"); return

            arm_path, gidx, gqs = preview["arm_path"], preview["gidx"], preview["gqs"]
            st["real_preview"] = None
            print(f"[real] executing {len(arm_path)} waypoints (grasp@{gidx}) ...")
            if hand_x is not None:
                hand_x.move_to(gc.sim_to_real_hand(open_hand_qs), speed=gc.HAND_SPEED)
            arm_x.stream_jnt_path(arm_path[:gidx + 1], control_freq=gc.ARM_CTRL_FREQ,
                                  max_jntvel=gc.ARM_MAX_JNTVEL)               # descend
            if hand_x is not None:
                # two-phase tactile close (preshape swing, then force-close flexion)
                required = gc.REQUIRED_FINGERS[gc.GRASP_PRIMITIVE]
                open12 = gc.sim_to_real_hand(open_hand_qs)
                target12 = gc.sim_to_real_hand(gqs).copy()
                contact_ids = sorted({i for f in required
                                      for i in gc.HAND_CONTACT_IDS[f]})
                preshape12 = target12.copy()
                for i in contact_ids:
                    preshape12[i] = open12[i]
                for f in required:
                    for i in gc.HAND_CONTACT_IDS[f]:
                        target12[i] = hand_hi[i] if target12[i] >= open12[i] else hand_lo[i]
                hand_x.move_to(preshape12, speed=gc.HAND_SPEED)
                contacted = gc.tactile_close(hand_x, preshape12, target12, required)
                miss = [f for f in required if not contacted[f]]
                print(f"[real] grasp {'secured' if not miss else 'LOOSE, no contact on '+str(miss)}")
            arm_x.stream_jnt_path(arm_path[gidx:], control_freq=gc.ARM_CTRL_FREQ,
                                  max_jntvel=gc.ARM_MAX_JNTVEL)               # lift
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
    print(f"{len(candidates)} candidates.  N/B: next/prev   G: play   F: step   "
          f"R: reset   C: spheres" +
          ("   ENTER: preview/execute on robot" if arm_x is not None else ""))
    base.schedule_interval(tick, interval=0.03)
    base.run()


if __name__ == "__main__":
    main()
