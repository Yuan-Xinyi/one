"""Grasp planning for the XHand-right on a 6 cm cube, with three grasp primitives:

    * PINCH  -- thumb + index (a 2-point opposing grasp)
    * TRIPOD -- thumb + index + middle (a genuine 3-finger / multi-finger grasp)
    * POWER  -- all five fingers envelop the cube against the palm

All three are planned with the SAME in-house planner, ``one.grasp.antipodal``.
Each is an OPPOSING grasp -- the thumb pad opposes one or more finger pads -- so
the dexterous hand is presented to ``antipodal`` as a parallel jaw via
``hand.spawn_jaw(primitive)``. That returns a calibrated, immutably-bound clone
exposing exactly the gripper interface antipodal consumes (jaw_range /
set_jaw_width / grip_at / grasp_center_at). No hand-rolled grasp math.

POWER is a five-finger ENVELOPE, not literally a parallel jaw; it is planned by
its DOMINANT opposition (thumb vs the central middle finger) so antipodal can
place it, while ``grasp(..., primitive='power')`` still curls all five fingers on
execution. (This is the same modelling the O6 hand uses; it required giving
XHand's 'power' primitive a thumb-vs-middle ``pads`` entry -- see xhand_right.py.)

``antipodal`` plans in the cube's LOCAL frame and returns, best score first, a list
of (pose, pre_pose, jaw_width, score) where ``pose`` is the grasp-center frame
(origin = contact-pair midpoint, +z = approach axis) and ``pre_pose`` is that pose
backed off along the approach axis -- the PRE-GRASP the hand swings to before
closing in. Here the cube sits at the origin so local == world.
``grip_at(pos, rotmat, jaw_width)`` then poses the hand. For each selected grasp
the example draws BOTH: the pre-grasp (at ``pre_pose``, jaw opened, translucent)
and the grasp (at ``pose``, closed, solid), joined by the red approach line.

Two tweaks make the fingers actually seat on the cube:
  * ``clearance=0`` -- antipodal's default leaves a per-side standoff; 0 closes
    the pads onto the faces.
  * RE-RANK by real pad seating -- antipodal scores grasps by normal alignment +
    jaw centering, NOT by whether the pads touch the object, so its top grasp can
    leave a finger (especially the curling thumb, which the parallel-jaw view only
    models by its single nearest vertex) standing well off a flat face. We re-sort
    by each grasping pad's true distance to the cube so the FIRST grasp shown is
    the best-contacting one. NOTE: the seating metric scores only the planned
    OPPOSITION pads (thumb + the opposing finger(s)). A flat cube suits the
    2-point opposition of PINCH / POWER; a 3-finger TRIPOD can never seat all
    three pads on two opposing flat faces (the third finger necessarily stands
    off ~1-2 cm) -- tripod is for rounded objects.

Note on ``polypodal`` (the N-point force-closure planner): it is a symmetric
front-N / back-N COPLANAR-opposition model (multi-finger pinch of a thin part),
not a five-finger envelope, and it needs a gripper with a native multi-point
``contact_pattern`` (shape (N, 3)) that the in-repo dexterous hands don't define.
So a power ENVELOPE is planned here through the antipodal opposition view, not
polypodal.

Keys: P = cycle primitive (pinch / tripod / power), N = next grasp, R = re-plan.
Headless validation (no window): set ONE_HEADLESS=1
"""
import os

import numpy as np

import one.utils.constant as ouc
import one.viewer.world as ovw
import one.scene.scene_object_primitive as ossop
from one.robots.end_effectors.xhand.xhand_right import XHandRight
from one.grasp.antipodal import antipodal

CUBE_SIZE = 0.06                       # 6 cm cube
HALF = CUBE_SIZE / 2
PRIMITIVES = ('pinch', 'tripod', 'power')   # 2-finger / 3-finger / 5-finger
# antipodal sampling: small cube -> fine surface density; a 6 cm box has only a
# few large faces, so sample densely and allow a generous normal tolerance.
# clearance=0 so the pads close onto the faces instead of standing off.
PLAN_KW = dict(density=0.0006, normal_tol_deg=25, roll_step_deg=30,
               max_grasps=40, clearance=0.0)
# pre-grasp jaw opening, as a fraction of the room between the grasp width and
# the max opening (same convention as antipodal's pre_open): the hand swings in
# wider, then closes to the grasp width. 0 = same as grasp, 1 = fully open.
PRE_OPEN = 0.5


def make_cube():
    """A 6 cm cube at the origin. MESH collision so antipodal can sample its
    surface (the planner reads ``obj.collisions``)."""
    return ossop.box(pos=(0.0, 0.0, 0.0),
                     xyz_lengths=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                     rgb=(0.3, 0.55, 0.85),
                     collision_type=ouc.CollisionType.MESH,
                     is_floating=True)


def pad_seating(jaw, pose, jaw_width):
    """Sum over the grasp's contact pads of each pad's MIN distance to the cube
    surface (cube is axis-aligned at the origin). 0 == that pad touches the cube;
    larger == a finger standing off. Used to re-rank grasps by real contact."""
    jaw.grip_at(pose[:3, 3], pose[:3, :3], jaw_width)
    pads = [jaw._spec.thumb_pad] + list(jaw._spec.opp_pads)
    total = 0.0
    for link in pads:
        v = jaw._world_vs(link)                     # world-frame pad vertices
        a = np.abs(v) - HALF                        # signed dist to cube box
        d = np.linalg.norm(np.maximum(a, 0.0), axis=1) + np.minimum(a.max(1), 0.0)
        total += float(d.min())
    return total


def plan(primitive, cube):
    """Plan grasps for one primitive, RE-RANKED by real pad seating. Returns
    (jaw, grasps): the spawned parallel-jaw view (holds the jaw_width<->closure
    calibration needed to pose the hand) and the list of
    (pose, pre_pose, jaw_width, score), best-seated first."""
    jaw = XHandRight().spawn_jaw(primitive)        # immutable parallel-jaw clone
    grasps = antipodal(jaw, cube, **PLAN_KW)       # cube-local, best score first
    grasps.sort(key=lambda g: pad_seating(jaw, g[0], g[2]))   # best contact first
    if grasps:
        gap = pad_seating(jaw, grasps[0][0], grasps[0][2])
        print(f'[{primitive}] antipodal grasps: {len(grasps)}; '
              f'best-seated pad gap (sum) = {gap * 1000:.1f} mm')
    else:
        print(f'[{primitive}] antipodal grasps: 0')
    return jaw, grasps


def main():
    headless = os.environ.get('ONE_HEADLESS')
    if headless:
        np.random.seed(0)

    cube = make_cube()
    plans = {p: plan(p, cube) for p in PRIMITIVES}

    if headless:
        for p in PRIMITIVES:
            jaw, grasps = plans[p]
            assert grasps, f'antipodal found no {p} grasps on the cube'
            pose, pre, jw, _sc = grasps[0]            # best-SEATED after re-rank
            gap = pad_seating(jaw, pose, jw)
            standoff = float(np.linalg.norm(pose[:3, 3] - pre[:3, 3]))
            assert standoff > 1e-3, 'pre-grasp not retreated from grasp'
            print(f'headless OK [{p}]: {len(grasps)} grasps, '
                  f'best-seated pad gap {gap * 1000:.1f} mm, '
                  f'jaw width {jw * 1000:.1f} mm, '
                  f'pre-grasp standoff {standoff * 1000:.0f} mm')
        return

    import pyglet.window.key as key

    base = ovw.World(cam_pos=(0.25, 0.20, 0.16),
                     cam_lookat_pos=(0.0, 0.0, 0.0))
    ossop.frame(length_scale=0.2, radius_scale=0.25).attach_to(base.scene)
    cube.attach_to(base.scene)

    state = {'prim': 0, 'cur': 0, 'items': []}

    def clear():
        for o in state['items']:
            o.detach_from(base.scene)
        state['items'] = []

    def show():
        clear()
        primitive = PRIMITIVES[state['prim']]
        jaw, grasps = plans[primitive]
        if not grasps:
            base.set_caption(f'{primitive}: no grasps'); return
        idx = state['cur'] % len(grasps)
        state['cur'] = idx
        pose, pre, jw, sc = grasps[idx]
        jaw_max = float(jaw.jaw_range[1])
        pre_jw = jw + PRE_OPEN * (jaw_max - jw)         # hand opened wider to swing in
        # PRE-GRASP: retreated along the approach axis, jaw opened -- translucent.
        # ``pre`` is exactly antipodal's pre_pose (pose backed off along -approach).
        pre_hand = jaw.clone()
        pre_hand.grip_at(pre[:3, 3], pre[:3, :3], pre_jw)
        pre_hand.alpha = 0.3
        pre_hand.attach_to(base.scene)
        # GRASP: closed on the cube -- solid.
        grasp_hand = jaw.clone()
        grasp_hand.grip_at(pose[:3, 3], pose[:3, :3], jw)
        grasp_hand.attach_to(base.scene)
        # approach line: pre-grasp center -> grasp center.
        seg = np.array([[pre[:3, 3], pose[:3, 3]]], dtype=np.float32)
        line = ossop.linsegs(seg, radius=0.0015,
                             srgbs=np.array([0.9, 0.2, 0.2], dtype=np.float32))
        line.attach_to(base.scene)
        state['items'] = [pre_hand, grasp_hand, line]
        standoff = float(np.linalg.norm(pose[:3, 3] - pre[:3, 3]))
        msg = (f'{primitive}  grasp {idx + 1}/{len(grasps)}  jaw={jw * 1000:.1f}mm  '
               f'pre-open={pre_jw * 1000:.1f}mm  standoff={standoff * 1000:.0f}mm  '
               f'score={sc:.3f}')
        base.set_caption(msg)
        print(msg)

    show()
    print('P = cycle primitive (pinch/tripod/power), N = next grasp, R = re-plan')

    def tick(dt):
        im = base.input_manager
        if im.is_key_pressed_edge(key.P):
            state['prim'] = (state['prim'] + 1) % len(PRIMITIVES)
            state['cur'] = 0
            show()
        if im.is_key_pressed_edge(key.N):
            state['cur'] += 1
            show()
        if im.is_key_pressed_edge(key.R):
            primitive = PRIMITIVES[state['prim']]
            plans[primitive] = plan(primitive, cube)
            state['cur'] = 0
            show()

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == '__main__':
    main()
