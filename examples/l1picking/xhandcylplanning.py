"""Plan XHandRight antipodal PINCH grasps on the cylinder loaded from
cylinder.stl, and save them for l1binpicking_xhand.py. Counterpart to
o6cylstlplanning.py (the O6 hand); same cylinder, same antipodal parameters, so
the only variable is the hand.

Headless: ONE_HEADLESS=1   Viewer keys: N = next grasp pair
"""
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS))
for p in (_PROJECT_ROOT, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import one.utils.constant as ouc                                  # noqa: E402
import one.scene.scene_object as osso                             # noqa: E402
from one.robots.end_effectors.xhand.xhand_right import XHandRight  # noqa: E402
from one.grasp.antipodal import antipodal                         # noqa: E402
from one.grasp.serialize import save_grasps                       # noqa: E402

CYL_STL = os.path.join(_THIS, "cylinder.stl")
OUT_JSON = os.path.join(_THIS, "xhand_cyl_stl_grasps.json")

# same cylinder as o6cylstlplanning.py (dia 0.025, height 0.075). The XHand pinch
# is a curling opposition, so allow a slightly looser normal tolerance / finer
# sampling than the O6 to find enough wall-clear pinches.
PLAN_KW = dict(density=0.0015, normal_tol_deg=25, roll_step_deg=30,
               max_grasps=60, clearance=0.003)


def main():
    cyl = osso.SceneObject.from_file(
        CYL_STL, collision_type=ouc.CollisionType.MESH, is_free=True,
        rgb=(0.6, 0.7, 0.5))
    hand = XHandRight()
    jaw = hand.spawn_jaw('pinch')
    grasps = antipodal(jaw, cyl, **PLAN_KW)
    print(f"[xhand] antipodal: {len(grasps)} pinch grasps")
    if not grasps:
        raise RuntimeError("[xhand] no antipodal grasp found")
    pos = np.array([g[0][:3, 3] for g in grasps])
    print(f"[xhand] grasp-pose pos bbox  x{np.round([pos[:,0].min(),pos[:,0].max()],4)}"
          f"  y{np.round([pos[:,1].min(),pos[:,1].max()],4)}"
          f"  z{np.round([pos[:,2].min(),pos[:,2].max()],4)}")
    save_grasps(grasps, OUT_JSON, gripper_name="XHandRight", object_name="stl")
    print(f"[xhand] saved {len(grasps)} grasps -> {os.path.basename(OUT_JSON)}")

    if os.environ.get("ONE_HEADLESS"):
        return

    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    import pyglet.window.key as key

    base = ovw.World(cam_pos=(0.25, 0.0, 0.08), cam_lookat_pos=(0.0, 0.0, 0.04))
    builtins.base = base
    ossop.frame(length_scale=0.3).attach_to(base.scene)
    cyl.attach_to(base.scene)
    jaw_open = float(jaw.jaw_range[1])
    jaw_pose = hand.spawn_jaw('pinch')
    jaw_pre = hand.spawn_jaw('pinch')
    jaw_pose.attach_to(base.scene)
    jaw_pre.attach_to(base.scene)
    state = {"i": 0}

    def show(i):
        pose, pre, jw, score = grasps[i]
        wpose = cyl.wd_tf @ pose
        wpre = cyl.wd_tf @ pre
        jaw_pose.grip_at(wpose[:3, 3], wpose[:3, :3], jw)
        jaw_pre.grip_at(wpre[:3, 3], wpre[:3, :3], jaw_open)
        jaw_pose.rgb = (0.20, 0.85, 0.25)
        jaw_pre.rgb = (0.95, 0.85, 0.15)
        base.scene.dirty = True
        base.set_caption(f"[xhand] pair {i}/{len(grasps)}  green=pose "
                         f"yellow=pre  jaw={jw*1000:.1f}mm   N: next")

    show(0)

    def tick(dt):
        if base.input_manager.is_key_pressed_edge(key.N):
            state["i"] = (state["i"] + 1) % len(grasps)
            show(state["i"])

    base.schedule_interval(tick, interval=0.05)
    base.run()


if __name__ == "__main__":
    main()
