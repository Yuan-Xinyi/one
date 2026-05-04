"""FR3 + official Franka hand + pen end-effector helpers.

The TCP we control here is defined as the PEN TIP, located
``HAND_TCP_OFFSET + PEN_LENGTH`` meters along the flange +z axis (the official
Franka hand's grasptarget plus a rigid pen extension).

Two helpers:
  - ``make_fr3_with_pen()``: factory returning ``(arm, hand)`` with the Franka
    hand engaged and the manipulator's local TCP overridden so that
    ``DLSController(arm)`` automatically tracks the pen tip.
  - ``attach_pen_visual(arm)``: attaches a thin cylinder + tip sphere to the
    flange link so viewers see the pen following the arm.

These are the only two helpers anywhere downstream that need to know about the
pen geometry. Everything else just reads ``arm._loc_tcp_tf``.
"""
from __future__ import annotations

import numpy as np


# Shared geometry constants.
HAND_TCP_OFFSET = 0.1034    # m, FR3 gripper acting center (panda_hand grasptarget)
PEN_LENGTH = 0.10            # m, rigid pen extension past the gripper acting center


def make_fr3_with_pen(rotmat=None, pos=None, jaw_width: float = 0.0,
                      use_pen_tcp: bool = True,
                      pen_length: float = PEN_LENGTH):
    """FR3 arm with Franka hand engaged and TCP shifted to the pen tip.

    Returns ``(arm, hand)``. With ``use_pen_tcp=False`` this is identical to
    ``one.robots.manipulators.franka.fr3.fr3_with_hand``.
    """
    from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand
    arm, hand = fr3_with_hand(rotmat=rotmat, pos=pos, jaw_width=jaw_width)
    if use_pen_tcp:
        # ``engage`` already set _loc_tcp_tf rotation (-pi/4 about z, hand's
        # grasptarget orientation) and position (0, 0, HAND_TCP_OFFSET).
        # Extend along flange +z by pen_length to put TCP at pen tip.
        arm._loc_tcp_tf[2, 3] += float(pen_length)
    return arm, hand


def attach_pen_visual(arm,
                      rgb=(0.15, 0.15, 0.15),
                      shaft_radius: float = 0.005,
                      tip_radius: float = 0.0065,
                      alpha: float = 0.95,
                      pen_length: float = PEN_LENGTH,
                      hand_tcp_offset: float = HAND_TCP_OFFSET):
    """Attach a thin cylinder + tip sphere to the flange link as the pen.

    The visuals follow the flange automatically across ``fk()`` updates.
    Returns ``(shaft, tip)`` primitive sobjs.
    """
    import one.scene.scene_object_primitive as ossop

    # link7's frame is ``runtime_lnks[-1]``; the flange is offset (0, 0, 0.107)
    # along link7's z, then hand center (0, 0, hand_tcp_offset) beyond that.
    # Pen sits between hand center and pen tip.
    flange_lnk = arm.runtime_lnks[-1]
    z0 = 0.107 + float(hand_tcp_offset)         # pen base in link7
    z1 = z0 + float(pen_length)                  # pen tip   in link7
    spos = (0.0, 0.0, z0)
    epos = (0.0, 0.0, z1)
    shaft = ossop.cylinder(spos=spos, epos=epos,
                           radius=shaft_radius,
                           rgb=np.asarray(rgb, dtype=np.float32),
                           alpha=alpha)
    shaft.attach_to(flange_lnk)
    tip = ossop.sphere(pos=epos,
                       radius=tip_radius,
                       rgb=np.asarray(rgb, dtype=np.float32),
                       alpha=alpha)
    tip.attach_to(flange_lnk)
    return shaft, tip


if __name__ == "__main__":
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=[1.5, 1.0, 0.9], cam_lookat_pos=[0.3, 0.0, 0.5])
    arm, hand = make_fr3_with_pen()
    arm.attach_to(base.scene)
    attach_pen_visual(arm)
    arm.toggle_tcp(length_scale=0.10, radius_scale=0.4)
    ossop.frame(length_scale=0.20, radius_scale=0.7).attach_to(base.scene)

    print(f"loc_tcp_tf:\n{arm._loc_tcp_tf}")
    print(f"gl_tcp_pos: {arm.gl_tcp_tf[:3, 3]}")
    builtins.base = base
    base.run()
