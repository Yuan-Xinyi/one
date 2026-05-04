"""FR3 + pen end-effector — re-export shim.

The actual implementation lives in
``one.robots.manipulators.franka.fr3_pen.fr3_with_pen``. This module reads the
RL pipeline's ``cfg.USE_PEN_TCP`` / ``cfg.PEN_LENGTH`` / ``cfg.HAND_TCP_OFFSET``
so existing call sites keep their config-driven behavior.
"""
from __future__ import annotations

import Yuan.RL.config as cfg
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    attach_pen_visual as _attach_pen_visual,
)


def make_fr3_with_pen(rotmat=None, pos=None, jaw_width: float = 0.0):
    """FR3 arm with Franka hand engaged and TCP shifted to the pen tip.

    Returns ``(arm, hand)``. With ``cfg.USE_PEN_TCP=False`` this is identical
    to ``one.robots.manipulators.franka.fr3.fr3.fr3_with_hand``.
    """
    from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
        make_fr3_with_pen as _make,
    )
    return _make(rotmat=rotmat, pos=pos, jaw_width=jaw_width,
                 use_pen_tcp=bool(cfg.USE_PEN_TCP),
                 pen_length=float(cfg.PEN_LENGTH))


def attach_pen_visual(arm, **kwargs):
    """No-op when ``cfg.USE_PEN_TCP`` is False; otherwise delegates."""
    if not cfg.USE_PEN_TCP:
        return None, None
    kwargs.setdefault("pen_length", float(cfg.PEN_LENGTH))
    kwargs.setdefault("hand_tcp_offset", float(cfg.HAND_TCP_OFFSET))
    return _attach_pen_visual(arm, **kwargs)


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
