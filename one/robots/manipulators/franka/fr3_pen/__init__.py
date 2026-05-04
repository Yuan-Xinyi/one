"""FR3 + Franka Hand + pen end-effector — shared building blocks.

CPU robot helpers (``make_fr3_with_pen``, ``attach_pen_visual``) and a
torch-batched FK (``BatchedFR3Kinematics``) live here so that downstream
packages (``Yuan/RL``, ``Yuan/fr3_dit``) can share one definition.
"""
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    HAND_TCP_OFFSET,
    PEN_LENGTH,
    attach_pen_visual,
    make_fr3_with_pen,
)
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
    DEFAULT_TCP_OFFSET,
    QDOT_MAX_DEFAULT,
    BatchedFR3Kinematics,
)
from one.robots.manipulators.franka.fr3_pen.sphere_collision_helpers import (
    PenCollisionHelpers,
    make_pen_collision_helpers,
)


TCP_OFFSET = DEFAULT_TCP_OFFSET  # alias for code that imported the constant directly

__all__ = [
    "HAND_TCP_OFFSET",
    "PEN_LENGTH",
    "TCP_OFFSET",
    "DEFAULT_TCP_OFFSET",
    "QDOT_MAX_DEFAULT",
    "attach_pen_visual",
    "make_fr3_with_pen",
    "BatchedFR3Kinematics",
    "PenCollisionHelpers",
    "make_pen_collision_helpers",
]
