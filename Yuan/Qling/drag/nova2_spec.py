"""Dobot Nova2 spec for the IJRR ``BatchedChainKinematics`` class.

Joint placements are transcribed verbatim from the WRS model
(/disk2/wrs_xinyi/wrs/robot_sim/manipulators/dobot_nova2/nova2.py,
author chen hao / weiwei). The local rotation matrices are the exact
numeric outputs of ``rm.rotmat_from_euler`` on the literals used there
(1.5708, not pi/2 -- hence the ~3.7e-6 off-diagonal terms), dumped from
the live WRS checkout so this file has no wrs dependency. Verified
against WRS FK in test_drag_env.py.

Joint 1 rotates about local -z (the only non-+z axis in the chain).

TCP: the WRSGripper3 acting center sits 0.16 m along flange +z
(wrs_gripper_v3.py: loc_acting_center_pos = [0, 0, .16]); the regrasp
robot mounts the gripper directly on the flange, so tcp_offset = 0.16.

qdot_max: the WRS model carries no joint speed limits. The official
Nova 2 spec (dobot-robots.com, checked 2026-08-27) lists 135 deg/s for
every joint J1-J6 (payload 2 kg, working radius 625 mm).
"""
import math

_R2 = [[-3.6732051033465739e-06, 9.9999999998650746e-01, -3.6732051033217936e-06],
       [0.0, -3.6732051033465739e-06, -9.9999999999325373e-01],
       [-9.9999999999325373e-01, -3.6732051033217936e-06, 1.3492435731251315e-11]]
_R4 = [[-3.673205103346574e-06, 9.999999999932537e-01, 0.0],
       [-9.999999999932537e-01, -3.673205103346574e-06, 0.0],
       [0.0, 0.0, 1.0]]
_R5 = [[1.0, 0.0, 0.0],
       [0.0, -3.673205103346574e-06, -9.999999999932537e-01],
       [0.0, 9.999999999932537e-01, -3.673205103346574e-06]]
_R6 = [[-1.0, 1.2246467991390914e-16, 4.4983788724251044e-22],
       [0.0, -3.6732051033465739e-06, 9.9999999999325373e-01],
       [1.2246467991473532e-16, 9.9999999999325373e-01, 3.6732051033465739e-06]]
_EYE3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

GRIPPER_TCP_OFFSET = 0.16   # WRSGripper3 acting center along flange +z

_PI = math.pi
NOVA2 = dict(
    name='nova2',
    joints=[
        (_EYE3, [0.0, 0.0, 0.2234],       [0.0, 0.0, -1.0]),
        (_R2,   [0.0, 0.0, 0.0],          [0.0, 0.0, 1.0]),
        (_EYE3, [-0.28, 0.0, 0.0],        [0.0, 0.0, 1.0]),
        (_R4,   [-0.22501, 0.0, 0.1175],  [0.0, 0.0, 1.0]),
        (_R5,   [0.0, -0.12, 0.0],        [0.0, 0.0, 1.0]),
        (_R6,   [0.0, 0.088004, 0.0],     [0.0, 0.0, 1.0]),
    ],
    lmt_lo=[-_PI, -_PI, -2.79, -2.0 * _PI, -2.0 * _PI, -2.0 * _PI],
    lmt_up=[_PI, _PI, 2.79, 2.0 * _PI, 2.0 * _PI, 2.0 * _PI],
    qdot_max=[math.radians(135.0)] * 6,   # official spec, all joints
    flange_pos=[0.0, 0.0, 0.0],
)

REACH_RADIUS = 0.625    # official working radius (m)
