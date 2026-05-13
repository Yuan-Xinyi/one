"""Hyper-parameters for the v18 backward-CFM pipeline."""
import numpy as np


# ---------- Control loop ----------
DT          = 0.02                  # control period [s] (50 Hz)
THETA_MAX   = np.deg2rad(5.0)       # z-axis vs plane-normal tolerance [rad]
EPS_POS_INIT = 5e-3                 # IK projection position tolerance [m]
INIT_IK_ORIENT_MODE = "z_axis"      # "z_axis" or "full_rot"

# ---------- Branch IK ----------
BRANCH_SWIVEL_GAIN   = 0.15         # null-space gain for elbow branch objective
BRANCH_FD_EPS        = 1e-3         # finite-difference step for branch gradient
BRANCH_IK_NUM_STARTS = 16           # deterministic IK start postures

# ---------- Cartesian DLS controller ----------
KP_LIN       = 5.0                  # position-error feedback gain [1/s]
KOMEGA       = 5.0                  # orientation-error feedback gain [1/s]
DLS_LAMBDA   = 0.05                 # DLS damping
POS_PRIORITY_ORIENT_MARGIN_RATIO = 0.5   # dead-zone ramp start at this * theta_max
POS_PRIORITY_JLIMIT_MARGIN       = 0.20  # rad; joint-limit repulsion activates inside this
POS_PRIORITY_JLIMIT_GAIN         = 4.0   # rad/s per rad-of-incursion
K_NULL       = 0.5                  # null-space pull toward q_ref
NULL_USE_MANIPULABILITY = True
NULL_MANIP_GAIN         = 0.6
NULL_MANIP_DAMPING      = 1e-3
NULL_JOINT_LIMIT_GAIN   = 0.2       # push toward joint-center in null-space
NULL_ANGLE_GAIN         = 0.4       # z-axis alignment pull near cone boundary
NULL_ANGLE_ATTRACT_GAIN = 0.0       # always-on z-axis attractor
NULL_ANGLE_MARGIN       = np.deg2rad(8.0)

# ---------- End-effector geometry (FR3 hand + pen) ----------
# BatchedFR3Kinematics already includes link7 → flange (0.107 m).
# TCP_OFFSET adds flange → pen-tip on top.
USE_PEN_TCP     = True
HAND_TCP_OFFSET = 0.1034            # m, FR3 gripper acting center
PEN_LENGTH      = 0.10              # m, pen extension
TCP_OFFSET      = (HAND_TCP_OFFSET + PEN_LENGTH) if USE_PEN_TCP else 0.0

# ---------- Self-collision ----------
USE_COLLISION_CHECK = True

# ---------- Batched rollout ----------
BATCHED_ROLLOUT_DEVICE   = "auto"   # "auto", "cuda", or "cpu"
BATCHED_COLLISION_CHECK  = True     # sphere self-collision in batched rollout
BATCHED_COLLISION_MARGIN = 0.0
BATCHED_IK_MAX_ITERS     = 50
BATCHED_IK_DAMPING       = 1e-4
