"""IK and TCP constants shared by the kinematics helpers."""
import numpy as np


THETA_MAX   = np.deg2rad(5.0)       # z-axis vs plane-normal tolerance [rad]
EPS_POS_INIT = 5e-3                 # IK projection position tolerance [m]
INIT_IK_ORIENT_MODE = "z_axis"      # "z_axis" or "full_rot"

# ---------- Branch IK ----------
BRANCH_SWIVEL_GAIN   = 0.15         # null-space gain for elbow branch objective
BRANCH_FD_EPS        = 1e-3         # finite-difference step for branch gradient


BATCHED_IK_MAX_ITERS     = 50
BATCHED_IK_DAMPING       = 1e-4

# ---------- TCP ----------
# Read via getattr() in kinematics/batched_fr3_kin.py, which falls back to 0.0.
# Deleting these does not raise: it silently moves the TCP from the pen tip
# back to the flange and every FK result shifts by 0.2034 m.
USE_PEN_TCP     = True
HAND_TCP_OFFSET = 0.1034            # m, FR3 gripper acting center
PEN_LENGTH      = 0.10              # m, pen extension
TCP_OFFSET      = (HAND_TCP_OFFSET + PEN_LENGTH) if USE_PEN_TCP else 0.0
