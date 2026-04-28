"""Hyper-parameters for the farsighted-seed RL pipeline."""
import numpy as np


# ---------- Task / rollout (defaults; randomised at training time) ----------
DT          = 0.02                  # control period [s] (50 Hz)
V_PATH      = 0.25                  # default path linear velocity [m/s]
MAX_STEPS   = 80                    # absolute upper bound on T per episode
EPS_POS     = 5e-3                  # default position tracking tolerance [m]
EPS_POS_INIT= 5e-3                  # tolerance for the seed->p0 IK projection
THETA_MAX   = np.deg2rad(5.0)       # z-axis vs plane-normal tolerance [rad]
PATH_STEP   = V_PATH * DT           # 0.005 m, kept for backward compat

# ---------- Domain randomisation (training only; eval uses defaults) ----------
DR_ENABLE   = True
DR_V_PATH   = (0.10, 0.40)          # m/s
DR_EPS_POS  = (3e-3, 1e-2)          # m
DR_T        = (40, MAX_STEPS)       # int, inclusive
DR_N_TILT   = (0.0, np.deg2rad(60.0))  # widen vs eval (45 deg)
DR_P0_BOX_LO = np.array([0.30, -0.30, 0.10], dtype=np.float32)
DR_P0_BOX_HI = np.array([0.60,  0.30, 0.55], dtype=np.float32)

# ---------- Controller (DLS Cartesian) ----------
KP_LIN      = 5.0                   # position-error feedback gain [1/s]
KOMEGA      = 5.0                   # orientation-error feedback gain [1/s]
DLS_LAMBDA  = 0.05                  # DLS damping
K_NULL      = 0.5                   # null-space gain (pull toward q_ref)
# FR3 datasheet joint-velocity limits (rad/s):
#   joints 1-4: 150 deg/s, joints 5-7: 180 deg/s
QDOT_MAX    = np.array([2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14],
                       dtype=np.float32)

# ---------- Self-collision ----------
USE_COLLISION_CHECK = True

# ---------- Batched rollout ----------
BATCHED_ROLLOUT = True              # GPU/torch batch rollout for training
BATCHED_ROLLOUT_DEVICE = "auto"     # "auto", "cuda", or "cpu"
BATCHED_COLLISION_CHECK = True      # sphere self-collision in batched rollout
BATCHED_COLLISION_MARGIN = 0.0
BATCHED_IK_MAX_ITERS = 50
BATCHED_IK_DAMPING   = 1e-4
BATCHED_IK_TOL_POS   = 1e-4
BATCHED_IK_TOL_ROT   = 1e-3

# ---------- Workspace sampling (FR3 base frame) ----------
P0_BOX_LO   = np.array([0.30, -0.30, 0.20], dtype=np.float32)
P0_BOX_HI   = np.array([0.60,  0.30, 0.55], dtype=np.float32)
N_TILT_MAX  = np.deg2rad(45.0)      # plane-normal tilt off world +z

# ---------- Network ----------
# state layout: 9 (p0,d,n) + 3 (v_path, eps_p, T_norm) + 2 (fk-aug) = 14
STATE_DIM    = 14
RAW_C_DIM    = 9                    # [p0, d, n] portion
TASK_PARAM_DIM = 3                  # v_path, eps_p, T_norm
FK_AUG_DIM   = 2                    # dist_home_p0, angle_z_home_n
HIDDEN_DIM   = 256
POLICY_TYPE  = "mixture"            # "gaussian" or "mixture"
MIXTURE_COMPONENTS = 8
LOG_STD_INIT = -1.0                 # ~ exp(-1)=0.37 rad
LOG_STD_MIN  = -5.0
LOG_STD_MAX  = 1.0
STATE_DEP_LOG_STD = True            # if False, fall back to a free Parameter

# ---------- PPO ----------
PPO_EPOCHS    = 8
PPO_CLIP      = 0.2
PPO_TARGET_KL = 0.05
MINIBATCH     = 32

# ---------- Optim ----------
LR_PI       = 3e-4
LR_V        = 1e-3
BATCH_SIZE  = 64
N_ITERS     = 3000
GRAD_CLIP   = 1.0
SEED        = 0

# ---------- Entropy annealing ----------
ENT_COEF        = 3e-4                  # legacy (used if no annealing)
ENT_COEF_INIT   = 2e-2                  # exploration-heavy at start
ENT_COEF_FINAL  = 3e-4
ENT_ANNEAL_END  = 1500                  # iter at which decay finishes

# ---------- Logging ----------
LOG_EVERY   = 10
CKPT_EVERY  = 100
CKPT_DIR    = "Yuan/RL/checkpoints_v5_mixture"
