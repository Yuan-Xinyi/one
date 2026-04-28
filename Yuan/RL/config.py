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
INIT_IK_ORIENT_MODE = "z_axis"      # "z_axis" or "full_rot"
SEED_MANIFOLD_REG = True            # penalize raw q_seed off the init manifold
SEED_MANIFOLD_COEF = 0.25
SEED_POS_ERR_SCALE = 0.20           # meters; penalty saturates above this
SEED_ORIENT_ERR_SCALE = np.deg2rad(60.0)
ACTION_MODE = "branch_descriptor"   # "joint_seed" or "branch_descriptor"
BRANCH_ACTION_DIM = 4               # [cos(phi), sin(phi), cos(psi), sin(psi)]
BRANCH_SWIVEL_GAIN = 0.15           # null-space gain for elbow branch objective
BRANCH_FD_EPS = 1e-3                # finite-difference step for branch gradient
BRANCH_IK_NUM_STARTS = 9            # deterministic IK starts for branch projection

# ---------- Domain randomisation (training only; eval uses defaults) ----------
DR_ENABLE   = True
DR_V_PATH   = (0.10, 0.40)          # m/s
DR_EPS_POS  = (3e-3, 1e-2)          # m
DR_T        = (40, MAX_STEPS)       # int, inclusive
DR_N_TILT   = (0.0, np.pi)          # full-sphere plane-normal sampling
DR_P0_BOX_LO = np.array([-0.20, -0.75, 0.02], dtype=np.float32)
DR_P0_BOX_HI = np.array([ 0.85,  0.75, 0.85], dtype=np.float32)
DR_P0_RADIUS = (0.22, 0.88)         # coarse FR3 reachable shell from base
DR_SAMPLE_REACHABLE_ONLY = True     # reject tasks with no branch IK solution
DR_REACHABILITY_TRIES = 4000
DR_REACHABILITY_PHI_SAMPLES = 4
DR_REACHABILITY_PSI_SAMPLES = 2

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
BATCH_SIZE  = 96
N_ITERS     = 6000
GRAD_CLIP   = 1.0
SEED        = 0

# ---------- Entropy annealing ----------
ENT_COEF        = 3e-4                  # legacy (used if no annealing)
ENT_COEF_INIT   = 2e-2                  # exploration-heavy at start
ENT_COEF_FINAL  = 3e-4
ENT_ANNEAL_END  = 3000                  # iter at which decay finishes

# ---------- Logging ----------
LOG_EVERY   = 10
CKPT_EVERY  = 200
CKPT_DIR    = "Yuan/RL/checkpoints_v6_branch"
WANDB_ENABLE  = False
WANDB_PROJECT = "fr3-rl-branch"
WANDB_ENTITY  = None
WANDB_RUN_NAME = "v6_branch_mixture"
