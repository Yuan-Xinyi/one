"""Hyper-parameters for the farsighted-seed RL pipeline."""
import numpy as np


# ---------- Task / rollout (defaults; randomised at training time) ----------
DT          = 0.02                  # control period [s] (50 Hz)
V_PATH      = 0.25                  # default path linear velocity [m/s]
MAX_STEPS   = 240                   # absolute upper bound on T per episode
                                    #   = 1.2 m max path at default v=0.25 m/s
                                    #   covers most of FR3's 1.7 m diameter workspace
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
BRANCH_IK_NUM_STARTS = 16           # deterministic IK starts (more = better
                                    # oracle ceiling on hard, full-sphere tasks)

# ---------- Domain randomisation (training only; eval uses defaults) ----------
DR_ENABLE   = True
# Task sampling mode (v10):
#   "random_q":     sample random q in (contracted) joint limits, FK -> (p0, n).
#                   Guaranteed reachable; no need for reachability rejection.
#                   d is then sampled uniform in the plane perp to n.
#   "workspace":    legacy — sample p0 from a box, n from a sphere, reject
#                   unreachable (p0, n) via batched-IK pre-check.
TASK_SAMPLE_MODE = "random_q"
RANDOM_Q_MARGIN  = 0.05               # stay this many rad off joint-limit edges
DR_V_PATH   = (0.10, 0.40)          # m/s
DR_EPS_POS  = (3e-3, 1e-2)          # m
DR_T        = (40, MAX_STEPS)       # int, inclusive  (40 -> ~0.2 m, 240 -> 1.2 m)
DR_N_TILT   = (0.0, np.pi)          # legacy field; unused — env now samples
                                     #   n uniformly on the FULL sphere via
                                     #   standard-normal + normalize.
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
NULL_USE_MANIPULABILITY = True      # null-space ascent of directional manipulability
NULL_MANIP_GAIN = 0.6
NULL_MANIP_DAMPING = 1e-3
NULL_JOINT_LIMIT_GAIN = 0.2         # push toward joint-limit center in null-space
NULL_ANGLE_GAIN = 0.4               # z-axis alignment pull near cone boundary
NULL_ANGLE_ATTRACT_GAIN = 0.0       # always-on z-axis attractor (0 keeps legacy feel)
NULL_ANGLE_MARGIN = np.deg2rad(8.0)
NULL_MANIP_FD_EPS = 1e-3            # serial-controller finite-difference step
# FR3 datasheet joint-velocity limits (rad/s):
#   joints 1-4: 150 deg/s, joints 5-7: 180 deg/s
QDOT_MAX    = np.array([2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14],
                       dtype=np.float32)

# ---------- End-effector geometry (FR3 hand + pen) ----------
# The TCP we control is the PEN TIP, not the bare-arm flange.
#   link7 → flange:        (0, 0, 0.107)              (FR3 fixed)
#   flange → hand center:  (0, 0, HAND_TCP_OFFSET)    (FR3Gripper grasptarget)
#   hand center → pen tip: (0, 0, PEN_LENGTH)         (rigid pen attached)
# Total link7 → TCP offset along z: 0.107 + HAND_TCP_OFFSET + PEN_LENGTH.
# BatchedFR3Kinematics ALREADY adds the 0.107; TCP_OFFSET adds the rest.
USE_PEN_TCP      = True
HAND_TCP_OFFSET  = 0.1034              # m, FR3 gripper acting center
PEN_LENGTH       = 0.10                # m, pen extension
TCP_OFFSET       = (HAND_TCP_OFFSET + PEN_LENGTH) if USE_PEN_TCP else 0.0

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
# Multi-start IK ranks 16 candidate q solutions by:
#   score = pos_err + rot_err + IK_SWIVEL_W * swivel_cost - IK_MARGIN_W * margin
# Joint-margin penalty pushes IK to choose solutions farther from joint
# limits, mitigating step-0 joint_limit terminations during rollout.
IK_SWIVEL_W = 0.05
IK_MARGIN_W = 0.20                  # 0 disables; 0.2 ≈ comparable to swivel

# ---------- Workspace sampling (FR3 base frame) ----------
P0_BOX_LO   = np.array([0.30, -0.30, 0.20], dtype=np.float32)
P0_BOX_HI   = np.array([0.60,  0.30, 0.55], dtype=np.float32)
N_TILT_MAX  = np.deg2rad(45.0)      # plane-normal tilt off world +z

# ---------- Network ----------
# v8 default: state_dim = 14 (no geom-aug). v9 added 3 geom-aug features
# (dist_p0_shoulder, reach_margin, n_dot_grav) + bumped MIXTURE to 8 +
# enabled UCB action selection — but eval showed -3.6 pp pol/orc regression,
# so we revert to v8 settings as the production config. v9 code paths in
# qnet.py / train.py / env._state_vec are kept but deactivated by these flags.
STATE_DIM    = 14
RAW_C_DIM    = 9                    # [p0, d, n] portion
TASK_PARAM_DIM = 3                  # v_path, eps_p, T_norm
FK_AUG_DIM   = 2                    # dist_home_p0, angle_z_home_n
GEOM_AUG_DIM = 3                    # (only added to state when STATE_DIM=17)
HIDDEN_DIM   = 256
POLICY_TYPE  = "flow"               # "gaussian" | "mixture" | "flow"
MIXTURE_COMPONENTS = 4              # only used if POLICY_TYPE == "mixture"
FLOW_LAYERS  = 4                    # number of conditional RealNVP coupling layers
LOG_STD_INIT = -1.0                 # ~ exp(-1)=0.37 rad
LOG_STD_MIN  = -5.0
LOG_STD_MAX  = 1.0
STATE_DEP_LOG_STD = True            # if False, fall back to a free Parameter

# ---------- Q ensemble + active sampling (v9, deactivated) ----------
Q_ENSEMBLE_M    = 5                 # number of bootstrap Q networks
ACTIVE_SAMPLING = False             # v9 ablation: -3.6 pp regression, off by default
ACTIVE_K        = 8                 # candidate actions per state when active
FR3_REACH_RADIUS = 0.855            # m, used for reach_margin feature
FR3_SHOULDER     = (0.0, 0.0, 0.333)  # used for dist_p0_shoulder feature

# ---------- PPO (legacy / unused in v8) ----------
PPO_EPOCHS    = 8
PPO_CLIP      = 0.3
PPO_TARGET_KL = 0.10
MINIBATCH     = 32

# ---------- SAC (v8) ----------
SAC_REPLAY_SIZE = 200_000           # FIFO buffer capacity (bigger to keep up with B=384)
SAC_BATCH       = 512               # Q / pi update minibatch from buffer
SAC_K_Q         = 8                 # Q updates per env iter (more, to consume the bigger inflow)
SAC_K_PI        = 2                 # policy updates per env iter
SAC_LR_Q        = 3e-4
SAC_LR_PI       = 3e-4
SAC_LR_ALPHA    = 3e-4
SAC_ALPHA_INIT  = 0.05              # initial entropy coefficient
SAC_TARGET_H    = -8.0              # tighter than -action_dim;
                                    # encourages policy to commit to a mode
                                    # (was -4.0, observed too-loose at v10c)
SAC_AUTO_ALPHA  = True              # learn alpha to hit target entropy
SAC_WARMUP_ROLLOUTS = 1024          # collect this many before training starts
SAC_ACTION_SAMPLES_PER_TASK = 8     # rollout K policy samples per task into replay
REWARD_USE_SAMPLED_ORACLE = True    # normalize each action by best of K actions on same task
REWARD_ORACLE_MIN_STEPS = 1.0       # avoid division by zero when every sampled branch fails
REWARD_FAIL_INIT_IK = 0.10
REWARD_FAIL_JOINT_LIMIT = 0.03
REWARD_FAIL_SELF_COLLISION = 0.05
REWARD_FAIL_ORIENT = 0.03
REWARD_FAIL_POS = 0.00              # keep position failures ranked by L/T
REWARD_CLIP_LO = -0.20
REWARD_CLIP_HI = 1.00

# ---------- Prioritized Experience Replay (PER, Schaul 2016) ----------
# Sample buffer entries by p_i ∝ |TD-error|^alpha, debias gradient by
# IS weight (1/(N*p_i))^beta. Beta anneals from PER_BETA -> PER_BETA_FINAL
# linearly over the first PER_BETA_ANNEAL_END iters.
PER_ENABLE          = True
PER_ALPHA           = 0.6
PER_BETA            = 0.4
PER_BETA_FINAL      = 1.0
PER_BETA_ANNEAL_END = 5000
PER_EPS             = 1e-3

# ---------- Optim ----------
LR_PI       = 3e-4
LR_V        = 1e-3
BATCH_SIZE  = 128                   # K=8 -> 1024 rollouts/iter; better wall-clock feedback
N_ITERS     = 5000
GRAD_CLIP   = 1.0
SEED        = 0

# ---------- Entropy annealing ----------
ENT_COEF        = 5e-5                  # legacy (used if no annealing)
ENT_COEF_INIT   = 2e-2                  # exploration-heavy at start
ENT_COEF_FINAL  = 5e-5                  # 6x stronger decay than v6
ENT_ANNEAL_END  = 1000                  # iter at which decay finishes

# ---------- Logging ----------
LOG_EVERY   = 10
CKPT_EVERY  = 200
CKPT_DIR    = "Yuan/RL/checkpoints_v11b_sampled_oracle_k8"
WANDB_ENABLE  = False
WANDB_PROJECT = "fr3-rl-branch"
WANDB_ENTITY  = None
WANDB_RUN_NAME = "v11b_sac_flow_sampled_oracle_k8"
