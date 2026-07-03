"""r12: higher-quality MPC labels (K=32, hold_H=16) on soup2 visitation.

mpc_distill does the expensive collection+labeling (datasets checkpointed per
round); the r8-style merges with r6 clean labels happen in a follow-up script.
mpc_chunk=768 keeps the tiled search env at 768*32=24576 envs (GPU shared).
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.self_improve.mpc_distill import mpc_distill

stats = mpc_distill(
    "Yuan/RL_controller/runs/distill_r12_mpc32",
    behavior_ckpt="Yuan/RL_controller/runs/distill_soup2",
    pi0_ckpt="Yuan/RL_controller/runs/p0_progress_only_30M_0520",
    n_tasks=12288, dagger_rounds=1,
    tau_hi=0.975, band=0.02,
    K=32, hold_H=16, mpc_chunk=768,
    seed=8710, epochs=80)
print("[r12] stats:", stats, flush=True)
