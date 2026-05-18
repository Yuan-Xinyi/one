# FR3 Position-only NSRL Path-following

PPO controller for FR3 (7-DOF). EE moves at constant `v · u_hat` along an
infinite ray via DLS pseudo-inverse; the 4-DOF nullspace action is the
policy output. Posture is shaped by the 30° rotation-cone hard-terminate
against `n_target`. Episode lifetime is the sole objective.
See [rules.md](rules.md) for the math spec.

## Layout
```
env/env.py                  NSRLBatchedEnv: torch-batched, 31-d obs, 4-d action
env/line_distribution.py    MC reachability sampler + feasibility filter
env/baseline_controller.py  GPM nullspace baseline + rollout helper
env/classical_nullspace.py  4-term hand-tuned NS (used for feasibility filter)
ppo.py                      cleanrl-style continuous PPO
train.py / eval.py / visualize.py    entry points
tests/test_reward.py        reward-shaping unit tests
config.yaml                 all hyperparameters
runs21/                     current main-line ckpt + logs
```

## Run
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m Yuan.RL_controller.train --config Yuan/RL_controller/config.yaml \
    --out-dir Yuan/RL_controller/runs21 --wandb
python -m Yuan.RL_controller.eval --config Yuan/RL_controller/config.yaml \
    --ckpt Yuan/RL_controller/runs21/agent.pt
python -m Yuan.RL_controller.visualize --config Yuan/RL_controller/config.yaml \
    --controller rl --ckpt Yuan/RL_controller/runs21/agent.pt
```

## Key design
- **Infinite-ray task** — no length, no success terminate, no
  position-relative obs. Avoids policy coupling to training-set lengths.
- **Reuse `one`** — `batched_fr3_kin.BatchedFR3Kinematics` for FK/Jacobian;
  `sphere_collision.FR3SphereCollision` for self-collision.
- **LineDistribution** pre-samples collision-free `(q, z_tool)` pairs at
  init; reset draws an index (no per-reset IK). Optional feasibility
  filter drops lines the classical controller can't survive 10 cm on.
- **auto_reset** flag on `env.step`: True for training
  (`info["terminal_obs"]` for PPO truncation bootstrap), False for eval
  (finished envs freeze).
- **Nullspace basis** continuity — Procrustes alignment of `V[:, -4:]` from
  `J_p` SVD against `B_prev`; first step uses a sign-convention seed.
- **Tanh-squashed action**, state-dep `log σ` clamped `[-5, 0]`; without
  the squash, μ grows unbounded and actions saturate.
- **Reward** — progress + telescoping JL/cone/dirmanip deltas (N-step
  lookback), weights renormalized to sum=1; running-std return scaling.
