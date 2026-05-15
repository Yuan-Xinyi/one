# FR3 Position-only NSRL Path-following

PPO controller for a 3-DOF position task on a Franka Research 3 (7-DOF non-SRS).
EE moves at constant `v · u_hat` along an infinite ray via DLS pseudo-inverse;
the 4-DOF nullspace action is the policy output. Posture is shaped entirely by
the 30° rotation cone hard-terminate vs `n_target`.

The agent's only objective is to maximize episode lifetime (no goal, no reward
shaping beyond `r_alive`). See [rules.md](rules.md) for the spec.

## Layout
```
env/
  env.py                 NSRLBatchedEnv: torch-batched env, 20-d obs, 4-d action
                         supports auto_reset=True (training) / False (eval)
  line_distribution.py   MC reachability sampler (q_0, u_hat, n_target)
  ik_init.py             SELIKSolver wrapper for external IK use; not on train hot path
  baseline_controller.py GPM nullspace baseline + rollout helper (auto_reset=False)
ppo.py                   cleanrl-style continuous PPO, truncation-correct bootstrap
train.py                 entry point — builds env + LineDistribution + runs PPO
eval.py                  200-line holdout, RL vs baseline T-ratio CSV
config.yaml              all hyperparameters
```

## Run
```bash
# Set conda lib path once per shell to avoid the libstdc++ ABI mismatch in `one`
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Train (writes runs/agent.pt and runs/train.log)
python -m Yuan.RL_controller.train --config Yuan/RL_controller/config.yaml

# Eval (writes runs/eval.csv)
python -m Yuan.RL_controller.eval \
    --config Yuan/RL_controller/config.yaml \
    --ckpt   Yuan/RL_controller/runs/agent.pt \
    --out    Yuan/RL_controller/runs/eval.csv
```

`--device` is auto-detected (cuda if available else cpu).

## Key design decisions

### Infinite-ray task
A "line" is `(p_0, u_hat, n_target)` with no length. There is no success
terminate, no goal distance, no progress; obs carries `u_hat` and `n_target`
but never anything position-relative. This avoids the policy learning a
spurious "approach-the-end" behaviour tied to the training-set length
distribution.

### Reuse over reimplementation
All forward kinematics, geometric Jacobian, link-frame transforms, joint limits,
and self-collision come from `one`:
- `one.robots.manipulators.franka.fr3_pen.batched_fr3_kin:BatchedFR3Kinematics`
- `one.robots.manipulators.franka.fr3.sphere_collision:FR3SphereCollision`
- `one.robots.manipulators.franka.fr3.fr3:FR3` + `ManipulatorBase.ik_tcp_nearest`

### Line sampling avoids per-reset IK
`LineDistribution` pre-samples `n_pool` random joint configs, FK's them, filters
self-collisions, and stores `(q, z_tool)`. At reset it draws an index, uses
the stored `q` as `q_0`, derives `n_target ≈ z_tool` with small angular noise,
and picks `u_hat` random ⊥ to `n_target`. No fresh IK at reset.

`env/ik_init.py` keeps a fresh-IK path for users who specify an arbitrary
`(p_target, n_target)`.

### auto_reset flag on `env.step`
- `auto_reset=True` (default, training): terminated envs are re-sampled
  in-place; `info["terminal_obs"]` snapshots the pre-reset obs for the PPO
  bootstrap branch.
- `auto_reset=False` (eval): finished envs freeze (state unchanged, zero
  reward, no new termination flags). Caller polls `env.done_persistent` to
  decide when to stop. This eliminates the race where eval's scripted line
  distribution would otherwise exhaust its cursor on auto-reset.

### Nullspace basis continuity
`align_nullspace_basis` SVDs `J_p` per env (batched), takes the last 4 columns
of V, then does Procrustes alignment to the previous `B`. On the first step of
each episode, a sign convention (first nonzero element positive per column)
gives a deterministic seed.

### Damping
Position-only DLS: `J_p^+ = J_p^T(J_p J_p^T + λ²I)^{-1}`. λ follows the
spec — constant `λ_0 = 0.05` above `σ_thr`, ramps to 0 via
`λ_0·√(1−(σ_min/σ_thr)²)` below. In practice σ_min stays well above 0.05 in
reachable configs.

### Truncation bootstrap (PPO)
Truncated episodes bootstrap `V(s_T)`; terminated episodes do not. The env
saves `info["terminal_obs"]` before auto-reset; the PPO GAE loop pulls
`V(terminal_obs)` for steps where `truncated & ¬terminated`.

### Reward / return magnitudes (estimate)
- `r_alive = 1.0` per step, all terminal penalties 0.
- Return per episode == episode length × 1.0.
- max_steps = 10000 = 100 s physical time; saturation return ≈ +10000.
- Typical untrained-policy episode is tens to a few hundred steps (cone /
  JL terminate fast).

## Open Questions
1. **MC pool size 10⁵**: gives roughly 10⁵ valid `(q, z)` samples (after
   self-collision filter). If reachability coverage looks holey in eval
   term-reason distributions, raise to 10⁶.
2. **Expected RL/baseline T-ratio**: baseline is intentionally weak (no
   posture term, cone is the dominant failure mode for it). RL should
   easily exceed 1; ratio 2–5× seems plausible. Empirical.
3. **Sample budget**: `n_envs × n_steps = 4096`/update, 244 updates @ 10
   epochs = 2440 gradient steps. PPO often wants more for continuous control.
   If learning curves are still improving at convergence, raise
   `total_timesteps` to 3–5e6.
4. **4-DOF action exploration**: `init_log_std = −0.5` (std ≈ 0.6) is the
   first-try value. If learning is flat, try `ent_coef = 1e-3` before
   raising init std (4-DOF nullspace has many flat reward directions; big
   init std wastes effort on those).
5. **Baseline strength**: pure GPM-JL has no `z_t` → `n_target` pull. If
   baseline collapses uniformly in < 50 steps, the ratio is uninformative.
   A stronger reference (GPM-JL + Frobenius posture term) would tighten the
   comparison.

## Known Issues / 待验证

1. **`SELIKSolver` is not natively position-only**. `env/ik_init.py` works
   around it by passing a 6-DOF target rotmat with `z_tool ← n_target` and
   varying twist seeds (10 retries) about that axis. The post-check rejects
   solutions that miss tolerance. A clean fix would modify
   `NumIKSolver._backward` to drop `delta_theta`; not done.

2. **`SELIKSolver` CVT database build is heavy**. First-time use of
   `ik_init.solve_q0` triggers a 200k-sample kmeans build at the data dir
   (minutes). Training does NOT trigger this — `LineDistribution` avoids
   fresh IK.

3. **`B_prev` semantics on auto-reset**. After an env resets mid-batch, its
   `B_prev_valid = False`, so the next step uses the sign-convention seed.
   Single-step discontinuity. No effect on current reward (no smoothness term).

4. **PPO rollout-boundary truncation**. The PPO GAE loop bootstraps
   `V(terminal_obs)` for within-rollout truncations. At the boundary step
   (`t == n_steps − 1`), `next_obs` is already the post-auto-reset obs of
   the next episode; if that boundary step is truncated-only, V is biased
   for that single step. With `max_steps = 10000` and typical episode length
   « 10000, truncations are rare; magnitude is negligible.

5. **`tcp_offset = 0.0`** matches the scalar `FR3()` class's identity TCP
   (bare flange). Switching to a pen tip requires also setting the
   corresponding `_loc_tcp_tf` on the scalar FR3 instance, otherwise
   `ik_init.py` and the batched env will disagree on TCP position.

6. **No obs / reward normalization**. PPO often benefits from these.
   Skipped per spec ("先把上述跑通"). Add later if learning is unstable.

7. **`actor_logstd` is state-independent**. cleanrl default. State-dependent
   log-std could help exploration in promising regions; not implemented.

8. **`one` library matplotlib ABI mismatch**. Setting
   `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` works around it. Required for both
   train and eval. Documented in the Run section above.

9. **`np.random.shuffle` for mini-batch indices**. Uses global numpy RNG;
   PPO mini-batch order is not strictly reproducible across runs.
