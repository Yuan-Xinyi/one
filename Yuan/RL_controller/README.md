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

## Why directional manipulability

The reward includes a directional-manipulability term in **log form**:
`r_dm = w_dm · log(w_u(q))` where

$$w_{\hat u}(q) = \frac{1}{\sqrt{\hat u^T (J_p J_p^T + \lambda^2 I)^{-1} \hat u}}$$

**Versus scalar (Yoshikawa) manipulability $\sqrt{\det(J_p J_p^T)}$**: scalar
manipulability measures overall isotropy of the position Jacobian. Our task
only moves along $\hat u$, so what matters is how easily the arm can move
*in that direction*. $w_{\hat u}$ measures exactly that — the inverse of the
length of the velocity ellipsoid's projection onto $\hat u$. Choosing scalar
manipulability would let the agent pursue "high overall reach" configurations
that happen to be bad for $\hat u$ specifically.

**Why log form, not raw value**: at large $w_u$ the log derivative is small
(diminishing returns — don't reward parking in obviously-good configs); at
small $w_u$ (approaching singularity) the gradient grows like $1/w_u$, giving
strong "escape" signal exactly where it matters. The earlier telescoping
variant `Δw_u(q)` had similar anti-farming intent but was harder to interpret
and required NaN-on-reset bookkeeping.

**All four terms always-on, no gating**: alive + JL center attractor + cone
alignment + log-manipulability. Each is action-discriminating at every step,
mirroring the per-step gradient of the classical nullspace controller's
implicit objective. No gated penalties (those were a safety-net design with
limited learning signal away from the cone/JL boundaries).

**Strong vs weak baseline**: `baseline.k_dm` upgrades the GPM controller to
also include $k_{dm} \cdot B^T \nabla w_{\hat u}(q)$ in its nullspace command.
- `k_dm = 0` (weak): pure GPM-JL (q-mid pull only); ≈ 23% of init random.
- `k_dm > 0` (strong): GPM-JL + directional manipulability ascent; uses the
  same signal the RL agent gets.

The strong baseline is the fair upper-bound on what hand-tuned reward shaping
can give. If RL doesn't exceed it, we've shown PPO can't find anything beyond
what the gradient of the same scalar provides.

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
6. **`w_dirmanip = 1.0`** is same order as alive. If too large it might dominate
   reward, pushing actor to ignore other terminations. If too small, the
   dirmanip signal is drowned out by alive constant. Sweep range: 0.1–5.0.
7. **Log-transform vs delta vs raw `w_u`** — three formulations of the same
   underlying scalar. Worth an ablation: raw value alone causes farming;
   delta should fix that; log-transform would emphasize relative changes
   over absolute. Not done here.
8. **Strong baseline lifetime**: if `baseline.k_dm = 1.0` pushes GPM-JL above
   693 (current classical_nullspace mean), the paper story changes — the
   `693` number is not a hard ceiling for "what hand-tuning can achieve",
   and we should report the strong baseline as the upper-bound RL needs to
   beat.
9. **Multi-step lookahead $\tilde w_{\hat u}(q, N)$**: instead of single-step
   $w_u$, evaluate $w_u$ averaged over the next N steps of the predicted
   task trajectory. ~10× more compute per step, but gives actor a longer
   horizon for the same signal. Not implemented.

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

5. **`tcp_offset = 0.2034`** (pen tip) is the env's EE. The position task
   drives the pen tip along `v · u_hat`. Visualization (`visualize.py`) uses
   `make_fr3_with_pen(use_pen_tcp=True)` so scalar `FR3`'s `_loc_tcp_tf` is
   also at the pen tip; `gl_tcp_tf` then matches the env's EE 1:1. Switching
   to bare flange would require setting `tcp_offset=0.0` here AND
   `use_pen_tcp=False` in `visualize.py`.

6. **No obs / reward normalization**. PPO often benefits from these.
   Skipped per spec ("先把上述跑通"). Add later if learning is unstable.

7. **`actor_logstd` is state-independent**. cleanrl default. State-dependent
   log-std could help exploration in promising regions; not implemented.

8. **`one` library matplotlib ABI mismatch**. Setting
   `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` works around it. Required for both
   train and eval. Documented in the Run section above.

9. **`np.random.shuffle` for mini-batch indices**. Uses global numpy RNG;
   PPO mini-batch order is not strictly reproducible across runs.
