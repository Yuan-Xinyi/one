# Unified Seed-Control RL

## Goal

Treat seed selection and null-space control as two stages of one SMDP. The
initial joint configuration is the first macro-action of an episode, not an
offline preprocessing result. Both stages optimize the same downstream
progress return.

```text
task c
  -> feasible seed action set Q(c)
  -> i ~ pi_seed(i | c, Q), q0 = Q_i(c)
  -> a_t ~ pi_ctrl(a_t | o_t)
  -> G = sum_t gamma^t r_t
```

The first implementation keeps diffusion + Newton IK as a constraint-aware
action proposal layer. It unifies *selection* and control before attempting
the harder problem of directly generating feasible 7-DoF configurations.

## Relation to bidirectional policy chaining

The loop has the same causal structure as Sequential Dexterity:

- Forward initialization: the current seed policy induces the initial-state
  distribution used to train the controller.
- Backward credit: complete controller rollouts train a transition-feasibility
  critic `F(task, q0)` and the seed actor.
- Iteration: update one stage against a lagged snapshot of the other to avoid
  chasing a moving target.

No gradient through MuJoCo or IK is required. The learned feasibility value
transports downstream credit.

## Models

### Seed stage

`CandidateSeedActorCritic` consumes `(B,K,D)` candidate features plus a valid
mask. A permutation-equivariant set encoder produces:

- a categorical policy over valid candidates;
- an action-independent set value;
- `F(task,q_i)` for every candidate, supervised from downstream returns.

The default backward update is full-action Monte-Carlo GPI. With the
controller frozen, it evaluates every valid candidate and forms

```text
pi_ref = (1-epsilon) pi_old + epsilon Uniform(valid)
q*(i|c) proportional to pi_ref(i|c) exp(A(c,i) / eta)
```

`eta` is solved so `KL(q* || pi_ref)` meets a trust-region target. The actor
fits `q*`, while the feasibility head receives dense Huber and gain-weighted
ranking losses. The uniform reference floor lets a proven-good candidate
recover even if its old float32 probability underflowed. The original
four-sample contextual-bandit PPO remains available through
`--backward-mode sampled-ppo` as an ablation.

The KL target is an expected KL over tasks with at least two valid actions;
single-action tasks cannot change policy and therefore do not dilute the
budget. Q regression is reduced within task before averaging, and ranking is
averaged over informative tasks, so tasks with nine proposals do not receive
quadratically more weight than difficult tasks with two proposals. Mean and
maximum update KL are both logged; the configured limit is an expected-KL
early stop, not a per-task hard guarantee or rollback.

Historical frozen-controller experiments use its exact 31-D observation at
`t=0`. Joint training uses 34 dimensions by appending the lateral component of
`FK(q)-p0`, normalized by the lateral safety radius; this exposes the finite IK
residual that drives termination without adding unbounded along-ray progress.
A 31-D controller is lifted to 34-D by zero-extending both input layers (and
their Adam moments), so the migration preserves its outputs exactly.

The seed policy additionally uses log positional manipulability by default
(32-D frozen gate, 35-D joint seed input). Valid-only per-dimension mean/std
are fitted on the training split and stored as policy buffers, so evaluation
and controller reset sampling use the identical transformation. Both choices
are explicit checkpoint schema fields; `--no-log-manip` is an ablation.

The production frozen-controller selector extends this input to 45 dimensions
with ten controller-aligned directional features: the damped line-tracking
joint velocity, its norm, joint-limit travel horizon, and directional
manipulability. Five geometry-bootstrap members are trained on within-task
range-normalized complete progress labels and aggregated by mean member log
probability. Source fit/model-selection/final-calibration geometries are
disjoint; two additional exact-return caches contribute only to fit. The
checkpoint stores all fit geometry fingerprints, so external evaluation can
audit the union rather than only the original source cache.
New bidirectional runs can enable the same feature schema with
`--directional-dynamics`; backward updates and forward reset sampling share
the identical flag and normalization buffers.

### Control stage

The existing tanh-Gaussian PPO actor remains the continuous low-level policy.
The frozen-controller gate supports either pure RL or the historical
RL/classical hysteresis system.

### Controller-consistent seed lookahead

For deployment with a deterministic dynamics model, an optional second seed
decision stage probes `actor top-k union first-valid` under the frozen
controller. Each branch runs for `H` virtual steps. The score follows the
checkpoint selector objective; for the production progress selector it is
endpoint ray progress divided by `v*dt`, plus the calibration-locked survival
bonus. The physical/primary episode is then restarted once from the selected
`q0`. Virtual branch rewards never enter the episode return.

This is explicitly model-based planning, not free observation and not several
physical trial executions. Artifacts record branch-prefix state, selected
index, active virtual steps, selected full-execution steps, and their sum.
The evaluator also rolls the static policy, first-valid baseline, and complete
candidate oracle in the same primary batch. Lookahead is disabled by default;
`top-k=3,H=128,task-chunk=256` (branch batch 1,024) is the balanced profile
and `top-k=5,H=128,task-chunk=256` (branch batch 1,536) is the high-compute
profile. Task/branch batch is part of the locked protocol because the current
GPU SVD path can exhibit small batch-dependent floating-point differences.

### Residual seed stage

After the discrete gate passes, a small contextual-bandit head may refine the
selected candidate in the controller's four-dimensional task-aligned basis.
Its joint displacement is capped at 0.08 rad and projected back toward the
task origin. A deterministic shield checks joint limits, 5 mm position error,
the 30-degree tool cone, self-collision margin, and branch distance at fixed
scales `(1,.5,.25,.125,0)`. Zero/disabled actions preserve the discrete seed
bit-for-bit; missing collision evidence or any invalid shield result fails
closed before an environment reset.

## Training stages and gates

1. **Frozen-controller seed gate.** Train only `pi_seed` from complete
   controller returns. It must approach the frozen supervised ranker before
   introducing controller non-stationarity.
2. **Forward adaptation gate.** Freeze `pi_seed`, adapt the controller on its
   selected initial-state distribution, and verify no loss on the original
   distribution.
3. **Bidirectional loop.** Alternate controller rollout/update blocks with
   seed feasibility/actor blocks. Each learner sees a lagged peer snapshot.
4. **Direct seed generation.** Replace the candidate index with a diffusion
   latent or projected residual action only after the closed-loop hypothesis
   passes the first three gates.

The dense default uses 28 tasks x 9 actions, approximately matching the old
64 x 4 rollout budget while removing duplicate samples and unobserved-action
bias. Task comparisons remain within task, so intrinsic task difficulty does
not dominate the macro-policy update.

## Evaluation

Always compare under equal candidate and rollout budgets:

- historical diffusion + frozen ranker + frozen controller;
- RL seed + frozen controller;
- frozen seed + adapted controller;
- joint updates without backward feasibility;
- full alternating bidirectional optimization.

Primary metrics are paired on a task-geometry-grouped validation split and a
separately generated holdout whose `LineDistribution` pool seed is new.
Diagnostics include seed capture/top-1 regret, raw controller lifetime,
failure modes, initial-q diversity, switch count, and perturbation robustness.

Oracle capture is reported twice. `metric_seed_return_capture` uses the exact
discounted or undiscounted macro return stored in the checkpoint and is the
optimization gate. `metric_progress_capture` independently uses final net
ray progress as a deployment diagnostic. Incremental clipped progress reward
and final net displacement are correlated but not strictly order-equivalent,
so they must not share one unnamed oracle.

The first *sampled-PPO* frozen-controller experiment reached 28.5% of
available oracle headroom after 500 updates (complete undiscounted deployment
return). This is now a baseline for dense GPI, not a passed gate. A fixed 50%
threshold is not assumed: historical numbers used different pure/hybrid
controllers. The matched-controller dense baseline should set the gate, with
oracle regret and worse-than-first rate reported alongside capture.

With the same frozen pure controller, log-manipulability feature, valid-only
normalization, and approximately equal episodes per update, version-3 dense
GPI reached 30.9% progress capture and 30.8% exact undiscounted-return capture
after 300 updates. Mean progress was 0.5204 m versus 0.4948 m for first-valid
and 0.5778 m for the complete-candidate progress oracle. At 100 updates its
progress capture was already 28.1%, approximately the 28.5% reached by sampled
PPO at 500 updates. A later audit found that this version-3 result used a
row-heldout split: the finite task pool was sampled with replacement, and
18.6% of validation rows had the same geometry as a training row. It remains
a sample-efficiency diagnostic, not a generalization claim.

Version 4 groups exact float32 `(p0,line_dir,n_target)` task signatures before
splitting. It retains 18,432 training and 2,048 validation rows with zero task
overlap. On this corrected Gate-100 protocol, the default mean-set encoder
reached 26.6% progress capture and 26.4% exact-return capture. An approximately
parameter-matched one-layer attention encoder (H=168, 404,211 parameters versus
mean H=256, 404,227 parameters) reached 22.5% and 22.4%; attention therefore
remains an ablation rather than the main model.

The first reset-distribution ablation held the 250k-step controller budget and
70/20/10 mixture fixed. Sampling the seed policy improved all-valid mean
progress by 1.82 mm over the initial controller; using its deployed argmax
improved it by 2.23 mm. Argmax versus sampling was only +0.42 mm on that
coverage metric and its paired interval crossed zero, so sampling remains the
default until replicated.

An equal-budget cadence ablation then compared one coarse
`250k controller + 50 seed` round against five `50k + 10` alternations. The
fine schedule reduced selected-policy progress by 4.92 mm (paired 95% interval
`[-9.48,-0.52]` mm) while leaving all-valid controller coverage unchanged.
Moving seed targets therefore outweighed any benefit from faster cycling, and
coarse frozen phases remain the default. Continuing the coarse run peaked at
round 2: 0.5172 m validation progress and 33.0% oracle headroom capture. Later
rounds were non-monotonic, so immutable validation snapshots and early model
selection are required.

That round-2 snapshot was evaluated once on a new 2,048-task development
holdout sampled without replacement from a new 100k line pool. The builder
verified zero exact geometry overlap against `rank_train`, `rank_train_b`,
`rank_train_c`, the 10k systematic evaluation set, and the supplied historical
holdout caches. It achieved 0.5301 m versus 0.4991 m for first-valid and
0.5885 m for the complete-candidate oracle: 34.6% progress capture and 34.5%
exact-return capture. The paired progress gain was 30.93 mm with a bootstrap
95% interval `[22.14,39.79]` mm. These seeds are explicitly development
defaults, not the sealed final-test seeds.

The same round-2 coarse protocol was repeated with training/split seeds
12,000, 13,000, and 14,000. Their validation progress captures were 24.5%,
33.0%, and 31.3% (mean 29.6%, seed-level standard deviation 4.5 percentage
points). Geometry-grouped progress gains over first-valid were 22.9, 27.5,
and 27.8 mm (mean 26.1 mm, standard deviation 2.7 mm); every run's grouped
bootstrap interval was strictly positive. This supports the schedule rather
than only its best random seed. Validation splits differ by run and come from
the same finite source pool, so these three values are a seed-robustness
summary rather than three independent final test sets.

The 45-D five-member selector was then trained without reading the version-4
validation split. On the 2,048-row grouped validation set, its paired static
policy reached 0.5385 m versus 0.4902 m for first-valid and 0.5716 m for the
complete-candidate oracle: 59.36% progress capture (60.40% geometry-macro).
The locked `top3/H128` virtual probe reached 0.5626 m, 89.00% capture (89.50%
geometry-macro), with 180.4 virtual branch steps plus 56.6 selected execution
steps per task. `top5/H128` reached 0.5659 m and 92.98% capture (93.42%
geometry-macro) at 319.9 active controller steps per task. Its
geometry-bootstrap 95% capture interval was `[91.16%,95.41%]`.

The same locked profiles were evaluated on the independent 2,048-task
development holdout after auditing zero geometry overlap against every
ensemble fit source. Static capture was 58.99%; `top3/H128` reached 89.18%
and `top5/H128` reached 95.05%, with total costs of 244.9 and 330.5 active
controller steps per task respectively. Dense GPU kernel work also depends on the
recorded branch batch/chunk because terminated lanes remain in the tensor
batch. The high-compute profile's capture interval
was `[93.47%,96.38%]`. These are extra-model-planning results, not equal-step
comparisons: the paired artifacts retain static results for the no-lookahead
operating point.

The first residual R-stage did not pass its promotion gate. With the discrete
selector and controller frozen, 100 updates at exploration standard deviation
0.05 produced a shielded deterministic-mean diagnostic of only +1.30 mm
progress (geometry-grouped 95% interval `[-1.19,+3.87]` mm); the exact-return
interval also crossed zero. All shield outputs were valid, active alpha-zero
fallback was 0.34%, and median alpha was 1.0, so the limitation was benefit
rather than feasibility. The learned deployment gate stayed closed, preserving
the discrete baseline exactly. The residual implementation remains an
ablation, and the more expensive all-candidate joint H-stage is not entered
without positive paired evidence.

`--seed-return discounted` makes the macro return exactly match the low-level
PPO gamma. The default `undiscounted` option instead targets full deterministic
deployment lifetime, which performed better in the initial gate but is a
bilevel deployment surrogate rather than literally the same discounted SMDP
objective. Both choices are checkpointed and cannot be changed on resume.

## Operational guarantees

- Candidate masks accept the historical `ik_ok` and `ok` schemas, but a cache
  without an explicit mask is rejected.
- The appended pilot/native action carries explicit fallback metadata. Caches
  without a pilot use first-valid as their deterministic fallback rather than
  treating the last slot as a special action; uniform-valid exploration stays
  a separate reset-mixture component.
- Failed-IK NaNs are sanitized before batched FK and never reach a masked set
  encoder. Strict physical validation removes tasks with no feasible action
  while retaining their source-cache indices.
- Seed and controller peers are frozen within each optimization phase. A
  checkpoint is atomically written after the controller phase and after the
  full round, so resume never repeats a completed forward update or replaces
  the last good file with a partial serialization.
- Residual checkpoints use schema v2: the architecture, shield, bandit, and
  controller gamma are bound into provenance. Historical v1 residuals remain
  evaluation-compatible but are intentionally not resumable under v2.
- Resume state includes both optimizers, reward normalization, local sampling
  generators, NumPy/torch/CUDA RNGs, split source indices, and SHA-256
  provenance for candidate/controller artifacts.
- Version-4 checkpoints bind the grouped split mode, seed architecture, reset
  semantics, derived validity masks, effective controller configuration, and
  paired final controller state.
  Evaluation rejects a changed action set, `agent.pt`, or `config.yaml` unless
  an explicit controller-mismatch ablation is requested.
- Bidirectional runs keep immutable, self-contained snapshots after warmup,
  controller, and backward phases. `--branch-from` starts a matched reset
  ablation from a phase boundary while preserving models, optimizers, scaler,
  and all RNG states; other provenance changes are rejected.
- External evaluation reconstructs the recorded training cache and rejects
  exact task-geometry overlap by default. Historical `fresh_holdout_final`
  overlaps `rank_train` on 435/2,048 rows and is transfer-only. The dedicated
  builder samples without replacement from a new pool and audits all supplied
  training caches before writing its artifact.
- Ensemble checkpoints bind the exact member aggregation, 45-D feature schema,
  controller/config hashes, objective, split indices, and every fit geometry.
  Probe artifacts additionally bind top-k, horizon, score, alive bonus,
  branch batch size, restart semantics, and explicit controller-step costs.

Checkpointing is currently exact at warmup/controller/backward phase
boundaries. An interruption inside a million-step controller phase can still
lose that in-progress phase's compute; it cannot silently combine incompatible
states or repeat a phase already marked complete. Finer update-boundary
checkpointing is an engineering follow-up for long production runs.

## Non-negotiable invariants

- Invalid IK candidates are action-masked and can never be selected.
- `env.p_start` is the task-defined ray origin, not approximate `FK(q0)`.
- Training tasks and all model-selection tasks remain disjoint from final
  evaluation sets.
- Existing baseline artifacts and code paths are not mutated.
