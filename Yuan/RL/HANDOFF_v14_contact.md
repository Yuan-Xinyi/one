# v14 contact-mode handoff (overnight 2026-05-04 → 05)

## TL;DR

- Built a contact-aware rollout in `batched_rollout.py:batched_rollout_contact`.
  - **v1 (default ON when `use_dynamics=False`)**: linear-spring static surface.
  - **v2 (opt-in via `use_dynamics=True`)**: 1-DOF mass-spring at pen tip
    along surface normal — adds tooling/sensor compliance + dynamics.
- Both v1 and v2 collapse phantom dominance: gap to oracle jumps from
  **~0.5pp (geo)** to **16-31pp (contact) at K=8/16**. RL has room to grow.
- **Honest finding on v2**: the dynamics machinery is correct, but in this
  task formulation it does NOT add ON TOP OF v1's tighter tolerance — the
  failure mode is dominated by joint-limit-driven steady-state pos_err,
  and lightly-damped mass-spring actually *low-pass-filters* tracking
  spikes. So v2 ≤ v1 phantom-gap on the same window. To make dynamics the
  load-bearing element you'd need sharper forcing (impact, resonance,
  stick-slip, bandwidth-limited control). Details in §"v2 honest finding".
- Did NOT touch train.py. Did NOT retrain. Decisions for you below.

## What changed (5 files)

| file | change |
|---|---|
| `config.py` | v1 knobs: `USE_CONTACT_MODE`, `CONTACT_K_N`, `CONTACT_PENETRATION_TARGET`, `CONTACT_PEN_MIN/MAX`, `CONTACT_GRACE_STEPS`. v2 knobs (opt-in): `CONTACT_USE_DYNAMICS`, `CONTACT_TIP_MASS`, `CONTACT_GRIP_K`, `CONTACT_GRIP_C`, `CONTACT_N_SUBSTEPS` |
| `batched_rollout.py` | added `batched_rollout_contact()`. Optional `use_dynamics=True` activates 1-DOF mass-spring at pen tip integrated with explicit-Euler substeps (default 20 sub per control step, dt_sub=1ms) |
| `env.py` | added `contact_mode` flag to `FarsightedSeedEnv.__init__`; `collect_batch` dispatches to `batched_rollout_contact` when True. Defaults to `cfg.USE_CONTACT_MODE` (currently False) |
| `eval_contact_compare.py` | NEW smoke test with `--use-dynamics`, `--tip-mass`, `--grip-k`, `--grip-c`, `--n-substeps` flags |
| (`phantom_eval.py` was earlier today's work — unchanged) |

## Contact model design (v1)

- Surface = plane through `p0` with outward normal `n` (already in `c[6:9]`).
- Pen tip target = `p0 + step·dt·v·d - pen_target·n`  (offset INTO surface).
- After IK: `penetration = max(0, -(p_tcp - p0)·n)`,
  `F = K_n · penetration` (linear spring, no damping yet).
- Failure modes (all in addition to existing ones):
  - `force_high`: `pen > pen_max` after grace steps
  - `force_low` : `pen < pen_min` after grace steps (lost contact)
  - `pos_err`: ONLY tangential to `n` — normal direction is the spring's job
  - `joint_limit`, `orient_err`, `self_collision`: unchanged
- `grace_steps = 10` (200 ms) so transient settle into contact doesn't fail.
- Returns extra `last_force` per task for diagnostics.

## Smoke test result (verified)

```
contact knobs: K_n=2000  pen_target=5.0mm  F_target=10.0N  F_range=[7.0, 13.0]N
n_tasks=32, K=16

[L_top mean over well-def] geo=72.1 steps  con=53.7 steps  con/geo=0.745
[con vs geo] frac of K samples per task where con < geo: 0.552
[diag last-step force] mean=6.2N  med=6.3N  p10=0.0N  p90=13.0N

GEO (existing kinematic):
  K=8   unif_orc=0.965  phant_sel=0.960  gap=0.42pp   ← phantom matches oracle
  K=16  unif_orc=1.000  phant_sel=0.994  gap=0.58pp

CON (NEW contact):
  K=8   unif_orc=0.919  phant_sel=0.753  gap=16.56pp  ← phantom can't predict force
  K=16  unif_orc=1.000  phant_sel=0.804  gap=19.62pp
```

Speed: contact 5.2 ms/sample (≈ same as geo 5.8 ms), phantom 2.6 ms.

## v2 honest finding

I added a 1-DOF mass-spring at the pen tip (representing tooling +
sensor compliance + arm inertia projected to TCP). Equations per substep:

```
F_grip = K_grip · (z_kin - z_dyn) - C_grip · z_dot
F_surf = -K_n · max(z_dyn, 0)
z_ddot = (F_grip + F_surf) / m_tip       (integrated 20× per control step)
F_actual = K_n · max(z_dyn, 0)
```

with defaults `m=0.5kg, K_grip=20kN/m, C=20Ns/m → ω_n=200 rad/s, ζ=0.10`.

**Smoke test on n=64, K=16, tight ±3N window**:

| config | phantom gap K=8 | phantom gap K=16 |
|---|---|---|
| v1 static | **20.34** | **31.47** |
| v2 dyn ζ=0.10 | 16.60 | 19.62 |
| v2 dyn ζ=0.02 (light damping) | 18.13 | 28.18 |

**v2 actually has a SMALLER gap than v1**. Reason: my mass-spring with
reasonable damping low-pass-filters tracking-error spikes that v1
captures exactly. The "transient overshoot" mechanism I expected to
matter requires SHARP forcing, but the controller is slow (KP·dt = 0.1
per step) so z_kin ramps gradually and never excites the mass-spring.

So the 16-31pp phantom gap on contact tasks comes mostly from
**tighter normal-direction position tolerance** (encoded as force window),
not from genuine dynamics. v2 is the right machinery but v1's task design
already does the work.

**To make dynamics the LOAD-BEARING ELEMENT**, the task would need at
least one of:
1. **Impact event**: start TCP off-surface, drive into surface with non-
   zero approach velocity → mass-spring rings up.
2. **Resonant path**: path with frequency content matching ω_n
   (e.g. weave pattern in welding) → sustained oscillation excitation.
3. **Stick-slip friction**: tangential velocity → friction model needed,
   creates dynamics-only failure mode.
4. **Bandwidth-limited force control**: replace position controller with
   admittance + sensor delay; controller can't track step force changes.

For overnight delivery I left v1 as the default (`CONTACT_USE_DYNAMICS=False`)
because it produces the same phantom-failure effect with less complexity.
v2 stays in the codebase for future work where dynamics matters.

## Decisions for you to make

### 1. Force tracking window

Default I committed (`±3N around 10N`) is **tight enough to make contact a
real RL task**, but may be too tight for early training (force failures
dominate, no successful samples to learn from). Options:
- Keep tight `±3N` and let SAC's exploration find good actions.
- Curriculum: start `±10N` (easy), tighten over training to `±3N`.
- Loose `±10N` at default, only tighten at eval.

My take: try tight first; if training collapses, fall back to curriculum.

### 2. State / reward changes for training

Currently the policy state has `(p0, d, n, v_path, eps_p, T_norm, ...)`.
For contact, the policy can't observe force history (one-shot decision).
Two options:
- **Don't extend state**: policy still picks `(φ, ψ)` based on geometry,
  hoping that geometry alone determines whether contact will be safe.
  This is the simplest and consistent with current single-shot bandit framing.
- **Extend state**: add `pen_target`, `F_min`, `F_max` to state vector
  (3 new dims). Lets policy adapt to varying force budgets in DR.

For v1 I left state untouched. Recommend extending **only if** you DR over
contact params (otherwise they're constants in state, useless feature).

### 3. Reward shaping for contact failures

Current `reward = length / T`. For contact:
- Same formula works (force failures terminate early → shorter L → lower r).
- OR add force-tracking-quality bonus during the rollout (continuous
  signal). I'd skip this for v1 — keep reward simple.

### 4. Should training switch to contact mode?

To retrain in contact mode:
```python
env = FarsightedSeedEnv(seed=0, randomize=True, contact_mode=True)
# OR set cfg.USE_CONTACT_MODE = True before constructing env
```
That's all train.py needs (env.collect_batch dispatches automatically).

CKPT_DIR/WANDB_RUN_NAME you'll want to bump to `v14_contact_*`.

### 5. Phantom rollout for contact?

The current `phantom_rollout` is geometric-only and **cannot** see force
failures. So phantom_select on contact tasks gets the 16.6pp gap shown
above. This is the WHOLE POINT — phantom no longer dominates → RL has
work to do.

If you want a phantom that's also force-aware (cheaper than full contact
rollout but predicts force), that's a follow-up. For now phantom stays
the geometric baseline that RL needs to beat.

## Suggested next session

1. **Sanity:** run `eval_contact_compare --n-tasks 200 --K 32` for tighter
   stats (~3 min).
2. **Train v14:** flip `cfg.USE_CONTACT_MODE = True`, bump
   `CKPT_DIR = "checkpoints_v14_contact_10k"`,
   `WANDB_RUN_NAME = "v14_contact_10k"`, run `python -m Yuan.RL.train`.
   ETA ~6 h with current `N_ITERS=10000`.
3. **Compare:** run a contact-aware version of `eval_heuristic_compare`
   (need to add — it currently only knows geo rollout). The eval should
   compare `policy_det / q_ranked / phantom_select / unif_oracle` on
   contact tasks and check if RL ratio actually beats phantom now.

## Open issues

- `batched_rollout_contact` drops normal-direction pos_err entirely. If
  IK puts TCP very far above the surface (no contact) AND tangential pos
  is fine, the rollout will not fail until force_low triggers after grace.
  Acceptable — `force_low` catches it. But worth noting.
- No friction. No damping. Pure normal spring. If that becomes limiting,
  add Coulomb friction (need tangential velocity → straightforward but
  ~50 more lines).
- Policy state doesn't see contact params; force budget changes between
  tasks would not be observable.

## Files I touched

```
M  Yuan/RL/config.py
M  Yuan/RL/batched_rollout.py
M  Yuan/RL/env.py
A  Yuan/RL/eval_contact_compare.py
A  Yuan/RL/HANDOFF_v14_contact.md   (this file)
```

`train.py`, `policy.py`, `qnet.py`, `eval_heuristic_compare.py` unchanged.

## Smoke test outputs

Saved to `/tmp/contact_smoke.out` (loose) and `/tmp/contact_smoke_tight.out`
(committed defaults). Re-running:

```
python -m Yuan.RL.eval_contact_compare --n-tasks 32 --K 16          # uses cfg defaults
python -m Yuan.RL.eval_contact_compare --n-tasks 32 --K 16 \
  --pen-target 0.005 --pen-min 0.0035 --pen-max 0.0065              # explicit
```
