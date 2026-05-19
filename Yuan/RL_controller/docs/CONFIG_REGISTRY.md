# FR3 NSRL Config Registry

**Last updated**: 2026-05-19
**Current canonical**: v4 (framing B + task-space PD feedback)
**Purpose**: Single source of truth for milestone-level hyperparam and component state. When `config.yaml` / `ppo.py` / `env.py` disagree with this file, **the registry wins** — fix the code, then update the entry.

## Workflow

- **Before a new experiment**: add an entry with `status: planned`. List config diff vs current canonical. Verify `config.yaml` / `ppo.py` / `env.py` match the entry before launching.
- **After the experiment**: set status (`current canonical` / `failed` / `deprecated`), fill in commit hash, ckpt path, baseline numbers.
- **On milestone switch**: flip old canonical → `deprecated`, new entry → `current canonical`, update the header above.
- **BO sweep trials are NOT registered.** Registry is milestone-granularity only.

## Milestone Index

- [v4 (current canonical)](#v4) — framing B + task-space PD feedback
- [v3 (deprecated)](#v3) — framing B initial, lateral in reward
- [5b+iss3 (historical)](#5b-iss3) — framing A reference (gauge fix + state-indep log_std)
- [5b only (historical)](#5b) — gauge fix only, state-dep log_std still active
- [old plateau (problem case)](#old) — pre-fix, RL worse than no-op

---

## v4 {#v4}

**Status**: current canonical
**Commit**: `f52bcafd673dadab7fff439551636278f8b89950`
**Created**: 2026-05-19

### Configuration

#### `config.yaml`
```yaml
env:
  dt: 0.05
  v: 0.2
  a_max: 2.5
  cone_deg: 30.0
  max_steps: 500
  tcp_offset: 0.2034
  w_progress: 0.6
  w_jl: 0.2
  w_cone: 0.1
  w_dm: 0.1
  delta_scale: 100.0

ppo:
  total_timesteps: 30000000
  learning_rate: 3.0e-4
  n_steps: 32
  n_envs: 128
  n_minibatches: 32
  update_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  ent_coef: 0.01
  ent_coef_floor: 0.01           # v4: hold floor at 1% (was 0.0)
  ent_coef_anneal_frac: 0.7      # v4: anneal over first 70% (was 1.0)
  anneal_ent_coef: true
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.02
  hidden_dim: 512
  init_log_std: -1.0             # v4: was -0.5
  normalize_returns: true
```

#### `ppo.py` (Agent class)
- `LOG_STD_MIN = -2.5`  (v4: was -2.0; floor σ ≈ 0.082)
- `LOG_STD_MAX = 0.5`   (σ ≤ ≈1.65 safety cap; nn.Parameter rarely saturates)
- state-independent `log_std = nn.Parameter(torch.full((act_dim,), init_log_std))` — iss3 fix is in effect

#### `env.py`
- `LATERAL_SAFETY_NET = 0.02`   (20 mm, hard terminate only as safety net; not an active learning constraint)
- `K_P = 5.0`                   (task-space PD pull-back gain, in `step()`)
- Reward: no lateral term (`W_LAT = 0` semantically — the constant is removed entirely from `env.py`; lateral is not in the reward at all)
- Reward shaping: **clamped delta** (issue 2 signed-delta rollback maintained)

### Component status (current canonical)

| Component | State | Note |
|---|---|---|
| 5b task-aligned Gram-Schmidt basis | ENABLED | fp64 SVD + twice-Gram-Schmidt; replaces raw SVD V[:, -4:] |
| iss3 state-indep `log_std` | ENABLED | `nn.Parameter`, init -1.0, clamp [-2.5, 0.5] |
| NH adaptive damping formula | ENABLED | corrected; stays away from singular λ=0 |
| Task-space PD feedback (K_P=5) | ENABLED | lateral correction is the controller's job |
| Lateral termination | SAFETY NET (20mm) | not an active constraint; PD keeps actuals ≪ this |
| Lateral in reward (`W_LAT`) | DISABLED | removed from reward formula entirely |
| Signed delta (issue 2) | DISABLED | rolled back to clamped delta |
| Mixture policy | DISABLED | intentionally abandoned, do not revert |
| `joint_seed` mode | DISABLED | intentionally abandoned |
| v9 active sampling | DISABLED | intentionally abandoned |
| geom-aug state | DISABLED | intentionally abandoned |

### Ckpts

- `Yuan/RL_controller/runs/framing_b_pd_smoke_v4/agent.pt` — 3M smoke
- `Yuan/RL_controller/runs/framing_b_pd_10M_v4/agent.pt` — 10M

### Baseline numbers (200-line holdout, seed=42, v4 controller setup)

| Controller | L_mean (m) | L_median (m) | max_lat (mm) | lat>5mm | lat>20mm |
|---|---|---|---|---|---|
| zero (a≡0) | 0.306 | 0.300 | 24.2 | 30/200 | 22/200 |
| classical (GPM-JL) | 0.321 | 0.310 | 24.0 | 30/200 | 23/200 |
| RL_v4 (3M smoke) | 0.395 | 0.342 | 25.5 | 150/200 | 30/200 |

### Per-line ratios

- L_RL / L_zero (mean):       1.368
- L_RL / L_zero (median):     1.111
- L_RL / L_classical (mean):  1.233

### Key findings

- All 4 nullspace dims active; per-dim |a| ≈ 0.348 / 0.525 / 0.771 / 0.454
- dim 2 (JL channel) frequently saturates at ±1
- policy does **not** see lateral state; PD controller hard-guarantees it
- 10M run ckpt exists at `framing_b_pd_10M_v4/agent.pt` — ratios pending re-eval

---

## v3 {#v3}

**Status**: deprecated (superseded by v4)
**Commit**: `3fcb2d2` (approx — predates v4 PD feedback)
**Created**: ~2026-05 (early framing B work)

### Configuration diff vs v4

- `LATERAL_TOLERANCE = 0.005`   (5 mm hard termination, replaced by 20 mm safety net in v4)
- `W_LAT = 0.1`                 (normalized quadratic lateral penalty in reward — removed in v4)
- No task-space PD feedback     (lateral was an active learning constraint, not a controller-enforced one)

### Component status
- 5b basis: ENABLED
- iss3 state-indep log_std: ENABLED
- Lateral termination: HARD (5 mm)
- Lateral in reward: ENABLED (W_LAT=0.1, quadratic)
- Task-space PD feedback: DISABLED

### Ckpts
- `Yuan/RL_controller/runs/framing_b_smoke_v3/agent.pt`

### Baseline numbers (v3 setup)
- L_zero  = 0.292 m
- L_RL_v3 = 0.296 m
- L_RL / L_zero ≈ 1.01

### Key findings
- Policy learned **trivial+ε** strategy (single bias on dim 2)
- Only matches zero baseline — lateral cap suppressed all useful exploration
- Motivated v4 redesign: move lateral correction out of policy's reward into a PD controller

---

## 5b+iss3 {#5b-iss3}

**Status**: historical (framing A reference)
**Commit**: predates current branch (`short_5b_iss3_*` runs)
**Created**: 2026-05-18

### Configuration diff vs v4

- Framing **A** reward (no lateral concept at all; classical 4-term shaping)
- No PD feedback; no lateral safety net
- `init_log_std = -0.5`
- `ent_coef = 0.003`, `ent_coef_floor = 1.0e-4`, `ent_coef_anneal_frac = 0.3`

### Component status
- 5b basis: ENABLED
- iss3 state-indep log_std: ENABLED (`-2.0`/`0.0` clamp at the time; v4 widened to `-2.5`/`0.5`)
- Framing A reward (no lateral)

### Ckpts
- `Yuan/RL_controller/runs/short_5b_iss3_10M/agent.pt`
- `Yuan/RL_controller/runs/short_5b_iss3_13M/agent.pt` — continuation, ent_coef held at floor

### Baseline numbers (framing A, ratios — see [feedback_eval_metrics](../../../../.claude/projects/-home-lqin-one/memory/feedback_eval_metrics.md))
- L_RL / L_zero:      1.77 mean, 1.22 median
- L_RL / L_classical: 1.48 mean, 1.18 median

### Key findings
- First evidence that gauge fix + state-indep log_std together break the old plateau
- Reference point for "framing A" results — not directly comparable to framing B numbers

---

## 5b only {#5b}

**Status**: historical
**Commit**: predates iss3 fix
**Created**: 2026-05

### Configuration diff vs 5b+iss3
- State-**dependent** log_std head still in place (the broken one that saturates)
- 1M training only

### Ckpts
- `Yuan/RL_controller/runs/short_5b/agent.pt`

### Baseline numbers
- L_RL / L_zero ≈ 1.624 mean (1M training)

### Key findings
- 5b basis fix alone is sufficient to break the old plateau
- But state-dep log_std still drifts toward saturation at longer horizons → motivated iss3

---

## Old plateau {#old}

**Status**: problem case (do not return to this configuration)
**Commit**: pre-fix branch state

### Configuration
- Broken 5b basis (raw `V[..., -4:]` from fp32 SVD — SO(4) gauge ambiguity)
- Broken state-dep log_std (saturates at upper clamp)

### Baseline numbers
- L_RL / L_zero ≈ 0.93 (RL **worse** than doing nothing)

### Key findings
- Diagnosed in [`runs/short_5b_iss3_10M/REPORT.md`](../runs/short_5b_iss3_10M/REPORT.md) §2–3
- Two independent root causes (gauge + log_std saturation), both required to fix
