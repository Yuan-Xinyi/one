# IJRR — frozen clean copy of the final pipeline

Everything here is a **copy**. Nothing was moved and nothing outside this
folder was edited, so the older directories still reproduce the older papers.
The only dependency outside `Yuan/IJRR/` is the top-level `one` package
(FR3 kinematics and the sphere collision model), which is deliberately not
duplicated.

Imports were rewritten `Yuan.<old> -> Yuan.IJRR.<new>`; file names were kept
unchanged so every file maps back to its origin (table below).

## The final system

Per `runs/iksel_final_freeze_v3.json` (frozen 2026-07-31, supersedes
`ikpool_final_freeze_v2.json`):

| | |
|---|---|
| Stage 1 candidate layer | faithful IKSel — CVT-sampled table of 201,600 entries, key `(pos/0.05, tool-z)`, 32 cone directions at 29.5°, k=200 query, full-6D Jacobian rerank to top-20, DLS with farthest-from-tried reshuffle retries (max 4), dedup 0.08 rad, classical `q0_pilot` fallback slot, K=32 |
| Stage 1 selector | 5-member SetSel, 45-D features, listwise CE (temperature 0.1) + Huber feasibility, trained on 20,000-task return tables labelled under the hybrid controller |
| Stage 2 | two variants only — **RL** (the plain PPO policy alone) and **Hybrid** (RL in the interior, `ClassicalNullspaceController` near the joint-limit boundary, hysteresis `tau_enter=0.985`, `tau_exit=0.96`) |
| Deployment | one table query, one selector forward, one seed, one rollout; no probes |

**Expert iteration is not part of this pipeline.** Freeze v3 named
`runs/r2_grouped_best` as C0, but that checkpoint is round 10 of a search-free
expert-iteration loop warm-started from round 9 (250k steps per round). It was
replaced by `runs/rl_smmstart_30M`, a plain 30M-step PPO policy trained from
cone-IK SMM start states — which also matches the start distribution stage 1
actually produces. Consequences:

* every label in `runs/iksel_final/` and the selector checkpoints trained on
  them were produced under the **old** C0 and must be regenerated;
* `tau_enter=0.985 / tau_exit=0.96` were tuned against the old C0 and should be
  re-swept.

## Layout

| IJRR path | copied from | role |
|---|---|---|
| `env/env.py` | `Yuan/RL_controller/env/env.py` | task definition, 31-D observation, 4-D null-space action, progress reward, termination |
| `env/classical_nullspace.py` | `Yuan/RL_controller/env/classical_nullspace.py` | classical secondary objective (manipulability, joint centering, cone gate) |
| `env/line_distribution.py` | `Yuan/RL_controller/env/line_distribution.py` | task sampling with feasibility prescreening; scripted replay for evaluation |
| `env/path_geometry.py` | `Yuan/RL_controller/env/path_geometry.py` | intrinsic path frame `(p0, d, n, kappa)`; straight line is `kappa = 0` |
| `env/rollout.py` | `Yuan/RL_controller/env/rollout.py` | single-episode rollout harness |
| `kinematics/batched_fr3_kin.py` | `Yuan/flow_connectivity/batched_fr3_kin.py` | batched FK/Jacobian |
| `kinematics/batched_rollout.py` | `Yuan/flow_connectivity/batched_rollout.py` | `_batched_ik_project`, the Newton/DLS pose projector |
| `kinematics/config.py` | `Yuan/flow_connectivity/config.py` | `EPS_POS_INIT=5e-3`, `THETA_MAX=5°` (IK projection tolerance, **not** the 30° execution cone) |
| `kinematics/pen_collision.py` | `Yuan/system_eval/pen_collision.py` | sphere model extended to the hand and the pen |
| `stage1_seed/iksel_clean_pilot.py` | `Yuan/unified_rl/iksel_clean_pilot.py` | CVT table construction and the retrieval constants |
| `stage1_seed/iksel_campaign.py` | `Yuan/unified_rl/iksel_campaign.py` | the campaign: generate, relabel, train selector, evaluate |
| `stage1_seed/setsel.py` | **extracted** from `Yuan/unified_rl/ikpool_bidir.py` | `SetSel` model, `_picks`, `_paired`; the rest of that module was the bidirectional expert-iteration loop |
| `stage1_seed/cone_ik.py` | `Yuan/seed_selection/smm/cone_ik.py` | cone sampling, `_build_R_with_z`, dedup |
| `stage1_seed/features.py` | `Yuan/unified_rl/features.py` | 45-D candidate features |
| `stage1_seed/feature_build.py` | **extracted** from `Yuan/unified_rl/offline_seed_ensemble_train.py` | just `_build_features`; see note below |
| `stage1_seed/candidate_batch.py` | `Yuan/unified_rl/candidate_batch.py` | candidate container and cached dataset |
| `stage1_seed/checkpoint.py` | `Yuan/unified_rl/checkpoint.py` | rebuild env and controller from a run directory |
| `stage1_seed/controller_rollout.py` | `Yuan/unified_rl/controller_rollout.py` | frozen RL and frozen hybrid controllers, batched seed rollout |
| `stage1_seed/validity.py` | `Yuan/unified_rl/validity.py` | candidate validity checks |
| `stage2_traj/ppo.py` | `Yuan/RL_controller/algorithms/ppo.py` | PPO, with the expert-iteration options removed (see below) |
| `stage2_traj/train.py` | `Yuan/RL_controller/algorithms/train.py` | training entry point, same removals |
| `stage2_traj/config.yaml` | `Yuan/RL_controller/config.yaml` | frozen training configuration |
| `eval/line_bound.py` | `Yuan/reachability/line_bound.py` | per-task pointwise-feasibility bound (the denominator) |
| `eval/reach_map.py` | `Yuan/reachability/reach_map.py` | voxel × direction reachability map, for workspace figures only |

### Data under `runs/`

| path | what it is |
|---|---|
| `rl_smmstart_30M/` | the C0 policy: plain 30M-step PPO from cone-IK SMM start states |
| `iksel_clean_v1/cvt_table_201600.npz` | the retrieval table |
| `iksel_final/` | IKSel candidates, hybrid return tables, selector checkpoints (`sel_enum_run0.pt`, `sel_iksel_run0.pt`), `eval_dev.json` |
| `tasks/` | task geometry (p0, line_dir, n_target, q0_pilot, task_indices) for the 18,432 train / 2,048 validation / 2,048 external splits |
| `eval_10k_systematic/` | the 10,000-task evaluation set and the controller-aware oracle |
| `reach_5cm_50dir.*` | voxel reachability map |
| `iksel_final_freeze_v3.json` | the governing preregistration |

## Things to know before quoting any number

1. **No sealed result exists for IKSel.** The sealed numbers in
   `Yuan/ikpool_paper/main.tex` (+29.2 mm, capture 67.3%) were produced by the
   *enumeration* arm on sealed v1. Sealed v3 was preregistered but never run —
   there is no `iksel_sealed_v3` on disk. Every table has to be regenerated.
2. **The enumeration arm is gone from this copy.** It was the internal
   candidate-layer ablation of freeze v3, and on the last measurement IKSel sat
   below it rather than at parity: validation 0.5537 m vs 0.5701 m (−16.4 mm,
   CI [−24.6, −8.4]), external 0.5726 m vs 0.5853 m (−12.7 mm, CI [−20.8,
   −4.8]). Those numbers survive only in `runs/iksel_final/eval_dev.json`; the
   code (`cone_constrained_ik_enumerate`, the `train-enum-sel` stage), the
   candidate pools and the enumeration selector were removed. Reinstating that
   comparison means re-running it from `Yuan/unified_rl/`.
   The diffusion baseline went the same way — freeze v3 declares IKSel the sole
   mainline with no external baseline, and the `OLD` paths were already dead
   code.
3. **The pointwise bound is not the voxel chord.** `Yuan/reachability/chord_bound.py`
   (not copied here) asks whether a point is reachable in *any* of 50 global
   directions and snaps to 5 cm voxels; the achieved length exceeds it on 5.3%
   of tasks. `eval/line_bound.py` replaces it, and reports two variants: use
   `L_hi` as the denominator. Its value is monotone in the IK search budget —
   going from 24×32 to 48×64 raises the mean bound by 1.55% — so the budget
   must be fixed and stated.
4. **Two collision models exist.** `env/env.py` uses the stock
   `FR3SphereCollision`, which does not see the hand or the pen;
   `kinematics/pen_collision.py` adds them. `eval/line_bound.py` defaults to
   the stock model so the bound stays valid against rollouts produced with it.
   Unify before the final tables.
5. `stage2_traj/train.py` re-execs the interpreter on import to fix
   `LD_LIBRARY_PATH`; it is an entry point, not an importable module.
6. **What was removed from `ppo.py` / `train.py`**, all of it expert-iteration
   machinery that the frozen configuration left switched off: the BC auxiliary
   loss (`bc_obs`, `bc_actions`, `bc_coef`, `bc_anneal`), the guided switch
   (`guide_action_fn`, `guide_tau_*`, `guide_anneal_*`), the critic floor
   (`critic_floor_fn`, `critic_floor_coef`), the phase-start actor anchor
   (`anchor_agent`, `actor_anchor_*`), and the danger-start mixture. This also
   removed the two lazy imports of `Yuan.RL_controller.self_improve.*`, so
   nothing here reaches outside `Yuan/IJRR/` any more.

   Verification: `ppo.py` went 718 -> 543 lines and is a **pure deletion** —
   every remaining line occurs, in order, in the original. `train.py` went
   238 -> 202 lines with exactly three modified lines, each a direct
   consequence of a deletion (dropping the `+ guide_str` concatenation and
   closing the `ppo_train(...)` call). A 40,960-step training run completes and
   writes a checkpoint.
7. **`kinematics/config.py` is read through `getattr` with a fallback.**
   `batched_fr3_kin.py` does `getattr(cfg, "TCP_OFFSET", 0.0)`, so deleting
   that constant does not raise — it silently moves the TCP from the pen tip
   to the flange and shifts every FK result by 0.2034 m. A dead-code pass
   removed it once; it is back, with a warning comment. Check `kin.tcp_offset
   == 0.2034` after touching that file.
8. `feature_build.py` holds the one function (`_build_features`) that the
   mainline used from `offline_seed_ensemble_train.py`. Importing the original
   module pulled in six further legacy modules the mainline never calls.

## Running it

```bash
# stage 1: candidates -> hybrid labels -> selector -> dev evaluation
python -m Yuan.IJRR.stage1_seed.iksel_campaign gen --source train
python -m Yuan.IJRR.stage1_seed.iksel_campaign relabel --source train --shard 0/16
python -m Yuan.IJRR.stage1_seed.iksel_campaign train-selector --run-seed 0
python -m Yuan.IJRR.stage1_seed.iksel_campaign eval-dev

# stage 2: policy
python -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/config.yaml

# evaluation: pointwise-feasibility bound
python -m Yuan.IJRR.eval.line_bound --n-tasks 500 --n-dirs 48 --n-try 64 --k-nn 600
```

All 24 importable modules were verified to import, and `eval/line_bound.py`
was run end to end from this copy.
