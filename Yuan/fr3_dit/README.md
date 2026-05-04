# fr3_dit — Farsighted IK via Task-Conditioned Diffusion Transformer

Pipeline for FR3 plane-constrained trajectory generation, topological stitching, and task tokenization. Feeds a downstream DiT-based policy.

## Structure

```
fr3_dit/
├── core/                 # Robot model + visualization helpers
│   ├── pen_fr3_robot.py     # PEN_LENGTH, PenFrankaResearch3 / PenFrankaResearch3GPU
│   └── viz_utils.py         # visualize_anime_path + visualize_anime_dual (multi-robot)
├── data_generation/      # Phase 0: collect raw plane-constrained straight trajectories
│   └── generate_fr3_plane_dataset.py
├── stitching/            # Phase 1+2: anchor-based stitching → composite tasks + tokens
│   ├── stitch_composite_tasks.py
│   ├── filter_min_subseg_length.py        # drop tasks with any subseg < threshold
│   └── add_spatial_anchor.py              # stamp path-start XY into start tokens
├── training/             # Phase 3: DiT training, inference, and tracker eval
│   ├── composite_task_dataset.py
│   ├── task_cond_dit.py                       # v1 full-trajectory DiT
│   ├── task_cond_dit_q0.py                    # v3+ q₀-only DiT (DDPM v-pred + CFG + joint-limit norm)
│   ├── task_cond_dit_q0_v6.py                 # v6 token-aligned per-keypoint q sequence prediction
│   ├── flow_matching_q0.py                    # CFM helpers + Euler ODE sampler
│   ├── train_dit.py / train_dit_q0.py         # DDPM training (v1 / v3+ ; v5 adds TCP-orient loss + q7 canon)
│   ├── train_dit_q0_fm.py                     # CFM training (v4 parallel branch ; v5 same upgrades)
│   ├── train_dit_q0_v6.py                     # v6 sequence prediction (v + tcp + orient + smooth + margin)
│   ├── infer_dit.py / infer_dit_q0.py         # DDPM inference + plotting (v5: q7→0 snap)
│   ├── infer_dit_q0_fm.py                     # CFM inference (Euler ODE sampler; v5: q7→0 snap)
│   ├── infer_dit_q0_v6.py                     # v6 keypoint sampling + IK refine + Cartesian IK interp
│   ├── ik_refine.py                           # farsighted-IK helper: refine q0 seed to exact target TCP
│   ├── eval_tracker.py                        # rollout each q₀ through tracker, report completion %
│   └── eval_v6.py                             # v6: validate IK-interpolated trajectory directly (no tracker)
├── calligraphy/          # Phase 5: write Chinese characters using DiT as stroke-feasibility prior
│   ├── character_def.py        # canonical polyline definitions (e.g., "万" = 3 strokes)
│   ├── polyline_to_tokens.py   # convert a polyline + scene placement → 32-D token sequence
│   ├── feasibility_check.py    # tokens → DiT N candidates → IK refine → tracker → bool
│   ├── dataset_bounds.py       # read training-data segment-length histogram → analytical bounds
│   ├── retrieve_strokes.py     # kNN over training tasks: closest-shape match + similarity transform
│   ├── find_max_line.py        # 1-D bisection on length of a single straight stroke
│   ├── find_max_size.py        # 1-D bisection on character size at fixed placement
│   └── draw_character.py       # orchestrator: full character execution + viz
├── visualization/        # Viewers for raw + composite trajectories + DiT predictions
│   ├── visualize_fr3_plane_trajectory.py
│   ├── visualize_composite_task.py
│   ├── visualize_q0_compare.py        # static GT vs predicted q0 robot poses
│   └── visualize_q0_rollout.py        # animate tracker rollout from a predicted q0
├── experiments/          # Ablations
│   ├── test_same_task_start_conf_gap.py
│   └── outputs/             # gitignored
├── data/                 # HDF5 datasets (gitignored)
│   ├── pen_fr3_plane_trajectories.hdf5
│   └── pen_fr3_composite_tasks.hdf5
└── README.md
```

## Run commands

All commands are run from the `CleanDiffuser/` directory using the `-m` module form.

```bash
cd /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser
```

### Phase 0 — Collect raw plane-constrained trajectories

```bash
python -m fr3_dit.data_generation.generate_fr3_plane_dataset \
    --num-trajectories 1000 \
    --batch-size 32
```

Writes `fr3_dit/data/pen_fr3_plane_trajectories.hdf5`.

Useful flags: `--theta-max-deg`, `--angle-margin-deg`, `--joint-margin-ratio`, `--max-steps`, `--seed`.

### Phase 1+2 — Stitch + tokenize composite tasks

Two-stage criterion, in priority order:

1. **Primary — Workspace intersection**: two anchors are candidates iff their TCP positions are within `--tcp-eps-m` meters. Guarantees composite trajectories are spatially connected.
2. **Secondary — C-space similarity**: among TCP-proximate candidates, keep only those with weighted joint-space distance

   $$\big\|W (q_A - q_B)\big\| < \epsilon_q, \qquad W = \mathrm{diag}(1/\mathrm{span}_i)$$

   This rejects pairs that share a TCP point but sit on different IK branches (elbow-up vs elbow-down etc.).
3. **Optional** — plane / seam physics, off by default.

```bash
python -m fr3_dit.stitching.stitch_composite_tasks \
    --tcp-eps-m 0.02 \
    --c-eps 0.5 \
    --anchor-stride 10 \
    --max-hops 3 \
    --include-singles

# Tighter — require co-planar trajectories + smooth seams
python -m fr3_dit.stitching.stitch_composite_tasks \
    --tcp-eps-m 0.02 --c-eps 0.3 \
    --enforce-plane --enforce-physics --include-singles

# Small-data fallback: split each long trajectory into N contiguous sub-segments
python -m fr3_dit.stitching.stitch_composite_tasks --self-slice 3 --include-singles
```

Reads `fr3_dit/data/pen_fr3_plane_trajectories.hdf5`,
writes `fr3_dit/data/pen_fr3_composite_tasks.hdf5`.

Key flags:
- `--tcp-eps-m` (default `0.02`) — **primary**: TCP-space seam radius in meters
- `--c-eps` (default `0.1`) — **secondary**: weighted C-space similarity bound (dimensionless). At `0.1` cross-trajectory stitches stay on the same IK branch; loosen (e.g. `0.3`) to admit branch switches and find more 3-hop composites at the cost of bigger joint jumps at the seam.
- `--anchor-stride` (default `10`) — sample an anchor every N steps along each trajectory
- `--enforce-plane` (off by default) — additionally require same plane normal (within `--plane-cos`) and same plane side
- `--enforce-physics` (off by default) — additionally require cubic-fit `v/a/jerk` at the seam under `--vel-ratio`/`--acc-ratio`/`--jerk-ratio`
- `--max-hops` (default `3`), `--max-per-seed` (default `16`)
- `--self-slice N` — fallback N-way split of each long trajectory (continuous, Δθ≈0)
- `--include-singles` — also emit single-segment tasks for curriculum

> **Data-density note**: with few independently-sampled trajectories, true 3-trajectory workspace hubs are rare. Expect tighter `--tcp-eps-m` (1–2 cm) to yield few 3-segment composites until the raw dataset grows to ≥ 5000 trajectories. Use `--self-slice 3` for continuous demo data, or loosen `--tcp-eps-m` to 5–10 cm to accept visible TCP gaps at seams.

### Phase 3 — Dataset sanity check

```bash
python -m fr3_dit.training.composite_task_dataset --batch-size 4 --num-batches 2
```

Use `CompositeTaskDataset` + `dit_collate` from `fr3_dit.training.composite_task_dataset`.

### Phase 4 — q₀-DiT training (DDPM v-pred + CFM)

```bash
# 1) Filter composite HDF5: keep only tasks with every sub-segment ≥ 10 cm
python -m fr3_dit.stitching.filter_min_subseg_length \
    --input fr3_dit/data/pen_fr3_composite_tasks_50k.hdf5 --min-m 0.10

# 2) Stamp the path-start XY into every START token (spatial anchor)
python -m fr3_dit.stitching.add_spatial_anchor \
    --input fr3_dit/data/pen_fr3_composite_tasks_50k_minseg10.hdf5

# 3a) DDPM v-prediction training (v5 — adds TCP-orient loss + q7 canonicalization)
python -m fr3_dit.training.train_dit_q0 \
    --data fr3_dit/data/pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5 \
    --num-steps 40000 --batch-size 512 \
    --lambda-tcp 5.0 --lambda-orient 2.0 --mirror-prob 0.5 \
    --ckpt-dir fr3_dit/experiments/outputs/dit_q0_v5_ckpts

# 3b) Conditional Flow Matching training (parallel branch ; v5 same upgrades, uniform t-weighting)
python -m fr3_dit.training.train_dit_q0_fm \
    --data fr3_dit/data/pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5 \
    --num-steps 40000 --batch-size 512 \
    --lambda-tcp 5.0 --lambda-orient 2.0 --mirror-prob 0.5 \
    --ckpt-dir fr3_dit/experiments/outputs/dit_q0_fm_v5_ckpts

# 3c) v6 — token-aligned per-keypoint q sequence prediction (5-loss objective:
#       v + tcp + orient + smoothness + joint-margin). Replaces v5's tracker-rollout
#       dependency: at inference each keypoint is IK-refined and Cartesian-interpolated
#       between, no tracker drift. Slower training (sps≈0.7 vs 1.2 for v5) due to
#       per-token output head + extra loss terms.
python -m fr3_dit.training.train_dit_q0_v6 \
    --data fr3_dit/data/pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5 \
    --num-steps 60000 --batch-size 512 \
    --lambda-tcp 5.0 --lambda-orient 2.0 --lambda-smooth 1.0 --lambda-margin 0.5 \
    --mirror-prob 0.5 \
    --ckpt-dir fr3_dit/experiments/outputs/dit_q0_v6_ckpts

# v7: same script, switched to hinge orient loss + tighter joint margin (only
# penalize cone-violators / out-of-margin samples). Pair with stronger λ since
# in-distribution samples now contribute zero loss.
python -m fr3_dit.training.train_dit_q0_v6 \
    --data fr3_dit/data/pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5 \
    --num-steps 50000 --batch-size 512 \
    --orient-loss hinge --theta-max-deg 30.0 \
    --lambda-tcp 5.0 --lambda-orient 10.0 \
    --lambda-smooth 2.0 --lambda-margin 2.0 --margin-threshold 0.80 \
    --mirror-prob 0.5 \
    --ckpt-dir fr3_dit/experiments/outputs/dit_q0_v7_ckpts

# 4c) v6 inference (DDIM sampling → keypoint sequence → IK refine → Cartesian IK interp)
python -m fr3_dit.training.infer_dit_q0_v6 --task-idx 234088 --n-samples 8 --cfg-w 3.0

# 5c) v6 evaluation (no tracker; validate IK-interpolated trajectory feasibility per frame)
python -m fr3_dit.training.eval_v6 \
    --task-indices 234088 127753 59086 \
    --prefix infer_q0_v6 \
    --report-out /tmp/eval_v6.json

# 4) Inference (one task, 8 candidates, classifier-free guidance w=3)
# Both inference scripts snap q7→0 by default (training canonicalizes q7=0; pass --no-snap-q7 to disable).
# Use --out-prefix to keep different versions' outputs side-by-side without clobbering.
python -m fr3_dit.training.infer_dit_q0    --task-idx 234088 --n-samples 8 --cfg-w 3.0 \
    --out-prefix infer_q0_v5     # DDPM v5, 50 DDIM steps
python -m fr3_dit.training.infer_dit_q0_fm --task-idx 234088 --n-samples 8 --cfg-w 3.0 \
    --out-prefix infer_q0_fm_v5  # CFM v5, 10 Euler steps

# 5) Tracker-based completion eval (real metric)
# --prefix matches whatever --out-prefix you used at inference time.
# --refine-ik runs wrs IK from each predicted q0 (seed) → exact path-start TCP before rollout,
# isolating "good IK seed" (DiT job) from "TCP precision" (IK job). Run both modes for the
# three-axis evaluation: TCP error / raw rollout / IK-refined rollout.
# --angle-attract-gain enables a stronger always-on interior attractor (pulls TCP_z toward
# -desk_normal proportional to angle deviation in radians) to suppress angle drift
# accumulation that otherwise causes late-segment angle_violation. Eval default 2.0
# (data-gen used 0.0; 5.0 turned out too aggressive after IK refine — wrist self-collided).
# --angle-null-gain ramps up the boundary brake (eval default 1.0; data-gen used 0.4).
python -m fr3_dit.training.eval_tracker \
    --task-indices 234088 127753 59086 \
    --prefix infer_q0_v5_task \
    --report-out /tmp/eval_v5.json
python -m fr3_dit.training.eval_tracker \
    --task-indices 234088 127753 59086 \
    --prefix infer_q0_v5_task \
    --refine-ik \
    --report-out /tmp/eval_v5_refined.json
```

Notes:
- Both training scripts share the same model architecture (`task_cond_dit_q0.TaskCondDiTq0`); the only difference is the noise objective (DDPM v-pred vs CFM linear-path velocity).
- CFM inference defaults to **10 Euler steps** vs DDPM's 50 DDIM steps → ~5× faster sampling.
- **v5 training** adds two improvements applied to both branches:
  - TCP-orient loss (`--lambda-orient`, default 2.0): supervises the predicted TCP_z direction toward `−desk_normal` (pen pointing into desk) to suppress `angle_violation` failures.
  - q7 canonicalization: `StartQDataset` zeroes q7 before normalization (pen self-rotation is task-irrelevant null-space). Inference scripts snap `q7→0` post-sample by default to eliminate q7 OOL violations; pass `--no-snap-q7` to disable.
  - The DDPM branch keeps `α²` time-weighting on TCP/orient (clean-side emphasis); the CFM branch uses **uniform** weighting since Euler integration from t=0 needs accurate `u` at every noise level.
- `eval_tracker.py` re-runs `PlaneConstrainedTracker` from each predicted q₀ segment-by-segment, reporting **per-task best-of-N completion %** and full-task success rate, which is the metric that ultimately matters.

### Visualization

```bash
# Single raw trajectory
python -m fr3_dit.visualization.visualize_fr3_plane_trajectory --seed 42

# Composite task (any seg count)
python -m fr3_dit.visualization.visualize_composite_task --seed 42

# Only 3-segment composites
python -m fr3_dit.visualization.visualize_composite_task --min-segs 3 --seed 42

# Animate robot along the path
python -m fr3_dit.visualization.visualize_composite_task --min-segs 3 --seed 42 --toggle-animate

# Specific task index
python -m fr3_dit.visualization.visualize_composite_task --task-idx 200

# === Predicted q0 demo: GT (green) + DiT-predicted candidates (red, ranked) ===
# Requires running infer_dit_q0[_fm] first with the same --out-prefix.
python -m fr3_dit.visualization.visualize_q0_compare \
    --task-idx 234088 --out-prefix infer_q0_v5 --n-show 4

# === Rollout demo: actually run the tracker from a predicted q0 and animate ===
# Black trace = rollout TCP path; red sphere = failure point (if any). Per-segment
# success/failure with termination label is printed to stdout. --rank-k 0 = best
# of N by RMSE; --rank-k -1 = use GT q0 instead (sanity baseline).
# --refine-ik runs wrs IK to snap TCP to the exact path-start (preserving seed
# orientation) before rollout — matches eval_tracker --refine-ik.
# --angle-attract-gain enables a stronger always-on interior attractor (pulls TCP_z
# toward -desk_normal proportional to angle deviation in radians) to suppress angle
# drift accumulation that otherwise causes late-segment angle_violation. Default 2.0
# (data-gen used 0.0; 5.0 was too aggressive after IK refine — wrist self-collided).
# --angle-null-gain ramps up the boundary brake (default 1.0; data-gen used 0.4).
# --playback-stride subsamples the rollout for animation (default 5 = 5× faster than
# full-rate playback; pass 1 to step through every tracker frame).
python -m fr3_dit.visualization.visualize_q0_rollout \
    --task-idx 234088 --out-prefix infer_q0_v5 --rank-k 0 --refine-ik
```

### Calligraphy: write a Chinese character (DiT-as-stroke-prior)

```bash
# Default: 中 character at 8 cm size, centered on the desk.
# For each stroke, oracle samples N=8 q0 candidates with DiT, IK-refines each to the
# exact stroke start TCP, runs the tracker, and picks whichever candidate completes.
# Between strokes, joint-space interpolation provides the pen-lift transition.
python -m fr3_dit.calligraphy.draw_character --char 中 --size 0.08

# Other characters defined in character_def.py: 一, 二, 十, 万, 日, 中
python -m fr3_dit.calligraphy.draw_character --char 万 --size 0.10 --theta-deg 0

# Skip animation (just print per-stroke feasibility) — useful for size sweeps.
python -m fr3_dit.calligraphy.draw_character --char 中 --size 0.06 --no-animate

# Faster oracle: rank candidates by DiT self-score (option B) and only roll out the top 2.
# ~4x speedup on 8 candidates with negligible hit-rate loss when score is informative.
python -m fr3_dit.calligraphy.draw_character --char 中 --size 0.08 --top-k-rollout 2

# Find the maximum writable size for a character via 1-D bisection (no grid search).
# Reports max size_m at the chosen placement; ~30s with --top-k-rollout=2.
python -m fr3_dit.calligraphy.find_max_size --char 中

# Simplest baseline: the longest single straight stroke writable from a fixed start
# point in a fixed in-plane direction. ~20s.
python -m fr3_dit.calligraphy.find_max_line --x 0.5 --y 0.0 --direction-deg 0

# Most elegant: read training-data segment-length distribution → analytical bounds.
# No oracle calls, no rollouts — under 1 second.
python -m fr3_dit.calligraphy.dataset_bounds                          # print bounds
python -m fr3_dit.calligraphy.dataset_bounds --query-line 0.25        # is 25cm line writable?
python -m fr3_dit.calligraphy.dataset_bounds --query-char 中 --size 0.15

# Retrieval: for a target stroke, find the closest training task by intrinsic shape
# (seg_count, lengths, corner angles) and align it (translate+rotate+uniform-scale)
# to the target geometry. The aligned polyline can then feed the existing DiT
# pipeline — guaranteed in-distribution because we started from a known-feasible
# training shape and only made a small similarity transform.

# Whole character: top-1 retrieval per stroke, single overlay figure
# (target = gray dashed; retrieved raw = dotted; adjusted = solid stroke colors).
python -m fr3_dit.calligraphy.retrieve_strokes --char 中 --size 0.15 --save-plot

# Single-stroke detail: top-K matches with per-stroke comparison plots.
python -m fr3_dit.calligraphy.retrieve_strokes --char 中 --size 0.15 --stroke 4 --save-plot --k 3
```

### Experiments

```bash
python -m fr3_dit.experiments.test_same_task_start_conf_gap --num-starts 64
```

Writes `fr3_dit/experiments/outputs/same_task_start_conf_gap_curves.svg`.

## HDF5 layout (composite tasks)

```
/meta                                # hyperparams used at stitch time
/raw_trajs
    q_flat, tcp_flat, offset         # packed raw trajectories
    direction, plane_normal, plane_point, plane_side, length
/tasks
    token_flat    : (T_total, 32)    # packed tokens, all tasks concatenated
    token_offset  : (M+1,) int64     # task m tokens at [offset[m], offset[m+1])
    token_kind    : (T_total,) u8    # 0=start, 1=segment, 2=corner
    qtraj_flat    : (Q_total, 7)     # packed joint-space trajectory
    qtraj_offset  : (M+1,) int64
    tcp_flat      : (Q_total, 3)
    start_q       : (M, 7)
    local_frame   : (M, 3, 3)        # columns = (x̂, ŷ, ẑ)
    local_origin  : (M, 3)
    plane_normal  : (M, 3)           # world frame
    seg_count     : (M,) u16
    total_length  : (M,) f32
    subseg_meta_flat   : (S_total, 3) int32   # (traj_id, start, end)
    subseg_offset      : (M+1,) int64
    seg_step_counts_flat : (S_total,) int32
```

## Token layout (`token_dim = 32`)

| offset | size | channel | semantics |
|---|---|---|---|
| 0..3   | 3 | `kind_onehot`        | `[start, segment, corner]` |
| 3..6   | 3 | `dir_local`          | segment direction in local frame |
| 6      | 1 | `len_norm`           | segment length / `length_ref` |
| 7..9   | 2 | `delta_theta_sincos` | `[sin Δθ, cos Δθ]` for corner |
| 9..12  | 3 | `axis_local`         | corner rotation axis |
| 12..15 | 3 | `bisector_local`     | corner angle bisector |
| 15..18 | 3 | `plane_normal_local` | plane normal in local frame |
| 18     | 1 | `cum_len_norm`       | cumulative arc length / `length_ref` |
| 19..27 | 8 | `fourier_time`       | 4-band Fourier features of `cum_len_norm` |
| 27..32 | 5 | `pad`                | zero, reserved |

Local frame: `x̂ = v̂₁` (first segment direction), `ẑ = n̂` (plane normal after Gram-Schmidt vs `x̂`), `ŷ = ẑ × x̂`.

## Physical limits used at seam check

FR3 per-joint maxima (conservative):
- `q̇_max` = `[2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]` rad/s
- `q̈_max` = `[15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]` rad/s²
- `⃛q_max` = `[7500, 3750, 5000, 6250, 7500, 10000, 10000]` rad/s³

Seam check fits a cubic per joint over a `window_steps` window on each side, evaluates `(v, a)` at the endpoints, and compares to `{vel, acc, jerk}_ratio × FR3 limits`.
