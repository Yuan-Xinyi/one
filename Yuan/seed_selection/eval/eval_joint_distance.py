"""Mode-coverage evaluation for a trained SeedQ0DiT ckpt.

For each multi-label task in the dataset:
  - Build c = (p0, line_dir, n_target).
  - Draw M samples from the diffusion model (DDIM).
  - Assign each sample to its nearest label in joint-space L2.
  - Compute:
      * per-sample min-distance to nearest label (fidelity)
      * per-task mode-coverage: which of n_labels are within `match_rad`
      * per-task assignment entropy: how evenly samples spread across labels
Aggregate across tasks; plot fidelity histogram + coverage by n_labels +
example per-task assignment bars for a few tasks.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from Yuan.fr3_dit.training.task_cond_dit_q0 import denormalize_q
from Yuan.seed_selection.diffusion.dataset import SeedSelectionDataset
from Yuan.seed_selection.diffusion.sampling import ddim_sample_q0, load_ckpt


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz'
DEFAULT_CKPT = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/q0_20k_cfg_mirror_ckpts/step_300000.pt'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--data', type=Path, default=DEFAULT_NPZ)
    p.add_argument('--n-samples', type=int, default=32,
                   help='samples per task')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='cap tasks evaluated (default: all multi-label kept/edge)')
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--match-rad', type=float, default=0.5,
                   help='L2 joint distance for "covered a mode" (rad)')
    p.add_argument('--use-model', action='store_true',
                   help='use raw model weights instead of EMA')
    p.add_argument('--out-prefix', default='eval_q0',
                   help='output filename prefix (saved next to ckpt)')
    p.add_argument('--which', choices=['train', 'val', 'all'], default='all',
                   help="restrict eval to the train or val side of split.json "
                        "(loaded from ckpt's dir unless --split-file given). "
                        "'all' ignores any split.")
    p.add_argument('--split-file', type=Path, default=None,
                   help='override split.json path (default: <ckpt-dir>/split.json)')
    p.add_argument('--shuffle-c', action='store_true',
                   help='replace each task\'s c with a random other task\'s c '
                        '(within the eval set) before sampling. Tests whether the '
                        'model is using c or just sampling the marginal q0 dist.')
    p.add_argument('--shuffle-seed', type=int, default=12345,
                   help='RNG seed for --shuffle-c permutation.')
    p.add_argument('--cfg-w', type=float, default=1.0,
                   help='classifier-free guidance weight (1.0 = no guidance).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, schedule, cfg, step = load_ckpt(args.ckpt, device, use_ema=not args.use_model)
    print(f'[eval] ckpt={args.ckpt} step={step} weights={"model" if args.use_model else "ema"}')

    ds = SeedSelectionDataset(args.data)
    print(f'[eval] dataset: {len(ds)} entries')

    # Apply train/val split filter (relative to ds entries).
    subset_mask = np.ones(len(ds), dtype=bool)
    if args.which != 'all':
        import json as _json
        split_path = args.split_file or (args.ckpt.parent / 'split.json')
        if not split_path.exists():
            raise SystemExit(f"--which={args.which} but no split.json at {split_path}")
        s = _json.loads(split_path.read_text())
        if int(s['n_total']) != len(ds):
            raise SystemExit(f"split n_total={s['n_total']} != ds size {len(ds)}")
        subset_idx = np.array(s[f'{args.which}_idx'], dtype=np.int64)
        subset_mask = np.zeros(len(ds), dtype=bool)
        subset_mask[subset_idx] = True
        print(f'[eval] using {args.which} split from {split_path}: {int(subset_mask.sum())} entries')

    # Pick all multi-label tasks (n>=2) within the chosen subset.
    multi_idx = np.where((ds.n_labels >= 2) & subset_mask)[0]
    if args.max_tasks is not None:
        multi_idx = multi_idx[:args.max_tasks]
    print(f'[eval] {len(multi_idx)} multi-label tasks to evaluate, {args.n_samples} samples each')

    # Per-task results
    all_min_dists = []                  # (T*M,) min L2 distance from each sample to any label
    per_task_coverage_frac = []         # fraction of labels covered, per task
    per_task_full_coverage = []         # 1 if all labels covered
    per_task_assignment = []            # list of (n_labels, [count_per_label, ...])
    per_task_n_labels = []
    per_task_L_seed_bucket = []         # for grouping
    per_task_idx_global = []

    # bucket helper
    L_seed = np.load(args.data, allow_pickle=False)['L_seed']  # original task indices
    # ds.labels_q0 was filtered to keep/edge tasks; we need L_seed in the same order.
    # Reload the filter to get the source indices.
    z = np.load(args.data, allow_pickle=False)
    keep_mask = np.isin(z['status'], ['kept', 'edge', 'edge_seed_fallback']) & (z['n_labels'] >= 1)
    src_idx = np.where(keep_mask)[0]
    L_seed_per_entry = L_seed[src_idx]  # aligned with ds entries

    def bucket(L):
        if L < 0.15: return 'weak'
        if L < 0.20: return 'medium-weak'
        if L < 0.30: return 'medium'
        return 'strong'

    # If --shuffle-c: derangement of multi_idx (no fixed points). The sample
    # for task i is drawn under c_{shuffle[i]}, but evaluated against task i's
    # labels. A model that actually uses c should produce near-random coverage.
    if args.shuffle_c:
        rng = np.random.default_rng(args.shuffle_seed)
        perm = rng.permutation(len(multi_idx))
        # ensure derangement (no fixed points) — if any, rotate by 1
        if (perm == np.arange(len(multi_idx))).any():
            perm = np.concatenate([perm[1:], perm[:1]])
        shuffle_src_idx = multi_idx[perm]   # for task i, use c from task shuffle_src_idx[i]
        n_fixed = int((perm == np.arange(len(multi_idx))).sum())
        print(f'[eval] --shuffle-c: deranged {len(multi_idx)} tasks (fixed points: {n_fixed})')
    else:
        shuffle_src_idx = multi_idx  # identity (use own c)

    # Batch eval: process tasks in groups for GPU efficiency.
    # Each task replicates c M times → (T*M, 9) condition matrix.
    BATCH_TASKS = 64
    for start in range(0, len(multi_idx), BATCH_TASKS):
        batch_t = multi_idx[start:start + BATCH_TASKS]
        c_src_t = shuffle_src_idx[start:start + BATCH_TASKS]
        Bt = len(batch_t)
        c_np = np.stack([
            np.concatenate([ds.cs_p0[i], ds.cs_line_dir[i], ds.cs_n_target[i]])
            for i in c_src_t              # ← sample under shuffled c (or own c if no shuffle)
        ], axis=0).astype(np.float32)
        c_t = torch.from_numpy(c_np).to(device)
        # repeat each c M times
        c_rep = c_t.repeat_interleave(args.n_samples, dim=0)  # (Bt*M, 9)
        q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                                num_steps=args.ddim_steps, eta=0.0,
                                cfg_w=args.cfg_w)
        q_raw = denormalize_q(q_norm).cpu().numpy()  # (Bt*M, 7) absolute joint angles

        for bi, ti in enumerate(batch_t):
            n = int(ds.n_labels[ti])
            labels = ds.labels_q0[ti, :n]  # (n, 7)
            samples = q_raw[bi*args.n_samples:(bi+1)*args.n_samples]  # (M, 7)
            # M×n distance matrix
            d = np.linalg.norm(samples[:, None, :] - labels[None, :, :], axis=-1)  # (M, n)
            nearest = d.argmin(axis=1)            # (M,) which label each sample picked
            min_dist = d.min(axis=1)              # (M,) distance to nearest label
            all_min_dists.append(min_dist)

            # Coverage: a label is "covered" if any sample is within match_rad
            covered = np.zeros(n, dtype=bool)
            for k in range(n):
                covered[k] = (d[:, k] < args.match_rad).any()
            per_task_coverage_frac.append(covered.mean())
            per_task_full_coverage.append(bool(covered.all()))
            counts = np.bincount(nearest, minlength=n)
            per_task_assignment.append((n, counts.tolist()))
            per_task_n_labels.append(n)
            per_task_L_seed_bucket.append(bucket(L_seed_per_entry[ti]))
            per_task_idx_global.append(int(src_idx[ti]))

    all_min_dists = np.concatenate(all_min_dists)
    per_task_coverage_frac = np.array(per_task_coverage_frac)
    per_task_full_coverage = np.array(per_task_full_coverage, dtype=bool)
    per_task_n_labels = np.array(per_task_n_labels)
    per_task_L_seed_bucket = np.array(per_task_L_seed_bucket)

    # ---------- report ----------
    print("\n" + "=" * 78)
    print("Sample fidelity — min L2 distance from sample to nearest label (rad)")
    print("=" * 78)
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:>2d}: {np.percentile(all_min_dists, q):.3f}")
    print(f"  mean: {all_min_dists.mean():.3f}   max: {all_min_dists.max():.3f}")
    for t in [0.1, 0.2, 0.3, 0.5, 1.0]:
        pct = 100 * (all_min_dists < t).mean()
        print(f"  < {t:.2f} rad: {pct:>5.1f}%")

    print("\n" + "=" * 78)
    print(f"Mode coverage (label covered if any sample within {args.match_rad:.2f} rad)")
    print("=" * 78)
    print(f"  full-coverage tasks: {int(per_task_full_coverage.sum())}/{len(per_task_full_coverage)} ({100*per_task_full_coverage.mean():.1f}%)")
    print(f"  per-task coverage fraction (mean): {per_task_coverage_frac.mean():.3f}")
    print(f"  per-task coverage fraction (median): {np.median(per_task_coverage_frac):.3f}")
    print()
    print("  by n_labels:")
    for n in sorted(set(per_task_n_labels.tolist())):
        m = per_task_n_labels == n
        cov_frac = per_task_coverage_frac[m].mean()
        full = per_task_full_coverage[m].mean()
        print(f"    n_labels={n} (N={int(m.sum())}): mean cov frac={cov_frac:.3f}, full coverage rate={100*full:.1f}%")
    print()
    print("  by L_seed bucket:")
    for b in ['weak', 'medium-weak', 'medium', 'strong']:
        m = per_task_L_seed_bucket == b
        if m.sum() == 0: continue
        cov = per_task_coverage_frac[m].mean()
        full = per_task_full_coverage[m].mean()
        print(f"    {b:<14}: N={int(m.sum()):>4}  mean cov frac={cov:.3f}  full coverage rate={100*full:.1f}%")

    # Diversity: per-task assignment entropy normalized to [0,1] (1 = uniform across labels).
    ent = []
    for (n, cnt) in per_task_assignment:
        p = np.array(cnt) / sum(cnt)
        p = p[p > 0]
        H = -(p * np.log(p)).sum() / math.log(n)  # normalized
        ent.append(H)
    ent = np.array(ent)
    print()
    print(f"Per-task normalized assignment entropy (1.0 = perfectly uniform across labels):")
    print(f"  mean: {ent.mean():.3f}   median: {np.median(ent):.3f}   p25={np.percentile(ent,25):.3f}  p75={np.percentile(ent,75):.3f}")
    print(f"  tasks with H > 0.9 (near-uniform): {int((ent > 0.9).sum())}/{len(ent)} ({100*(ent>0.9).mean():.1f}%)")
    print(f"  tasks with H < 0.3 (mode-collapsed): {int((ent < 0.3).sum())}/{len(ent)} ({100*(ent<0.3).mean():.1f}%)")

    # Save plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = args.ckpt.parent
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: min-distance distribution (sample fidelity)
    ax = axes[0, 0]
    ax.hist(all_min_dists, bins=60, color='steelblue', edgecolor='k', alpha=0.85)
    ax.axvline(args.match_rad, color='red', ls='--', label=f'match_rad={args.match_rad}')
    ax.axvline(0.08, color='orange', ls=':', label='smm_dedup_rad=0.08')
    ax.set_xlabel('min L2 dist (rad) from sample to nearest label')
    ax.set_ylabel('# samples')
    ax.set_title(f'Sample fidelity ({len(all_min_dists)} samples)')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: per-task coverage fraction histogram
    ax = axes[0, 1]
    ax.hist(per_task_coverage_frac, bins=[0, 0.34, 0.5, 0.67, 0.85, 1.01],
            color='seagreen', edgecolor='k', alpha=0.85)
    ax.set_xlabel('fraction of labels covered per task')
    ax.set_ylabel('# tasks')
    ax.set_title('Mode coverage per task')
    ax.grid(alpha=0.3)

    # Panel 3: assignment entropy histogram
    ax = axes[1, 0]
    ax.hist(ent, bins=30, color='goldenrod', edgecolor='k', alpha=0.85)
    ax.axvline(np.median(ent), color='red', ls='--', label=f'median={np.median(ent):.2f}')
    ax.set_xlabel('normalized assignment entropy')
    ax.set_ylabel('# tasks')
    ax.set_title('Mode diversity (1.0=uniform across labels)')
    ax.set_xlim(0, 1.05)
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 4: per-task assignment bars for 6 example tasks (n_labels=3, mix of high+low entropy)
    ax = axes[1, 1]
    triples = [(i, per_task_assignment[i], ent[i]) for i in range(len(per_task_assignment)) if per_task_assignment[i][0] == 3]
    triples.sort(key=lambda t: t[2])  # by entropy ascending
    pick = []
    if len(triples) >= 6:
        idx_pick = [0, 1, len(triples)//4, len(triples)//2, 3*len(triples)//4, len(triples)-1]
        pick = [triples[i] for i in idx_pick]
    elif len(triples) > 0:
        pick = triples[:6]
    width = 0.25
    x = np.arange(len(pick))
    if pick:
        counts = np.array([t[1][1] for t in pick])  # (npicked, 3)
        ax.bar(x - width, counts[:, 0], width, label='label 0', color='#1f77b4')
        ax.bar(x,         counts[:, 1], width, label='label 1', color='#ff7f0e')
        ax.bar(x + width, counts[:, 2], width, label='label 2', color='#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels([f'H={t[2]:.2f}' for t in pick])
        ax.set_xlabel('example tasks (sorted by entropy)')
        ax.set_ylabel('# samples assigned to each label')
        ax.set_title('Per-task sample assignment (6 example n=3 tasks)')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    suffix = f'step{step}_{args.which}' if args.which != 'all' else f'step{step}'
    if args.shuffle_c:
        suffix = f'{suffix}_shuffled'
    if args.cfg_w != 1.0:
        suffix = f'{suffix}_cfg{args.cfg_w}'
    fig_path = out_dir / f'{args.out_prefix}_{suffix}.png'
    plt.savefig(fig_path, dpi=120)
    print(f'\nSaved: {fig_path}')

    # Save numerical results
    np.savez(
        out_dir / f'{args.out_prefix}_{suffix}.npz',
        min_dists=all_min_dists,
        per_task_coverage_frac=per_task_coverage_frac,
        per_task_full_coverage=per_task_full_coverage,
        per_task_n_labels=per_task_n_labels,
        per_task_entropy=ent,
        per_task_L_seed_bucket=per_task_L_seed_bucket.astype('U16'),
        per_task_idx_global=np.array(per_task_idx_global, dtype=np.int64),
        match_rad=args.match_rad,
        n_samples=args.n_samples,
    )
    print(f'Saved: {out_dir / f"{args.out_prefix}_{suffix}.npz"}')


if __name__ == '__main__':
    main()
