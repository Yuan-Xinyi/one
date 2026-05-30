"""Construct eval_set_10k.npz: stratified safe-task subset for the 5-cell ablation.

Selection rules (mirrors the user's spec):
  - source: pilot_20k.npz, status in {kept, edge, edge_seed_fallback}
            and n_labels >= 1   (same default keep as SeedSelectionDataset)
  - safe filter: any_label_collides == False  (matches val eval default;
            "labels don't pierce the bounded plane")
  - flat random draw of total_target (default 10000) from all eligible tasks
            with L_seed >= weak.lo (=0.10); deterministic via task_seed (42).
  - bucket label is attached per task for reporting strata only — it does
            NOT set a per-bucket quota.

Saved fields (all aligned to row index i in [0, n_selected)):
  src_idx        (n,) int64    original index into pilot_20k.npz
  bucket         (n,) <U16     'weak' / 'medium-weak' / 'medium' / 'strong'
  cs_p0          (n, 3) f32
  cs_line_dir    (n, 3) f32
  cs_n_target    (n, 3) f32
  q0_seed        (n, 7) f32    pilot's q0 seed (used by cells A and C)
  L_seed         (n,)  f32
  max_label_L    (n,)  f32     max_k labels_L_clean[i, :n_labels[i]]
  max_label_q    (n, 7) f32    labels_q0[i, argmax_k labels_L_clean[i, :]]
  n_labels       (n,) int32

Usage:
    python -m Yuan.system_eval.build_eval_set --config Yuan/system_eval/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


DEFAULT_KEEP_STATUSES = ('kept', 'edge', 'edge_seed_fallback')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--max-total', type=int, default=None,
                   help='cap the total selected (e.g. 2000 for pilot).')
    p.add_argument('--out-name', default=None,
                   help='override output filename (default from config).')
    p.add_argument('--pilot-frac', type=float, default=None,
                   help='shortcut: scale each bucket target by this fraction.')
    return p.parse_args()


def assign_bucket(L_seed: float, bucket_def: dict) -> str | None:
    for name, (lo, hi) in bucket_def.items():
        # closed-open on lo, except the strong bucket whose upper is inf
        hi_v = float('inf') if hi in (None, 'inf', '.inf') else float(hi)
        if lo <= L_seed < hi_v:
            return name
    return None


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    npz_path = Path(cfg['dataset']['pilot_npz'])
    pc_path  = Path(cfg['dataset']['plane_collision_npz'])
    bucket_def = {k: tuple(v) for k, v in cfg['dataset']['buckets'].items()}
    total_target = int(cfg['dataset']['total_target'])
    if args.pilot_frac is not None:
        total_target = max(1, int(round(total_target * args.pilot_frac)))
        print(f'[build_eval_set] pilot fraction {args.pilot_frac} → total_target={total_target}')

    z = np.load(npz_path, allow_pickle=False)
    pc = np.load(pc_path, allow_pickle=False)

    status = z['status']
    n_labels = z['n_labels'].astype(np.int32)
    keep = np.isin(status, DEFAULT_KEEP_STATUSES) & (n_labels >= 1)
    safe = ~pc['any_label_collides'].astype(bool)        # safe = label-side not pierced
    eligible = keep & safe
    eligible_idx = np.where(eligible)[0]
    print(f'[build_eval_set] eligible (kept ∧ safe) = {len(eligible_idx)} / {len(status)}')

    L_seed = z['L_seed'].astype(np.float32)
    # Per-task max label L and its argmax (oracle seed)
    labels_q0 = z['labels_q0'].astype(np.float32)
    labels_L  = z['labels_L_clean'].astype(np.float32)

    def per_task_max_L_q(i):
        n = int(n_labels[i])
        ls = labels_L[i, :n]
        mask = np.isfinite(ls)
        if not mask.any():
            return float('nan'), labels_q0[i, 0].copy()
        valid = np.where(mask)[0]
        kbest = int(valid[np.argmax(ls[valid])])
        return float(ls[kbest]), labels_q0[i, kbest].copy()

    # Flat random sampling: draw `total_target` from all eligible tasks that
    # fall in a bucket (L_seed >= weak.lo). Buckets are tagged afterwards for
    # reporting strata only — they no longer set per-bucket quotas.
    rng = np.random.default_rng(int(cfg['dataset']['task_seed']))

    bucket_of: dict[int, str] = {}
    for i in eligible_idx:
        b = assign_bucket(float(L_seed[i]), bucket_def)
        if b is not None:
            bucket_of[int(i)] = b
    pool = np.fromiter(bucket_of.keys(), dtype=np.int64, count=len(bucket_of))
    print(f'[build_eval_set] sampling pool (eligible ∧ in-bucket) = {len(pool)}')

    rng.shuffle(pool)
    actual = min(total_target, len(pool))
    if actual < total_target:
        print(f'[build_eval_set] WARN only {actual} in pool (target {total_target})')
    else:
        print(f'[build_eval_set] drew {actual} of {len(pool)} (target {total_target})')
    chosen = pool[:actual].astype(np.int64)
    chosen_bucket = np.asarray([bucket_of[int(i)] for i in chosen], dtype='<U16')

    # Optional global cap (mostly for pilot)
    if args.max_total is not None and len(chosen) > args.max_total:
        # Random subset preserving bucket fractions
        idx = np.arange(len(chosen))
        rng.shuffle(idx)
        idx = idx[:args.max_total]
        chosen = chosen[idx]
        chosen_bucket = chosen_bucket[idx]
        print(f'[build_eval_set] capped at {len(chosen)} (--max-total {args.max_total})')

    # Build output fields
    max_label_L = np.zeros(len(chosen), dtype=np.float32)
    max_label_q = np.zeros((len(chosen), 7), dtype=np.float32)
    for k, i in enumerate(chosen):
        L, q = per_task_max_L_q(int(i))
        max_label_L[k] = L
        max_label_q[k] = q

    out_dir = Path(cfg['output']['root'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or cfg['output']['eval_set_name']
    out_path = out_dir / out_name

    np.savez(
        out_path,
        src_idx=chosen.astype(np.int64),
        bucket=chosen_bucket,
        cs_p0=z['cs_p0'][chosen].astype(np.float32),
        cs_line_dir=z['cs_line_dir'][chosen].astype(np.float32),
        cs_n_target=z['cs_n_target'][chosen].astype(np.float32),
        q0_seed=z['q0_seeds'][chosen].astype(np.float32),
        L_seed=L_seed[chosen],
        max_label_L=max_label_L,
        max_label_q=max_label_q,
        n_labels=n_labels[chosen],
        # tag bucket counts for the report
        bucket_counts=np.array(
            [(b, int((chosen_bucket == b).sum())) for b in bucket_def.keys()],
            dtype=[('bucket', '<U16'), ('count', 'i8')]),
    )
    print(f'[build_eval_set] wrote {out_path} ({len(chosen)} tasks)')

    # Sidecar JSON for human-readable provenance
    meta = {
        'pilot_npz': str(npz_path),
        'plane_collision_npz': str(pc_path),
        'buckets': bucket_def,
        'total_target': total_target,
        'task_seed': int(cfg['dataset']['task_seed']),
        'eligible_n': int(eligible.sum()),
        'selected_n': int(len(chosen)),
        'per_bucket': {b: int((chosen_bucket == b).sum()) for b in bucket_def.keys()},
    }
    (out_dir / (out_name.rsplit('.', 1)[0] + '.meta.json')).write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
