"""Multi-process launcher for dataset build.

Splits a total N tasks into K equal-ish contiguous chunks of the
line_distribution.valid_idx pool, launches K subprocesses (each runs
`Yuan.seed_selection.smm.build_worker` on its slice with an isolated
cache_name), waits for all to finish, then concatenates the K chunk NPZs
into a single final NPZ at `<out_dir>/<cache_name>.npz`.

Resume:
  * Re-running with the same `--cache-name` is the resume path.
  * Each chunk has its own cache_name (`<root>_chunk{i}`). On re-run:
      - if `<chunk>.npz` exists → chunk already done; skip launch.
      - if `<chunk>.partial-*.npz` exist → launch subprocess; it picks up
        where it left off (handled inside dataset_builder.build_dataset).
      - otherwise → launch fresh.
  * SIGINT/SIGTERM to the parent is forwarded to all live children so
    partial NPZs flush and Ctrl-C doesn't leak orphan PIDs.

Usage:
    python -m Yuan.seed_selection.smm.build_parallel \\
        --total-n-tasks 10000 --n-procs 4 --cache-name pilot_10k
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parents[3] / "Yuan/seed_selection/runs/pilot_20k"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total-n-tasks', type=int, default=100,
                   help='total tasks across all workers (click-run default: 100 for smoke test)')
    p.add_argument('--n-procs', type=int, default=4)
    p.add_argument('--cache-name', default='click_run_smoke',
                   help='name of the final merged NPZ (without .npz)')
    p.add_argument('--start-seed', type=int, default=0,
                   help='line_distribution offset for chunk 0 (others continue contiguously)')
    p.add_argument('--checkpoint-interval', type=int, default=100,
                   help='speedtest --checkpoint-interval: save partial every N tasks per chunk')
    p.add_argument('--stagger-seconds', type=float, default=3.0,
                   help='delay between launching procs to avoid simultaneous CUDA init')
    p.add_argument('--keep-chunks', action='store_true',
                   help='do not delete the per-chunk NPZ/meta after merging')
    p.add_argument('--task-npz', default=None,
                   help='source tasks from this NPZ instead of LineDistribution; '
                        'chunks slice it contiguously by --start-seed offset.')
    p.add_argument('--out-dir', default=None,
                   help='override default output directory (pilot_20k).')
    return p.parse_args()


def merge_chunk_npzs(chunk_paths: list[Path], out_path: Path) -> None:
    """Concatenate per-chunk NPZ files. Scalar fields (hyperparams_json, norm_*
    stats) are dropped and re-derived from the merged dataset so they reflect
    the full N, not just chunk 0's slice."""
    from Yuan.seed_selection.smm.dataset_builder import _compute_norm_stats

    arrays: dict[str, list] = {}
    hyperparams_json = None
    skip_concat_keys = {
        'hyperparams_json',
        'norm_p0_mean', 'norm_p0_std',
        'norm_line_dir_mean', 'norm_line_dir_std',
        'norm_n_target_mean', 'norm_n_target_std',
        'norm_labels_q0_mean', 'norm_labels_q0_std',
    }
    for path in chunk_paths:
        z = np.load(path, allow_pickle=False)
        for k in z.keys():
            if k == 'hyperparams_json':
                if hyperparams_json is None:
                    hyperparams_json = str(z[k])
                continue
            if k in skip_concat_keys:
                continue
            arrays.setdefault(k, []).append(z[k])
    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    norm_stats = _compute_norm_stats(merged)
    merged.update(norm_stats)
    if hyperparams_json is not None:
        merged['hyperparams_json'] = np.array(hyperparams_json)
    np.savez(out_path, **merged)


def merge_chunk_metas(chunk_meta_paths: list[Path], out_path: Path,
                       n_procs: int, total_n: int, wall_seconds: float) -> None:
    agg_counters: dict[str, int] = {}
    agg_errors: list = []
    total_build_time = 0.0
    per_chunk = []
    for path in chunk_meta_paths:
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        for k, v in (m.get('status_counts') or m.get('counters') or {}).items():
            agg_counters[k] = agg_counters.get(k, 0) + int(v)
        agg_errors.extend(m.get('errors', []))
        chunk_elapsed = float(m.get('elapsed_seconds') or m.get('wall_seconds') or 0.0)
        total_build_time += chunk_elapsed
        per_chunk.append({
            'n_tasks': m.get('n_tasks'),
            'elapsed_seconds': chunk_elapsed,
            'status_counts': m.get('status_counts') or m.get('counters'),
        })
    out_path.write_text(json.dumps({
        'n_tasks': int(total_n),
        'n_procs': int(n_procs),
        'wall_seconds_parallel': float(wall_seconds),
        'wall_seconds_summed': float(total_build_time),
        'parallel_speedup': float(total_build_time / max(wall_seconds, 1e-6)),
        'status_counts': agg_counters,
        'errors': agg_errors,
        'per_chunk': per_chunk,
    }, indent=2))


def _detect_chunk_state(chunk_cache: str) -> tuple[str, int]:
    """Inspect on-disk artifacts for a chunk.

    Returns (state, n_done) where:
      state ∈ {'done', 'partial', 'fresh'}
      n_done is the number of tasks already serialized (0 if 'fresh',
      the last partial-N index if 'partial', or full N if 'done').
    """
    final = OUT_DIR / f'{chunk_cache}.npz'
    if final.exists():
        try:
            z = np.load(final, allow_pickle=False)
            n = int(z['L_seed'].shape[0])
            return ('done', n)
        except Exception:
            pass
    # Look for partials.
    partials = sorted(OUT_DIR.glob(f'{chunk_cache}.partial-*.npz'),
                      key=lambda p: int(p.stem.split('-')[-1]))
    if partials:
        last = partials[-1]
        n = int(last.stem.split('-')[-1])
        return ('partial', n)
    return ('fresh', 0)


def main():
    global OUT_DIR
    args = parse_args()
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split into K chunks (last chunk gets the remainder).
    per = args.total_n_tasks // args.n_procs
    rem = args.total_n_tasks - per * args.n_procs
    chunks = []
    seed = args.start_seed
    for i in range(args.n_procs):
        n = per + (1 if i < rem else 0)
        chunks.append({'i': i, 'seed': seed, 'n': n})
        seed += n

    print(f'[parallel] {args.n_procs} chunks of total {args.total_n_tasks} tasks:')
    chunk_cache_names = []
    for c in chunks:
        chunk_cache = f'{args.cache_name}_chunk{c["i"]}'
        chunk_cache_names.append(chunk_cache)
        state, n_done = _detect_chunk_state(chunk_cache)
        c['state'] = state
        c['n_done'] = n_done
        c['cache'] = chunk_cache
        if state == 'done':
            print(f'   chunk{c["i"]}: seed={c["seed"]:>6}  n={c["n"]:>5}  ✓ already DONE ({n_done} tasks)')
        elif state == 'partial':
            print(f'   chunk{c["i"]}: seed={c["seed"]:>6}  n={c["n"]:>5}  ↻ RESUME from partial-{n_done}')
        else:
            print(f'   chunk{c["i"]}: seed={c["seed"]:>6}  n={c["n"]:>5}  • fresh')

    to_launch = [c for c in chunks if c['state'] != 'done']
    if not to_launch:
        print(f'[parallel] all {args.n_procs} chunks already done; skipping launch, merging now.')
    else:
        print(f'[parallel] launching {len(to_launch)} chunks '
              f'(skipping {args.n_procs - len(to_launch)} that are done)')

    log_dir = OUT_DIR / f'{args.cache_name}_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # LD_LIBRARY_PATH fix for matplotlib's libstdc++ on this conda env.
    conda_lib = os.path.join(sys.prefix, "lib")
    child_env = dict(os.environ)
    child_env["LD_LIBRARY_PATH"] = conda_lib + ":" + child_env.get("LD_LIBRARY_PATH", "")

    t0 = time.time()
    procs = []
    # Signal forwarding: on SIGINT/SIGTERM to parent, send same signal to all
    # live children so dataset_builder flushes its current partial cleanly.
    def _forward_signal(signum, frame):
        print(f'\n[parallel] received signal {signum}; forwarding to {len(procs)} children...', flush=True)
        for pinfo in procs:
            try:
                if pinfo['proc'].poll() is None:
                    pinfo['proc'].send_signal(signum)
            except Exception as e:
                print(f'  failed to signal pid={pinfo["proc"].pid}: {e}')
        # Wait briefly for children to flush, then re-raise.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if all(p['proc'].poll() is not None for p in procs):
                break
            time.sleep(0.5)
        sys.exit(130 if signum == signal.SIGINT else 143)
    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    for c in to_launch:
        log_path = log_dir / f'chunk{c["i"]}.log'
        # Append (not truncate) so a resume preserves the previous run's history.
        mode = 'a' if c['state'] == 'partial' else 'w'
        cmd = [
            sys.executable, '-u', '-m', 'Yuan.seed_selection.smm.build_worker',
            '--n-tasks', str(c['n']),
            '--seed', str(c['seed']),
            '--cache-name', c['cache'],
            '--checkpoint-interval', str(args.checkpoint_interval),
        ]
        if args.task_npz is not None:
            cmd += ['--task-npz', args.task_npz]
        if args.out_dir is not None:
            cmd += ['--out-dir', args.out_dir]
        log_file = open(log_path, mode)
        if c['state'] == 'partial':
            log_file.write(f'\n\n===== RESUME from partial-{c["n_done"]} at {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n\n')
            log_file.flush()
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=child_env,
                             cwd=str(OUT_DIR.parents[3]))
        procs.append({'proc': p, 'cache': c['cache'], 'log': log_path, 'log_file': log_file,
                      'i': c['i'], 'n': c['n'], 'state': c['state'], 'n_done': c['n_done']})
        msg = f'   chunk{c["i"]}: launched pid={p.pid} → {log_path}'
        if c['state'] == 'partial':
            msg += f'  (resume from {c["n_done"]}/{c["n"]})'
        print(msg, flush=True)
        if args.stagger_seconds > 0 and c is not to_launch[-1]:
            time.sleep(args.stagger_seconds)

    # Wait for all
    if procs:
        print(f'[parallel] waiting for {len(procs)} processes...', flush=True)
    failed = []
    for pinfo in procs:
        pinfo['proc'].wait()
        pinfo['log_file'].close()
        if pinfo['proc'].returncode != 0:
            failed.append(pinfo)
            print(f'   chunk{pinfo["i"]}: FAILED rc={pinfo["proc"].returncode}, see {pinfo["log"]}', flush=True)
        else:
            print(f'   chunk{pinfo["i"]}: done', flush=True)

    wall = time.time() - t0
    if failed:
        print(f'[parallel] {len(failed)}/{len(procs)} chunks failed; not merging.', flush=True)
        print(f'[parallel] re-run the same command to resume the failed chunks.', flush=True)
        sys.exit(1)

    # Belt-and-suspenders: even if every child exited with rc=0, verify each
    # chunk produced its final NPZ. (Graceful-shutdown exits with rc=130, so
    # the failed[] check above already covers that; this catches a child that
    # erroneously returned 0 without finishing.)
    missing_finals = [c for c in chunk_cache_names
                       if not (OUT_DIR / f'{c}.npz').exists()]
    if missing_finals:
        print(f'[parallel] missing final NPZ for: {missing_finals}', flush=True)
        print(f'[parallel] not merging. Re-run the same command to resume.', flush=True)
        sys.exit(1)

    # Merge
    print(f'[parallel] merging {args.n_procs} chunks into {args.cache_name}.npz ...', flush=True)
    chunk_npz_paths = [OUT_DIR / f'{c}.npz' for c in chunk_cache_names]
    chunk_meta_paths = [OUT_DIR / f'{c}.meta.json' for c in chunk_cache_names]
    final_npz = OUT_DIR / f'{args.cache_name}.npz'
    final_meta = OUT_DIR / f'{args.cache_name}.meta.json'
    merge_chunk_npzs(chunk_npz_paths, final_npz)
    merge_chunk_metas(chunk_meta_paths, final_meta, args.n_procs, args.total_n_tasks, wall)

    # Verify total N
    z = np.load(final_npz, allow_pickle=False)
    actual_n = len(z['L_seed'])
    assert actual_n == args.total_n_tasks, f'merged N {actual_n} != requested {args.total_n_tasks}'
    print(f'[parallel] merged: {final_npz} ({actual_n} tasks)')
    print(f'[parallel] meta:   {final_meta}')

    serial_eq = json.loads(final_meta.read_text())['wall_seconds_summed']
    speedup = serial_eq / max(wall, 1e-6)
    print(f'[parallel] WALL = {wall:.1f}s  (summed serial {serial_eq:.1f}s, speedup {speedup:.2f}x)')

    if not args.keep_chunks:
        for p in chunk_npz_paths + chunk_meta_paths:
            if p.exists():
                p.unlink()
        for c in chunk_cache_names:
            for pp in OUT_DIR.glob(f'{c}.partial-*.npz'):
                pp.unlink()
        print(f'[parallel] cleaned per-chunk files (keep with --keep-chunks)')


if __name__ == '__main__':
    main()
