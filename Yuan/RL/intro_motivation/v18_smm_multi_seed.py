"""Multi-seed SMM path-following sweep.

For N task seeds, runs path_following → saves smm_path scatter plot as
task_NN_seedSS_path_following.png (sequential numbering). Outputs a
manifest mapping each task number to its rollout_world command.

By default picks the top N hardest tasks from hardness_scan.jsonl.
Or pass --seeds 118,123,124,125,114 to specify.

After the script finishes you can launch the ONE viewer for any task:
    python -m Yuan.RL.intro_motivation.v18_smm_rollout_world --seed <S>
The task_NN number in the PNG filename matches the table printed.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_multi_seed
    python -m Yuan.RL.intro_motivation.v18_smm_multi_seed --seeds 118,123,124,125
    python -m Yuan.RL.intro_motivation.v18_smm_multi_seed --task-dof 6 --n-tasks 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def pick_hardest_seeds(jsonl_path: Path, n: int) -> list[int]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
    hard = [r for r in rows if r.get('is_hard', False)]
    hard.sort(key=lambda r: -float(r['L_self_rel_spread']))
    return [int(r['seed']) for r in hard[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='',
                        help='comma-separated seeds; if empty, uses top-n hard tasks')
    parser.add_argument('--n-tasks', type=int, default=4)
    parser.add_argument('--task-dof', type=str, choices=['5', '5strict', '6'],
                        default='5',
                        help='rollout mode passed to path_following')
    parser.add_argument('--n-per-branch', type=int, default=30)
    parser.add_argument('--hardness-jsonl', type=str,
                        default='Yuan/RL/intro_motivation/data/hardness_scan.jsonl')
    parser.add_argument('--out-dir', type=str,
                        default='Yuan/RL/intro_motivation/data/multi_seed')
    args = parser.parse_args()

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    else:
        hp = Path(args.hardness_jsonl)
        if not hp.exists():
            raise FileNotFoundError(
                f'{hp} not found. Run v18_smm_multi_seed --seeds=... or first run '
                f'v18_hardness_scan.')
        seeds = pick_hardest_seeds(hp, args.n_tasks)
    print(f'will process {len(seeds)} task(s): {seeds}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'dof{args.task_dof}'

    manifest: list[dict] = []
    for i, seed in enumerate(seeds):
        n = i + 1
        out_png = out_dir / f'task_{n:02d}_seed{seed}_{tag}_path_following.png'
        print(f'\n{"=" * 70}\n=== task_{n:02d}  seed={seed}  ({tag}) ===\n{"=" * 70}')
        cmd = [
            sys.executable, '-m', 'Yuan.RL.intro_motivation.v18_smm_path_following',
            '--seed', str(seed),
            '--task-dof', args.task_dof,
            '--n-per-branch', str(args.n_per_branch),
            '--out-png', str(out_png),
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f'  task_{n:02d} FAILED (exit {ret.returncode})')
            continue
        manifest.append({'task': n, 'seed': int(seed), 'png': str(out_png)})

    print(f'\n\n{"=" * 70}\n=== MANIFEST ({len(manifest)}/{len(seeds)} succeeded) ===\n{"=" * 70}')
    print(f'{"task":<7}{"seed":<7}{"plot file":<60}')
    for m in manifest:
        print(f'task_{m["task"]:02d} {m["seed"]:<7}{Path(m["png"]).name}')
    print('\nlaunch ONE viewer for any task:')
    for m in manifest:
        print(f'  task_{m["task"]:02d}: python -m Yuan.RL.intro_motivation.v18_smm_rollout_world --seed {m["seed"]}')

    manifest_txt = out_dir / f'manifest_{tag}.txt'
    with open(manifest_txt, 'w') as f:
        f.write(f'# Multi-seed sweep manifest\n')
        f.write(f'# task-dof: {args.task_dof}\n')
        f.write(f'# n-per-branch: {args.n_per_branch}\n')
        f.write(f'# {len(manifest)} task(s)\n\n')
        for m in manifest:
            f.write(f'task_{m["task"]:02d}\tseed={m["seed"]}\t'
                    f'plot={Path(m["png"]).name}\t'
                    f'rollout_world_cmd="python -m '
                    f'Yuan.RL.intro_motivation.v18_smm_rollout_world '
                    f'--seed {m["seed"]}"\n')
    print(f'\nsaved manifest: {manifest_txt}')


if __name__ == '__main__':
    main()
