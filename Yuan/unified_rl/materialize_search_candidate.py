"""Materialize an immutable controller candidate from a search-distill epoch.

Search-distill keeps strict publication semantics and rolls back when a
controller-only confidence interval crosses zero.  Bidirectional optimization
still needs to test whether such a controller becomes useful after the
backward seed-policy update.  This utility turns one already-evaluated epoch
snapshot into a provenance-bound ``round_complete`` source without changing
its weights, so exhaustive candidate relabeling can form the matched 2x2.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import yaml

from Yuan.unified_rl.checkpoint import (
    atomic_torch_save,
    build_env_from_run,
    load_controller_agent,
    load_run_config,
)
from Yuan.unified_rl.provenance import (
    file_fingerprint,
    state_dict_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Publish one evaluated search-distill controller epoch.')
    parser.add_argument('--search-run', required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.epoch < 1:
        raise ValueError('--epoch must be positive')

    search_dir = Path(args.search_run).expanduser().resolve(strict=True)
    source_path = search_dir / 'unified.pt'
    snapshot_path = (
        search_dir / 'snapshots' / f'agent_epoch_{args.epoch:03d}.pt')
    if not source_path.is_file() or not snapshot_path.is_file():
        raise ValueError('search run lacks unified.pt or requested snapshot')
    out_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    if os.path.lexists(out_dir):
        raise FileExistsError(f'refusing to overwrite output: {out_dir}')
    out_dir.mkdir(parents=True)

    source = torch.load(source_path, map_location='cpu', weights_only=False)
    if not isinstance(source, dict):
        raise ValueError('search unified.pt must contain a dictionary')
    search_record = source.get('joint_controller_search_distill')
    if not isinstance(search_record, dict):
        raise ValueError('run is not a joint controller search-distill run')
    evaluation = next((
        item for item in search_record.get('evaluations', [])
        if int(item.get('epoch', -1)) == args.epoch
    ), None)
    if evaluation is None:
        raise ValueError('requested epoch has no immutable evaluation record')
    candidate_state = torch.load(
        snapshot_path, map_location='cpu', weights_only=True)
    if not isinstance(candidate_state, dict):
        raise ValueError('controller snapshot must contain a state dictionary')
    candidate_hash = state_dict_fingerprint(candidate_state)

    output_config = copy.deepcopy(load_run_config(search_dir))
    output_config.setdefault('unified', {})[
        'bidirectional_controller_candidate'] = {
            'format': 'search-distill-controller-candidate-v1',
            'source_search_run': str(search_dir),
            'epoch': args.epoch,
            'controller_only_evaluation': copy.deepcopy(evaluation),
            'purpose': 'backward-relabel-and-matched-joint-promotion',
            'inference': 'one-static-seed-one-controller-rollout-v1',
        }
    with open(out_dir / 'config.yaml', 'x') as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
    config_hash = file_fingerprint(out_dir / 'config.yaml')['sha256']

    # Reset Adam moments because the supervised actor update invalidates the
    # PPO optimizer carried by the rolled-back source.
    env = build_env_from_run(search_dir, 1, torch.device('cpu'))
    agent = load_controller_agent(search_dir, env, torch.device('cpu'))
    agent.load_state_dict(candidate_state)
    resume_lr = float(source.get('args', {}).get('controller_lr', 3e-4))
    optimizer = torch.optim.Adam(agent.parameters(), lr=resume_lr, eps=1e-5)

    result = copy.deepcopy(source)
    result['phase'] = 'round_complete'
    result['controller'] = {
        key: value.detach().cpu() for key, value in candidate_state.items()
    }
    result['controller_state_sha256'] = candidate_hash
    result['controller_run_config_sha256'] = config_hash
    result['controller_optimizer'] = optimizer.state_dict()
    candidate_record = {
        'format': 'search-distill-controller-candidate-v1',
        'source_search_checkpoint': file_fingerprint(source_path),
        'source_snapshot': file_fingerprint(snapshot_path),
        'epoch': args.epoch,
        'controller_only_evaluation': copy.deepcopy(evaluation),
        'controller_state_sha256': candidate_hash,
        'requires_matched_backward_promotion': True,
        'deployment_uses_search': False,
    }
    result['bidirectional_controller_candidate'] = candidate_record
    provenance = copy.deepcopy(result['provenance'])
    provenance['bidirectional_controller_candidate'] = copy.deepcopy(
        candidate_record)
    result['provenance'] = provenance
    atomic_torch_save(result['controller'], out_dir / 'agent.pt')
    atomic_torch_save(result, out_dir / 'unified.pt')
    print(
        f'[controller-candidate] epoch={args.epoch}; '
        f'progress={evaluation["policy_progress_mean_m"]:.6f} m; '
        f'controller_sha256={candidate_hash}; saved -> {out_dir}')


if __name__ == '__main__':
    main()
