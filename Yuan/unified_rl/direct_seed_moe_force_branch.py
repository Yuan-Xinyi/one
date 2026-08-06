"""Create a training-only checkpoint that always deploys one MoE branch."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from Yuan.unified_rl.checkpoint import atomic_torch_save
from Yuan.unified_rl.direct_seed_rl import (
    direct_seed_moe_checkpoint,
    load_direct_seed_moe_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expert-index', required=True, type=int)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    actor, _, payload = load_direct_seed_moe_checkpoint(source)
    if not 0 <= args.expert_index < actor.config.n_experts:
        raise ValueError('--expert-index is outside the actor expert range')
    with torch.no_grad():
        actor.gate.weight.zero_()
        actor.gate.bias.fill_(-1.0)
        actor.gate.bias[args.expert_index] = 1.0
    metadata = {
        'method': 'forced-hard-moe-branch-training-collection-only',
        'forced_expert_index': args.expert_index,
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': _sha256(source),
        'source_update_step': int(payload['update_step']),
        'deployment_protocol': {
            'one_seed': True,
            'candidate_enumeration': 0,
            'controller_probes': 0,
        },
    }
    checkpoint = direct_seed_moe_checkpoint(
        actor, update_step=int(payload['update_step']),
        metadata=metadata)
    atomic_torch_save(checkpoint, output)
    print(
        f'[force-moe-branch] expert={args.expert_index} -> {output}',
        flush=True)


if __name__ == '__main__':
    main()
