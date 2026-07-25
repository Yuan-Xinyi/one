"""Content-addressed provenance for resumable unified-RL runs."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from Yuan.unified_rl.checkpoint import resolve_controller_dir


def file_fingerprint(path: str | Path) -> dict[str, str | int]:
    """Return a canonical path plus a SHA-256 content identity."""
    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with open(resolved, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {
        'path': str(resolved),
        'size': resolved.stat().st_size,
        'sha256': digest.hexdigest(),
    }


def controller_fingerprint(ckpt_dir: str | Path) -> dict[str, Any]:
    """Fingerprint the two files that define a controller checkpoint."""
    directory = resolve_controller_dir(ckpt_dir)
    return {
        'path': str(directory),
        'agent': file_fingerprint(directory / 'agent.pt'),
        'config': file_fingerprint(directory / 'config.yaml'),
    }


def state_dict_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Stable SHA-256 identity for a tensor state dict."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_same_provenance(saved: dict[str, Any],
                           current: dict[str, Any]) -> None:
    """Reject a resume whose immutable inputs or settings have changed."""
    differences: list[str] = []

    def compare(prefix: str, old: Any, new: Any) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            old_keys = set(old)
            new_keys = set(new)
            for key in sorted(old_keys | new_keys):
                label = f'{prefix}.{key}' if prefix else key
                if key not in old:
                    differences.append(f'{label}: missing in checkpoint')
                elif key not in new:
                    differences.append(f'{label}: missing in current run')
                else:
                    compare(label, old[key], new[key])
            return
        if old != new:
            differences.append(f'{prefix}: checkpoint={old!r}, current={new!r}')

    compare('', saved, current)
    if differences:
        details = '\n  - '.join(differences[:20])
        suffix = '\n  - ...' if len(differences) > 20 else ''
        raise ValueError(
            'resume provenance mismatch:\n  - ' + details + suffix)
