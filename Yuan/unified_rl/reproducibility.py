"""Global random-state helpers for reproducible unified-RL training."""
from __future__ import annotations

import numpy as np
import torch


def device_identity(device: torch.device) -> str:
    """Canonical device identity used by strict resume provenance."""
    device = torch.device(device)
    if device.type == 'cuda':
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f'cuda:{index}'
    return str(device)


def seed_global_rng(seed: int) -> None:
    """Seed NumPy and torch's process-global CPU/CUDA generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def global_rng_state(device: torch.device) -> dict:
    """Capture process-global RNGs used by the unified training loops."""
    return {
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': (torch.cuda.get_rng_state(device)
                           if device.type == 'cuda' else None),
    }


def restore_global_rng(state: dict, device: torch.device) -> None:
    """Restore a state produced by :func:`global_rng_state`.

    Missing keys are accepted for backward compatibility with early unified
    checkpoints, which did not persist NumPy's state.
    """
    if state.get('numpy_rng_state') is not None:
        np.random.set_state(state['numpy_rng_state'])
    if state.get('torch_rng_state') is not None:
        torch.set_rng_state(state['torch_rng_state'].cpu())
    if device.type == 'cuda' and state.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device)
