"""45-D candidate features for the SetSel selector.

Extracted verbatim from ``Yuan/unified_rl/offline_seed_ensemble_train.py``
(``_build_features``). Only this one function was on the final path; importing
the original module dragged in six further legacy modules (offline_seed_train,
provenance, reproducibility, seed_policy, seed_distribution, seed_deployment)
that the IKSel mainline never calls.
"""
from __future__ import annotations

import torch

from Yuan.IJRR.stage1_seed.candidate_batch import CachedSeedCandidateDataset
from Yuan.IJRR.stage1_seed.features import initial_observation_features


def _build_features(kin, dataset: CachedSeedCandidateDataset,
                    chunk_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        batch = dataset.batch.index_select(torch.arange(start, end)).to(
            kin.device, dtype=kin.dtype)
        parts.append(initial_observation_features(
            kin, batch, include_ray_error=True, include_log_manip=True,
            include_directional_dynamics=True).cpu())
    result = torch.cat(parts, dim=0)
    if result.shape[-1] != 45:
        raise RuntimeError(f'expected 45-D ensemble features, got {result.shape[-1]}')
    return result
