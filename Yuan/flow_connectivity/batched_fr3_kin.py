"""Torch batched FR3 FK and TCP Jacobian — re-export shim.

The actual implementation lives in
``one.robots.manipulators.franka.fr3_pen.batched_fr3_kin``. This module reads
``cfg.TCP_OFFSET`` so RL call sites keep their config-driven default.
"""
from __future__ import annotations

import torch

import Yuan.flow_connectivity.config as cfg
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
    BatchedFR3Kinematics as _BatchedFR3Kinematics,
)


class BatchedFR3Kinematics(_BatchedFR3Kinematics):
    """RL-flavored FR3 batched FK: ``tcp_offset`` defaults to ``cfg.TCP_OFFSET``."""

    def __init__(self, device=None, dtype=torch.float32,
                 tcp_offset: float | None = None):
        if tcp_offset is None:
            tcp_offset = float(getattr(cfg, "TCP_OFFSET", 0.0))
        super().__init__(device=device, dtype=dtype, tcp_offset=tcp_offset)
