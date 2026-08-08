"""The SetSel selector and the two helpers the campaign needs.

Extracted from ``Yuan/unified_rl/ikpool_bidir.py``. That module also carried
the bidirectional expert-iteration loop (forward controller adaptation, C1
relabelling, the 2x2 arms); the final system keeps only the RL and the hybrid
controller, so the loop and its dependencies (``seed_distribution``,
``seed_deployment`` and the PPO re-entry) are not part of this copy.
"""
import torch
import torch.nn as nn


class SetSel(nn.Module):
    """Per-candidate score + feasibility(metres) with a mean-pool set context."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.feas = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, X, V):
        e = self.enc(X)
        vf = V.unsqueeze(-1).float()
        ctx = (e * vf).sum(1) / vf.sum(1).clamp_min(1)
        h = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
        return self.score(h).squeeze(-1), self.feas(h).squeeze(-1)



@torch.no_grad()
def _picks(nets, mu, sd, X, V):
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    score = torch.stack([n(Xz, V)[0].masked_fill(~V, -1e9) for n in nets]).mean(0)
    return score.argmax(1)




# ---------------------------------------------------------------- stages
