"""E driver: run the standard train entry point with LineDistribution.sample
monkey-patched to draw tasks proportionally to precomputed weights
(pool_weights.npz, aligned to pool row order — same seed/params → same
cached pool)."""
import sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch
import Yuan.IJRR.env.line_distribution as LD

_W = torch.tensor(np.load(
    '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/pool_weights.npz'
)['w'])


def sample_weighted(self, n, generator=None):
    gen = generator if generator is not None else self._gen
    valid_idx = torch.nonzero(self.valid_mask, as_tuple=False).squeeze(-1)
    w = _W.to(self.device)[valid_idx].clamp_min(1e-9)
    pick = torch.multinomial(w, n, replacement=True, generator=gen)
    idx = valid_idx[pick]
    return {
        "q0": self.q_pool[idx],
        "line_dir": self.line_dir_pool[idx],
        "n_target": self.n_target_pool[idx],
        "amp": self.amp_pool[idx],
        "wavelen": self.wavelen_pool[idx],
    }


LD.LineDistribution.sample = sample_weighted
from Yuan.IJRR.stage2_traj import train
train.main()
