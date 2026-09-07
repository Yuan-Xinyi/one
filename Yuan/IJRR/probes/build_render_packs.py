"""Per-task render packs for the three-way video: each trajectory
resampled on a 1cm arc grid + TCP tip / tool axis from FK."""
import sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from Yuan.IJRR.eval import line_bound as lb
OUT = MAIN/'runs/paper_fill/search_compare'
dev = torch.device('cuda')
env = lb.build_env(dev, 'stock', 512)
kin = env.kin
tz = np.load(MAIN/'runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
TASKS = [6030, 6142, 3512, 5087, 3953, 8942, 9771, 7179, 653, 6036]
GRID = 0.01

def resample(qs, ss):
    ss = np.asarray(ss, np.float64); qs = np.asarray(qs, np.float64)
    keep = np.concatenate([[True], np.diff(ss) > 1e-9])
    ss, qs = ss[keep], qs[keep]
    g = np.arange(0.0, ss[-1] + 1e-9, GRID)
    qg = np.stack([np.interp(g, ss, qs[:, j]) for j in range(7)], 1)
    return qg.astype(np.float32)

def fk(qg):
    q = torch.tensor(qg, dtype=kin.dtype, device=dev)
    p, R, _, _ = kin.tcp_fk_jac(q)
    return (p.cpu().numpy().astype(np.float32),
            R[:, :, 2].cpu().numpy().astype(np.float32))

for ti in TASKS:
    d = np.load(OUT/f't{ti}_three_way.npz')
    pack = {'p0': tz['cs_p0'][ti].astype(np.float32),
            'd': (tz['cs_line_dir'][ti]
                  / np.linalg.norm(tz['cs_line_dir'][ti])).astype(np.float32),
            'lpw': np.float32(d['ref'])}
    for tag, qk, sk in (('cls', 'cls_q', 'cls_s'), ('rl', 'rl_q', 'rl_s'),
                        ('srch', 'search_q', 'search_s')):
        qg = resample(d[qk], d[sk])
        tip, zax = fk(qg)
        pack[f'{tag}_q'], pack[f'{tag}_tip'], pack[f'{tag}_zax'] = qg, tip, zax
    np.savez_compressed(OUT/f't{ti}_render_pack.npz', **pack)
    print(f't{ti}: cls {len(pack["cls_q"])} rl {len(pack["rl_q"])} '
          f'srch {len(pack["srch_q"])} frames', flush=True)
print('packs done')
