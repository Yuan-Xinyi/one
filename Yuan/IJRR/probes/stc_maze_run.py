"""Roll one STC maze at density M and save the render pack, including the
spanning-tree skeleton for the underlay."""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
import onestroke_lib as osl
import onestroke_run as osr
import stc_lib
from print_raster import start_pool

NDOWN = np.float32([0, 0, -1])
CONE, S = 30, 0.60


def tree_segments(M, seed):
    rng = np.random.default_rng(seed)
    tree, seen, stack = set(), {(0, 0)}, [(0, 0)]
    while stack:
        c = stack[-1]
        nb = [(c[0]+d[0], c[1]+d[1]) for d in ((1,0),(-1,0),(0,1),(0,-1))]
        nb = [n for n in nb if 0 <= n[0] < M and 0 <= n[1] < M
              and n not in seen]
        if not nb:
            stack.pop(); continue
        n = nb[rng.integers(len(nb))]
        tree.add((c, n)); seen.add(n); stack.append(n)
    s = 1.0 / M
    ctr = lambda c: (-0.5+(c[0]+0.5)*s, -0.5+(c[1]+0.5)*s)
    return np.array([[ctr(a), ctr(b)] for a, b in tree], np.float32)


def main(M, seed):
    p2 = stc_lib.stc_path(M, seed=seed)[:, :2]
    Lu = float(np.linalg.norm(np.diff(p2, axis=0), axis=1).sum())
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    z = np.load(OUT / 'down_field.npz')
    gx, gz, st = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    LAD = np.round(np.arange(0.04, 1.001, 0.02), 3)
    pw = osr.pointwise_scale(z[f'F{CONE}'], gx, gz, st, p2, places, LAD)
    for place in [places[c] for c in np.argsort(-pw)[:4]
                  if pw[c] >= S - 1e-9]:
        q1 = osl.resample(p2, 0.002 / S)
        W = (np.concatenate([S*q1, np.zeros((len(q1), 1), np.float32)], 1)
             + place).astype(np.float64)
        T = np.gradient(W, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
        need = S * Lu
        B = 8
        Wb = np.repeat(W[None].astype(np.float32), B, 0)
        Tb = np.repeat(T[None].astype(np.float32), B, 0)
        npt = np.full(B, len(W), np.int64)
        d0 = np.repeat(T[:1].astype(np.float32), B, 0)
        CQ, CT = start_pool(env0, tree, T0,
                            W[:1].astype(np.float32).repeat(4, 0),
                            np.tile(NDOWN, (4, 1)), CONE)
        if not len(CQ):
            continue
        envp, _, ag = osr.build_env(CONE, B,
                                    max_steps=int(need/0.005*2*1.3))
        pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
        q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
        t0 = time.time()
        a, (tr, ar) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                               want_traj=True)
        del envp; torch.cuda.empty_cache()
        ok = a[0] >= need - 3*osr.STEPW
        print(f'M={M}: drew {a[0]:.2f}/{need:.2f} m '
              f'-> {"COMPLETE" if ok else "no"} ({time.time()-t0:.0f}s)',
              flush=True)
        if not ok:
            continue
        stop = int(np.searchsorted(ar[0], need - 3*osr.STEPW)) + 1
        Q = tr[0][:stop].astype(np.float32)
        dt_ = env0.kin.dtype
        tips, zaxs = [], []
        for lo in range(0, len(Q), 4096):
            p_, R_, _, _ = env0.kin.tcp_fk_jac(
                torch.as_tensor(Q[lo:lo+4096], device='cuda', dtype=dt_))
            tips.append(p_.cpu().numpy()); zaxs.append(R_[:,:,2].cpu().numpy())
        seg = tree_segments(M, seed) * S
        seg3 = np.concatenate([seg, np.zeros((*seg.shape[:2], 1),
                                             np.float32)], -1) + place
        np.savez_compressed(
            OUT/'ompl'/f'stc_maze{M}_pack.npz',
            figure=W.astype(np.float32), size=np.float32([S, *place]),
            need=np.float32(need), tree=seg3.astype(np.float32),
            q_b=Q, tip_b=np.concatenate(tips).astype(np.float32),
            zax_b=np.concatenate(zaxs).astype(np.float32),
            arc_b=ar[0][:stop].astype(np.float32))
        print(f'SAVED M={M}: {4*M*M} subcells, {need:.1f} m', flush=True)
        return


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
