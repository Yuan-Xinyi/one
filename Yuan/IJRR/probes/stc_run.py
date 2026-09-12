"""Draw a Spanning Tree Coverage cycle with the proposed method.

Same protocol as the one-stroke figures: pointwise ladder over placements for
the largest admissible size, candidate pool at the cycle's start, critic
start selection, one continuous rollout, completion verified by arc length.
"""
import sys, math, time
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
CONE, M = 30, 4

def main():
    p2 = stc_lib.stc_path(M)[:, :2]
    Lu = float(np.linalg.norm(np.diff(p2, axis=0), axis=1).sum())
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    z = np.load(OUT / 'down_field.npz')
    gx, gz, st = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    LAD = np.round(np.arange(0.04, 1.001, 0.02), 3)
    t0 = time.time()
    pw = osr.pointwise_scale(z[f'F{CONE}'], gx, gz, st, p2, places, LAD)
    print(f'pointwise max {pw.max()*100:.0f} cm ({time.time()-t0:.0f}s)',
          flush=True)
    # size ladder downward from the bound until one placement completes
    for S in LAD[LAD <= pw.max() + 1e-9][::-1]:
        for c in np.argsort(-pw)[:4]:
            if pw[c] < S - 1e-9:
                continue
            place = places[c]
            W, T = osr.figure_world(p2, S, place)
            B = 8
            Wb = np.repeat(W[None], B, 0); Tb = np.repeat(T[None], B, 0)
            npt = np.full(B, len(W), np.int64)
            d0 = np.repeat(T[:1], B, 0)
            p5 = osl.resample(p2, 0.005 / S)
            W05 = (np.concatenate([S * p5,
                                   np.zeros((len(p5), 1), np.float32)], 1)
                   + place).astype(np.float32)
            CQ, CT = start_pool(env0, tree, T0, W05[:1].repeat(4, 0),
                                np.tile(NDOWN, (4, 1)), CONE)
            if not len(CQ):
                continue
            envp, _, ag = osr.build_env(CONE, B)
            pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
            q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
            a, (tr, ar) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                                   want_traj=True)
            del envp; torch.cuda.empty_cache()
            need = S * Lu
            ok = a[0] >= need - 3 * osr.STEPW
            print(f'S={S*100:.0f}cm at ({place[0]:+.2f},{place[1]:+.2f}) '
                  f'bed {place[2]:+.2f}: drew {a[0]:.2f}/{need:.2f} m '
                  f'-> {"COMPLETE" if ok else "no"} ({time.time()-t0:.0f}s)',
                  flush=True)
            if not ok:
                continue
            # truncate at completion, save the pack
            stop = int(np.searchsorted(ar[0], need - 3 * osr.STEPW)) + 1
            Q = tr[0][:stop].astype(np.float32)
            dt_ = env0.kin.dtype
            pfk, R, _, _ = env0.kin.tcp_fk_jac(
                torch.as_tensor(Q, device='cuda', dtype=dt_))
            np.savez_compressed(
                OUT / 'ompl' / 'stc_pack.npz', figure=W.astype(np.float32),
                size=np.float32([S, *place]), need=np.float32(need),
                q_b=Q, tip_b=pfk.cpu().numpy().astype(np.float32),
                zax_b=R[:, :, 2].cpu().numpy().astype(np.float32),
                arc_b=ar[0][:stop].astype(np.float32),
                subcells=np.int32(4 * M * M))
            print(f'saved: STC M={M}, {4*M*M} subcells, side {S*100:.0f} cm, '
                  f'stroke {need:.2f} m', flush=True)
            return
    print('no completing size found')

if __name__ == '__main__':
    main()
