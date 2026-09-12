"""One-stroke three-dimensional print: a stacked Spanning Tree Coverage slab.

The coverage cycle is closed, so it stacks: the reference path repeats the
cycle with a continuous z ramp of one layer height per lap. The pen never
leaves the surface, the branch never changes, and after eight laps a solid
slab with full coverage infill has been "printed" in one uninterrupted
stroke -- the vase-mode trick, which real printers can only do for hollow
shells, applied to a fully infilled solid.
"""
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
CONE, M, S, LAYERS, LAYER_H = 30, 5, 0.60, 8, 0.005


def main():
    p2 = stc_lib.stc_path(M)[:, :2]
    lap_u = float(np.linalg.norm(np.diff(p2, axis=0), axis=1).sum())
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    z = np.load(OUT / 'down_field.npz')
    gx, gz, st = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    LAD = np.round(np.arange(0.04, 1.001, 0.02), 3)
    pw = osr.pointwise_scale(z[f'F{CONE}'], gx, gz, st, p2, places, LAD)
    cand = [places[c] for c in np.argsort(-pw)[:4] if pw[c] >= S - 1e-9]
    print(f'pointwise max {pw.max()*100:.0f} cm; slab side {S*100:.0f} cm, '
          f'{4*M*M} subcells/layer, {LAYERS} layers', flush=True)

    # one lap in world coords, then stacked with the continuous ramp
    q1 = osl.resample(p2, 0.002 / S)
    for place in cand:
        lap = (np.concatenate([S * q1, np.zeros((len(q1), 1), np.float32)], 1)
               + place).astype(np.float64)
        rows = []
        for k in range(LAYERS):
            seg = lap.copy() if k == 0 else lap[1:].copy()
            rows.append(seg)
        W = np.concatenate(rows)
        sg = np.linalg.norm(np.diff(W[:, :2], axis=1+0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(W, axis=0), axis=1))])
        lap_len = S * lap_u
        W[:, 2] += LAYER_H * (arc / lap_len)          # continuous climb
        T = np.gradient(W, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
        need = float(arc[-1])
        print(f'  placement ({place[0]:+.2f},{place[1]:+.2f}) bed '
              f'{place[2]:+.2f}: total stroke {need:.1f} m, '
              f'{len(W)} table points', flush=True)

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
        max_steps = int(need / 0.005 * 2 * 1.3)
        envp, _, ag = osr.build_env(CONE, B, max_steps=max_steps)
        pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
        q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
        t0 = time.time()
        a, (tr, ar) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                               want_traj=True)
        del envp; torch.cuda.empty_cache()
        ok = a[0] >= need - 3 * osr.STEPW
        print(f'  drew {a[0]:.2f}/{need:.2f} m '
              f'({a[0]/need*100:.1f}%) -> {"COMPLETE" if ok else "no"} '
              f'({time.time()-t0:.0f}s roll)', flush=True)
        if not ok:
            continue
        stop = int(np.searchsorted(ar[0], need - 3 * osr.STEPW)) + 1
        Q = tr[0][:stop].astype(np.float32)
        dt_ = env0.kin.dtype
        CH = 4096
        tips, zaxs = [], []
        for lo in range(0, len(Q), CH):
            p_, R_, _, _ = env0.kin.tcp_fk_jac(
                torch.as_tensor(Q[lo:lo + CH], device='cuda', dtype=dt_))
            tips.append(p_.cpu().numpy()); zaxs.append(R_[:, :, 2].cpu().numpy())
        np.savez_compressed(
            OUT / 'ompl' / 'stc_mega_pack.npz',
            figure=W.astype(np.float32), lap_len=np.float32(lap_len),
            layers=np.int32(LAYERS), size=np.float32([S, *place]),
            need=np.float32(need), q_b=Q,
            tip_b=np.concatenate(tips).astype(np.float32),
            zax_b=np.concatenate(zaxs).astype(np.float32),
            arc_b=ar[0][:stop].astype(np.float32))
        print(f'SAVED: {LAYERS}-layer slab, {need:.1f} m in one stroke',
              flush=True)
        return
    print('no placement completed')


if __name__ == '__main__':
    main()
