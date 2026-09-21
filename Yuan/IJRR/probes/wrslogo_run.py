"""Roll the WRS mark: ten closed square strokes at the largest height the
layer search admits, each square through the full two-stage protocol."""
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
import onestroke_run as osr
import wrslogo_lib, wrs_lib
from print_raster import start_pool

NDOWN = np.float32([0, 0, -1])
CONE = 30


def main():
    squares = wrslogo_lib.mark()
    all2 = np.concatenate([wrs_lib.densify(p, 0.02) for _, p in squares])
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    z = np.load(OUT / 'down_field.npz')
    gx, gz, st = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    LAD = np.round(np.arange(0.10, 1.201, 0.02), 3)
    pw = osr.pointwise_scale(z[f'F{CONE}'], gx, gz, st, all2, places, LAD)

    for H in LAD[LAD <= pw.max() + 1e-9][::-1]:
        for c in np.argsort(-pw)[:4]:
            if pw[c] < H - 1e-9:
                continue
            place = places[c]
            packs, good = [], True
            t00 = time.time()
            for name, p2 in squares:
                q1 = wrs_lib.densify(p2, 0.002 / H) * H
                W = (np.concatenate([q1, np.zeros((len(q1), 1))], 1)
                     + place).astype(np.float64)
                T = np.gradient(W, axis=0)
                T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)
                need = float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum())
                B = 8
                Wb = np.repeat(W[None].astype(np.float32), B, 0)
                Tb = np.repeat(T[None].astype(np.float32), B, 0)
                npt = np.full(B, len(W), np.int64)
                d0 = np.repeat(T[:1].astype(np.float32), B, 0)
                CQ, CT = start_pool(env0, tree, T0,
                                    W[:1].astype(np.float32).repeat(4, 0),
                                    np.tile(NDOWN, (4, 1)), CONE)
                if not len(CQ):
                    good = False; break
                envp, _, ag = osr.build_env(
                    CONE, B, max_steps=int(max(need, 0.4) / 0.005 * 2 * 1.5))
                pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B,
                                            Wb, Tb, npt)
                q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
                a, (tr, ar) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                                       want_traj=True)
                del envp; torch.cuda.empty_cache()
                if a[0] < need - 3 * osr.STEPW:
                    print(f'  H={H*100:.0f} {name}: {a[0]:.2f}/{need:.2f} FAIL',
                          flush=True)
                    good = False; break
                stop = int(np.searchsorted(ar[0], need - 3*osr.STEPW)) + 1
                packs.append((name, W.astype(np.float32),
                              tr[0][:stop].astype(np.float32), need))
            if not good:
                continue
            dt_ = env0.kin.dtype
            save = {'size': np.float32([H, *place]),
                    'names': np.array([n for n, *_ in packs])}
            for name, W, Q, need in packs:
                p_, R_, _, _ = env0.kin.tcp_fk_jac(
                    torch.as_tensor(Q, device='cuda', dtype=dt_))
                save[f'{name}_fig'] = W
                save[f'{name}_q'] = Q
                save[f'{name}_tip'] = p_.cpu().numpy().astype(np.float32)
                save[f'{name}_zax'] = R_[:, :, 2].cpu().numpy().astype(np.float32)
                save[f'{name}_need'] = np.float32(need)
            np.savez_compressed(OUT / 'ompl' / 'wrslogo_pack.npz', **save)
            tot = sum(need for *_, need in packs)
            print(f'SAVED: WRS mark height {H*100:.0f} cm '
                  f'({1.10*H*100:.0f} x {0.89*H*100:.0f} cm), '
                  f'{len(packs)} squares, strokes {tot:.2f} m, '
                  f'bed z={place[2]:+.2f} ({time.time()-t00:.0f}s)', flush=True)
            return
    print('nothing completed')


if __name__ == '__main__':
    main()
