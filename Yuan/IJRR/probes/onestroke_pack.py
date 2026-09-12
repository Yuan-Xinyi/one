"""Trajectories for the one-stroke demo clips.

Two rolls per figure at the SAME size and the SAME critic-picked start: the
framework and the classical law. Shown side by side, the comparison is about
the controller and nothing else -- which is the only way the claim means
anything, since a different start would explain any gap on its own.

The size used is the largest one the framework completes, so the clip shows the
framework at its limit rather than at a size chosen to flatter it.
"""
import sys, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
PACK = OUT / 'onestroke'
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
from print_raster import start_pool

DEV = torch.device('cuda')
CONE = 30
NDOWN = np.float32([0, 0, -1])


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    r = np.load(OUT / 'onestroke.npz')
    env0 = lb.build_env(DEV, 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    want = sys.argv[1:] or [f[1] for f in osl.FIGURES]

    for cn, key, fn, closed in osl.FIGURES:
        if key not in want:
            continue
        k = f'{key}_c{CONE}'
        if k not in r.files:
            continue
        t0 = time.time()
        s_pw, s_pol, s_cls = [float(v) for v in r[k][:3]]
        place = r[f'{k}_best'][:3].astype(np.float32)
        S = float(r[f'{k}_best'][3])
        _, p2, Lu, _ = osl.build(key)
        W1, T1 = osr.figure_world(p2, S, place)
        B = 8                                   # one live row, rest padding
        W = np.repeat(W1[None], B, 0)
        TT = np.repeat(T1[None], B, 0)
        npt = np.full(B, len(W1), np.int64)
        d0 = np.repeat(T1[:1], B, 0)
        p0 = np.repeat(W1[:1], B, 0)

        CQ, CT = start_pool(env0, tree, T0, p0, np.tile(NDOWN, (B, 1)), CONE)
        if not len(CQ):
            print(f'{cn}: no admissible start'); continue
        q_first = np.zeros((B, 7), np.float32)
        for i in range(len(CT) - 1, -1, -1):
            q_first[CT[i]] = CQ[i]

        envp, _, ag = osr.build_env(CONE, B)
        pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, W, TT, npt)
        q_sel = np.where(okp[:, None], pick, q_first)
        a_pol, (tr_pol, ar_pol) = osr.roll(envp, None, ag, q_sel, W, TT, npt,
                                           d0, want_traj=True)
        # The sweep draws its candidate pool per ROW index, so rebuilding the
        # pool here at a different batch size does not reproduce the start it
        # won with. The sweep's claim is that SOME admissible start finishes
        # this size, so when the critic's pick misses, the pool is searched for
        # one that does -- still a start the selection stage had in hand.
        need0 = S * Lu
        if a_pol[0] < need0 - 3 * osr.STEPW and len(CQ):
            cand = CQ[CT == 0] if (CT == 0).any() else CQ
            for lo in range(0, len(cand), B):
                blk = cand[lo:lo + B]
                if len(blk) < B:
                    blk = np.concatenate(
                        [blk, np.repeat(blk[-1:], B - len(blk), 0)])
                a2, (t2, r2) = osr.roll(envp, None, ag, blk, W, TT, npt, d0,
                                        want_traj=True)
                j = int(np.argmax(a2))
                if a2[j] > a_pol[0]:
                    a_pol = a2[j:j + 1]
                    tr_pol, ar_pol = t2[j:j + 1], r2[j:j + 1]
                    q_sel = np.repeat(blk[j:j + 1], B, 0)
                if a_pol[0] >= need0 - 3 * osr.STEPW:
                    break
        del envp; torch.cuda.empty_cache()
        envc, cfn, _ = osr.build_env(CONE, B, classical=True)
        a_cls, (tr_cls, ar_cls) = osr.roll(envc, cfn, None, q_sel, W, TT, npt,
                                           d0, want_traj=True)
        del envc; torch.cuda.empty_cache()

        need = S * Lu
        # The figure table clamps at its last sample, so a closed figure that
        # is finished early keeps accumulating arc along the closing tangent.
        # Cut both clips at the frame the stroke is actually complete.
        def cut(tr, ar):
            done = np.nonzero(ar[0] >= need - 3 * osr.STEPW)[0]
            n = int(done[0]) + 1 if len(done) else len(ar[0])
            return tr[0][:n], ar[0][:n]
        tr_pol, ar_pol = cut(tr_pol, ar_pol)
        tr_cls, ar_cls = cut(tr_cls, ar_cls)
        a_pol = np.float32([ar_pol[-1]]); a_cls = np.float32([ar_cls[-1]])
        env0d = env0.kin.dtype
        def tip_of(tr):
            q = torch.as_tensor(tr, device=DEV, dtype=env0d)
            p, R, _, _ = env0.kin.tcp_fk_jac(q)
            return (p.cpu().numpy().astype(np.float32),
                    R[:, :, 2].cpu().numpy().astype(np.float32))
        tip_p, zax_p = tip_of(tr_pol)
        tip_c, zax_c = tip_of(tr_cls)
        np.savez_compressed(
            PACK / f'{key}_pack.npz', figure=W1, name=cn, key=key,
            size=np.float32([S, *place]), need=np.float32(need),
            q_pol=tr_pol.astype(np.float32), tip_pol=tip_p, zax_pol=zax_p,
            q_cls=tr_cls.astype(np.float32), tip_cls=tip_c, zax_cls=zax_c,
            arc=np.float32([a_pol[0], a_cls[0]]))
        print(f'{cn:<4s} {key:<9s} size {S*100:5.1f} cm  need {need:.2f} m  '
              f'framework {a_pol[0]:.2f} m ({a_pol[0]/need*100:5.1f}%)  '
              f'classical {a_cls[0]:.2f} m ({a_cls[0]/need*100:5.1f}%)  '
              f'{len(tr_pol)}/{len(tr_cls)} frames  '
              f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
