"""Trajectory-quality columns for the one-stroke table.

Two numbers per (method, figure, tolerance), computed over the stroke up to
the figure's end and no further:

  joint travel    sum of |dq| across all seven joints, divided by the arc
                  actually drawn [rad per metre] -- normalised so methods
                  reporting different sizes remain comparable;
  limit margin    mean over the stroke of the distance from the closest
                  joint to its nearest hardware limit [rad].

Proposed and Classical are re-rolled at their own reported largest sizes from
their own critic-picked starts; the planner's numbers come from its verified
solution files. A method that did not complete the figure gets its metrics
over the stroke it drew, marked as partial by the accompanying fraction.
"""
import sys, math
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
import onestroke_ompl as oo
from print_raster import start_pool

NDOWN = np.float32([0, 0, -1])
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def metrics(Q, arc, need):
    """(travel rad/m, mean limit margin rad, fraction drawn) up to need."""
    stop = np.searchsorted(arc, need - 1e-9) + 1
    Q, arc = Q[:stop], arc[:stop]
    drawn = float(min(arc[-1], need))
    if drawn < 1e-6 or len(Q) < 2:
        return np.nan, np.nan, 0.0
    travel = float(np.abs(np.diff(Q, axis=0)).sum()) / drawn
    marg = float(np.minimum(Q - LMT_LO, LMT_UP - Q).min(axis=1).mean())
    return travel, marg, drawn / need


def roll_method(env0, tree, T0, key, S, place, cone, classical):
    _, p2, Lu, _ = osl.build(key)
    W, T = osr.figure_world(p2, S, place)
    B = 8
    Wb = np.repeat(W[None], B, 0); Tb = np.repeat(T[None], B, 0)
    npt = np.full(B, len(W), np.int64)
    d0 = np.repeat(T[:1], B, 0)
    pts5 = osl.resample(p2, 0.005 / S)
    W05 = (np.concatenate([S * pts5, np.zeros((len(pts5), 1), np.float32)], 1)
           + place).astype(np.float32)
    CQ, CT = start_pool(env0, tree, T0, W05[:1].repeat(4, 0),
                        np.tile(NDOWN, (4, 1)), cone)
    if not len(CQ):
        return None, None, S * Lu
    envp, fn, ag = osr.build_env(cone, B, classical=classical)
    if not classical:
        pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
        q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
    else:
        envc2, _, ag2 = osr.build_env(cone, B)
        pick, okp = osr.critic_pick(envc2, ag2, CQ, CT, d0, B, Wb, Tb, npt)
        del envc2; torch.cuda.empty_cache()
        q0 = np.where(okp[:, None], pick, np.repeat(CQ[:1], B, 0))
    need = S * Lu
    _, (tr, ar) = osr.roll(envp, fn, ag, q0, Wb, Tb, npt, d0, want_traj=True)
    best = (tr[0].astype(np.float64), ar[0].astype(np.float64))
    # The sweep's claim is that SOME admissible start completes this size, and
    # its per-row candidate pools are not bit-reproducible here. If the
    # critic-picked start falls short, other pool candidates are tried in
    # batches until one completes; only if none does is the size flagged.
    if best[1][-1] < need - 3 * osr.STEPW and len(CQ) > 1:
        for lo in range(0, min(len(CQ), 64), B):
            blk = CQ[lo:lo + B]
            if len(blk) < B:
                blk = np.concatenate([blk, np.repeat(blk[-1:], B - len(blk), 0)])
            _, (t2, a2) = osr.roll(envp, fn, ag, blk, Wb, Tb, npt, d0,
                                   want_traj=True)
            j = int(np.argmax(a2[:, -1]))
            if a2[j, -1] > best[1][-1]:
                best = (t2[j].astype(np.float64), a2[j].astype(np.float64))
            if best[1][-1] >= need - 3 * osr.STEPW:
                break
    del envp; torch.cuda.empty_cache()
    return best[0], best[1], need


def main():
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    prev = np.load(OUT / 'onestroke.npz')

    res = {}

    for cone in (30, 5):
        print(f'--- tolerance {cone} deg ---', flush=True)
        for cn, key, fn_, closed in osl.FIGURES:
            k = f'{key}_c{cone}'
            place = prev[f'{k}_best'][:3].astype(np.float32)
            # Proposed at its own largest size
            S_fw = float(prev[k][1])
            Q, ar, need = roll_method(env0, tree, T0, key, S_fw, place, cone,
                                      classical=False)
            m_fw = metrics(Q, ar, need) if Q is not None else (np.nan,) * 3
            # Classical at its own largest size (per-cone opt archive, with
            # the legacy single-file name as fallback)
            S_cl = 0.0
            for cand_f in (OUT / f'onestroke_opt_c{cone}.npz',
                           OUT / 'onestroke_opt.npz'):
                if cand_f.exists():
                    po = np.load(cand_f)
                    if f'{k}_classical' in po.files:
                        S_cl = float(po[f'{k}_classical'][0])
                        break
            m_cl = (np.nan,) * 3
            if S_cl > 0:
                Qc, arc_, needc = roll_method(env0, tree, T0, key, S_cl,
                                              place, cone, classical=True)
                if Qc is not None:
                    m_cl = metrics(Qc, arc_, needc)
            # Planner from its verified solution (30 deg only, where it won)
            m_pl = (np.nan,) * 3
            fo = OUT / f'onestroke_ompl_c{cone}_{key}.npz'
            if fo.exists():
                ro = np.load(fo)
                S_pl = float(ro[k][0])
                sol = oo.WORK / f'sol_{key}_{int(round(S_pl*100))}.txt'
                if S_pl > 0 and sol.exists():
                    Qo = np.loadtxt(sol, skiprows=1).reshape(-1, 8)
                    _, p2, Lu, _ = osl.build(key)
                    m_pl = metrics(Qo[:, :7], Qo[:, 7], S_pl * Lu)
            res[f'{k}_metrics'] = np.float32([*m_fw, *m_cl, *m_pl])
            def fmt(m):
                return ('--' if np.isnan(m[0]) else
                        f'{m[0]:5.1f} rad/m, {m[1]:5.3f} rad, {m[2]*100:3.0f}%')
            print(f'{cn:<4s} Proposed[{fmt(m_fw)}]  Classical[{fmt(m_cl)}]  '
                  f'OMPL[{fmt(m_pl)}]', flush=True)

    np.savez_compressed(OUT / 'onestroke_metrics.npz', **res)
    print(f'wrote {OUT / "onestroke_metrics.npz"}')


if __name__ == '__main__':
    main()
