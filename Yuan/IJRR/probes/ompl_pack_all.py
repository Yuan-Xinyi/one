"""Render packs for every figure of the fair comparison.

One pack per figure at the LARGEST size the planner solved: the planner's
verified path against the framework rolled at the same size and placement from
its own critic-picked start. Both trajectories go onto one arc-length grid
(5 mm per frame), so in every video frame the two arms are at the same point
of the figure and the joint plots are comparable column by column.

The sweep does not record which placement the winning solution used, so it is
reconstructed: the candidate placements are re-derived in the sweep's own
order and the saved path is verified against each until one passes. A pack is
only written for a (figure, size) whose reconstruction passes the full check,
which also re-confirms the solution itself.
"""
import sys, math, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
PACK = OUT / 'ompl'
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
FEED = 0.005
CONE = 30  # overridden by argv[1]


def on_grid(q, s, u):
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s, q = s[keep], q[keep]
    out = np.stack([np.interp(u, s, q[:, j]) for j in range(7)], 1)
    return out.astype(np.float32), float(s[-1])


def main(only=None, cone=30):
    global CONE
    CONE = cone
    PACK.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    prev = np.load(OUT / 'onestroke.npz')
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))

    for cn, key, fn_, closed in osl.FIGURES:
        if only and key not in only:
            continue
        f = OUT / f'onestroke_ompl_c{CONE}_{key}.npz'
        if not f.exists():
            print(f'{cn}: no sweep result yet'); continue
        r = np.load(f)
        S = float(r[f'{key}_c{CONE}'][0])
        if S <= 0:
            print(f'{cn}: planner solved nothing'); continue
        sol = oo.WORK / f'sol_{key}_{int(round(S*100))}.txt'
        if not sol.exists():
            print(f'{cn}: missing {sol.name}'); continue
        Qo = np.loadtxt(sol, skiprows=1).reshape(-1, 8)

        _, p2, Lu, _ = osl.build(key)
        pw = osr.pointwise_scale(z[f'F{CONE}'], gx, gz, step, p2, places, LAD)
        cand = [places[c] for c in np.argsort(-pw)[:5] if pw[c] >= S - 1e-9][:2]
        fwb = (prev[f'{key}_c{CONE}_best'][:3].astype(np.float32)
               if f'{key}_c{CONE}_best' in prev.files else None)
        if fwb is not None and not any(np.allclose(fwb, c, atol=1e-6)
                                       for c in cand):
            cand = [fwb] + cand
        place = None
        for c in cand:
            pts = osl.resample(p2, oo.STEP / S)
            W_orig = (np.concatenate(
                [S * pts, np.zeros((len(pts), 1), np.float32)], 1)
                + c).astype(np.float64)
            Wf = oo.fillet(W_orig)
            chk = oo.verify(env0.kin, env0.collision, Qo, Wf,
                            math.cos(math.radians(CONE)), W_orig=W_orig)
            if chk['ok']:
                place = np.asarray(c, np.float32)
                break
        if place is None:
            print(f'{cn}: no candidate placement verifies -- skipped')
            continue

        # framework at the same size and placement, its own critic start
        W, T = osr.figure_world(p2, S, place)
        B = 8
        Wb = np.repeat(W[None], B, 0); Tb = np.repeat(T[None], B, 0)
        npt = np.full(B, len(W), np.int64)
        d0 = np.repeat(T[:1], B, 0)
        pts5 = osl.resample(p2, 0.005 / S)
        W05 = (np.concatenate([S * pts5,
                               np.zeros((len(pts5), 1), np.float32)], 1)
               + place).astype(np.float32)
        CQ, CT = start_pool(env0, tree, T0, W05[:1].repeat(4, 0),
                            np.tile(NDOWN, (4, 1)), CONE)
        envp, _, ag = osr.build_env(CONE, B)
        if len(CQ):
            pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
            q0 = np.where(okp[:, None], pick,
                          np.repeat(Qo[:1, :7].astype(np.float32), B, 0))
        else:
            q0 = np.repeat(Qo[:1, :7].astype(np.float32), B, 0)
        a_fw, (tr_fw, ar_fw) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                                        want_traj=True)
        del envp; torch.cuda.empty_cache()

        s_pl = Qo[:, 7].astype(np.float64)
        s_fw = ar_fw[0].astype(np.float64)
        need = S * Lu
        # the drawing ends where the figure ends: a closed figure that has
        # been completed must not keep accumulating arc along the closing
        # tangent, so the common grid is clipped at the figure's length
        end = min(max(float(s_pl[-1]), float(s_fw[-1])), need)
        u = np.arange(0.0, end + FEED, FEED)
        q_pl, e_pl = on_grid(Qo[:, :7], s_pl, u)
        q_fw, e_fw = on_grid(tr_fw[0].astype(np.float64), s_fw, u)

        dt_ = env0.kin.dtype
        def fk(Q):
            p, R, _, _ = env0.kin.tcp_fk_jac(
                torch.as_tensor(Q, device='cuda', dtype=dt_))
            return (p.cpu().numpy().astype(np.float32),
                    R[:, :, 2].cpu().numpy().astype(np.float32))
        tp, zp = fk(q_pl)
        tf, zf = fk(q_fw)
        np.savez_compressed(
            PACK / f'{key}_c{CONE}_pack.npz', figure=W.astype(np.float32),
            size=np.float32([S, *place]), need=np.float32(need),
            q_a=q_pl, tip_a=tp, zax_a=zp, arc=u.astype(np.float32),
            q_b=q_fw, tip_b=tf, zax_b=zf,
            ends=np.float32([e_pl, e_fw]))
        print(f'{cn:<4s} S={S*100:.0f}cm at ({place[0]:+.2f},{place[1]:+.2f}) '
              f'bed {place[2]:+.2f}: grid {len(u)} frames | '
              f'OMPL ends {e_pl:.2f} ({e_pl/need*100:.0f}%) | '
              f'framework ends {e_fw:.2f} ({e_fw/need*100:.0f}%) '
              f'[need {need:.2f} m]', flush=True)


if __name__ == '__main__':
    cone = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(sys.argv[2:] or None, cone=cone)
