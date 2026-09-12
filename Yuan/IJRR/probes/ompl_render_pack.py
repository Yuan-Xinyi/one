"""Render packs for the planner-vs-framework pair.

Two figures, chosen because they split the verdict: on the circle at 70 cm the
planner finishes and the framework does not, and on the square at 60 cm the
framework finishes and the planner gets a fifth of the way. Both are drawn from
the same start at the same feed, so the clip is about the method and nothing
else.

The planner's path is resampled to a uniform 5 mm of arc per frame -- the
commanded feed -- because OMPL returns states spaced by its own geodesic
discretisation, and playing those back directly would show the two arms moving
at different speeds.
"""
import sys, math
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

from Yuan.IJRR.eval import line_bound as lb
import onestroke_lib as osl
import onestroke_run as osr
import onestroke_ompl as oo

DEV = torch.device('cuda')
CASES = [('circle', 0.70, 30)]
FEED = 0.005            # metres of arc per animation frame


def on_grid(q, s, u):
    """Put a trajectory on a common arc grid.

    The two methods are recorded at different rates -- the planner at its own
    geodesic spacing, the rollout at two integration steps per stored frame --
    so played back raw the arms move at different speeds and nothing about the
    postures is comparable. Re-sampling both against arc length puts them at
    the SAME point of the figure in the same frame; past its own end a
    trajectory holds its last posture."""
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s, q = s[keep], q[keep]
    out = np.stack([np.interp(u, s, q[:, j]) for j in range(7)], 1)
    return out.astype(np.float32), float(s[-1])


def main():
    PACK.mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    env0 = lb.build_env(DEV, 'stock', 64)

    for key, S, cone in CASES:
        sol = oo.WORK / f'sol_{key}_{int(S*100)}.txt'
        if not sol.exists():
            print(f'{key}: no planner file'); continue
        _, p2, Lu, _ = osl.build(key)
        pw = osr.pointwise_scale(z[f'F{cone}'], gx, gz, step, p2, places, LAD)
        place = places[[c for c in np.argsort(-pw)[:5]
                        if pw[c] >= S - 1e-9][0]]
        Qo = np.loadtxt(sol, skiprows=1).reshape(-1, 8)

        # the framework, from the planner's own start, on the same figure
        W, T = osr.figure_world(p2, S, place)
        B = 8
        Wb = np.repeat(W[None], B, 0); Tb = np.repeat(T[None], B, 0)
        npt = np.full(B, len(W), np.int64)
        d0 = np.repeat(T[:1], B, 0)
        # Each method gets the start IT chose: the planner sampled its own,
        # the framework's selection stage picks from its candidate pool. Forcing
        # one onto the other is not a controller comparison -- the circle shows
        # why, since the framework finishes 70 cm from the planner's start but
        # not from the one its own critic picked at this placement.
        from print_raster import start_pool
        from scipy.spatial import cKDTree
        from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE as _PS
        T0 = np.load(REPO / lb.TABLE)
        tree = cKDTree(np.concatenate([T0['pos'] * _PS, T0['zax']], 1)
                       .astype(np.float32))
        NDOWN = np.float32([0, 0, -1])
        CQ, CT = start_pool(env0, tree, T0, W[:1].repeat(B, 0),
                            np.tile(NDOWN, (B, 1)), cone)
        envp, _, ag = osr.build_env(cone, B)
        if len(CQ):
            pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B,
                                        Wb, Tb, npt)
            q0 = np.where(okp[:, None], pick,
                          np.repeat(Qo[:1, :7].astype(np.float32), B, 0))
        else:
            q0 = np.repeat(Qo[:1, :7].astype(np.float32), B, 0)
        a_fw, (tr_fw, ar_fw) = osr.roll(envp, None, ag, q0, Wb, Tb, npt, d0,
                                        want_traj=True)
        del envp; torch.cuda.empty_cache()
        q_fw_raw, s_fw = tr_fw[0].astype(np.float32), ar_fw[0].astype(np.float64)
        s_pl_raw = Qo[:, 7].astype(np.float64)
        end = max(float(s_pl_raw[-1]), float(s_fw[-1]))
        u = np.arange(0.0, end + FEED, FEED)
        q_pl, e_pl = on_grid(Qo[:, :7], s_pl_raw, u)
        q_fw, e_fw = on_grid(q_fw_raw, s_fw, u)
        s_pl = u.astype(np.float32)

        dt_ = env0.kin.dtype
        def fk(Q):
            p, R, _, _ = env0.kin.tcp_fk_jac(
                torch.as_tensor(Q, device=DEV, dtype=dt_))
            return (p.cpu().numpy().astype(np.float32),
                    R[:, :, 2].cpu().numpy().astype(np.float32))
        tp, zp = fk(q_pl)
        tf, zf = fk(q_fw)
        need = S * Lu
        np.savez_compressed(
            PACK / f'{key}_pack.npz', figure=W.astype(np.float32),
            size=np.float32([S, *place]), need=np.float32(need),
            q_a=q_pl, tip_a=tp, zax_a=zp, arc=u.astype(np.float32),
            q_b=q_fw, tip_b=tf, zax_b=zf,
            ends=np.float32([e_pl, e_fw]))
        print(f'{key} {S*100:.0f} cm: common grid {len(u)} frames at '
              f'{FEED*1000:.0f} mm/frame | planner ends {e_pl:.2f} m '
              f'({e_pl/need*100:.0f}%) | framework ends {e_fw:.2f} m '
              f'({e_fw/need*100:.0f}%)  [need {need:.2f} m]', flush=True)


if __name__ == '__main__':
    main()
