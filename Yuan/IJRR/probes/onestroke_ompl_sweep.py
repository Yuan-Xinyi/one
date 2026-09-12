"""Does a planner beat the framework on the one-stroke figures?

Three sizes are asked of OMPL per figure: the largest the framework completes,
one rung above it, and the pointwise geometric bound. That is the whole
question -- not the size curve, but whether a probabilistically complete
planner with the entire figure in hand can draw something the causal
controllers cannot.

Each size is attempted from several admissible starts, because a planner is
entitled to the same choice of start the framework's selection stage gets. A
size counts as solved only when the returned path passes the independent torch
check: on the figure, in the cone, in limits, collision-free, monotone along
the figure, and inside the velocity box.
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
from scipy.spatial import cKDTree

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
import onestroke_lib as osl
import onestroke_run as osr
import onestroke_ompl as oo
from print_raster import start_pool

NDOWN = np.float32([0, 0, -1])
BUDGET = 300.0


def main():
    cone = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    only = sys.argv[2:] or [f[1] for f in osl.FIGURES]
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    prev = np.load(OUT / 'onestroke.npz')
    popt = (np.load(OUT / 'onestroke_opt.npz')
            if (OUT / 'onestroke_opt.npz').exists() else None)
    env = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    res = {}

    for cn, key, fn_, closed in osl.FIGURES:
        if key not in only:
            continue
        _, p2, Lu, _ = osl.build(key)
        t0 = time.time()
        pw = osr.pointwise_scale(z[f'F{cone}'], gx, gz, step, p2, places, LAD)
        s_pw = float(pw.max())
        fw = float(prev[f'{key}_c{cone}'][1])
        sg = (float(popt[f'{key}_c{cone}_sgngrad'][0])
              if popt is not None and f'{key}_c{cone}_sgngrad' in popt.files
              else float('nan'))
        lad = LAD[LAD <= s_pw + 1e-9]
        up = lad[lad > fw + 1e-9]
        probes = sorted({round(x, 3) for x in
                         ([fw] + ([float(up[0])] if len(up) else []) + [s_pw])
                         if x > 0})
        # placement parity: the framework's winning placement is always in
        # the planner's candidate list, so the planner is never asked to solve
        # a size only at placements the framework never won at
        fw_best = prev[f'{key}_c{cone}_best'][:3].astype(np.float32) \
            if f'{key}_c{cone}_best' in prev.files else None
        verdict, timing = {}, {}
        for S in probes:
            cand = [places[c] for c in np.argsort(-pw)[:5]
                    if pw[c] >= S - 1e-9][:2]
            if fw_best is not None and not any(
                    np.allclose(fw_best, c, atol=1e-6) for c in cand):
                cand = [fw_best] + cand
            hit, t_pool_s, t_plan_s = False, 0.0, 0.0
            for place in cand:
                tp0 = time.time()
                pts = osl.resample(p2, 0.005 / S)
                W0 = (np.concatenate([S * pts,
                                      np.zeros((len(pts), 1), np.float32)], 1)
                      + place).astype(np.float32)
                CQ, CT = start_pool(env, tree, T0, W0[:1].repeat(4, 0),
                                    np.tile(NDOWN, (4, 1)), cone)
                t_pool_s += time.time() - tp0
                if not len(CQ):
                    continue
                # the WHOLE pool as start states of one query; in a tight
                # cone the projected sampler starves, so the sampler draws
                # from a bank of cone-admissible states built with the same
                # IK machinery as everyone's start pool (informed sampling)
                # informed sampling at EVERY tolerance: the bank rescued the
                # sharp-cornered figures at 5 degrees, so withholding it at 30
                # would understate the planner; one sampler config for the
                # whole table
                bank, dl = None, None
                if True:
                    Wd = (np.concatenate(
                        [S * osl.resample(p2, oo.STEP / S),
                         np.zeros((len(osl.resample(p2, oo.STEP / S)), 1),
                                  np.float32)], 1) + place).astype(np.float64)
                    bank = oo.build_bank(env, tree, T0, oo.fillet(Wd),
                                         float(cone))
                    dl = 0.02
                _Q, chk = oo.plan(key, S, place, CQ, float(cone),
                                  budget=BUDGET, kin=env.kin,
                                  coll=env.collision, verbose=False,
                                  bank=bank, delta=dl)
                t_plan_s += chk['t_plan'] if chk else 0.0
                if chk and chk['ok']:
                    hit = True
                    break
            verdict[S] = hit
            timing[S] = (t_pool_s, t_plan_s)
            print(f'   {cn} S={S*100:.0f}cm -> {"SOLVED" if hit else "no"}  '
                  f'pool {t_pool_s:.1f}s  plan {t_plan_s:.1f}s', flush=True)
        best = max([S for S, v in verdict.items() if v], default=0.0)
        res[f'{key}_c{cone}'] = np.float32(
            [best] + [v for S, v in sorted(verdict.items()) for v in (S, v)])
        res[f'{key}_c{cone}_times'] = np.float32(
            [[S, float(verdict[S]), *timing[S]] for S in sorted(verdict)])
        print(f'{cn:<4s} pointwise {s_pw*100:5.1f}  framework {fw*100:5.1f}  '
              f'sgngrad {sg*100:5.1f}  |  OMPL {best*100:5.1f} cm   '
              f'probes {[(round(S*100), int(v)) for S, v in sorted(verdict.items())]}'
              f'  {time.time()-t0:.0f}s', flush=True)

    tag = '_'.join(only) if len(only) < 7 else 'all'
    f = OUT / f'onestroke_ompl_c{cone}_{tag}.npz'
    np.savez_compressed(f, **res)
    print(f'\nwrote {f}')


if __name__ == '__main__':
    main()
