"""One-stroke figures against the optimisation baselines, not just the
fixed-gain gradient law.

The gradient redundancy-resolution law is the weakest member of its own class,
and reporting only it makes the comparison look better than it is. Three
stronger non-learned controllers are added here, all rolled on the SAME figure
at the SAME size from the SAME critic-picked start:

  classical  fixed-gain gradient of the manipulability field  (current state)
  sgngrad    sign of the MARGIN field's gradient              (current state)
  myopic     one-step exact optimisation: enumerate the action-set vertices,
             simulate one command forward, keep the best margin  (privileged,
             it needs a forward model the others do not)
  beam(H)    the same objective optimised over an H-step horizon

sgngrad and myopic do not assume a straight path: their margin terms are the
joint-limit and cone margins, and the lateral term -- the only one that refers
to a ray -- is excluded. The forward model does hold the tangent fixed across
a command, which on a curved figure is an approximation; at 5 mm per command
the tangent turns by well under a degree on every figure here.
"""
import sys, dataclasses, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
import onestroke_lib as osl
import onestroke_run as osr
from print_raster import start_pool

DEV = torch.device('cuda')
NDOWN = np.float32([0, 0, -1])
hl.SUB = 2


def make_model(env):
    m = hl.StraightModel(env)
    m.cfg = dataclasses.replace(env.cfg, dt=env.cfg.dt * 2)   # un-halve
    m.terms = [0, 1]                    # joint limit + cone; no ray term
    return m


def controllers(env, which):
    from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                                   cn_action_fn)
    model = make_model(env)
    out = {}
    if 'classical' in which:
        f = cn_action_fn(ClassicalNullspaceController(env.kin))
        out['classical'] = lambda e, d: f(e)
    if 'sgngrad' in which:
        out['sgngrad'] = hl.make_sgngrad(model)
    if 'myopic' in which:
        out['myopic'] = hl.make_myopic(model)
    if 'beam' in which:
        out['beam'] = hl.make_beam(model, width=8, H=4)
    return out


@torch.no_grad()
def roll2(env, fn, q0s, W, T, n_pt, d0):
    from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
    B, rdt = env.n_envs, env.kin.dtype
    env.set_figure(torch.as_tensor(W, device=DEV, dtype=rdt),
                   torch.as_tensor(T, device=DEV, dtype=rdt),
                   torch.as_tensor(n_pt, device=DEV))
    env.line_dist = ScriptedLineDistribution({
        'q0': torch.as_tensor(q0s, device=DEV, dtype=rdt),
        'line_dir': torch.as_tensor(d0, device=DEV, dtype=rdt),
        'n_target': torch.as_tensor(np.tile(NDOWN, (B, 1)), device=DEV,
                                    dtype=rdt)})
    env.reset()
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    for _ in range(env.cfg.max_steps // 2):
        a = fn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return env.arc_progress.float().cpu().numpy()


def main():
    cones = [int(x) for x in (sys.argv[1:] or ['30', '5'])]
    WHICH = ['classical', 'sgngrad', 'myopic']
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    prev = np.load(OUT / 'onestroke.npz')
    env0 = lb.build_env(DEV, 'stock', 256)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    res = {}

    for cone in cones:
        print(f'\n######## nozzle tilt tolerance {cone} deg ########', flush=True)
        for cn, key, fn_, closed in osl.FIGURES:
            _, p2, Lu, _ = osl.build(key)
            t0 = time.time()
            pw = osr.pointwise_scale(z[f'F{cone}'], gx, gz, step, p2, places, LAD)
            s_pw = float(pw.max())
            top = np.argsort(-pw)[:osr.TOPK]
            combos = [(int(t), float(S)) for t in top for S in LAD
                      if S <= pw[t] + 1e-9]
            Wl, Tl, d0, p0 = [], [], [], []
            for t, S in combos:
                w, tg = osr.figure_world(p2, S, places[t])
                Wl.append(w); Tl.append(tg); d0.append(tg[0]); p0.append(w[0])
            B = len(combos)
            M = max(len(w) for w in Wl)
            W = np.zeros((B, M, 3), np.float32); TT = np.zeros((B, M, 3), np.float32)
            npt = np.zeros(B, np.int64)
            for i, (w, tg) in enumerate(zip(Wl, Tl)):
                W[i, :len(w)] = w; W[i, len(w):] = w[-1]
                TT[i, :len(tg)] = tg; TT[i, len(tg):] = tg[-1]
                npt[i] = len(w)
            d0 = np.stack(d0); p0 = np.stack(p0)
            need = np.float32([S * Lu for _, S in combos])
            sc = np.float32([S for _, S in combos])

            CQ, CT = start_pool(env0, tree, T0, p0, np.tile(NDOWN, (B, 1)), cone)
            has = np.bincount(CT, minlength=B) > 0
            q_first = np.zeros((B, 7), np.float32)
            for i in range(len(CT) - 1, -1, -1):
                q_first[CT[i]] = CQ[i]
            envp, _, ag = osr.build_env(cone, B)
            pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, W, TT, npt)
            q_sel = np.where(okp[:, None], pick, q_first)
            del envp; torch.cuda.empty_cache()

            envv, _, _ = osr.build_env(cone, B, classical=True)
            ctl = controllers(envv, WHICH)
            line = f'{cn:<4s} pointwise {s_pw*100:5.1f}  '
            fw = float(prev[f'{key}_c{cone}'][1]) if f'{key}_c{cone}' in prev.files \
                else float('nan')
            line += f'framework {fw*100:5.1f}  |  '
            for nm, f in ctl.items():
                a = roll2(envv, f, q_sel, W, TT, npt, d0)
                done = has & (a >= need - 3 * osr.STEPW)
                s_max = float(sc[done].max()) if done.any() else 0.0
                frac = float(np.mean((a / need)[has].clip(0, 1)))
                res[f'{key}_c{cone}_{nm}'] = np.float32([s_max, frac])
                line += f'{nm} {s_max*100:5.1f} ({frac:.2f})  '
            del envv; torch.cuda.empty_cache()
            print(line + f' {time.time()-t0:.0f}s', flush=True)

    tag = '_'.join(str(c) for c in cones)
    f = OUT / f'onestroke_opt_c{tag}.npz'
    np.savez_compressed(f, **res)
    print(f'\nwrote {f}')


if __name__ == '__main__':
    main()
