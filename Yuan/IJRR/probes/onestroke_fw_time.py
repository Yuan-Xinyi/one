"""Per-query cost of the framework, measured in the ledger the planner is
measured in.

Three numbers per figure, at the framework's own winning size and placement:

  t_pool     building the IK candidate pool at the figure origin -- shared
             infrastructure, the planner consumes the same pool as its start
             states, so it appears on both sides of the table at the same
             price;
  t_critic   scoring the pool and picking the start: everything the framework
             must finish BEFORE the pen moves;
  t_policy   the summed actor forward passes over the whole stroke, kept
             separate from the rollout wall clock because the latter is
             dominated by stepping the simulator, which on hardware is the
             physical robot and costs no computation.

Execution time is arc length over feed and is identical for every method, so
it is recorded once per figure rather than per method.
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
from print_raster import start_pool

NDOWN = np.float32([0, 0, -1])
V = 0.2


def main(cone=30):
    prev = np.load(OUT / 'onestroke.npz')
    env0 = lb.build_env('cuda', 'stock', 64)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    res = {}
    for cn, key, fn_, closed in osl.FIGURES:
        k = f'{key}_c{cone}'
        if f'{k}_best' not in prev.files:
            continue
        S = float(prev[f'{k}_best'][3])
        place = prev[f'{k}_best'][:3].astype(np.float32)
        _, p2, Lu, _ = osl.build(key)
        need = S * Lu

        t0 = time.time()
        pts = osl.resample(p2, 0.005 / S)
        W0 = (np.concatenate([S * pts, np.zeros((len(pts), 1), np.float32)], 1)
              + place).astype(np.float32)
        CQ, CT = start_pool(env0, tree, T0, W0[:1].repeat(4, 0),
                            np.tile(NDOWN, (4, 1)), cone)
        torch.cuda.synchronize()
        t_pool = time.time() - t0
        if not len(CQ):
            print(f'{cn}: empty pool'); continue

        W, T = osr.figure_world(p2, S, place)
        B = 8
        Wb = np.repeat(W[None], B, 0); Tb = np.repeat(T[None], B, 0)
        npt = np.full(B, len(W), np.int64)
        d0 = np.repeat(T[:1], B, 0)
        envp, _, ag = osr.build_env(cone, B)
        t0 = time.time()
        pick, okp = osr.critic_pick(envp, ag, CQ, CT, d0, B, Wb, Tb, npt)
        torch.cuda.synchronize()
        t_critic = time.time() - t0
        q_sel = np.where(okp[:, None], pick,
                         np.repeat(CQ[:1], B, 0))

        # rollout with the actor forward passes timed separately
        from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
        rdt = envp.kin.dtype
        envp.set_figure(torch.as_tensor(Wb, device='cuda', dtype=rdt),
                        torch.as_tensor(Tb, device='cuda', dtype=rdt),
                        torch.as_tensor(npt, device='cuda'))
        envp.line_dist = ScriptedLineDistribution({
            'q0': torch.as_tensor(q_sel, device='cuda', dtype=rdt),
            'line_dir': torch.as_tensor(d0, device='cuda', dtype=rdt),
            'n_target': torch.as_tensor(np.tile(NDOWN, (B, 1)), device='cuda',
                                        dtype=rdt)})
        envp.reset()
        t_roll0, t_pol, n_steps = time.time(), 0.0, 0
        with torch.no_grad():
            for _ in range(envp.cfg.max_steps // 2):
                tp = time.time()
                a = ag.actor_mean(envp.current_obs())
                torch.cuda.synchronize()
                t_pol += time.time() - tp
                n_steps += 1
                for _ in range(2):
                    envp.step(a, auto_reset=False)
                if bool(envp.done_persistent.all()):
                    break
        t_roll = time.time() - t_roll0
        arc = float(envp.arc_progress.float().cpu().numpy()[0])
        del envp; torch.cuda.empty_cache()

        t_exec = need / V
        res[key] = np.float32([S, t_pool, t_critic, t_pol, t_pol / max(n_steps, 1),
                               t_roll, t_exec, arc, need])
        print(f'{cn:<4s} S={S*100:5.1f}cm  pool {t_pool:5.2f}s  '
              f'critic {t_critic:5.2f}s  policy-forward total {t_pol:5.2f}s '
              f'({t_pol/max(n_steps,1)*1000:5.2f} ms/ctrl-step, budget 50 ms)  '
              f'rollout-wall {t_roll:5.2f}s  exec {t_exec:5.1f}s  '
              f'drawn {arc:.2f}/{need:.2f} m', flush=True)

    np.savez_compressed(OUT / f'onestroke_fw_time_c{cone}.npz', **res)
    print(f'\nwrote {OUT / f"onestroke_fw_time_c{cone}.npz"}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
