"""Classic one-stroke figures, drawn without lifting the pen.

The rule the earlier object demos broke: a primitive has to be finished in one
continuous motion. A seam in the middle of a circle is not a circle, so the
arm is not allowed to change branch anywhere inside the figure. That makes the
question binary per (figure, size, placement) -- did the whole stroke come out
in one go -- and turns "how big" into "the largest size at which it did".

Four answers per figure:
  pointwise   every point admits SOME admissible configuration (field lookup)
  greedy IK   seed-preserving continuation covers the figure without ever
              changing branch; a lower bound, since a planner could have
              chosen a different branch earlier
  classical   the gradient redundancy-resolution law executes it
  framework   the learned policy from a critic-picked start

The path frame is taken at the COMMANDED arc length rather than the nearest
point on the curve. Five of these seven figures cross themselves, and a
nearest-point frame flips to the wrong branch at the crossing; commanding by
arc length is both unambiguous and what a machine tool actually does.
"""
import sys, math, dataclasses, time
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

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent
import onestroke_lib as osl
from print_raster import start_pool, CASES

DEV = torch.device('cuda')
STEPW = 0.002           # world-frame resample step of the figure table [m]
R_COL = 0.20
TOPK = 24               # placements carried into the rollouts
NDOWN = np.float32([0, 0, -1])


class FigureEnv(NSRLBatchedEnv):
    """Path frame read off the commanded arc length of a per-env figure."""

    def set_figure(self, pts, tans, n_pt):
        self._fp, self._ft, self._fn = pts, tans, n_pt

    def _path_frame(self, p):
        idx = (self.arc_progress / STEPW).round().long()
        idx = torch.minimum(idx.clamp_min(0), self._fn - 1)
        ar = torch.arange(idx.shape[0], device=p.device)
        ref, tan = self._fp[ar, idx], self._ft[ar, idx]
        lat = ref - p
        lat = lat - (lat * tan).sum(-1, keepdim=True) * tan   # keep the contract
        return tan, lat, lat.norm(dim=-1)


def place_grid():
    pg = np.arange(-0.75, 0.751, 0.05, np.float32)
    beds = np.arange(-0.45, 0.601, 0.05, np.float32)
    CX, CY, ZB = np.meshgrid(pg, pg, beds, indexing='ij')
    return np.stack([CX.ravel(), CY.ravel(), ZB.ravel()], 1).astype(np.float32)


def pointwise_scale(F, gx, gz, step, p2, places, ladder, chunk=900):
    """Largest ladder scale whose every sample is nozzle-down admissible."""
    NX, NZ = len(gx), len(gz)
    best = np.zeros(len(places), np.float32)
    alive = np.arange(len(places))
    for S in ladder:
        if not len(alive):
            break
        q = osl.resample(p2, STEPW / S)
        pl = np.concatenate([S * q, np.zeros((len(q), 1), np.float32)], 1)
        keep = []
        for lo in range(0, len(alive), chunk):
            rows = alive[lo:lo + chunk]
            w = places[rows][:, None, :] + pl[None, :, :]
            i = np.rint((w[..., 0] - gx[0]) / step).astype(np.int32)
            j = np.rint((w[..., 1] - gx[0]) / step).astype(np.int32)
            k = np.rint((w[..., 2] - gz[0]) / step).astype(np.int32)
            inb = ((i >= 0) & (i < NX) & (j >= 0) & (j < NX)
                   & (k >= 0) & (k < NZ))
            ok = np.zeros(w.shape[:2], bool)
            ok[inb] = F[i[inb], j[inb], k[inb]]
            ok &= np.hypot(w[..., 0], w[..., 1]) >= R_COL
            keep.append(rows[ok.all(1)])
        alive = np.concatenate(keep) if keep else np.array([], np.int64)
        best[alive] = S
    return best


def figure_world(p2, S, place):
    """World-frame table + unit tangents for one (figure, scale, placement)."""
    q = osl.resample(p2, STEPW / S)
    w = np.concatenate([S * q, np.zeros((len(q), 1), np.float32)], 1) + place
    t = np.gradient(w, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(1e-9)
    return w.astype(np.float32), t.astype(np.float32)


def build_env(cone, B, classical=False, max_steps=4000):
    cfg = ('config_vertex_line.yaml' if classical else CASES[cone]['cfg'])
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfg))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2
    kw['max_steps'] = max_steps
    kw['cone_deg'] = float(cone)
    kw['k_lateral'] = 5.0            # curved paths need the feedback term
    env = FigureEnv(EnvConfig(**{**kw, 'n_envs': B}), None, DEV)
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin)), None
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(DEV)
    ag.load_state_dict(torch.load(REPO / CASES[cone]['ckpt'], map_location=DEV))
    ag.eval()
    return env, None, ag


@torch.no_grad()
def roll(env, fn, ag, q0s, W, T, n_pt, d0, want_traj=False):
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
    traj = [env.q.cpu().numpy().copy()] if want_traj else None
    aseq = [env.arc_progress.float().cpu().numpy().copy()] if want_traj else None
    for _ in range(env.cfg.max_steps // 2):
        a = fn(env) if fn else ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
        if want_traj:
            traj.append(env.q.cpu().numpy().copy())
            aseq.append(env.arc_progress.float().cpu().numpy().copy())
        if bool(env.done_persistent.all()):
            break
    arc = env.arc_progress.float().cpu().numpy()
    if not want_traj:
        return arc, None
    return arc, (np.stack(traj, 1), np.stack(aseq, 1))


@torch.no_grad()
def critic_pick(env, ag, CQ, CT, d0, N, W, TT, npt):
    """Score every candidate start of every (placement, scale) combo.

    Each candidate belongs to a different figure, so the env's figure table is
    re-pointed per chunk to the rows those candidates came from -- scoring a
    start against the wrong curve would make the selection meaningless."""
    B, rdt = env.n_envs, env.kin.dtype
    V = np.zeros(len(CQ), np.float32)
    for lo in range(0, len(CQ), B):
        hi = min(lo + B, len(CQ)); pad = B - (hi - lo)
        ids = np.concatenate([CT[lo:hi], np.full(pad, CT[hi - 1])]) if pad \
            else CT[lo:hi]
        env.set_figure(torch.as_tensor(W[ids], device=DEV, dtype=rdt),
                       torch.as_tensor(TT[ids], device=DEV, dtype=rdt),
                       torch.as_tensor(npt[ids], device=DEV))
        s = {'q0': torch.as_tensor(CQ[lo:hi], dtype=rdt),
             'line_dir': torch.as_tensor(d0[CT[lo:hi]], dtype=rdt),
             'n_target': torch.as_tensor(np.tile(NDOWN, (hi - lo, 1)), dtype=rdt)}
        if pad:
            s = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                 for k, v in s.items()}
        env.line_dist = ScriptedLineDistribution({k: v.to(DEV)
                                                  for k, v in s.items()})
        env.reset()
        V[lo:hi] = ag.critic(env.current_obs()).squeeze(-1) \
            .float().cpu().numpy()[:hi - lo]
    pick = np.zeros((N, 7), np.float32)
    best = np.full(N, -1e9, np.float32)
    for i, t in enumerate(CT):
        if V[i] > best[t]:
            best[t], pick[t] = V[i], CQ[i]
    return pick, best > -1e8


def main():
    cones = [int(x) for x in (sys.argv[1:] or ['30', '5'])]
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = place_grid()
    env0 = lb.build_env(DEV, 'stock', 256)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    res = {}

    for cone in cones:
        print(f'\n######## nozzle tilt tolerance {cone} deg ########', flush=True)
        for cn, key, fn, closed in osl.FIGURES:
            _, p2, Lu, _ = osl.build(key)
            t0 = time.time()
            pw = pointwise_scale(z[f'F{cone}'], gx, gz, step, p2, places, LAD)
            s_pw = float(pw.max())
            if s_pw <= 0:
                print(f'{cn}: nowhere admissible'); continue
            top = np.argsort(-pw)[:TOPK]

            # every (placement, scale) pair that the pointwise bound allows
            combos = [(int(t), float(S)) for t in top for S in LAD
                      if S <= pw[t] + 1e-9]
            Wl, Tl, d0, p0 = [], [], [], []
            for t, S in combos:
                w, tg = figure_world(p2, S, places[t])
                Wl.append(w); Tl.append(tg); d0.append(tg[0]); p0.append(w[0])
            B = len(combos)
            M = max(len(w) for w in Wl)
            W = np.zeros((B, M, 3), np.float32)
            TT = np.zeros((B, M, 3), np.float32)
            npt = np.zeros(B, np.int64)
            for i, (w, tg) in enumerate(zip(Wl, Tl)):
                W[i, :len(w)] = w; W[i, len(w):] = w[-1]
                TT[i, :len(tg)] = tg; TT[i, len(tg):] = tg[-1]
                npt[i] = len(w)
            d0 = np.stack(d0); p0 = np.stack(p0)
            need = np.float32([S * Lu for _, S in combos])

            CQ, CT = start_pool(env0, tree, T0, p0,
                                np.tile(NDOWN, (B, 1)), cone)
            has = np.bincount(CT, minlength=B) > 0
            q_first = np.zeros((B, 7), np.float32)
            for i in range(len(CT) - 1, -1, -1):
                q_first[CT[i]] = CQ[i]

            envp, _, ag = build_env(cone, B)
            pick, okp = critic_pick(envp, ag, CQ, CT, d0, B, W, TT, npt)
            q_sel = np.where(okp[:, None], pick, q_first)
            a_pol, _ = roll(envp, None, ag, q_sel, W, TT, npt, d0)
            del envp; torch.cuda.empty_cache()
            envc, cfn, _ = build_env(cone, B, classical=True)
            a_cls, _ = roll(envc, cfn, None, q_sel, W, TT, npt, d0)
            del envc; torch.cuda.empty_cache()

            done_p = has & (a_pol >= need - 3 * STEPW)
            done_c = has & (a_cls >= need - 3 * STEPW)
            sc = np.float32([S for _, S in combos])
            s_pol = float(sc[done_p].max()) if done_p.any() else 0.0
            s_cls = float(sc[done_c].max()) if done_c.any() else 0.0
            frac_p = float(np.mean((a_pol / need)[has].clip(0, 1)))
            frac_c = float(np.mean((a_cls / need)[has].clip(0, 1)))
            res[f'{key}_c{cone}'] = np.float32([s_pw, s_pol, s_cls,
                                                frac_p, frac_c])
            best_i = (int(np.argmax(np.where(done_p, sc, -1))) if done_p.any()
                      else int(np.argmax(a_pol / need)))
            res[f'{key}_c{cone}_best'] = np.float32(
                [*places[combos[best_i][0]], combos[best_i][1]])
            print(f'{cn:<4s} L={Lu:.2f}u  {B:4d} combos  '
                  f'pointwise {s_pw*100:5.1f}  framework {s_pol*100:5.1f}  '
                  f'classical {s_cls*100:5.1f} cm   '
                  f'(mean fraction drawn {frac_p:.2f} vs {frac_c:.2f})  '
                  f'{time.time()-t0:.0f}s', flush=True)

    np.savez_compressed(OUT / 'onestroke.npz', **res)
    print(f'\nwrote {OUT / "onestroke.npz"}')


if __name__ == '__main__':
    main()
