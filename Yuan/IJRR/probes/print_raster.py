"""How much of a print layer goes down in one continuous pass.

The pointwise field says the whole cube is admissible -- every point of every
layer has an admissible configuration. That is exactly why it is the wrong
question for a printer: what a printer needs is that consecutive points be
reachable by ONE continuous motion, and the gap between those two statements is
the entire subject of the paper.

A raster layer is a stack of straight passes, each the width of the object,
with the nozzle held down. That is the paper's own task family with the task
parameters read off the print instead of drawn at random, so it is measured
here with the paper's own protocol and no new environment: rays at the cube's
own location, 5-degree or 30-degree cone about straight down, admissible start
pool, classical vs policy vs critic-picked policy. What comes out is passes
completed per attempt, which converts directly into seams per layer.

Speed note: the mainline env runs at 0.2 m/s, far above any real print feed.
The v-sweep found the saturation ratio threshold-like and 0.05 m/s neutral on
the FR3, so a slower feed does not flatter these numbers -- it is the same
regime.
"""
import sys, math, dataclasses, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
OUT = MAIN / 'runs/paper_fill/print_analysis'
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z, _sample_in_cone
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent

DEV = torch.device('cuda')
TUBE = 0.002
DZ = 0.04             # layer spacing of the probe (not the print's layer height)
DY = 0.04             # pass spacing of the probe
CASES = {
    5:  dict(cfg='config_line_cont_dirfrac_e8kXXL_cone5.yaml',
             ckpt='Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5/agent.pt'),
    30: dict(cfg='config_line_cont_dirfrac_e8kXXL_rm.yaml',
             ckpt='Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'),
}


def passes(c, side, zb):
    """Every straight pass a raster layer is made of, both directions."""
    P0, D = [], []
    ys = np.arange(c[1] - side / 2, c[1] + side / 2 + 1e-9, DY)
    zs = np.arange(zb, zb + side + 1e-9, DZ)
    for z in zs:
        for y in ys:
            P0.append([c[0] - side / 2, y, z]); D.append([1.0, 0.0, 0.0])
            P0.append([c[0] + side / 2, y, z]); D.append([-1.0, 0.0, 0.0])
    return np.array(P0, np.float32), np.array(D, np.float32)


def start_pool(env, tree, T, p0, nt, cone, n_dirs=4, k_nn=96):
    """All admissible configurations found at each pass start: the Stage 1
    candidate pool, built exactly as the paper builds it."""
    dt_ = env.kin.dtype
    hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt_, device=DEV)
    cos_lim = math.cos(math.radians(cone))
    N = len(p0)
    CQ, CT = [], []
    for m in range(n_dirs):
        zs = nt.copy() if m == 0 else np.stack([
            _sample_in_cone(torch.as_tensor(nt[i]), cone, 1,
                            np.random.default_rng(m * 977 + i)).numpy()[0]
            for i in range(N)])
        feat = np.concatenate([p0 * POS_SCALE, zs], 1).astype(np.float32)
        _, ids = tree.query(feat, k=k_nn, workers=-1)
        for lo in range(0, N, 64):
            hi = min(lo + 64, N)
            fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, 7),
                                 device=DEV, dtype=dt_)
            fp = torch.as_tensor(np.repeat(p0[lo:hi], k_nn, 0), device=DEV, dtype=dt_)
            fz = torch.as_tensor(np.repeat(zs[lo:hi], k_nn, 0), device=DEV, dtype=dt_)
            fn = torch.as_tensor(np.repeat(nt[lo:hi], k_nn, 0), device=DEV, dtype=dt_)
            q_o, _, _ = _batched_ik_project(env.kin, fq, fp,
                                            _build_R_with_z(fz, hint), None)
            coll = env.collision.is_collided(env.kin.link_transforms(q_o))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
            in_l = ((q_o >= env.kin.lmt_lo - 1e-5)
                    & (q_o <= env.kin.lmt_up + 1e-5)).all(-1)
            fine = ((~coll) & in_l & ((p_fk - fp).norm(dim=-1) <= TUBE)
                    & ((R_fk[:, :, 2] * fn).sum(-1) >= cos_lim))
            f = fine.cpu().numpy()
            CQ.append(q_o[fine].cpu().numpy())
            CT.append((((np.arange(len(f)) // k_nn) + lo))[f])
    CQ = np.concatenate(CQ).astype(np.float32); CT = np.concatenate(CT)
    o = np.argsort(CT, kind='stable'); CQ, CT = CQ[o], CT[o]
    key = np.concatenate([CT[:, None], np.round(CQ, 2)], 1)
    _, ui = np.unique(key, axis=0, return_index=True)
    return CQ[np.sort(ui)], CT[np.sort(ui)]


def build(cone, classical=False, B=1024):
    # The classical law emits a B-coordinate action, so it needs the vertex
    # env, not the dir-frac one the policy was trained in; same kinematics,
    # same cone, same integration -- only the action parameterisation differs.
    cfg = ('config_vertex_line.yaml' if classical else CASES[cone]['cfg'])
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfg))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw['cone_deg'] = float(cone)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, DEV)
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin)), None
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(DEV)
    ag.load_state_dict(torch.load(REPO / CASES[cone]['ckpt'], map_location=DEV))
    ag.eval()
    return env, None, ag


def roll(env, fn, ag, q0s, d, nt):
    B, rdt = env.n_envs, env.kin.dtype
    out = np.zeros(len(q0s), np.float32)
    with torch.no_grad():
        for lo in range(0, len(q0s), B):
            hi = min(lo + B, len(q0s)); pad = B - (hi - lo)
            s = {'q0': torch.tensor(q0s[lo:hi], dtype=rdt),
                 'line_dir': torch.tensor(d[lo:hi], dtype=rdt),
                 'n_target': torch.tensor(nt[lo:hi], dtype=rdt)}
            if pad:
                s = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                     for k, v in s.items()}
            env.line_dist = ScriptedLineDistribution({k: v.to(DEV)
                                                      for k, v in s.items()})
            env.reset()
            for _ in range(env.cfg.max_steps // 2):
                a = fn(env) if fn else ag.actor_mean(env.current_obs())
                for _ in range(2):
                    env.step(a, auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
            out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


def critic_pick(env, ag, CQ, CT, d, nt, N):
    B, rdt = env.n_envs, env.kin.dtype
    V = np.zeros(len(CQ), np.float32)
    with torch.no_grad():
        for lo in range(0, len(CQ), B):
            hi = min(lo + B, len(CQ)); pad = B - (hi - lo)
            ids = CT[lo:hi]
            s = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt),
                 'line_dir': torch.tensor(d[ids], dtype=rdt),
                 'n_target': torch.tensor(nt[ids], dtype=rdt)}
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
    return pick


def main():
    q = np.load(OUT / 'cube_query.npz')
    gx, gz = q['gx'], q['gz']
    env0 = lb.build_env(DEV, 'stock', 512)
    T = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
                   .astype(np.float32))
    rows, out = [], {}

    for cone in (5, 30):
        A = q[f'cube_c{cone}_solid']
        b = np.unravel_index(A.argmax(), A.shape)
        c = (float(gx[b[0]]), float(gx[b[1]]))
        zb, side = float(gz[b[2]]) - 0.01, float(A.max())
        p0, d = passes(c, side, zb)
        nt = np.tile(np.float32([0, 0, -1]), (len(p0), 1))
        N = len(p0)
        print(f'\n=== cone {cone} deg: cube {side*100:.0f} cm at '
              f'({c[0]:+.2f},{c[1]:+.2f}) bed {zb:+.2f}, {N} passes of '
              f'{side*100:.0f} cm ===', flush=True)

        t0 = time.time()
        CQ, CT = start_pool(env0, tree, T, p0, nt, cone)
        cnt = np.bincount(CT, minlength=N)
        q_first = np.zeros((N, 7), np.float32)
        for i in range(len(CT) - 1, -1, -1):
            q_first[CT[i]] = CQ[i]                 # first candidate per pass
        print(f'[raster] start pools: {(cnt>0).mean()*100:.1f}% of passes, '
              f'median {int(np.median(cnt[cnt>0]))} candidates '
              f'({time.time()-t0:.0f}s)', flush=True)
        has = cnt > 0

        envc, fn, _ = build(cone, classical=True)
        p_cls = roll(envc, fn, None, q_first, d, nt)
        del envc; torch.cuda.empty_cache()
        envp, _, ag = build(cone)
        p_pol = roll(envp, None, ag, q_first, d, nt)
        pick = critic_pick(envp, ag, CQ, CT, d, nt, N)
        p_sel = roll(envp, None, ag, pick, d, nt)
        envc2, fn2, _ = build(cone, classical=True)
        p_cls_sel = roll(envc2, fn2, None, pick, d, nt)
        del envp, envc2; torch.cuda.empty_cache()

        for name, p in (('classical @first', p_cls),
                        ('classical @critic', p_cls_sel),
                        ('policy @first', p_pol),
                        ('policy @critic', p_sel)):
            v = p[has]
            print(f'[raster] {name:<20s} stroke {v.mean():.3f} m  '
                  f'= {v.mean()/side*100:5.1f}% of a pass   '
                  f'passes completed {np.mean(v >= side - 1e-3)*100:5.1f}%',
                  flush=True)
            rows.append((cone, name, float(v.mean()), float(v.mean() / side),
                         float(np.mean(v >= side - 1e-3))))
        out[f'c{cone}'] = np.stack([p_cls, p_cls_sel, p_pol, p_sel])
        out[f'meta_c{cone}'] = np.float32([side, c[0], c[1], zb, N])
        out[f'has_c{cone}'] = has

    np.savez_compressed(OUT / 'raster_query.npz', **out)
    print(f'\nwrote {OUT / "raster_query.npz"}')


if __name__ == '__main__':
    main()
