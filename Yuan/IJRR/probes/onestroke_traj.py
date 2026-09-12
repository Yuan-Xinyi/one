"""Trajectory optimisation over the whole stroke.

Every controller compared so far decides one command at a time. This one does
not: the entire joint trajectory along the figure is a decision variable, and
the whole stroke is optimised at once. It therefore sees the end of the figure
while choosing what to do at the beginning, which no reactive law and no
one-step lookahead can, and it is the right ceiling to hold the framework
against -- if an offline planner with the whole path in hand cannot draw a
figure in one stroke, no causal controller will.

Formulation. Nodes are the figure resampled at one command of travel. The free
variable is q at every node. The cost is the constraint violation itself --
tracking error, cone, joint limits, self-collision, and the per-joint velocity
box between consecutive nodes -- plus a barrier that pushes the minimum margin
up once the trajectory is feasible. Adam on GPU, warm-started from a
seed-preserving IK continuation so the initial guess is already continuous.

The verdict is not the loss: after optimising, every node is checked against
exactly the conditions the environment terminates on, and the figure counts as
drawn only if all of them hold everywhere. That makes a success a certificate
and not a claim about convergence.
"""
import sys, math, time, dataclasses
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
from Yuan.IJRR.env.env import LATERAL_SAFETY_NET
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
import onestroke_lib as osl
import onestroke_run as osr
from print_render_pack import march

DEV = torch.device('cuda')
NDOWN = np.float32([0, 0, -1])
QD = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], device=DEV)
V, DT = 0.2, 0.025                # one command of travel -> 5 mm per node
TUBE = LATERAL_SAFETY_NET         # 20 mm, the env's own admissibility
ITERS = 2400


def feasible(kin, coll, q, path, cos_lim):
    """Exactly the conditions the environment ends a stroke on."""
    p, R, _, _ = kin.tcp_fk_jac(q)
    track = (p - path).norm(dim=-1)
    cone = (R[:, :, 2] * torch.as_tensor(NDOWN, device=DEV,
                                         dtype=q.dtype)).sum(-1)
    lim = ((q >= kin.lmt_lo) & (q <= kin.lmt_up)).all(-1)
    cm = coll.min_margin(kin.link_transforms(q))
    dq = (q[1:] - q[:-1]).abs()
    vel = (dq <= (QD * DT).to(q.dtype)).all(-1)
    return dict(track=track, cone=cone, lim=lim, cm=cm, vel=vel,
                ok=bool((track <= TUBE).all() and (cone >= cos_lim).all()
                        and lim.all() and (cm >= 0).all() and vel.all()))


def optimise(kin, coll, path, q_init, cos_lim, iters=ITERS, chunk=4096):
    """Adam on the whole stroke, in three stages.

    Everything at once does not converge: the collision and velocity terms
    fight the tracking term before the trajectory is anywhere near the path.
    So the path and the joint limits come first, the cone and self-collision
    are switched on next, and the velocity box and the margin barrier last,
    with the learning rate decayed at each handover."""
    q = q_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=1.2e-2)
    nd = torch.as_tensor(NDOWN, device=DEV, dtype=q.dtype)
    vb = (QD * DT).to(q.dtype)
    for it in range(iters):
        ph = 0 if it < iters // 3 else (1 if it < 2 * iters // 3 else 2)
        if it == iters // 3:
            for g in opt.param_groups:
                g['lr'] = 5e-3
        if it == 2 * iters // 3:
            for g in opt.param_groups:
                g['lr'] = 2e-3
        opt.zero_grad(set_to_none=True)
        loss = q.new_zeros(())
        for lo in range(0, len(q), chunk):
            s = slice(lo, min(lo + chunk, len(q)))
            p, R, _, _ = kin.tcp_fk_jac(q[s])
            track = ((p - path[s]) ** 2).sum(-1)
            cone = (R[:, :, 2] * nd).sum(-1)
            l = 1500.0 * track.sum()
            over = torch.relu(q[s] - kin.lmt_up) + torch.relu(kin.lmt_lo - q[s])
            l = l + 800.0 * (over ** 2).sum()
            cm = coll.min_margin(kin.link_transforms(q[s]))
            if ph >= 1:
                l = l + 400.0 * torch.relu(cos_lim + 0.03 - cone).pow(2).sum()
                l = l + 600.0 * torch.relu(0.006 - cm).pow(2).sum()
            if ph < 2:
                loss = loss + l
                continue
            # margin barrier: reward headroom, saturating so it cannot fight
            # the hard terms once satisfied
            l = l - 0.6 * torch.tanh(6.0 * cm.clamp(min=0)).sum()
            l = l - 0.6 * torch.tanh(6.0 * (cone - cos_lim).clamp(min=0)).sum()
            qh = 0.5 * (kin.lmt_up - kin.lmt_lo)
            qm = 0.5 * (kin.lmt_up + kin.lmt_lo)
            hl_ = (qh - (q[s] - qm).abs()) / qh
            l = l - 0.4 * torch.tanh(4.0 * hl_.clamp(min=0)).amin(-1).sum()
            loss = loss + l
        dq = (q[1:] - q[:-1]).abs()
        if ph >= 1:
            loss = loss + 4000.0 * torch.relu(dq - 0.92 * vb).pow(2).sum()
        loss = loss + 3.0 * ((q[1:] - q[:-1]) ** 2).sum()      # smoothness
        loss.backward()
        opt.step()
    return q.detach()


def seed_from_controller(cone, W, TT, q0, name='sgngrad'):
    """Warm start from a controller's own trajectory.

    Seeding a trajectory optimiser with a simple controller's output is
    standard practice and it is what makes the result a ceiling rather than a
    coin flip: the optimum found is at least as good as the seed. Where the
    seed ran out (it died before the end) the last posture is held and the
    optimiser takes over from there."""
    import onestroke_opt as oo
    B = 8
    envv, _, _ = osr.build_env(cone, B, classical=True)
    ctl = oo.controllers(envv, [name])[name]
    from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
    rdt = envv.kin.dtype
    N = len(W)
    Wb = np.repeat(W[None], B, 0); Tb = np.repeat(TT[None], B, 0)
    envv.set_figure(torch.as_tensor(Wb, device=DEV, dtype=rdt),
                    torch.as_tensor(Tb, device=DEV, dtype=rdt),
                    torch.as_tensor(np.full(B, N), device=DEV))
    envv.line_dist = ScriptedLineDistribution({
        'q0': torch.as_tensor(np.repeat(q0[None], B, 0), device=DEV, dtype=rdt),
        'line_dir': torch.as_tensor(np.repeat(TT[:1], B, 0), device=DEV, dtype=rdt),
        'n_target': torch.as_tensor(np.tile(NDOWN, (B, 1)), device=DEV, dtype=rdt)})
    envv.reset()
    seq = [envv.q[0].cpu().numpy().copy()]
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    for _ in range(N + 8):
        a = ctl(envv, done)
        for _ in range(2):
            envv.step(a, auto_reset=False)
        seq.append(envv.q[0].cpu().numpy().copy())
        done = envv.done_persistent.clone()
        if bool(done.all()):
            break
    del envv; torch.cuda.empty_cache()
    Q = np.stack(seq).astype(np.float32)
    if len(Q) >= N:
        return Q[:N]
    return np.concatenate([Q, np.repeat(Q[-1:], N - len(Q), 0)])


def run_figure(env, tree, T0, p2, Lu, S, place, cos_lim, cone=30,
               verbose=False):
    step = V * DT
    pts = osl.resample(p2, step / S)
    W = (np.concatenate([S * pts, np.zeros((len(pts), 1), np.float32)], 1)
         + place).astype(np.float32)
    TT = np.gradient(W, axis=0)
    TT /= np.linalg.norm(TT, axis=1, keepdims=True).clip(1e-9)
    TT = TT.astype(np.float32)
    Q0, miss, _ = march(env, tree, T0, W, math.degrees(math.acos(cos_lim)))
    good = np.isfinite(Q0[:, 0])
    if not good.any():
        return None, 0.0, miss
    Q0 = np.where(good[:, None], Q0, Q0[good][0])
    # two seeds: the IK continuation, and a controller trajectory; keep both
    seeds = [Q0]
    try:
        seeds.append(seed_from_controller(cone, W, TT, Q0[0]))
    except Exception as e:                       # noqa: BLE001
        if verbose:
            print(f'      seed rollout unavailable: {e}', flush=True)
    path = torch.as_tensor(W, device=DEV, dtype=env.kin.dtype)
    q, f = None, None
    for sd in seeds:
        qi = torch.as_tensor(sd, device=DEV, dtype=env.kin.dtype)
        qo = optimise(env.kin, env.collision, path, qi, cos_lim)
        with torch.no_grad():
            fo = feasible(env.kin, env.collision, qo, path, cos_lim)
        if f is None or fo['ok'] and not f['ok']:
            q, f = qo, fo
        if f['ok']:
            break
    if verbose:
        print(f'      nodes {len(W)}  IK-miss {miss}  '
              f'track max {float(f["track"].max())*1000:.1f} mm  '
              f'cone min {math.degrees(math.acos(min(1.0, float(f["cone"].min())))):.1f} deg '
              f'coll min {float(f["cm"].min())*1000:.1f} mm  '
              f'vel ok {bool(f["vel"].all())}  -> {f["ok"]}', flush=True)
    return q.cpu().numpy(), f['ok'], miss


def main():
    cones = [int(x) for x in (sys.argv[1:] or ['30', '5'])]
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    places = osr.place_grid()
    prev = np.load(OUT / 'onestroke.npz')
    popt = (np.load(OUT / 'onestroke_opt.npz')
            if (OUT / 'onestroke_opt.npz').exists() else None)
    env = lb.build_env(DEV, 'stock', 256)
    T0 = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T0['pos'] * POS_SCALE, T0['zax']], 1)
                   .astype(np.float32))
    LAD = np.round(np.arange(0.04, 1.401, 0.02), 3)
    res = {}

    for cone in cones:
        cos_lim = math.cos(math.radians(cone))
        print(f'\n######## nozzle tilt tolerance {cone} deg ########', flush=True)
        for cn, key, fn_, closed in osl.FIGURES:
            _, p2, Lu, _ = osl.build(key)
            t0 = time.time()
            pw = osr.pointwise_scale(z[f'F{cone}'], gx, gz, step, p2, places, LAD)
            s_pw = float(pw.max())
            fw = float(prev[f'{key}_c{cone}'][1])
            sg = (float(popt[f'{key}_c{cone}_sgngrad'][0])
                  if popt is not None and f'{key}_c{cone}_sgngrad' in popt.files
                  else 0.0)
            # The decisive question is not the whole size curve, it is
            # whether a global optimiser beats the framework at all. So three
            # sizes are tested: the framework's own maximum, one rung above it,
            # and the pointwise bound. Each is tried at the best few placements.
            lad = LAD[LAD <= s_pw + 1e-9]
            up = lad[lad > fw + 1e-9]
            probes = [fw] + ([float(up[0])] if len(up) else []) + [s_pw]
            probes = sorted(set(round(x, 3) for x in probes if x > 0))
            verdict, bestp = {}, None
            for S in probes:
                cand = [c for c in np.argsort(-pw)[:6] if pw[c] >= S - 1e-9][:3]
                hit = False
                for c in cand:
                    _q, ok, _m = run_figure(env, tree, T0, p2, Lu, S,
                                            places[c], cos_lim, cone=cone)
                    if ok:
                        hit, bestp = True, places[c]
                        break
                verdict[S] = hit
            best = max([S for S, v in verdict.items() if v], default=0.0)
            res[f'{key}_c{cone}'] = np.float32([best, *(bestp if bestp is not None
                                                        else (0, 0, 0))])
            res[f'{key}_c{cone}_probes'] = np.float32(
                [[S, float(v)] for S, v in sorted(verdict.items())])
            print(f'{cn:<4s} pointwise {s_pw*100:5.1f}  framework {fw*100:5.1f}  '
                  f'sgngrad {sg*100:5.1f}  |  traj-opt {best*100:5.1f} cm  '
                  f'probes {[(round(S*100), int(v)) for S, v in sorted(verdict.items())]}  '
                  f'{time.time()-t0:.0f}s', flush=True)

    np.savez_compressed(OUT / 'onestroke_traj.npz', **res)
    print(f'\nwrote {OUT / "onestroke_traj.npz"}')


if __name__ == '__main__':
    main()
