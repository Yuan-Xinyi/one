"""Final eval + PHYSICS AUDIT for bottle-slope policies.

Beyond success rate, verifies on every frame of every episode:
  1. bottle-box corner penetration vs table+ramp (max / p99 / med)
  2. body-attitude deviation from the resting family (roll included)
  3. final pose precision vs the success tolerance
  4. joint-velocity-limit compliance and per-step jump bound
  5. arm-vs-bottle / arm-vs-surface collision margins

Usage:
    cd /home/lqin/one/Yuan/Qling
    /home/lqin/miniconda3/envs/one/bin/python -m drag.audit_slope \
        <ckpt.pt> <config.yaml> [n_envs seed]
"""
import matplotlib  # noqa: F401
import math
import sys

import numpy as np
import torch
import yaml

from .drag_env import DragEnvConfig
from .bottle_slope_env import BottleSlopeEnv
from .bottle_hill_env import BottleHillEnv
from .ijrr_root import add_ijrr_path
add_ijrr_path()
from Yuan.IJRR.stage2_traj.ppo import Agent  # noqa: E402

KINDS = {'bottle_slope': BottleSlopeEnv, 'bottle_hill': BottleHillEnv}


def main():
    ckpt, cfg_path = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 91
    DEV = 'cuda'
    cfg = yaml.safe_load(open(cfg_path))
    kind = cfg['env'].get('kind', 'bottle_slope')
    kw = {k: v for k, v in cfg['env'].items()
          if k not in ('kind', 'scenario')}
    kw.update(n_envs=N, seed=SEED, device=DEV, start_mode='wp0')
    env = KINDS[kind](DragEnvConfig(**kw))
    if hasattr(env, 'terrain'):
        ky = torch.tensor(env.terrain.ky, dtype=torch.float32,
                          device=DEV)
        kz = torch.tensor(env.terrain.kz, dtype=torch.float32,
                          device=DEV)

        def surf_at(yv):
            yv = yv.clamp(float(ky[0]), float(ky[-1]))
            i = torch.bucketize(yv, ky).clamp(1, len(ky) - 1)
            w = (yv - ky[i - 1]) / (ky[i] - ky[i - 1])
            return kz[i - 1] * (1 - w) + kz[i] * w
    else:
        def surf_at(yv):
            return (math.tan(env.THETA)
                    * (yv - env.Y_FOLD)).clamp(min=0.0)
    agent = Agent(env.obs_dim, env.act_dim, hidden_dim=512,
                  init_log_std=-0.5, squashed_entropy=True).to(DEV)
    agent.load_state_dict(torch.load(ckpt, map_location=DEV))
    agent.eval()

    def vfn(o):
        with torch.no_grad():
            return agent.get_value(o)
    env.set_value_fn(vfn)
    env.gen.manual_seed(SEED)
    obs = env.reset()
    R_, L_ = env.BOTTLE_R, env.BOTTLE_L
    corners = torch.tensor([[sx, sy, sz] for sx in (-R_, R_)
                            for sy in (-R_, R_) for sz in (0.0, L_)],
                           device=DEV)
    p, R, _, _ = env._frames(env.q)
    c0, ps0 = env._center(p, R)
    _, sphi, _, _ = env._fields(c0[:, 1], ps0)
    s_slope = sphi > 0.9 * env.THETA
    g_slope = env.goal_phi > 0.9 * env.THETA

    succ = torch.zeros(N, dtype=torch.bool, device=DEV)
    pen_all, roll_all, qd_viol, dq_max, marg_min = [], [], 0, 0.0, 1.0
    q_prev = env.q.clone()
    with torch.no_grad():
        for t in range(env.cfg.max_steps):
            a = agent.actor_mean(obs)
            alive = ~env.done_persistent
            obs, r, term, trunc, info = env.step(a, auto_reset=False)
            succ |= info['success'] & alive
            if alive.any():
                pp, RR, _, _ = env._frames(env.q)
                Rg = env.g_R[env.grasp_idx]
                R_obj = RR @ Rg.transpose(-1, -2)
                p_obj = pp - (R_obj @ env.g_p[env.grasp_idx]
                              .unsqueeze(-1)).squeeze(-1)
                cw = (torch.einsum('bij,cj->bci', R_obj, corners)
                      + p_obj.unsqueeze(1))
                surf = surf_at(cw[..., 1])
                pen = -(cw[..., 2] - surf).amin(dim=1)
                pen_all += pen[alive].tolist()
                cc, ps = env._center(pp, RR)
                _, phi, _, _ = env._fields(cc[:, 1], ps)
                u_b = R_obj[:, :, 1]
                n = torch.stack([torch.zeros_like(phi),
                                 -torch.sin(phi), torch.cos(phi)], dim=1)
                roll = torch.cross(u_b, n, dim=1).norm(dim=1)
                roll_all += roll[alive].tolist()
                qd_viol += int(((env.qdot.abs() > env.qd_limit * 1.001)
                                .any(dim=1) & alive).sum())
                dq = (env.q - q_prev).abs().amax(dim=1)
                dq_max = max(dq_max, float(dq[alive].max()))
                marg_min = min(marg_min,
                               float(env._coll_margin[alive].min()))
            q_prev = env.q.clone()
            if env.done_persistent.all():
                break
    pen_all = np.array(pen_all)
    roll_all = np.degrees(np.arcsin(np.clip(np.array(roll_all), 0, 1)))
    # final precision
    pp, RR, _, _ = env._frames(env.q)
    Rg = env.g_R[env.grasp_idx]
    R_obj = RR @ Rg.transpose(-1, -2)
    p_obj = pp - (R_obj @ env.g_p[env.grasp_idx].unsqueeze(-1)).squeeze(-1)
    hd = torch.atan2(R_obj[:, 1, 2], R_obj[:, 0, 2])
    d_end = (p_obj[:, :2] - env.goal_xy).norm(dim=1)
    y_end = (torch.remainder(hd - env.goal_yaw + math.pi, 2 * math.pi)
             - math.pi).abs()

    print(f'success {float(succ.float().mean()):.4f} ({int(succ.sum())}/{N})')
    for nm, m in (('f->f', ~s_slope & ~g_slope), ('f->s', ~s_slope & g_slope),
                  ('s->f', s_slope & ~g_slope), ('s->s', s_slope & g_slope)):
        print(f'  {nm}: n {int(m.sum()):>3d} succ '
              f'{float(succ[m].float().mean()):.3f}')
    print(f'[1] penetration: max {pen_all.max()*1000:.2f}mm  '
          f'p99 {np.percentile(pen_all, 99)*1000:.3f}mm  '
          f'med {np.median(pen_all)*1000:.4f}mm')
    print(f'[2] body-attitude dev: max {roll_all.max():.2f}deg  '
          f'p99 {np.percentile(roll_all, 99):.3f}deg')
    print(f'[3] final precision (successes): pos med '
          f'{float(d_end[succ].median())*1000:.1f}mm '
          f'p90 {float(d_end[succ].quantile(0.9))*1000:.1f}mm | yaw med '
          f'{math.degrees(float(y_end[succ].median())):.2f}deg '
          f'p90 {math.degrees(float(y_end[succ].quantile(0.9))):.2f}deg '
          f'(tol {env.cfg.goal_eps*1000:.0f}mm/{math.degrees(env.yaw_tol):.0f}deg)')
    print(f'[4] qdot-limit violations: {qd_viol} frames  '
          f'max per-step joint jump {math.degrees(dq_max):.2f}deg')
    print(f'[5] min collision margin over run: {marg_min*1000:.2f}mm')


if __name__ == '__main__':
    main()
