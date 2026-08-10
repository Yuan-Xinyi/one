"""Does the actor act on the critic's switching vector?

The HJB analysis says the greedy command with respect to any critic W is the
vertex sgn(sigma_W), with sigma_W = G(q)^T grad_q W. If the trained actor and
critic have reached the policy-improvement fixed point that Proposition 1
describes, the vertex the actor selects should agree, coordinate by
coordinate, with the sign of the switching vector computed from its own critic
by automatic differentiation. That is what this measures.

Two refinements pin the claim down:

  * The critic estimates V^pi, the value of the current policy, not V*.
    Agreement therefore tests actor-greedy-with-respect-to-own-critic
    consistency -- exactly the fixed point of Proposition 1 -- and says
    nothing about optimality.
  * Where a component of sigma is near zero the greedy action is close to
    indifferent, so mismatches there are expected and uninformative. The
    match rate is therefore reported per decile of |sigma_i|, with two
    predictions: match rate rises with |sigma_i|, and the coordinate flip
    rate between consecutive steps falls with it.

The gradient is taken through the observation map: obs(q) is built from the
same differentiable kinematics the environment uses, with the task quantities
(line_dir, n_target, a_prev) held fixed, which on the straight family they
are. The basis B(q) is evaluated, not differentiated through.

Result (rl_vertex_line_30M, 1024 tasks, 2026-08-10): the two critic readouts
agree with each other on 77% of coordinates, but with the actor on only
54-58% (chance 50%), rising monotonically to ~63% in the top |sigma| decile.
So the critic expresses a coherent local preference and the actor does NOT
pointwise follow it -- the policy is a learned policy over the vertex set,
not a greedy realization of its critic. Reported as a diagnostic in the
paper's discussion; none of the Proposition-1 claims depend on it.

Usage:
    python -m Yuan.IJRR.eval.switching_signal --n-tasks 1024
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis, damped_pinv)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = Path(__file__).resolve().parents[3]


def obs31(q, line_dir, n_target, a_prev, q_mid, q_half, kin):
    """The 31-D observation for an arbitrary batch, mirroring
    NSRLBatchedEnv._compute_obs (asserted against it on the first step).
    Needed because the environment's buffers are sized to its own batch and
    the one-step lookahead evaluates 16 successor states per environment."""
    p, R, _, _ = kin.tcp_fk_jac(q)
    z_tool = R[:, :, 2]
    q_norm = (q - q_mid) / q_half
    cos_angle = (z_tool * n_target).sum(-1, keepdim=True)
    z_cross_n = torch.linalg.cross(z_tool, n_target, dim=-1)
    return torch.cat([q_norm, q_norm * q_norm, line_dir, z_tool, n_target,
                      cos_angle, z_cross_n, a_prev], dim=-1)


def load_agent(ckpt_dir: Path, obs_dim: int, device):
    ck = torch.load(ckpt_dir / 'agent.pt', map_location=device,
                    weights_only=False)
    sd = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck
    if any(k.startswith('_logits_head') for k in sd):
        from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
        a = VertexAgent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(device)
    else:
        a = Agent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(device)
    a.load_state_dict(sd)
    return a.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='Yuan/IJRR/runs/rl_vertex_line_30M')
    ap.add_argument('--config',
                    default='Yuan/IJRR/stage2_traj/config_vertex_line.yaml')
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / a.config))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    line_cfg = y['line_distribution']
    N = a.n_tasks

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=max(3 * N, 20000),
        n_target_noise_deg=line_cfg['n_target_noise_deg'], seed=a.seed,
        env_cfg=env.cfg,
        feasibility_threshold_m=line_cfg['feasibility_threshold_m'],
        verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    env.line_dist = ScriptedLineDistribution(
        {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
         'n_target': pool.n_target_pool[idx]})

    agent = load_agent(REPO / a.ckpt, env.obs_dim, dev)

    with torch.no_grad():
        env.reset()

    vertices = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'),
                 -1).reshape(-1, 4), dtype=torch.float32, device=dev)  # (16,4)
    a_max, dt = env.cfg.a_max, env.cfg.dt

    sig_all, act_all, greedy_all, alive_all, flip_all = [], [], [], [], []
    prev_act = None
    checked = False
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(env.max_steps):
        # ---- switching vector from the critic, by autograd through obs(q)
        q = env.q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            p, R, _, _ = env.kin.tcp_fk_jac(q)
            obs = env._compute_obs(p, R, q=q)
            v = agent.get_value(obs)
            g, = torch.autograd.grad(v.sum(), q)
        with torch.no_grad():
            if not checked:   # local obs builder must equal the env's
                o2 = obs31(env.q, env.line_dir, env.n_target, env.a_prev,
                           env.kin.q_mid, env.q_half, env.kin)
                assert torch.allclose(o2, obs.detach(), atol=1e-5), \
                    'obs31 drifted from NSRLBatchedEnv._compute_obs'
                checked = True
            _, _, J, _ = env.kin.tcp_fk_jac(env.q)
            J_plus, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                                    env.cfg.sigma_thr)
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping)
            sigma = torch.einsum('bij,bi->bj', B_basis, g)   # (N, 4)
            act = agent.actor_mean(obs.detach())             # (N, 4) in {-1,1}

            # ---- discrete one-step lookahead greedy over the 16 vertices.
            # The advance term is vertex-independent (exact null space), so
            # greedy = argmax_j V(q + (qdot_task + B a_max v_j) dt).
            x_dot = (env.cfg.v * env.line_dir).unsqueeze(-1)
            qdot_task = (J_plus @ x_dot).squeeze(-1)          # (N, 7)
            qn = (env.q.unsqueeze(1) + qdot_task.unsqueeze(1) * dt
                  + torch.einsum('bij,kj->bki', B_basis,
                                 vertices) * a_max * dt)      # (N, 16, 7)
            rep = lambda t: t.unsqueeze(1).expand(-1, 16, *t.shape[1:]) \
                             .reshape(N * 16, *t.shape[1:])
            on = obs31(qn.reshape(N * 16, 7), rep(env.line_dir),
                       rep(env.n_target),
                       vertices.unsqueeze(0).expand(N, -1, -1).reshape(-1, 4),
                       env.kin.q_mid, env.q_half, env.kin)
            vn = agent.get_value(on).reshape(N, 16)
            greedy = vertices[vn.argmax(-1)]                  # (N, 4)

            live = ~done
            sig_all.append(sigma.cpu())
            act_all.append(act.cpu())
            greedy_all.append(greedy.cpu())
            alive_all.append(live.cpu())
            flip_all.append(((act != prev_act) & live.unsqueeze(-1)).cpu()
                            if prev_act is not None
                            else torch.zeros_like(act, dtype=torch.bool).cpu())
            prev_act = act
            env.step(act, auto_reset=False)
            done = env.done_persistent.clone()
        if bool(done.all()):
            break

    sig = torch.stack(sig_all).numpy()            # (T, N, 4)
    act = torch.stack(act_all).numpy()
    grd = torch.stack(greedy_all).numpy()
    alive = torch.stack(alive_all).numpy()        # (T, N)
    flip = torch.stack(flip_all).numpy()
    m = alive[..., None].repeat(4, axis=-1).astype(bool)
    ma = alive.astype(bool)

    s = sig[m].reshape(-1)
    v_act = act[m].reshape(-1)
    v_grd = grd[m].reshape(-1)
    fl = flip[m].reshape(-1)
    match_sig = (np.sign(s) == v_act)
    match_grd = (v_grd == v_act)

    print(f'ckpt {a.ckpt}')
    print(f'{alive.sum()} live steps x 4 coordinates = {len(s)} decisions\n')
    print(f'A. actor vs one-step lookahead greedy on its own critic')
    print(f'  per-coordinate match            : {match_grd.mean():.1%}   (chance 50%)')
    print(f'  full-vertex match (all 4)       : '
          f'{(grd == act).all(-1)[ma].mean():.1%}   (chance 6.25%)')
    print(f'\nB. actor vs sign of the continuous-time switching vector')
    print(f'  per-coordinate match            : {match_sig.mean():.1%}')
    print(f'  full-vertex match (all 4)       : '
          f'{(np.sign(sig) == act).all(-1)[ma].mean():.1%}')
    print(f'\nC. lookahead greedy vs sign(sigma)  (discretization gap)')
    print(f'  per-coordinate match            : '
          f'{(np.sign(s) == v_grd).mean():.1%}')
    for k in range(4):
        print(f'    coordinate {k}: greedy {100 * (grd[..., k][ma] == act[..., k][ma]).mean():.1f}%'
              f'   sigma {100 * (np.sign(sig[..., k][ma]) == act[..., k][ma]).mean():.1f}%'
              f'   median|sigma_{k}| {np.median(np.abs(sig[..., k][ma])):.4f}')

    print(f'\n  by decile of |sigma_i|  (prediction: match rises, flips fall)')
    q10 = np.quantile(np.abs(s), np.linspace(0, 1, 11))
    print(f'  {"decile":>8s}{"|sigma| range":>24s}{"match(grd)":>11s}'
          f'{"match(sig)":>11s}{"flip rate":>11s}')
    for d in range(10):
        sel = (np.abs(s) >= q10[d]) & (np.abs(s) <= q10[d + 1])
        print(f'  {d + 1:>8d}{q10[d]:>11.4f} - {q10[d + 1]:<10.4f}'
              f'{match_grd[sel].mean():>11.1%}'
              f'{match_sig[sel].mean():>11.1%}{fl[sel].mean():>11.3f}')

    dst = Path(a.out) if a.out else REPO / a.ckpt / 'switching_signal.npz'
    np.savez_compressed(dst, sigma=sig.astype(np.float32),
                        act=act.astype(np.int8), greedy=grd.astype(np.int8),
                        alive=alive, flip=flip)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
