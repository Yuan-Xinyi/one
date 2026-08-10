"""Finite-difference smoothness audit of the executed joint trajectories.

The 16.3 Hz action switching rate says the command jumps; what decides
hardware risk is what arrives in joint space after the null-space projection
and the shared task term. This measures it directly on the executed
trajectories: velocity from consecutive configurations (exact, since the
simulator integrates q_{k+1} = q_k + qdot dt), then first and second finite
differences for acceleration and jerk, per joint, within live segments only.

Reported per resolution as |qdot| / |qddot| / |qdddot| quantiles over all
(step, joint) pairs, plus the per-episode worst joint. The vertex policy is
the suspect; the classical law is the continuity reference; the hybrid adds
switch discontinuities on top.

Usage:
    python -m Yuan.IJRR.eval.smoothness_audit --n-tasks 1024
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = Path(__file__).resolve().parents[3]
CONT = 'Yuan/RL_controller/runs/rl_smmstart_30M'
VTX = 'Yuan/IJRR/runs/rl_vertex_line_30M'


def load(path, obs_dim, dev):
    ck = torch.load(REPO / path / 'agent.pt', map_location=dev,
                    weights_only=False)
    sd = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck
    if any(k.startswith('_logits_head') for k in sd):
        from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
        m = VertexAgent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(dev)
    else:
        m = Agent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(dev)
    m.load_state_dict(sd)
    return m.eval()


def hybrid_fn(agent, classical, te=0.98, tx=0.94):
    st = {}

    @torch.no_grad()
    def fn(env):
        qn = ((env.q - env.q_mid) / env.q_half).abs().max(dim=-1).values
        u = st.get('u')
        u = (qn < te) if (u is None or u.shape[0] != qn.shape[0]) \
            else torch.where(u, qn < te, qn < tx)
        st['u'] = u
        return torch.where(u.unsqueeze(-1),
                           agent.actor_mean(env.current_obs()).clamp(-1, 1),
                           cn_action_fn(classical)(env))
    return fn


@torch.no_grad()
def rollout_qdot(env, action_fn):
    """Executed joint velocities per step, with the live mask."""
    env.reset()
    qd, live = [], []
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    for _ in range(env.max_steps):
        q_before = env.q.clone()
        a = action_fn(env)
        env.step(a, auto_reset=False)
        qd.append(((env.q - q_before) / env.cfg.dt).cpu())
        live.append((~done).cpu())
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return torch.stack(qd).numpy(), torch.stack(live).numpy()  # (T,N,7),(T,N)


def stats(qd, live, dt):
    v = qd
    a = np.diff(qd, axis=0) / dt
    j = np.diff(qd, n=2, axis=0) / dt ** 2
    # a_k uses steps (k, k+1): valid where both live. j_k uses (k, k+1, k+2).
    lv = live.astype(bool)
    la = lv[:-1] & lv[1:]
    lj = lv[:-2] & lv[1:-1] & lv[2:]
    out = {}
    for name, arr, m in (('|qdot| (rad/s)', v, lv),
                         ('|qddot| (rad/s^2)', a, la),
                         ('|qdddot| (rad/s^3)', j, lj)):
        x = np.abs(arr[m]).reshape(-1)          # over (step, joint) pairs
        out[name] = [np.median(x), np.percentile(x, 95),
                     np.percentile(x, 99), x.max()]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    N = a.n_tasks

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}
    fresh = lambda: setattr(env, 'line_dist', ScriptedLineDistribution(
        {k: v.clone() for k, v in spec.items()}))

    cont = load(CONT, env.obs_dim, dev)
    vtx = load(VTX, env.obs_dim, dev)
    cl = ClassicalNullspaceController(env.kin)

    arms = {
        'classical': cn_action_fn(cl),
        'continuous RL': lambda e: cont.actor_mean(e.current_obs()).clamp(-1, 1),
        'projected to vertex': lambda e: torch.sign(
            cont.actor_mean(e.current_obs())).clamp(-1, 1),
        'vertex RL': lambda e: vtx.actor_mean(e.current_obs()),
        'hybrid (vertex)': hybrid_fn(vtx, cl),
    }

    dt = env.cfg.dt
    print(f'{a.n_tasks} straight tasks, dt = {dt * 1000:.0f} ms, '
          f'a_max = {env.cfg.a_max} rad/s\n')
    hdr = f'{"resolution":<22s}{"quantity":<20s}' \
          f'{"median":>9s}{"p95":>9s}{"p99":>9s}{"max":>10s}'
    print(hdr)
    for nm, fn in arms.items():
        fresh()
        qd, live = rollout_qdot(env, fn)
        for qty, row in stats(qd, live, dt).items():
            print(f'{nm:<22s}{qty:<20s}'
                  f'{row[0]:>9.3f}{row[1]:>9.3f}{row[2]:>9.3f}{row[3]:>10.3f}')
        print()


if __name__ == '__main__':
    main()
