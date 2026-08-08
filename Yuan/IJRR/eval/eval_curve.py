"""Evaluate a policy on the serpentine task family, alone and inside the hybrid.

Three resolutions are compared on identical tasks and identical start
configurations, so any difference is attributable to the resolution alone:

    classical   the tuned null-space law of Eq. (7)
    rl          the learned null-space policy
    hybrid      rl in the interior, classical once max|q_norm| >= tau_enter,
                back to rl below tau_exit

Only ratios are reported. Absolute lengths are not comparable across swing
buckets, because a task whose bends are sharper has a shorter attainable stroke
to begin with; the ratio to the classical law on the same task is.

Usage:
    python -m Yuan.IJRR.eval.eval_curve --ckpt <run-dir> --n-tasks 2048
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.env.rollout import rollout_first_episode
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = Path(__file__).resolve().parents[3]


def _agent(ckpt_dir: Path, obs_dim: int, device):
    ck = torch.load(ckpt_dir / 'agent.pt', map_location=device,
                    weights_only=False)
    sd = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck
    # A vertex-action checkpoint is recognised by its categorical head.
    if any(k.startswith('_logits_head') for k in sd):
        from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
        a = VertexAgent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(device)
    else:
        a = Agent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(device)
    a.load_state_dict(sd)
    return a.eval()


def _hybrid_fn(agent, classical, tau_enter, tau_exit):
    """Variant-B hysteresis on max|q_norm|, read straight off the observation."""
    state = {}

    @torch.no_grad()
    def fn(env):
        qn = ((env.q - env.q_mid) / env.q_half).abs().max(dim=-1).values
        using_rl = state.get('using_rl')
        if using_rl is None or using_rl.shape[0] != qn.shape[0]:
            using_rl = qn < tau_enter
        stay = torch.where(using_rl, qn < tau_enter, qn < tau_exit)
        state['using_rl'] = stay
        a_rl = agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
        a_cl = cn_action_fn(classical)(env)
        return torch.where(stay.unsqueeze(-1), a_rl, a_cl)

    return fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='Yuan/IJRR/runs/rl_curve_6M')
    ap.add_argument('--config', default='Yuan/IJRR/stage2_traj/config_curve.yaml')
    ap.add_argument('--n-tasks', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--tau-enter', type=float, default=0.98)
    ap.add_argument('--tau-exit', type=float, default=0.94)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    # Cross-condition overrides. Neither touches the dynamics: the curvature
    # channel is observation-only, and the swing bound belongs to the task
    # distribution, so a policy trained under one setting can be scored under
    # the other against a classical arm recomputed in the very same env.
    ap.add_argument('--observe-curvature', type=int, default=-1,
                    help='-1 keeps the config value; 0/1 overrides it, which '
                         'is what lets a 31-D line-trained policy run on '
                         'curved tasks')
    ap.add_argument('--swing-max-deg', type=float, default=-1.0,
                    help='-1 keeps the config value; 0 forces straight tasks, '
                         'which is what lets a curve-trained policy be scored '
                         'on the straight family')
    ap.add_argument('--k-lateral', type=float, default=-1.0,
                    help='-1 keeps the config value. Unlike the two above this '
                         'does change the closed loop, so it is only for '
                         'checking that the gain is inert on straight paths')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / a.config))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    line_cfg = dict(y['line_distribution'])
    if a.observe_curvature >= 0:
        kw['observe_curvature'] = bool(a.observe_curvature)
    if a.swing_max_deg >= 0.0:
        line_cfg['swing_max_deg'] = a.swing_max_deg
    if a.k_lateral >= 0.0:
        kw['k_lateral'] = a.k_lateral
    N = a.n_tasks

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=max(3 * N, 20000),
        n_target_noise_deg=line_cfg['n_target_noise_deg'], seed=a.seed,
        env_cfg=env.cfg, feasibility_threshold_m=line_cfg['feasibility_threshold_m'],
        swing_max_deg=line_cfg.get('swing_max_deg', 0.0),
        wavelen_range=tuple(line_cfg.get('wavelen_range', (0.4, 1.2))),
        min_radius_m=line_cfg.get('min_radius_m', 0.15), verbose=True)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    assert idx.numel() == N, f'only {idx.numel()} feasible tasks available'
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx],
            'amp': pool.amp_pool[idx], 'wavelen': pool.wavelen_pool[idx]}

    agent = _agent(REPO / a.ckpt, env.obs_dim, dev)
    classical = ClassicalNullspaceController(env.kin)
    arms = {
        'classical': cn_action_fn(classical),
        'rl': lambda e: agent.actor_mean(e.current_obs()).clamp(-1.0, 1.0),
        'hybrid': _hybrid_fn(agent, classical, a.tau_enter, a.tau_exit),
    }

    out = {}
    for name, fn in arms.items():
        env.line_dist = ScriptedLineDistribution({k: v.clone()
                                                  for k, v in spec.items()})
        st = rollout_first_episode(env, fn)
        out[name] = {'progress': st['episode_progress'].cpu().numpy().copy(),
                     'term': st['term_reason'].cpu().numpy().copy()}
        print(f'[{name}] done')

    # swing angle is the difficulty variable; 0 is the straight ray
    swing = np.degrees(np.arctan(
        2 * math.pi * pool.amp_pool[idx].cpu().numpy()
        / pool.wavelen_pool[idx].cpu().numpy()))
    base = out['classical']['progress']
    ok = base > 1e-6
    edges = [(0, 1e-6), (1e-6, 10), (10, 20), (20, 31)]
    names = ['straight', 'swing 0-10', 'swing 10-20', 'swing 20-30']

    report = {'n_tasks': int(N), 'tau': [a.tau_enter, a.tau_exit],
              'ckpt': a.ckpt, 'config': a.config, 'seed': a.seed,
              'observe_curvature': bool(kw.get('observe_curvature', False)),
              'swing_max_deg': float(line_cfg.get('swing_max_deg', 0.0)),
              'k_lateral': float(kw.get('k_lateral', 0.0)),
              'buckets': {}}
    print(f'\n{"bucket":<14s}{"n":>6s}' + ''.join(f'{k:>22s}' for k in arms))
    print(f'{"":14s}{"":6s}' + ''.join(f'{"ratio to classical":>22s}' for _ in arms))
    for (lo, hi), nm in zip(edges, names):
        sel = ok & (swing >= lo) & (swing < hi)
        if sel.sum() < 10:
            continue
        row, cells = {}, ''
        for k in arms:
            r = out[k]['progress'][sel] / base[sel]
            row[k] = {'mean': float(r.mean()), 'median': float(np.median(r))}
            cells += f'{r.mean():>10.4f} /{np.median(r):>9.4f}'
        report['buckets'][nm] = {'n': int(sel.sum()), **row}
        print(f'{nm:<14s}{int(sel.sum()):>6d}' + cells)

    print('\ntermination reasons (%)')
    for k in arms:
        t = out[k]['term']
        frac = {TERM_NAMES[c]: float((t == c).mean() * 100)
                for c in TERM_NAMES if (t == c).any()}
        report.setdefault('term', {})[k] = frac
        print(f'  {k:<10s}' + '  '.join(f'{n}={v:.1f}' for n, v in frac.items()))

    dst = Path(a.out) if a.out else REPO / a.ckpt / 'eval_curve.json'
    dst.write_text(json.dumps(report, indent=1))
    np.savez_compressed(dst.with_suffix('.npz'), swing=swing,
                        **{f'{k}_progress': out[k]['progress'] for k in arms},
                        **{f'{k}_term': out[k]['term'] for k in arms})
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
