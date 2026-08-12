"""Single-task PPO vs the myopic margin law: the cleanest possible RL test.

Question: on ONE fixed task (q0, line_dir, n_target), with the same 16-vertex
action space, the same 50 ms decision period and the same dynamics, can PPO
find a better command sequence than the deployed one-step margin lookahead
(horizon_ladder.make_myopic, margin terms jl+cone)?

Everything runs at SUB=1 — one 50 ms integration step per held command — so
the myopic arm's planning model is bit-identical to the environment step:
neither arm has a model error, and every arm is scored by the same
rollout_first_episode. Numbers here are therefore NOT comparable to the
horizon-ladder paper numbers (those integrate at 25 ms under a 50 ms hold);
the comparison is internal to this protocol.

Stages:
    select  run zero / classical / myopic over the standard eval pool
            (seed 4242, same recipe as horizon_ladder) and pick the task
            where myopic travels farthest without hitting the step cap
    train   PPO (VertexAgent, config_vertex_line.yaml hyperparameters, the
            recipe that produced the deployed 30M multi-task arm) with every
            reset replaying that single task
    report  deterministic-policy progress on the task vs myopic

Usage:
    python -m Yuan.IJRR.eval.single_task_ppo --stage select
    python -m Yuan.IJRR.eval.single_task_ppo --stage train --total-steps 5000000
    python -m Yuan.IJRR.eval.single_task_ppo --stage report
"""
from __future__ import annotations

# Same self-relaunch preamble as stage2_traj/train.py: put the conda
# libstdc++ on LD_LIBRARY_PATH before `one` pulls in matplotlib.
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, TERM_TRUNCATED)
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.env.rollout import rollout_first_episode
from Yuan.IJRR.stage2_traj.ppo import PPOConfig, train as ppo_train
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent

REPO = Path(__file__).resolve().parents[3]
CFG = REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'
OUT = REPO / 'Yuan/IJRR/runs/single_task_ppo'
MYOPIC_TERMS = [0, 1]     # jl + cone, the deployed combination

hl.SUB = 1                # model.step == env.step exactly at the 50 ms period


class SingleTaskDistribution:
    """Every reset replays the same task."""

    def __init__(self, spec: dict[str, torch.Tensor]):
        self._spec = {k: v.clone() for k, v in spec.items()}   # (1, ...) each

    def sample(self, n: int, generator=None) -> dict[str, torch.Tensor]:
        return {k: v.expand(n, *v.shape[1:]).clone()
                for k, v in self._spec.items()}


def _env_and_yaml(n_envs: int, dev: torch.device):
    y = yaml.safe_load(open(CFG))
    env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': n_envs}),
                         None, dev)
    return y, env


def _arms(env):
    model = hl.StraightModel(env)
    model.terms = MYOPIC_TERMS
    fcl = cn_action_fn(ClassicalNullspaceController(env.kin))
    myo = hl.make_myopic(model)
    return {
        'zero': lambda e: torch.zeros((e.n_envs, e.act_dim), device=e.device),
        'classical': fcl,
        'myopic': lambda e: myo(e, e.done_persistent),
    }


def _load_task(dev, dtype):
    task = np.load(OUT / 'task.npz')
    spec = {k: torch.tensor(task[k], device=dev, dtype=dtype).unsqueeze(0)
            for k in ('q0', 'line_dir', 'n_target')}
    return task, spec


@torch.no_grad()
def stage_select(a, dev):
    y, env = _env_and_yaml(a.n_tasks, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:a.n_tasks]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}

    res = {}
    for name, fn in _arms(env).items():
        t0 = time.time()
        env.line_dist = ScriptedLineDistribution(
            {k: v.clone() for k, v in spec.items()})
        res[name] = rollout_first_episode(env, fn)
        p = res[name]['episode_progress'].cpu().numpy()
        print(f"{name:<10s} mean progress {p.mean():.4f} m   "
              f"({time.time() - t0:.0f}s)", flush=True)

    myo = res['myopic']['episode_progress'].cpu().numpy()
    cl = res['classical']['episode_progress'].cpu().numpy()
    ze = res['zero']['episode_progress'].cpu().numpy()
    term = res['myopic']['term_reason'].cpu().numpy()
    ok = term != TERM_TRUNCATED
    order = np.argsort(-np.where(ok, myo, -np.inf))

    print('\ntop tasks by myopic progress (step-cap tasks excluded):')
    print(f"{'rank':>4} {'task':>5} {'myopic':>8} {'classical':>9} "
          f"{'zero':>8} {'len':>5}  term")
    for r in range(min(10, len(order))):
        i = order[r]
        print(f"{r:>4} {i:>5} {myo[i]:>8.4f} {cl[i]:>9.4f} {ze[i]:>8.4f} "
              f"{int(res['myopic']['episode_len'][i]):>5}  "
              f"{TERM_NAMES[int(term[i])]}")

    chosen = int(order[a.task_rank])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'task.npz',
             q0=spec['q0'][chosen].cpu().numpy(),
             line_dir=spec['line_dir'][chosen].cpu().numpy(),
             n_target=spec['n_target'][chosen].cpu().numpy(),
             task_index=chosen, pool_seed=a.seed,
             myopic_progress=float(myo[chosen]),
             classical_progress=float(cl[chosen]),
             zero_progress=float(ze[chosen]),
             myopic_len=int(res['myopic']['episode_len'][chosen]),
             myopic_term=int(term[chosen]),
             all_myopic=myo, all_classical=cl, all_zero=ze, all_term=term)
    print(f"\nchosen task {chosen}: myopic {myo[chosen]:.4f} m over "
          f"{int(res['myopic']['episode_len'][chosen])} steps "
          f"(term {TERM_NAMES[int(term[chosen])]}) -> {OUT / 'task.npz'}")


def stage_train(a, dev):
    y, env = _env_and_yaml(a.n_envs, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    myopic_ref = float(task['myopic_progress'])

    _, eval_env = _env_and_yaml(2, dev)

    def eval_fn(agent):
        @torch.no_grad()
        def fn(e):
            return agent.actor_mean(e.current_obs()).clamp(-1.0, 1.0)
        eval_env.line_dist = SingleTaskDistribution(spec1)
        stats = rollout_first_episode(eval_env, fn)
        p = float(stats['episode_progress'][0])
        return {'eval/progress_m': p,
                'eval/ratio_to_myopic': p / myopic_ref,
                'eval/episode_len': int(stats['episode_len'][0]),
                'eval/term': TERM_NAMES[int(stats['term_reason'][0])]}

    ppo_cfg = PPOConfig(**{**y['ppo'], 'total_timesteps': a.total_steps})
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=ppo_cfg.hidden_dim).to(dev)
    print(f"[train] single task {int(task['task_index'])}, myopic ref "
          f"{myopic_ref:.4f} m; PPO {a.total_steps} steps on "
          f"{a.n_envs} envs, {agent.n_actions} vertex actions", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT / 'train.log', 'w')
    t0 = time.time()

    def log_fn(d):
        log_file.write(repr({'wall_s': time.time() - t0, **d}) + '\n')
        log_file.flush()
        if 'update' in d and d['update'] % 10 == 0:
            print(f"upd {d['update']:>5}  step {d['global_step']:>9}  "
                  f"ep_prog {d.get('episode/progress_mean_m', 0):.4f} m  "
                  f"ep_len {d.get('episode/length_mean', 0):6.1f}  "
                  f"entropy {d.get('train/entropy', 0):.2f}", flush=True)
        elif 'eval_at_step' in d:
            print(f"  eval @ {d['eval_at_step']:>9}  "
                  f"progress {d.get('eval/progress_m', 0):.4f} m  "
                  f"ratio-to-myopic {d.get('eval/ratio_to_myopic', 0):.4f}  "
                  f"len {d.get('eval/episode_len', 0)}  "
                  f"term {d.get('eval/term')}", flush=True)

    ppo_train(ppo_cfg, env, device=dev, agent=agent,
              eval_fn=eval_fn, eval_every=a.eval_every, log_fn=log_fn,
              ckpt_path=str(OUT / 'agent.pt'), ckpt_every_n_updates=25)
    log_file.close()
    print(f"[train] done -> {OUT / 'agent.pt'}")


@torch.no_grad()
def stage_report(a, dev):
    y, env = _env_and_yaml(4, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(OUT / 'agent.pt', map_location=dev))
    agent.eval()

    arms = _arms(env)
    arms['ppo'] = lambda e: agent.actor_mean(e.current_obs())
    print(f"task {int(task['task_index'])} (pool seed "
          f"{int(task['pool_seed'])}), 50 ms period, margin terms jl+cone")
    prog = {}
    for name, fn in arms.items():
        env.line_dist = SingleTaskDistribution(spec1)
        stats = rollout_first_episode(env, fn)
        prog[name] = float(stats['episode_progress'][0])
        print(f"{name:<10s} progress {prog[name]:.4f} m  "
              f"len {int(stats['episode_len'][0]):>4}  "
              f"term {TERM_NAMES[int(stats['term_reason'][0])]}")
    print(f"\nPPO / myopic = {prog['ppo'] / prog['myopic']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['select', 'train', 'report'])
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--task-rank', type=int, default=0,
                    help='pick the k-th best myopic task instead of the best')
    ap.add_argument('--n-envs', type=int, default=128)
    ap.add_argument('--total-steps', type=int, default=5_000_000)
    ap.add_argument('--eval-every', type=int, default=100_000)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)
    {'select': stage_select, 'train': stage_train,
     'report': stage_report}[a.stage](a, dev)


if __name__ == '__main__':
    main()
