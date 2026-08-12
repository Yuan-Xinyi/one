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


@torch.no_grad()
def stage_select2(a, dev):
    """Pick the task with the largest HEADROOM instead of the farthest
    myopic: rank by (pointwise kinematic bound L_hi) - (myopic progress).

    Two-pass: the first call dumps all candidate tasks in eval-set format
    and prints the line_bound command that produces the bounds; once that
    npz exists, the second call runs the arms, ranks by gap and writes
    task.npz exactly like stage select.
    """
    y, env = _env_and_yaml(a.n_tasks, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:a.n_tasks]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}

    OUT.mkdir(parents=True, exist_ok=True)
    bound_npz = OUT / 'bound_all.npz'
    if not bound_npz.exists():
        p0, _, _, _ = env.kin.tcp_fk_jac(spec['q0'])
        cs = OUT / 'tasks_cs_all.npz'
        np.savez(cs,
                 cs_p0=p0.cpu().numpy().astype(np.float32),
                 cs_line_dir=spec['line_dir'].cpu().numpy().astype(np.float32),
                 cs_n_target=spec['n_target'].cpu().numpy().astype(np.float32))
        print(f"wrote {cs}; now produce the bounds with:\n"
              f"  python -m Yuan.IJRR.eval.line_bound --tasks "
              f"{cs.relative_to(REPO)} --n-tasks {a.n_tasks} --step 0.02 "
              f"--max-len 2.0 --out {bound_npz.relative_to(REPO)}\n"
              f"then re-run --stage select2.")
        return

    res = {}
    for name, fn in _arms(env).items():
        env.line_dist = ScriptedLineDistribution(
            {k: v.clone() for k, v in spec.items()})
        res[name] = rollout_first_episode(env, fn)
    myo = res['myopic']['episode_progress'].cpu().numpy()
    cl = res['classical']['episode_progress'].cpu().numpy()
    ze = res['zero']['episode_progress'].cpu().numpy()
    term = res['myopic']['term_reason'].cpu().numpy()

    b = np.load(bound_npz, allow_pickle=True)
    L_hi = b['L_hi'].astype(np.float64)
    cens = b['censored'].astype(bool)
    gap = L_hi - myo
    order = np.argsort(-gap)

    print('\ntop tasks by (bound - myopic) headroom:')
    print(f"{'rank':>4} {'task':>5} {'myopic':>8} {'L_hi':>7} {'gap':>7} "
          f"{'myo/L_hi':>9} {'cens':>5}  term")
    for r in range(min(15, len(order))):
        i = order[r]
        print(f"{r:>4} {i:>5} {myo[i]:>8.4f} {L_hi[i]:>7.3f} {gap[i]:>7.3f} "
              f"{myo[i] / L_hi[i]:>9.3f} {str(bool(cens[i])):>5}  "
              f"{TERM_NAMES[int(term[i])]}")

    chosen = int(order[a.task_rank])
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
             bound_L_hi=float(L_hi[chosen]),
             bound_censored=bool(cens[chosen]),
             all_myopic=myo, all_classical=cl, all_zero=ze, all_term=term,
             all_L_hi=L_hi)
    print(f"\nchosen task {chosen}: myopic {myo[chosen]:.4f} m, bound L_hi "
          f"{L_hi[chosen]:.3f} m (gap {gap[chosen]:.3f} m) -> "
          f"{OUT / 'task.npz'}")


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


@torch.no_grad()
def stage_ceiling(a, dev):
    """How much room is above myopic on this task?

    Three probes, weakest to strongest claim:
      - exact 2-step tree + beam ladder on the margin objective (single task,
        so the beam can be very wide): what deterministic exact-ish search
        achieves — a lower bound on the ceiling;
      - best-of-N stochastic rollouts of the trained PPO policy: is there a
        better command sequence in the policy's own neighbourhood;
      - dumps the task in eval-set format for line_bound.py, whose IK march
        gives the kinematics-only upper bound (run separately).
    """
    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    myo_ref = float(task['myopic_progress'])
    print(f"task {int(task['task_index'])}: myopic ref {myo_ref:.4f} m")

    model = hl.StraightModel(env)
    model.terms = MYOPIC_TERMS
    arms = {'mtree2': hl.make_margin_tree2(model)}
    if a.beams:
        for s in a.beams.split(','):
            w, h = s.split('x')
            arms[f'beam{s}'] = hl.make_beam(model, int(w), int(h))
    for name, fn in arms.items():
        env.line_dist = SingleTaskDistribution(spec1)
        t0 = time.time()
        stats = rollout_first_episode(
            env, lambda e, f=fn: f(e, e.done_persistent))
        p = float(stats['episode_progress'][0])
        print(f"{name:<12s} progress {p:.4f} m  ratio {p / myo_ref:.4f}  "
              f"len {int(stats['episode_len'][0]):>4}  "
              f"term {TERM_NAMES[int(stats['term_reason'][0])]}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    _, env_n = _env_and_yaml(a.best_of, dev)
    agent = VertexAgent(obs_dim=env_n.obs_dim, act_dim=env_n.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(OUT / 'agent.pt', map_location=dev))
    agent.eval()

    def stoch(e):
        logits = agent._logits_head(agent._actor_trunk(e.current_obs()))
        idx = torch.distributions.Categorical(logits=logits).sample()
        return agent.vertices[idx]

    env_n.line_dist = SingleTaskDistribution(spec1)
    stats = rollout_first_episode(env_n, stoch)
    p = stats['episode_progress'].cpu().numpy()
    print(f"ppo-stoch    best-of-{a.best_of}: max {p.max():.4f} m "
          f"(ratio {p.max() / myo_ref:.4f})  "
          f"median {np.median(p):.4f}  min {p.min():.4f}")

    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    np.savez(OUT / 'task_cs.npz',
             cs_p0=env.p_start[0].cpu().numpy()[None].astype(np.float32),
             cs_line_dir=task['line_dir'][None].astype(np.float32),
             cs_n_target=task['n_target'][None].astype(np.float32),
             cs_q0=task['q0'][None].astype(np.float32))
    print(f"\nwrote {OUT / 'task_cs.npz'}; kinematic upper bound via:\n"
          f"  python -m Yuan.IJRR.eval.line_bound --tasks "
          f"{(OUT / 'task_cs.npz').relative_to(REPO)} --n-tasks 1 "
          f"--step 0.01 --n-dirs 48")


@torch.no_grad()
def stage_reachtree(a, dev):
    """Exhaustive-ish forward search for the farthest CONTINUOUS trajectory.

    From the task's q0, every kept state is expanded through all 2^m vertex
    commands with the exact environment step; states are kept alive iff all
    four normalized margins stay positive, deduplicated on a joint-space grid
    and (only if still over the width cap) randomly thinned. Selection never
    looks at the margin value, so the search cannot inherit the softmin
    objective's blind spot. Anything it returns is a genuinely executable
    open-loop command sequence; the deepest state is a continuous
    lower-bound witness for the task's reachability ceiling.
    """
    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    p0 = env.p_start[0]
    d = env.line_dir[0]
    n = env.n_target[0]

    model = hl.StraightModel(env)          # terms=None: all four margins
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    K = verts.shape[0]
    W = a.tree_width
    grid = a.tree_dedupe

    q = env.q[:1].clone()                  # (1, 7) the task's start
    parents = []                           # per depth: (P,) parent index
    actions = []                           # per depth: (P,) vertex index
    pools = [q.cpu()]                      # per depth: kept configs
    rng = np.random.default_rng(0)
    t0 = time.time()
    depth = 0
    while q.shape[0] > 0 and depth < env.max_steps:
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, K, -1).reshape(P * K, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * K, -1)
        qn = torch.cat([model.step(qe[i:i + 32768],
                                   d.expand(min(32768, P * K - i), 3),
                                   n.expand(min(32768, P * K - i), 3),
                                   ae[i:i + 32768])
                        for i in range(0, P * K, 32768)])
        m = torch.cat([model.margins(qn[i:i + 32768],
                                     p0.expand(min(32768, P * K - i), 3),
                                     d.expand(min(32768, P * K - i), 3),
                                     n.expand(min(32768, P * K - i), 3))
                       for i in range(0, P * K, 32768)])
        alive = (m.amin(dim=-1) > 0).nonzero(as_tuple=False).squeeze(-1)
        if alive.numel() == 0:
            break
        qn = qn[alive]
        par = (alive // K)
        act = (alive % K)
        # dedupe on a joint-space grid: cheap dispersion-keeping selection
        key = torch.round(qn / grid).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > W:
            keep = keep[torch.as_tensor(
                np.sort(rng.choice(keep.numel(), W, replace=False)),
                device=dev)]
        q = qn[keep]
        parents.append(par[keep].cpu())
        actions.append(act[keep].cpu())
        pools.append(q.cpu())
        depth += 1
        if depth % 20 == 0:
            print(f"depth {depth:>4}  pool {q.shape[0]:>5}  "
                  f"s ~ {depth * env.cfg.dt * env.cfg.v:.3f} m  "
                  f"{time.time() - t0:.0f}s", flush=True)

    # backtrack the deepest state with the largest actual progress
    qf = pools[depth].to(dev)
    pf, _, _, _ = env.kin.tcp_fk_jac(qf)
    prog = ((pf - p0) * d).sum(-1)
    best = int(prog.argmax())
    traj_q = [pools[depth][best]]
    traj_a = []
    i = best
    for r in range(depth - 1, -1, -1):
        traj_a.append(int(actions[r][i]))
        i = int(parents[r][i])
        traj_q.append(pools[r][i])
    traj_q = torch.stack(traj_q[::-1]).numpy()
    traj_a = np.array(traj_a[::-1], dtype=np.int64)

    myo_ref = float(task['myopic_progress'])
    print(f"\nreachtree: died out at depth {depth} "
          f"({depth * env.cfg.dt:.2f} s); best progress "
          f"{float(prog[best]):.4f} m  (myopic {myo_ref:.4f}, "
          f"ratio {float(prog[best]) / myo_ref:.2f})")
    np.savez(OUT / 'reachtree.npz', q=traj_q, action_idx=traj_a,
             progress=float(prog[best]), depth=depth,
             tree_width=W, dedupe=grid)
    print(f"wrote {OUT / 'reachtree.npz'}")


@torch.no_grad()
def _record_traj(env, spec1, fn):
    """Roll one episode on the single task, recording q after every step."""
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    qs = [env.q[0].clone()]
    for _ in range(env.max_steps):
        _, _, _, _, info = env.step(fn(env), auto_reset=False)
        qs.append(env.q[0].clone())
        if bool(info['episode_done'][0]):
            return torch.stack(qs), int(info['term_reason'][0])
    return torch.stack(qs), TERM_TRUNCATED


@torch.no_grad()
def stage_traj(a, dev):
    """Joint-angle trajectories of myopic vs the trained PPO on the task."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    model = hl.StraightModel(env)
    model.terms = MYOPIC_TERMS

    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(OUT / 'agent.pt', map_location=dev))
    agent.eval()

    myo = hl.make_myopic(model)
    trajs = {}
    for name, fn in (('myopic', lambda e: myo(e, e.done_persistent)),
                     ('PPO', lambda e: agent.actor_mean(e.current_obs()))):
        q, term = _record_traj(env, spec1, fn)
        # env.p_start/line_dir/n_target of the recording episode
        p0 = env.p_start[0].expand(q.shape[0], 3)
        d = env.line_dir[0].expand(q.shape[0], 3)
        n = env.n_target[0].expand(q.shape[0], 3)
        m = model.margins(q, p0, d, n)          # (T, 4) jl/cone/lat/coll
        trajs[name] = (q.cpu().numpy(), m.cpu().numpy(), term)
        print(f"{name:<7s} {q.shape[0] - 1} steps, term "
              f"{TERM_NAMES[term]}")

    dt = env.cfg.dt
    q_mid = env.q_mid.cpu().numpy()
    q_half = env.q_half.cpu().numpy()
    deg = 180.0 / np.pi
    colors = {'myopic': 'tab:blue', 'PPO': 'tab:red'}

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for j in range(7):
        ax = axes.flat[j]
        for name, (q, _, _) in trajs.items():
            t = np.arange(q.shape[0]) * dt
            ax.plot(t, q[:, j] * deg, color=colors[name], lw=1.4,
                    label=name)
        for s in (-1, 1):
            ax.axhline((q_mid[j] + s * q_half[j]) * deg, color='k',
                       lw=0.7, ls='--', alpha=0.6)
        ax.set_title(f'joint {j + 1}', fontsize=10)
        ax.set_ylabel('deg', fontsize=8)
        ax.tick_params(labelsize=8)
    for k, (mi, lab) in enumerate(((0, 'margin m_jl'),
                                   (1, 'margin m_cone'))):
        ax = axes.flat[7 + k]
        for name, (q, m, _) in trajs.items():
            t = np.arange(q.shape[0]) * dt
            ax.plot(t, m[:, mi], color=colors[name], lw=1.4, label=name)
        ax.axhline(0.0, color='k', lw=0.7, ls='--', alpha=0.6)
        ax.set_title(lab, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('time (s)', fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    steps = {k: v[0].shape[0] - 1 for k, v in trajs.items()}
    fig.legend(handles,
               [f'{l} ({steps[l]} steps, '
                f'term {TERM_NAMES[trajs[l][2]]})' for l in labels],
               loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=2, fontsize=10, frameon=False)
    fig.suptitle(f"task {int(task['task_index'])}: joint trajectories, "
                 f"myopic vs single-task PPO", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / 'traj_compare.png'
    fig.savefig(out, dpi=160)
    np.savez(OUT / 'traj_compare.npz',
             **{f'{k}_q': v[0] for k, v in trajs.items()},
             **{f'{k}_margins': v[1] for k, v in trajs.items()})
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['select', 'select2', 'train', 'report',
                             'ceiling', 'traj', 'reachtree'])
    ap.add_argument('--tree-width', type=int, default=4096)
    ap.add_argument('--tree-dedupe', type=float, default=0.03,
                    help='joint-space grid (rad) for reachtree dedup')
    ap.add_argument('--run-dir', default='single_task_ppo',
                    help='subdirectory of Yuan/IJRR/runs for all artifacts')
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--task-rank', type=int, default=0,
                    help='pick the k-th best myopic task instead of the best')
    ap.add_argument('--n-envs', type=int, default=128)
    ap.add_argument('--total-steps', type=int, default=5_000_000)
    ap.add_argument('--eval-every', type=int, default=100_000)
    ap.add_argument('--beams', default='16x4,16x8,64x8,64x16,256x16,1024x16')
    ap.add_argument('--best-of', type=int, default=2048)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    global OUT
    OUT = REPO / 'Yuan/IJRR/runs' / a.run_dir
    dev = torch.device(a.device)
    torch.manual_seed(0)
    {'select': stage_select, 'select2': stage_select2, 'train': stage_train,
     'report': stage_report, 'ceiling': stage_ceiling,
     'traj': stage_traj, 'reachtree': stage_reachtree}[a.stage](a, dev)


if __name__ == '__main__':
    main()
