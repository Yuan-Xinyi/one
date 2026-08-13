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


class CurriculumBankDistribution(SingleTaskDistribution):
    """Reverse-order curriculum over an RRT-style archive: bank resets start
    from the DEEPEST states only, and the admissible depth window slides back
    toward q0 as training proceeds. Each stage asks the policy to extend an
    already-competent region by one notch, sidestepping the V^pi bootstrap
    trap (mid-corridor states look worthless while the policy dies there)."""

    def __init__(self, spec, q_bank, depths, frac, total_resets):
        super().__init__(spec)
        self._bank = q_bank.clone()
        self._depths = depths.clone()
        self._dmax = int(depths.max())
        self._frac = float(frac)
        self._total = float(total_resets)
        self._seen = 0

    def sample(self, n, generator=None):
        out = super().sample(n)
        prog = min(1.0, self._seen / self._total)
        self._seen += n
        d_lo = int(round((1.0 - prog) * 0.9 * self._dmax))
        elig = (self._depths >= d_lo).nonzero(as_tuple=False).squeeze(-1)
        use = torch.rand(n, device=self._bank.device) < self._frac
        if bool(use.any()) and elig.numel() > 0:
            rows = elig[torch.randint(0, elig.numel(), (int(use.sum()),),
                                      device=self._bank.device)]
            q0 = out['q0'].clone()
            q0[use] = self._bank[rows]
            out['q0'] = q0
        return out


class NoveltyEnv(NSRLBatchedEnv):
    """Count-based intrinsic exploration: r += beta / sqrt(N(cell(q))).

    Cells are the configuration rounded to `cell` radians, hashed into a
    fixed table. Novel joint configurations pay a bonus that decays with
    visitation, steering exploration toward WHERE it has not been instead
    of merely adding noise. Only the reward changes; dynamics, termination
    and the evaluation metric are untouched.
    """

    def __init__(self, *args, novelty_beta=0.3, novelty_cell=0.15,
                 table_bits=22, **kwargs):
        super().__init__(*args, **kwargs)
        self._nov_beta = float(novelty_beta)
        self._nov_cell = float(novelty_cell)
        self._nov_size = 1 << table_bits
        self._nov_counts = torch.zeros(self._nov_size, device=self.device)
        g = torch.Generator(device='cpu').manual_seed(12345)
        self._nov_primes = torch.randint(
            1, 2 ** 61, (7,), generator=g, dtype=torch.int64
        ).to(self.device)

    def _nov_idx(self, q):
        cells = torch.round(q / self._nov_cell).to(torch.int64)
        return ((cells * self._nov_primes).sum(-1) % self._nov_size).abs()

    def step(self, actions, auto_reset=True):
        obs, rew, term, trunc, info = super().step(actions, auto_reset)
        idx = self._nov_idx(self.q)
        self._nov_counts.scatter_add_(
            0, idx, torch.ones_like(idx, dtype=self._nov_counts.dtype))
        bonus = self._nov_beta / self._nov_counts[idx].sqrt()
        return obs, rew + bonus, term, trunc, info


class RestartBankDistribution(SingleTaskDistribution):
    """Same task, but a fraction of resets start from bank states (e.g. the
    reachtree corridor) instead of q0: the Kakade-Langford restart
    distribution. Every bank state is itself reachable from q0, so nothing
    unreachable is being trained on; evaluation still starts from q0 only."""

    def __init__(self, spec, q_bank: torch.Tensor, frac: float):
        super().__init__(spec)
        self._bank = q_bank.clone()
        self._frac = float(frac)

    def sample(self, n: int, generator=None) -> dict[str, torch.Tensor]:
        out = super().sample(n, generator)
        use = torch.rand(n, device=self._bank.device) < self._frac
        if bool(use.any()):
            rows = torch.randint(0, self._bank.shape[0],
                                 (int(use.sum()),),
                                 device=self._bank.device)
            q0 = out['q0'].clone()
            q0[use] = self._bank[rows]
            out['q0'] = q0
        return out


def _env_and_yaml(n_envs: int, dev: torch.device, extra: dict | None = None):
    y = yaml.safe_load(open(CFG))
    env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': n_envs,
                                      **(extra or {})}), None, dev)
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
    extra = None
    levels = None
    if a.speed_levels:
        levels = tuple(float(x) for x in a.speed_levels.split(','))
        extra = {'speed_levels': levels}
    y = yaml.safe_load(open(CFG))
    if a.novelty_beta:
        env = NoveltyEnv(EnvConfig(**{**y['env'], 'n_envs': a.n_envs,
                                      **(extra or {})}), None, dev,
                         novelty_beta=a.novelty_beta,
                         novelty_cell=a.novelty_cell)
        print(f"[train] count-based novelty bonus: beta {a.novelty_beta}, "
              f"cell {a.novelty_cell} rad")
    else:
        _, env = _env_and_yaml(a.n_envs, dev, extra)
    task, spec1 = _load_task(dev, env.kin.dtype)
    if a.restart_bank:
        bk = np.load(REPO / a.restart_bank)
        q_bank = torch.tensor(bk['q'], device=dev, dtype=env.kin.dtype)
        if a.restart_window:
            lo, hi = (int(x) for x in a.restart_window.split(','))
            m = (bk['depth'] >= lo) & (bk['depth'] <= hi)
            q_bank = q_bank[torch.tensor(m, device=dev)]
            env.line_dist = RestartBankDistribution(spec1, q_bank,
                                                    a.restart_frac)
            print(f"[train] fixed-window bank [{lo},{hi}]: "
                  f"{q_bank.shape[0]} states, frac {a.restart_frac}")
        elif a.restart_curriculum:
            depths = torch.tensor(bk['depth'], device=dev)
            env.line_dist = CurriculumBankDistribution(
                spec1, q_bank, depths, a.restart_frac,
                a.restart_curriculum)
            print(f"[train] curriculum bank: {q_bank.shape[0]} states, "
                  f"deep-first window over {a.restart_curriculum} resets, "
                  f"frac {a.restart_frac}")
        else:
            env.line_dist = RestartBankDistribution(spec1, q_bank,
                                                    a.restart_frac)
            print(f"[train] restart bank: {q_bank.shape[0]} states from "
                  f"{a.restart_bank}, frac {a.restart_frac}")
    else:
        env.line_dist = SingleTaskDistribution(spec1)
    myopic_ref = float(task['myopic_progress'])

    _, eval_env = _env_and_yaml(2, dev, extra)

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

    ppo_kw = {**y['ppo'], 'total_timesteps': a.total_steps}
    if a.ent_coef is not None:
        ppo_kw['ent_coef'] = a.ent_coef
    if a.norm_returns is not None:
        ppo_kw['normalize_returns'] = bool(a.norm_returns)
    ppo_cfg = PPOConfig(**ppo_kw)
    if levels:
        from Yuan.IJRR.stage2_traj.vertex_agent import SpeedVertexAgent
        agent = SpeedVertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                                 hidden_dim=ppo_cfg.hidden_dim,
                                 speed_levels=levels).to(dev)
        print(f"[train] speed levels {levels}: {agent.n_actions} actions")
    else:
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

    anchor = None
    if a.anchor_data:
        gd = np.load(REPO / a.anchor_data)
        anchor = {'obs': torch.tensor(gd['obs'], device=dev,
                                      dtype=torch.float32),
                  'act': torch.tensor(gd['act'], device=dev,
                                      dtype=torch.long),
                  'coef': a.anchor_coef}
        print(f"[train] self-imitation anchor: {len(gd['act'])} pairs, "
              f"coef {a.anchor_coef}")
    ppo_train(ppo_cfg, env, device=dev, agent=agent,
              eval_fn=eval_fn, eval_every=a.eval_every, log_fn=log_fn,
              ckpt_path=str(OUT / 'agent.pt'), ckpt_every_n_updates=25,
              resume_from_ckpt=a.resume_from_ckpt, anchor=anchor)
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
    # RRT-style archive: a subsample of every depth's surviving pool, i.e.
    # states covering the whole reachable frontier, not just the best route.
    rng2 = np.random.default_rng(1)
    bank, bank_depth = [], []
    for di, pool in enumerate(pools):
        k = min(128, pool.shape[0])
        bank.append(pool[np.sort(rng2.choice(pool.shape[0], k,
                                             replace=False))].numpy())
        bank_depth.append(np.full(k, di, dtype=np.int64))
    bank = np.concatenate(bank)
    np.savez(OUT / 'reachtree_bank.npz', q=bank,
             depth=np.concatenate(bank_depth))
    print(f"wrote {OUT / 'reachtree_bank.npz'} ({bank.shape[0]} states "
          f"across {len(pools)} depths)")


@torch.no_grad()
def stage_goexplore(a, dev):
    """Combination-lock paradigm, oracle-free: E3/Go-Explore phase 1.

    The agent builds its OWN archive from scratch: cells are joint
    configurations rounded to --ge-cell radians; each generation samples
    archive entries biased toward deep & rarely-visited cells, resets there,
    probes --ge-k random vertex steps, and archives every new surviving cell
    with a parent pointer. No reachtree data, no policy, no reward: the only
    privileges are the exact step model and reset-to-visited-state. The
    deepest entry's action chain is replayed through the real env at the end.
    """
    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    p0, d, n = env.p_start[0], env.line_dir[0], env.n_target[0]
    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    B, K, CELL = a.ge_batch, a.ge_k, a.ge_cell
    primes = torch.randint(1, 2 ** 61, (7,), dtype=torch.int64,
                           generator=torch.Generator().manual_seed(7)).to(dev)

    def cells_of(q):
        c = torch.round(q / CELL).to(torch.int64)
        return ((c * primes).sum(-1)).cpu().numpy()

    # archive arrays (grown in chunks)
    Q = [env.q[0].clone()]
    depth = [0]
    visits = [0]
    parent = [-1]
    acts = [np.zeros(0, dtype=np.int64)]     # action chunk from parent
    index = {int(cells_of(env.q[:1])[0]): 0}

    rng = np.random.default_rng(0)
    t0 = time.time()
    best_i = 0
    stall = 0
    for gen in range(a.ge_generations):
        dep = np.array(depth, dtype=np.float64)
        vis = np.array(visits, dtype=np.float64)
        w = (dep + 1.0) ** 2 / (vis + 1.0)
        sel = rng.choice(len(w), size=B, p=w / w.sum())
        # keep the frontier hot: half the batch comes from the deepest
        # cells regardless of how large the archive has grown (otherwise
        # the exploding number of shallow cells dilutes frontier sampling
        # and the expansion freezes)
        top = np.argsort(dep)[-256:]
        if a.ge_bias_j7:
            # ONE BIT of oracle knowledge: among the deepest ~2048 ties,
            # prefer high-j7 entries (the viable edge). Tests whether the
            # bottleneck is exploration effort or directional information.
            cand = np.argsort(dep + rng.uniform(0, 0.5, len(dep)))[-2048:]
            j7 = np.array([float(Q[i][6]) for i in cand])
            top = cand[np.argsort(j7)[-256:]]
        sel[: B // 2] = rng.choice(top, size=B // 2)
        for i in np.unique(sel):
            visits[i] += int((sel == i).sum())
        q = torch.stack([Q[i] for i in sel])
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        a_hist = np.zeros((B, K), dtype=np.int64)
        new_before = len(Q)
        for k in range(K):
            a_idx = torch.randint(0, 16, (B,), device=dev)
            if a.ge_drift_j7 > 0:
                # ACTION-level directional push: for a fraction of the
                # biased half-batch, take the vertex that raises joint 7
                # fastest — extends the fan itself instead of merely
                # re-sampling its existing edge
                nd = B // 2
                qe16 = q[:nd].unsqueeze(1).expand(-1, 16, -1).reshape(-1, 7)
                ae16 = verts.unsqueeze(0).expand(nd, -1, -1).reshape(-1, 4)
                qn16 = torch.cat(
                    [model.step(qe16[i:i + 32768],
                                d.expand(min(32768, nd * 16 - i), 3),
                                n.expand(min(32768, nd * 16 - i), 3),
                                ae16[i:i + 32768])
                     for i in range(0, nd * 16, 32768)]).reshape(nd, 16, 7)
                best_a = qn16[:, :, 6].argmax(-1)
                use = torch.rand(nd, device=dev) < a.ge_drift_j7
                a_idx[:nd] = torch.where(use, best_a, a_idx[:nd])
            a_hist[:, k] = a_idx.cpu().numpy()
            qn = model.step(q, d.expand(B, 3), n.expand(B, 3), verts[a_idx])
            m = model.margins(qn, p0.expand(B, 3), d.expand(B, 3),
                              n.expand(B, 3))
            alive = alive & (m.amin(-1) > 0)
            q = torch.where(alive.unsqueeze(-1), qn, q)
            if not bool(alive.any()):
                break
            live = alive.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            hs = cells_of(q[live])
            for j, h in zip(live, hs):
                h = int(h)
                if h not in index:
                    index[h] = len(Q)
                    Q.append(q[j].clone())
                    dnew = depth[sel[j]] + k + 1
                    depth.append(dnew)
                    visits.append(0)
                    parent.append(int(sel[j]))
                    acts.append(a_hist[j, :k + 1].copy())
                    if dnew > depth[best_i]:
                        best_i = len(Q) - 1
        stall = 0 if len(Q) > new_before else stall + 1
        if gen % 10 == 0 or stall >= a.ge_stall:
            print(f"gen {gen:>4}  archive {len(Q):>7}  frontier depth "
                  f"{depth[best_i]:>4} (~{depth[best_i] * 0.01:.2f} m)  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if stall >= a.ge_stall:
            break

    # backtrack the deepest entry and replay through the REAL env
    chain = []
    i = best_i
    while parent[i] >= 0:
        chain.append(acts[i])
        i = parent[i]
    seq = np.concatenate(chain[::-1]) if chain else np.zeros(0, np.int64)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    for ai in seq:
        env.step(verts[int(ai)][None], auto_reset=False)
    p_end, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = float(((p_end[0] - p0) * d).sum())
    print(f"\ngo-explore: archive {len(Q)} cells, frontier depth "
          f"{depth[best_i]} ; deepest chain replayed through env: "
          f"{prog:.4f} m (alive: {not bool(env.done_persistent[0])})")
    np.savez(OUT / 'goexplore.npz', action_idx=seq, progress=prog,
             archive_size=len(Q), frontier_depth=depth[best_i],
             cell=CELL, k=K, batch=B)
    print(f"wrote {OUT / 'goexplore.npz'}")
    # dump a depth-stratified sample of the archive itself (the EXPLORED
    # region) for visualization against the optimal corridor
    dep_arr = np.array(depth)
    rng3 = np.random.default_rng(3)
    keep_idx = []
    for dv in range(dep_arr.max() + 1):
        ids = np.nonzero(dep_arr == dv)[0]
        if len(ids):
            keep_idx.append(rng3.choice(ids, min(256, len(ids)),
                                        replace=False))
    keep_idx = np.concatenate(keep_idx)
    np.savez(OUT / 'goexplore_archive.npz',
             q=torch.stack([Q[i] for i in keep_idx]).cpu().numpy(),
             depth=dep_arr[keep_idx])
    print(f"wrote {OUT / 'goexplore_archive.npz'} "
          f"({len(keep_idx)} states)")


@torch.no_grad()
def stage_goexplore_env(a, dev):
    """Interaction-only Go-Explore: NO model, NO state teleport.

    Every probe starts at q0 in the REAL batched env and REPLAYS its action
    prefix (the env is deterministic, so the archive is reachable by
    construction); cells allow up to 3 representatives to break the
    doomed-representative aliasing seen at the 0.85 pinch. The archive
    stores action sequences only — everything the agent knows, it learned
    by acting from q0.
    """
    y = yaml.safe_load(open(CFG))
    B = a.ge_batch
    env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': B}), None, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    primes = torch.randint(1, 2 ** 61, (7,), dtype=torch.int64,
                           generator=torch.Generator().manual_seed(7)).to(dev)
    CELL, K, RMAX = a.ge_cell, a.ge_k, 3

    def cells_of(q):
        c = torch.round(q / CELL).to(torch.int64)
        return ((c * primes).sum(-1)).cpu().numpy()

    seqs = [np.zeros(0, dtype=np.int16)]
    depth = [0]
    visits = [0]
    cellcount: dict = {}
    rng = np.random.default_rng(a.torch_seed)
    best_i, stall, t0 = 0, 0, time.time()

    for gen in range(a.ge_generations):
        dep = np.array(depth, dtype=np.float64)
        vis = np.array(visits, dtype=np.float64)
        w = (dep + 1.0) ** 2 / (vis + 1.0)
        sel = rng.choice(len(w), size=B, p=w / w.sum())
        # deepest ~2048 entries with RANDOM tie-breaking: focuses probes on
        # the frontier while mixing postures among equal-depth ties (index-
        # order ties concentrate on the newest, barely-alive walkers)
        order = np.argsort(dep + rng.uniform(0.0, 0.5, len(dep)))
        front = order[-2048:]
        sel[: B // 2] = rng.choice(front, size=B // 2)
        fan = (gen % 5 == 4) and depth[best_i] > 60
        if fan:
            # exhaustive 3-step fan of ONE frontier entry (rotating through
            # the deepest 32): the full 16^3 = 4096 three-step fan with
            # random tails — 100% coverage where the corridor needs a short
            # exact sequence rather than luck
            sel[:] = order[-1 - (gen // 5) % 32]
        for i in np.unique(sel):
            visits[i] += int((sel == i).sum())
        L = np.array([len(seqs[i]) for i in sel])
        Lmax = int(L.max()) + K
        A = rng.integers(0, 16, (B, Lmax)).astype(np.int64)
        for r in range(B):
            if L[r]:
                A[r, :L[r]] = seqs[sel[r]]
        if fan:
            r_ = np.arange(B)
            A[r_, L] = r_ % 16
            A[r_, L + 1] = (r_ // 16) % 16
            A[r_, L + 2] = (r_ // 256) % 16
        env.reset()
        new_before = len(seqs)
        for t in range(Lmax):
            a_idx = torch.as_tensor(A[:, t], device=dev)
            env.step(verts[a_idx], auto_reset=False)
            alive = (~env.done_persistent).cpu().numpy()
            rec = alive & (t >= L)
            if not alive.any():
                break
            if rec.any():
                rows = np.nonzero(rec)[0]
                hs = cells_of(env.q[torch.as_tensor(rows, device=dev)])
                for row, h in zip(rows, hs):
                    h = int(h)
                    c = cellcount.get(h, 0)
                    # near the frontier keep EVERY surviving variant: at the
                    # pinch the corridor is thinner than a cell and the few
                    # capped representatives are usually doomed variants
                    if c < RMAX or (depth[best_i] > 60
                                    and t + 1 >= depth[best_i] - 5):
                        cellcount[h] = c + 1
                        seqs.append(A[row, :t + 1].astype(np.int16))
                        depth.append(t + 1)
                        visits.append(0)
                        if t + 1 > depth[best_i]:
                            best_i = len(seqs) - 1
        stall = 0 if len(seqs) > new_before else stall + 1
        if gen % 10 == 0 or stall >= a.ge_stall:
            print(f"gen {gen:>4}  archive {len(seqs):>7}  frontier "
                  f"{depth[best_i]:>4} (~{depth[best_i] * 0.01:.2f} m)  "
                  f"{time.time() - t0:.0f}s", flush=True)
            np.savez(OUT / 'goexplore_env.npz',
                     action_idx=seqs[best_i].astype(np.int64),
                     frontier_depth=depth[best_i], archive_size=len(seqs),
                     gen=gen, cell=CELL)
        if stall >= a.ge_stall:
            break

    env.reset()
    for ai in seqs[best_i]:
        env.step(verts[int(ai)][None].expand(B, -1), auto_reset=False)
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = float(((p[0] - env.p_start[0]) * env.line_dir[0]).sum())
    print(f"\ngoexplore-env: frontier {depth[best_i]} steps, replay "
          f"{prog:.4f} m (alive {not bool(env.done_persistent[0])}); "
          f"archive {len(seqs)}")
    np.savez(OUT / 'goexplore_env.npz',
             action_idx=seqs[best_i].astype(np.int64), progress=prog,
             frontier_depth=depth[best_i], archive_size=len(seqs), cell=CELL)
    print(f"wrote {OUT / 'goexplore_env.npz'}")


def stage_selfimitate(a, dev):
    """Self-imitation: distill the agent's own best discovered episode
    (goexplore_env.npz, pure interaction data) into a policy, then evaluate
    deterministically from q0. Also fits the critic to the episode's
    discounted returns so a later PPO fine-tune can resume stably."""
    y, env = _env_and_yaml(2, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    src = a.imitate_from or str(OUT / 'goexplore_env.npz')
    ge = np.load(REPO / src if not Path(src).is_absolute() else src)
    seq = ge['action_idx']
    print(f"[selfimitate] expert sequence from {src}: {len(seq)} steps")
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    obs_l, act_l, rew_l = [], [], []
    with torch.no_grad():
        for ai in seq:
            obs_l.append(env.current_obs()[0].clone())
            _, r, _, _, _ = env.step(verts[int(ai)][None].expand(2, -1),
                                     auto_reset=False)
            act_l.append(int(ai))
            rew_l.append(float(r[0]))
    obs = torch.stack(obs_l)
    acts = torch.tensor(act_l, device=dev)
    ret = np.zeros(len(rew_l), dtype=np.float64)
    acc = 0.0
    for i in range(len(rew_l) - 1, -1, -1):
        acc = rew_l[i] + 0.99 * acc
        ret[i] = acc
    ret_t = torch.tensor(ret, device=dev, dtype=torch.float32)
    print(f"[selfimitate] episode: {len(seq)} steps, replayed reward "
          f"{sum(rew_l):.1f}")
    np.savez(OUT / 'golden_dataset.npz', obs=obs.cpu().numpy(),
             act=np.array(act_l), ret=ret)

    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    for ep in range(a.bc_epochs):
        logits = agent._logits_head(agent._actor_trunk(obs))
        v = agent.critic(obs).squeeze(-1)
        loss = (torch.nn.functional.cross_entropy(logits, acts)
                + 0.5 * torch.nn.functional.mse_loss(v, ret_t))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 500 == 0:
            acc_frac = float((logits.argmax(-1) == acts).float().mean())
            print(f"  bc {ep:>5}  loss {float(loss):.4f}  "
                  f"argmax-match {acc_frac:.3f}", flush=True)
    torch.save(agent.state_dict(), OUT / 'agent_bc.pt')

    agent.eval()
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    with torch.no_grad():
        for t in range(env.max_steps):
            env.step(agent.actor_mean(env.current_obs()), auto_reset=False)
            if bool(env.done_persistent.all()):
                break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = float(((p[0] - env.p_start[0]) * env.line_dir[0]).sum())
    print(f"[selfimitate] deterministic policy from q0: {prog:.4f} m "
          f"-> {OUT / 'agent_bc.pt'}")


def stage_fqi(a, dev):
    """Method 3 evidence run: offline fitted Q-iteration with the Bellman
    OPTIMALITY backup (max over actions) on data collected from archive
    restarts with random behavior. Tests whether max-semantics value
    propagation alone (no policy-gradient, no imitation) can rank the
    filament correctly and act on it from q0."""
    import torch.nn as nn
    y, env = _env_and_yaml(128, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    bk = np.load(REPO / a.restart_bank)
    q_bank = torch.tensor(bk['q'], device=dev, dtype=env.kin.dtype)
    env.line_dist = RestartBankDistribution(spec1, q_bank, 0.7)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)

    N_STEPS = a.fqi_data // 128
    obs_b, act_b, rew_b, nxt_b, dn_b = [], [], [], [], []
    env.reset()
    with torch.no_grad():
        o = env.current_obs()
        for t in range(N_STEPS):
            a_idx = torch.randint(0, 16, (128,), device=dev)
            o2, r, term, trunc, info = env.step(verts[a_idx])
            done = (term | trunc)
            nxt = torch.where(done.unsqueeze(-1), info['terminal_obs'], o2)
            obs_b.append(o.clone()); act_b.append(a_idx.clone())
            rew_b.append(r.clone()); nxt_b.append(nxt.clone())
            dn_b.append(done.clone())
            o = o2
    OBS = torch.cat(obs_b); ACT = torch.cat(act_b); REW = torch.cat(rew_b)
    NXT = torch.cat(nxt_b); DN = torch.cat(dn_b).float()
    print(f"[fqi] dataset {OBS.shape[0]} transitions, "
          f"mean r {float(REW.mean()):.3f}, done frac {float(DN.mean()):.3f}")

    def mknet():
        return nn.Sequential(
            nn.Linear(env.obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 16)).to(dev)
    qnet, tgt = mknet(), mknet()
    tgt.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=3e-4)
    M = OBS.shape[0]
    for it in range(a.fqi_iters):
        idx = torch.randint(0, M, (4096,), device=dev)
        with torch.no_grad():
            target = REW[idx] + 0.99 * (1 - DN[idx]) * tgt(NXT[idx]).max(-1).values
        qsa = qnet(OBS[idx]).gather(1, ACT[idx][:, None]).squeeze(-1)
        loss = nn.functional.smooth_l1_loss(qsa, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            tgt.load_state_dict(qnet.state_dict())
            print(f"  fqi {it:>6}  loss {float(loss):.4f}", flush=True)

    _, env1 = _env_and_yaml(2, dev)
    env1.line_dist = SingleTaskDistribution(spec1)
    env1.reset()
    with torch.no_grad():
        for t in range(env1.max_steps):
            a_idx = qnet(env1.current_obs()).argmax(-1)
            env1.step(verts[a_idx], auto_reset=False)
            if bool(env1.done_persistent.all()):
                break
    p, _, _, _ = env1.kin.tcp_fk_jac(env1.q)
    prog = float(((p[0] - env1.p_start[0]) * env1.line_dir[0]).sum())
    torch.save(qnet.state_dict(), OUT / 'fqi_qnet.pt')
    print(f"[fqi] greedy policy from q0: {prog:.4f} m -> "
          f"{OUT / 'fqi_qnet.pt'}")


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
                             'ceiling', 'traj', 'reachtree', 'goexplore',
                             'goexplore_env', 'selfimitate', 'fqi'])
    ap.add_argument('--fqi-data', type=int, default=400_000)
    ap.add_argument('--fqi-iters', type=int, default=20_000)
    ap.add_argument('--ge-generations', type=int, default=400)
    ap.add_argument('--ge-batch', type=int, default=4096)
    ap.add_argument('--ge-k', type=int, default=15)
    ap.add_argument('--ge-cell', type=float, default=0.04)
    ap.add_argument('--ge-stall', type=int, default=30,
                    help='stop after this many generations w/o new cells')
    ap.add_argument('--ge-drift-j7', type=float, default=0.0,
                    help='probability per probe step of taking the vertex '
                         'that raises joint 7 fastest (action-level fan '
                         'extension toward the winning filament)')
    ap.add_argument('--ge-bias-j7', action='store_true',
                    help='bias frontier probes toward high joint-7 (one bit '
                         'of oracle knowledge about the viable edge)')
    ap.add_argument('--bc-epochs', type=int, default=5000)
    ap.add_argument('--imitate-from', default=None,
                    help='npz with action_idx to imitate (selfimitate)')
    ap.add_argument('--anchor-data', default=None,
                    help='golden_dataset.npz for the self-imitation anchor')
    ap.add_argument('--anchor-coef', type=float, default=1.0)
    ap.add_argument('--resume-from-ckpt', default=None,
                    help='load policy weights before PPO training')
    ap.add_argument('--norm-returns', type=int, default=None,
                    help='override PPO normalize_returns (0/1)')
    ap.add_argument('--ent-coef', type=float, default=None,
                    help='override the config entropy coefficient')
    ap.add_argument('--novelty-beta', type=float, default=None,
                    help='count-based intrinsic bonus weight (train only)')
    ap.add_argument('--novelty-cell', type=float, default=0.15,
                    help='joint-space cell size (rad) for novelty counts')
    ap.add_argument('--speed-levels', default=None,
                    help='e.g. "1.0,0.5": the policy also picks a tangential '
                         'speed each step; reward stays arc length')
    ap.add_argument('--restart-bank', default=None,
                    help='npz with a q array (e.g. reachtree.npz); a '
                         'fraction of training resets start from these '
                         'states instead of q0')
    ap.add_argument('--restart-frac', type=float, default=0.5)
    ap.add_argument('--restart-window', default=None,
                    help='"lo,hi": fixed depth window into the restart bank '
                         '(competence-gated curriculum driver sets this)')
    ap.add_argument('--restart-curriculum', type=int, default=0,
                    help='>0: reverse-order curriculum — bank resets start '
                         'deepest-only and the window slides to the full '
                         'bank over this many resets')
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
    ap.add_argument('--torch-seed', type=int, default=0,
                    help='policy init + action sampling seed (basin lottery)')
    a = ap.parse_args()
    global OUT
    OUT = REPO / 'Yuan/IJRR/runs' / a.run_dir
    dev = torch.device(a.device)
    torch.manual_seed(a.torch_seed)
    {'select': stage_select, 'select2': stage_select2, 'train': stage_train,
     'report': stage_report, 'ceiling': stage_ceiling,
     'traj': stage_traj, 'reachtree': stage_reachtree,
     'goexplore': stage_goexplore, 'goexplore_env': stage_goexplore_env,
     'selfimitate': stage_selfimitate, 'fqi': stage_fqi}[a.stage](a, dev)


if __name__ == '__main__':
    main()
