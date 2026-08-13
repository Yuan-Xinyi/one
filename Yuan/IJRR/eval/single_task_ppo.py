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


class RNDEnv(NSRLBatchedEnv):
    """Method-5 evidence run: optimism-in-the-face-of-uncertainty proxy via
    Random Network Distillation — intrinsic bonus = prediction error of a
    trainable net against a frozen random target, i.e. epistemic novelty in
    feature space rather than visit counts."""

    def __init__(self, *args, rnd_beta=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        import torch.nn as nn

        def mk():
            return nn.Sequential(nn.Linear(self.obs_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 64)).to(self.device)
        self._rnd_tgt = mk()
        for p in self._rnd_tgt.parameters():
            p.requires_grad_(False)
        self._rnd_prd = mk()
        self._rnd_opt = torch.optim.Adam(self._rnd_prd.parameters(), lr=1e-4)
        self._rnd_beta = float(rnd_beta)
        self._rnd_var = 1.0

    def step(self, actions, auto_reset=True):
        obs, rew, term, trunc, info = super().step(actions, auto_reset)
        with torch.enable_grad():
            err = ((self._rnd_prd(obs) - self._rnd_tgt(obs)) ** 2).mean(-1)
            self._rnd_opt.zero_grad()
            err.mean().backward()
            self._rnd_opt.step()
        e = err.detach()
        self._rnd_var = 0.99 * self._rnd_var + 0.01 * float(e.mean())
        bonus = self._rnd_beta * e / (self._rnd_var + 1e-8)
        return obs, rew + bonus.clamp(0, 2.0), term, trunc, info


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
    if a.rnd_beta:
        env = RNDEnv(EnvConfig(**{**y['env'], 'n_envs': a.n_envs,
                                  **(extra or {})}), None, dev,
                     rnd_beta=a.rnd_beta)
        print(f"[train] RND intrinsic bonus: beta {a.rnd_beta}")
    elif a.novelty_beta:
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
    opt_value = None
    if a.opt_value:
        vnet = _vstar_net(env.obs_dim, dev)
        vnet.load_state_dict(torch.load(REPO / a.opt_value,
                                        map_location=dev))
        vnet.eval()

        @torch.no_grad()
        def opt_value(o):
            return vnet(o).squeeze(-1)
        print(f"[train] V*-guided advantages from {a.opt_value}")
    ppo_train(ppo_cfg, env, device=dev, agent=agent,
              eval_fn=eval_fn, eval_every=a.eval_every, log_fn=log_fn,
              ckpt_path=str(OUT / 'agent.pt'), ckpt_every_n_updates=25,
              resume_from_ckpt=a.resume_from_ckpt, anchor=anchor,
              opt_value=opt_value)
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


def stage_vmask(a, dev):
    """Method 1: vanilla PPO (environment reward, GAE, no BC, no expert
    labels) but with DOOMED ACTIONS MASKED at rollout time: all 16 vertex
    successors are enumerated with the exact model, scored by
    Qhat = alive * (1 + gamma * Vhat_search(s')), and actions with
    Qhat < mask_alpha * max_a Qhat are removed from the categorical before
    sampling (and identically in the update). Tests whether policy-gradient
    credit assignment succeeds once exploration is restricted to the
    non-doomed set.
    """
    y, env = _env_and_yaml(a.n_envs, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    myopic_ref = float(task['myopic_progress'])
    model = hl.StraightModel(env)
    B = a.n_envs
    verts16 = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    env.reset()
    dvec = env.line_dir[0].clone(); nvec = env.n_target[0].clone()
    p0 = env.p_start[0].clone()

    vnet = _vstar_net(env.obs_dim, dev)
    vnet.load_state_dict(torch.load(REPO / a.mask_value, map_location=dev))
    vnet.eval()
    fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': B * 16}),
                          None, dev)
    d_all = dvec[None].expand(B * 16, 3).clone()
    n_all = nvec[None].expand(B * 16, 3).clone()

    @torch.no_grad()
    def mask_fn():
        qs = env.q
        qe = qs.unsqueeze(1).expand(-1, 16, -1).reshape(B * 16, -1)
        ae = verts16.unsqueeze(0).expand(B, -1, -1).reshape(B * 16, -1)
        qn = model.step(qe, d_all, n_all, ae)
        mg = model.margins(qn, p0.expand(B * 16, 3), d_all, n_all)
        alive = (mg.amin(-1) > 0).float().reshape(B, 16)
        fenv.line_dist = ScriptedLineDistribution(
            {'q0': qn, 'line_dir': d_all, 'n_target': n_all})
        fenv.reset()
        v_next = vnet(fenv.current_obs()).squeeze(-1).reshape(B, 16)
        qhat = alive * (1.0 + 0.99 * v_next.clamp_min(0.0))
        keep = qhat >= a.mask_alpha * qhat.amax(-1, keepdim=True)
        dead_row = alive.sum(-1) == 0
        keep[dead_row] = True                  # let the env terminate them
        return keep

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

    ppo_kw = {**y['ppo'], 'total_timesteps': a.total_steps}
    if a.ent_coef is not None:
        ppo_kw['ent_coef'] = a.ent_coef
    ppo_cfg = PPOConfig(**ppo_kw)
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=ppo_cfg.hidden_dim).to(dev)
    print(f"[vmask] doomed-action-masked PPO: alpha {a.mask_alpha}, "
          f"Vhat {a.mask_value}, {a.total_steps} steps on {B} envs",
          flush=True)
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
              ckpt_path=str(OUT / 'agent.pt'), ckpt_every_n_updates=25,
              mask_fn=mask_fn)
    print(f"[vmask] done -> {OUT / 'agent.pt'}")


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
def stage_reachtree_bundle(a, dev):
    """stage_reachtree rerun (same seed/params) that backtracks EVERY leaf
    trajectory whose final progress exceeds --bundle-min, not just the best.

    A leaf is a pool state with no surviving child in the next depth's pool
    (dedupe/thinning victims count as leaves: their prefix is still a genuine
    executable sequence with that progress). All kept trajectories are saved
    concatenated with per-trajectory lengths in reachtree_bundle.npz.
    """
    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    p0 = env.p_start[0]
    d = env.line_dir[0]
    n = env.n_target[0]

    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    K = verts.shape[0]
    W = a.tree_width
    grid = a.tree_dedupe

    q = env.q[:1].clone()
    parents, actions, pools = [], [], [q.cpu()]
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
                  f"{time.time() - t0:.0f}s", flush=True)

    # collect leaves per depth, keep those with progress > bundle_min
    trajs, progs, depths_kept = [], [], []
    n_leaves_total = 0
    for r in range(len(pools)):
        P = pools[r].shape[0]
        has_child = np.zeros(P, dtype=bool)
        if r < len(parents):
            has_child[np.unique(parents[r].numpy())] = True
        leaf = np.nonzero(~has_child)[0]
        if leaf.size == 0:
            continue
        n_leaves_total += leaf.size
        pf, _, _, _ = env.kin.tcp_fk_jac(pools[r][leaf].to(dev))
        pr = ((pf - p0) * d).sum(-1).cpu().numpy()
        sel = np.nonzero(pr > a.bundle_min)[0]
        if sel.size == 0:
            continue
        idx = leaf[sel]
        traj = np.empty((idx.size, r + 1, 7), dtype=np.float32)
        traj[:, r] = pools[r].numpy()[idx]
        cur = idx
        for rr in range(r - 1, -1, -1):
            cur = parents[rr].numpy()[cur]
            traj[:, rr] = pools[rr].numpy()[cur]
        trajs.extend(traj)
        progs.extend(pr[sel].tolist())
        depths_kept.extend([r] * idx.size)
    progs = np.asarray(progs)
    depths_kept = np.asarray(depths_kept, dtype=np.int64)
    print(f"\nsearch died at depth {depth}; {n_leaves_total} leaves total, "
          f"{len(trajs)} with progress > {a.bundle_min:.2f} m "
          f"(max {progs.max() if len(trajs) else 0:.4f})")

    if len(trajs) > a.bundle_max_save:      # stratified thin, keep argmax
        order = np.argsort(progs)
        pick = order[np.unique(np.linspace(0, len(order) - 1,
                                           a.bundle_max_save).astype(int))]
        pick = np.union1d(pick, [int(progs.argmax())])
        trajs = [trajs[i] for i in pick]
        depths_kept = depths_kept[pick]
        progs = progs[pick]
        print(f"thinned to {len(trajs)} for saving (stratified by progress)")

    flat = np.concatenate([t.reshape(-1, 7) for t in trajs])
    lens = np.array([t.shape[0] for t in trajs], dtype=np.int64)
    np.savez(OUT / 'reachtree_bundle.npz', q_flat=flat, lens=lens,
             progress=progs, depth=depths_kept,
             n_leaves_total=n_leaves_total, bundle_min=a.bundle_min)
    print(f"wrote {OUT / 'reachtree_bundle.npz'} ({len(trajs)} trajectories)")


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


def _vstar_net(obs_dim, dev):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(),
                         nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 1)).to(dev)


def stage_vstarfit(a, dev):
    """Fit Vhat*(obs) on search-probe labels (units: survivable steps)."""
    import torch.nn as nn
    y, env = _env_and_yaml(2, dev)
    dat = np.load(OUT / 'vstar_labels.npz')
    obs = torch.tensor(dat['obs'], device=dev, dtype=torch.float32)
    lab = torch.tensor(dat['label'], device=dev, dtype=torch.float32)
    net = _vstar_net(env.obs_dim, dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = obs.shape[0]
    hold = torch.arange(n, device=dev) % 5 == 0
    for ep in range(6000):
        idx = torch.randint(0, n, (512,), device=dev)
        m = ~hold[idx]
        loss = nn.functional.smooth_l1_loss(
            net(obs[idx][m]).squeeze(-1), lab[idx][m])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 1000 == 0:
            with torch.no_grad():
                pred = net(obs[hold]).squeeze(-1)
                ss = 1 - ((pred - lab[hold]) ** 2).mean() / lab[hold].var()
            print(f"  vfit {ep:>5}  loss {float(loss):.3f}  "
                  f"holdout R2 {float(ss):.3f}", flush=True)
    torch.save(net.state_dict(), OUT / 'vstar_net.pt')
    print(f"wrote {OUT / 'vstar_net.pt'}")


def stage_vguide(a, dev):
    """16-action enumeration + Vhat_search-guided policy improvement
    (user-specified design).

    States come from on-policy rollouts from q0 ONLY (no BC, no expert
    action labels, no resets). At every visited state the model enumerates
    all 16 vertex successors; Qhat(s,a_i) = alive_i * (1 + gamma *
    Vhat_search(s'_i)); the actor maximizes J = E_s[sum_a pi(a|s) Qhat] —
    exact expectation over the 16 actions, so no action-sampling problem.
    The question: does swapping Q^pi for approximate optimal-continuation
    information alone let the policy cross the wall from q0?
    """
    import torch.nn as nn
    y, env = _env_and_yaml(a.n_envs, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    model = hl.StraightModel(env)
    B, NS = a.n_envs, 32
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    d = env.line_dir  # set after reset; refreshed below
    # featurizer env: computes the same reset-style obs the labels used
    featN = B * NS * 16
    fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': featN}),
                          None, dev)
    _, eval_env = _env_and_yaml(2, dev)

    vnet = _vstar_net(env.obs_dim, dev)
    vnet.load_state_dict(torch.load(REPO / a.opt_value, map_location=dev))
    vnet.eval()
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    env.reset()
    dvec = env.line_dir[0].clone()
    nvec = env.n_target[0].clone()
    p0 = env.p_start[0].clone()
    log = open(OUT / 'vguide.log', 'w')
    for upd in range(a.vguide_updates):
        obs_l, q_l = [], []
        with torch.no_grad():
            for t in range(NS):
                o = env.current_obs()
                obs_l.append(o.clone())
                q_l.append(env.q.clone())
                logits = agent._logits_head(agent._actor_trunk(o))
                ai = torch.distributions.Categorical(logits=logits).sample()
                env.step(verts[ai])
        OBS = torch.cat(obs_l)                       # (B*NS, obs)
        QS = torch.cat(q_l)                          # (B*NS, 7)
        M = QS.shape[0]
        with torch.no_grad():
            qe = QS.unsqueeze(1).expand(-1, 16, -1).reshape(M * 16, -1)
            ae = verts.unsqueeze(0).expand(M, -1, -1).reshape(M * 16, -1)
            CH = 32768
            qn = torch.cat([model.step(qe[i:i+CH],
                                       dvec.expand(min(CH, M*16-i), 3),
                                       nvec.expand(min(CH, M*16-i), 3),
                                       ae[i:i+CH])
                            for i in range(0, M*16, CH)])
            mg = torch.cat([model.margins(qn[i:i+CH],
                                          p0.expand(min(CH, M*16-i), 3),
                                          dvec.expand(min(CH, M*16-i), 3),
                                          nvec.expand(min(CH, M*16-i), 3))
                            for i in range(0, M*16, CH)])
            alive = (mg.amin(-1) > 0).float()
            # bulk reset trick: featurize all successors exactly the way
            # the Vhat_search labels were generated (reset-style obs)
            fenv.line_dist = ScriptedLineDistribution(
                {'q0': qn, 'line_dir': dvec[None].expand(M*16, 3).clone(),
                 'n_target': nvec[None].expand(M*16, 3).clone()})
            fenv.reset()
            v_next = vnet(fenv.current_obs()).squeeze(-1)
            qhat = alive * (1.0 + 0.99 * v_next)
            qhat = qhat.reshape(M, 16)
            qhat = (qhat - qhat.mean(-1, keepdim=True)) / 40.0
        for ep in range(3):
            logits = agent._logits_head(agent._actor_trunk(OBS))
            logp = torch.log_softmax(logits, -1)
            probs = logp.exp()
            ent = -(probs * logp).sum(-1).mean()
            loss = -(probs * qhat).sum(-1).mean() - a.ent_coef2 * ent
            opt.zero_grad(); loss.backward(); opt.step()
        if upd % 20 == 0:
            eval_env.line_dist = SingleTaskDistribution(spec1)
            eval_env.reset()
            with torch.no_grad():
                for t in range(eval_env.max_steps):
                    eval_env.step(agent.actor_mean(eval_env.current_obs()),
                                  auto_reset=False)
                    if bool(eval_env.done_persistent.all()):
                        break
            p, _, _, _ = eval_env.kin.tcp_fk_jac(eval_env.q)
            prog = float(((p[0] - eval_env.p_start[0])
                          * eval_env.line_dir[0]).sum())
            msg = (f"vguide upd {upd:>4}  q0-eval {prog:.4f} m  "
                   f"ent {float(ent):.2f}")
            print(msg, flush=True)
            log.write(msg + '\n'); log.flush()
            torch.save(agent.state_dict(), OUT / 'agent_vguide.pt')
    log.close()


def stage_vdagger(a, dev):
    """Method 2: value-DAgger. Fix vguide's value quality by (a) grounding
    Vhat_search on tree-descendant-depth labels (no 40-step probe cap, no
    max-backup bubbles: every label is the depth of a certified-executable
    continuation inside the exhaustive search tree) and (b) DAgger rounds
    that probe-relabel exactly the states the greedy-wrt-Vhat policy visits.

    Output: vstar_net_vd.pt + vdagger.log. Success criterion: greedy-wrt-
    Vhat rollout from q0 crosses the 0.73 wall.
    """
    import torch.nn as nn
    y, env = _env_and_yaml(a.vd_tube, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    dvec = env.line_dir[0].clone(); nvec = env.n_target[0].clone()
    p0 = env.p_start[0].clone()
    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    log = open(OUT / 'vdagger.log', 'w')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()

    # ---- (a) search-tree rerun with descendant-depth labels ----
    q = env.q[:1].clone()
    parents, pools, any_alive = [], [q.cpu()], []
    rng = np.random.default_rng(0)
    t0 = time.time()
    depth = 0
    while q.shape[0] > 0 and depth < env.max_steps:
        P = q.shape[0]
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
        ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
        CH = 32768
        qn = torch.cat([model.step(qe[i:i + CH],
                                   dvec.expand(min(CH, P * 16 - i), 3),
                                   nvec.expand(min(CH, P * 16 - i), 3),
                                   ae[i:i + CH])
                        for i in range(0, P * 16, CH)])
        m = torch.cat([model.margins(qn[i:i + CH],
                                     p0.expand(min(CH, P * 16 - i), 3),
                                     dvec.expand(min(CH, P * 16 - i), 3),
                                     nvec.expand(min(CH, P * 16 - i), 3))
                       for i in range(0, P * 16, CH)])
        alive_b = (m.amin(dim=-1) > 0)
        any_alive.append(alive_b.reshape(P, 16).any(-1).cpu().numpy())
        alive = alive_b.nonzero(as_tuple=False).squeeze(-1)
        if alive.numel() == 0:
            break
        qn = qn[alive]
        par = (alive // 16)
        key = torch.round(qn / a.tree_dedupe).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        keep = torch.as_tensor(np.sort(first), device=dev)
        if keep.numel() > a.tree_width:
            keep = keep[torch.as_tensor(
                np.sort(rng.choice(keep.numel(), a.tree_width,
                                   replace=False)), device=dev)]
        q = qn[keep]
        parents.append(par[keep].cpu().numpy())
        pools.append(q.cpu())
        depth += 1
    say(f"[vdagger] tree rebuilt: depth {depth}  ({time.time() - t0:.0f}s)")

    # backward pass: remaining survivable steps within the tree (used only
    # to find the backbone for proposal weighting; labels come from probes —
    # rem is a tree artifact, not a consistent function of the state)
    rem = [np.zeros(p.shape[0], dtype=np.int64) for p in pools]
    for r in range(depth - 1, -1, -1):
        np.maximum.at(rem[r], parents[r], rem[r + 1] + 1)

    # state PROPOSALS: per-depth uniform sample + the deep backbone in full.
    # Each proposal carries its tree label rem (exact along the backbone,
    # biased low elsewhere); the final label is max(tree, probe) — both are
    # lower bounds of V*, and only the tree sees the filament from afar.
    S_l, treelab_l = [], []
    per_depth = max(50, a.vd_states // max(depth, 1))
    rng2 = np.random.default_rng(1)
    for r in range(depth + 1):
        P = pools[r].shape[0]
        bb = np.nonzero(rem[r] >= max(depth - r - 10, 5))[0]   # backbone
        k = min(P, per_depth)
        idx = np.union1d(bb, rng2.choice(P, k, replace=False))
        S_l.append(pools[r][idx])
        treelab_l.append(rem[r][idx])
    S = torch.cat(S_l).to(dev)
    TREELAB = torch.tensor(np.concatenate(treelab_l), device=dev,
                           dtype=torch.float32)
    say(f"[vdagger] state proposals: {S.shape[0]}  "
        f"(backbone labels max {float(TREELAB.max()):.0f})")

    # ---- featurizer (reset-style obs, same as the vguide runtime) ----
    def featurize(qbatch):
        outs = []
        FB = 65536
        for i in range(0, qbatch.shape[0], FB):
            chunk = qbatch[i:i + FB]
            fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                                  'n_envs': chunk.shape[0]}), None, dev)
            fenv.line_dist = ScriptedLineDistribution(
                {'q0': chunk,
                 'line_dir': dvec[None].expand(chunk.shape[0], 3).clone(),
                 'n_target': nvec[None].expand(chunk.shape[0], 3).clone()})
            fenv.reset()
            outs.append(fenv.current_obs().clone())
            del fenv
        return torch.cat(outs)

    # ---- batched probe labeler: local re-search from K states at once ----
    @torch.no_grad()
    def probe_batch(states, H, W=256, grid=0.02, K=64):
        out = torch.zeros(states.shape[0], dtype=torch.float32, device=dev)
        for base in range(0, states.shape[0], K):
            chunk = states[base:base + K]
            kk = chunk.shape[0]
            m0 = model.margins(chunk, p0.expand(kk, 3),
                               dvec.expand(kk, 3), nvec.expand(kk, 3))
            alive0 = (m0.amin(-1) > 0)
            qq = chunk[alive0]
            sid = alive0.nonzero(as_tuple=False).squeeze(-1)
            for dpt in range(H):
                if qq.shape[0] == 0:
                    break
                N = qq.shape[0]
                qe = qq.unsqueeze(1).expand(-1, 16, -1).reshape(N * 16, -1)
                ae = verts.unsqueeze(0).expand(N, -1, -1).reshape(N * 16, -1)
                se = sid.repeat_interleave(16)
                CH = 32768
                qn = torch.cat([model.step(qe[i:i + CH],
                                           dvec.expand(min(CH, N*16-i), 3),
                                           nvec.expand(min(CH, N*16-i), 3),
                                           ae[i:i + CH])
                                for i in range(0, N * 16, CH)])
                m = torch.cat([model.margins(qn[i:i + CH],
                                             p0.expand(min(CH, N*16-i), 3),
                                             dvec.expand(min(CH, N*16-i), 3),
                                             nvec.expand(min(CH, N*16-i), 3))
                               for i in range(0, N * 16, CH)])
                ok = (m.amin(-1) > 0)
                # states whose sid vanishes here survived exactly dpt steps
                pre = torch.unique(sid)
                post = torch.unique(se[ok])
                gone = pre[~torch.isin(pre, post)]
                out[base + gone] = dpt
                qn, se = qn[ok], se[ok]
                # per-sid dedupe + width cap
                key = np.concatenate(
                    [se.cpu().numpy()[:, None],
                     torch.round(qn / grid).to(torch.int32).cpu().numpy()],
                    axis=1)
                _, first = np.unique(key, axis=0, return_index=True)
                keep = torch.as_tensor(np.sort(first), device=dev)
                qn, se = qn[keep], se[keep]
                r = torch.rand(qn.shape[0], device=dev)
                order = torch.argsort(se.float() * 2.0 + r)   # sid-major
                se_s = se[order]
                idx_arange = torch.arange(se_s.shape[0], device=dev)
                bnd = torch.nn.functional.pad(
                    (se_s[1:] != se_s[:-1]).long(), (1, 0), value=1)
                starts = torch.cummax(bnd * idx_arange, 0).values
                sel = order[(idx_arange - starts) < W]
                qq, sid = qn[sel], se[sel]
            if qq.shape[0] > 0:
                out[base + torch.unique(sid)] = H
        return out

    probe_h = a.vd_probe_h if a.vd_probe_h > 0 else min(
        env.max_steps, 200)
    OBS = featurize(S)
    say(f"[vdagger] features ready; probing proposal labels (H={probe_h})")
    t1 = time.time()
    LAB = torch.maximum(probe_batch(S, H=probe_h), TREELAB)
    say(f"[vdagger] labels (max of probe, tree): mean {float(LAB.mean()):.1f} "
        f"max {float(LAB.max()):.0f} zeros "
        f"{float((LAB == 0).float().mean()):.2f}  "
        f"({time.time() - t1:.0f}s)")

    vnet = _vstar_net(env.obs_dim, dev)
    opt = torch.optim.Adam(vnet.parameters(), lr=1e-3)

    def refit(steps):
        n = OBS.shape[0]
        hold = torch.arange(n, device=dev) % 10 == 0
        for ep in range(steps):
            idx = torch.randint(0, n, (1024,), device=dev)
            msk = ~hold[idx]
            loss = nn.functional.smooth_l1_loss(
                vnet(OBS[idx][msk]).squeeze(-1), LAB[idx][msk])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = vnet(OBS[hold]).squeeze(-1)
            r2 = 1 - ((pred - LAB[hold]) ** 2).mean() / LAB[hold].var()
        return float(r2)

    # ---- greedy-wrt-Vhat rollout (B tube envs, env 0 pure greedy) ----
    @torch.no_grad()
    def greedy_tube(eps):
        env.line_dist = SingleTaskDistribution(spec1)
        env.reset()
        B = env.n_envs
        visited = []
        for t in range(env.max_steps):
            qs = env.q.clone()
            live = ~env.done_persistent
            if not bool(live.any()):
                break
            visited.append(qs[live])
            qe = qs.unsqueeze(1).expand(-1, 16, -1).reshape(B * 16, -1)
            ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(B * 16, -1)
            qn = model.step(qe, dvec.expand(B * 16, 3),
                            nvec.expand(B * 16, 3), ae)
            mg = model.margins(qn, p0.expand(B * 16, 3),
                               dvec.expand(B * 16, 3),
                               nvec.expand(B * 16, 3))
            alive = (mg.amin(-1) > 0).float().reshape(B, 16)
            v_next = vnet(featurize(qn)).squeeze(-1).reshape(B, 16)
            qhat = alive * (1.0 + 0.99 * v_next.clamp_min(0.0))
            ai = qhat.argmax(-1)
            if eps > 0:                    # tube: eps-greedy among alive
                rnd = torch.randint(0, 16, (B,), device=dev)
                flip = ((torch.rand(B, device=dev) < eps)
                        & (alive.gather(1, rnd[:, None]).squeeze(-1) > 0))
                flip[0] = False            # env 0 stays pure greedy
                ai = torch.where(flip, rnd, ai)
            env.step(verts[ai], auto_reset=False)
        pf, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((pf - env.p_start) * env.line_dir).sum(-1)
        return (float(prog[0]), float(prog.max()),
                torch.cat(visited) if visited else env.q[:0])

    r2 = refit(a.vd_fit_steps)
    prog0, progmax0, _ = greedy_tube(0.0)
    say(f"[vdagger] round 0 (proposal probes only): holdout R2 {r2:.3f}  "
        f"greedy-from-q0 {prog0:.4f} m")

    best = prog0
    for rnd_i in range(1, a.vd_rounds + 1):
        # collect the tube the greedy policy actually visits
        _, _, vis = greedy_tube(a.vd_eps)
        key = torch.round(vis / 0.02).to(torch.int32)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        vis = vis[torch.as_tensor(np.sort(first), device=dev)]
        if vis.shape[0] > a.vd_probe_budget:
            vis = vis[torch.randperm(vis.shape[0],
                                     device=dev)[:a.vd_probe_budget]]
        t1 = time.time()
        labs = probe_batch(vis, H=probe_h)
        say(f"[vdagger] round {rnd_i}: probed {vis.shape[0]} visited states "
            f"(mean {float(labs.mean()):.1f}, {time.time() - t1:.0f}s)")
        OBS = torch.cat([OBS, featurize(vis)])
        LAB = torch.cat([LAB, labs])
        r2 = refit(a.vd_fit_steps // 2)
        prog, progmax, _ = greedy_tube(0.0)
        say(f"[vdagger] round {rnd_i}: R2 {r2:.3f}  greedy-from-q0 "
            f"{prog:.4f} m  (tube max {progmax:.4f})")
        torch.save(vnet.state_dict(), OUT / 'vstar_net_vd.pt')
        if prog > best:
            best = prog
    torch.save(vnet.state_dict(), OUT / 'vstar_net_vd.pt')
    np.savez(OUT / 'vdagger_labels.npz', obs=OBS.cpu().numpy(),
             label=LAB.cpu().numpy())
    say(f"[vdagger] done: best greedy-from-q0 {best:.4f} m "
        f"-> {OUT / 'vstar_net_vd.pt'}")
    log.close()


def stage_exit(a, dev):
    """Method 3: iterated expert iteration with a SHRINKING search budget.

    Round r: tree search from q0 where thinning keeps the top-W_r candidates
    by cumulative policy log-prob (round 0 starts from an untrained policy =
    noisy/random thinning); the best found trajectory is replayed in the real
    env, appended to the demo set, and the policy is BC-trained on the
    aggregate. If the policy amortizes the search, later rounds keep finding
    the long route at widths where an unguided search cannot.
    """
    import torch.nn as nn
    y, env = _env_and_yaml(1, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    dvec = env.line_dir[0].clone(); nvec = env.n_target[0].clone()
    p0 = env.p_start[0].clone()
    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    myopic_ref = float(task['myopic_progress'])
    log = open(OUT / 'exit.log', 'w')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()

    def featurize(qbatch, aprev):
        outs = []
        FB = 65536
        for i in range(0, qbatch.shape[0], FB):
            chunk = qbatch[i:i + FB]
            fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                                  'n_envs': chunk.shape[0]}), None, dev)
            fenv.line_dist = ScriptedLineDistribution(
                {'q0': chunk,
                 'line_dir': dvec[None].expand(chunk.shape[0], 3).clone(),
                 'n_target': nvec[None].expand(chunk.shape[0], 3).clone()})
            fenv.reset()
            o = fenv.current_obs().clone()
            del fenv
            outs.append(o)
        obs = torch.cat(outs)
        obs[:, -4:] = aprev            # exact obs: only a_prev differs
        return obs

    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)

    @torch.no_grad()
    def prior_search(W, gumbel):
        """Tree search from q0, thinning by cumulative policy log-prob."""
        qq = torch.tensor(task['q0'], device=dev,
                          dtype=env.kin.dtype)[None]
        score = torch.zeros(1, device=dev)
        aprev = torch.zeros(1, env.act_dim, device=dev)
        parents, actions, pools = [], [], [qq.cpu()]
        depth = 0
        while qq.shape[0] > 0 and depth < env.max_steps:
            P = qq.shape[0]
            obs = featurize(qq, aprev)
            logp = torch.log_softmax(
                agent._logits_head(agent._actor_trunk(obs.float())), -1)
            qe = qq.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
            ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
            CH = 32768
            qn = torch.cat([model.step(qe[i:i + CH],
                                       dvec.expand(min(CH, P*16-i), 3),
                                       nvec.expand(min(CH, P*16-i), 3),
                                       ae[i:i + CH])
                            for i in range(0, P * 16, CH)])
            m = torch.cat([model.margins(qn[i:i + CH],
                                         p0.expand(min(CH, P*16-i), 3),
                                         dvec.expand(min(CH, P*16-i), 3),
                                         nvec.expand(min(CH, P*16-i), 3))
                           for i in range(0, P * 16, CH)])
            alive = (m.amin(-1) > 0).nonzero(as_tuple=False).squeeze(-1)
            if alive.numel() == 0:
                break
            child_score = (score[:, None] + logp.to(score.dtype)
                           ).reshape(-1)[alive]
            qn = qn[alive]
            par = alive // 16
            act = alive % 16
            key = torch.round(qn / a.tree_dedupe).to(torch.int32)
            _, first = np.unique(key.cpu().numpy(), axis=0,
                                 return_index=True)
            keep = torch.as_tensor(np.sort(first), device=dev)
            if keep.numel() > W:
                if gumbel is None:          # round 0: unguided (random thin)
                    keep = keep[torch.randperm(keep.numel(),
                                               device=dev)[:W]]
                else:
                    noisy = child_score[keep] + gumbel * (
                        -torch.log(-torch.log(
                            torch.rand(keep.numel(), device=dev,
                                       dtype=score.dtype))))
                    keep = keep[noisy.topk(W).indices]
            qq = qn[keep]
            score = child_score[keep]
            aprev = verts[act[keep]].to(env.kin.dtype)
            parents.append(par[keep].cpu())
            actions.append(act[keep].cpu())
            pools.append(qq.cpu())
            depth += 1
        qf = pools[depth].to(dev)
        pf, _, _, _ = env.kin.tcp_fk_jac(qf)
        prog = ((pf - p0) * dvec).sum(-1)
        best = int(prog.argmax())
        traj_a = []
        i = best
        for r in range(depth - 1, -1, -1):
            traj_a.append(int(actions[r][i]))
            i = int(parents[r][i])
        return float(prog.max()), np.array(traj_a[::-1], dtype=np.int64)

    @torch.no_grad()
    def replay(traj_a):
        env.line_dist = SingleTaskDistribution(spec1)
        env.reset()
        obs_l, act_l, rew_l = [], [], []
        for t in range(len(traj_a)):
            obs_l.append(env.current_obs()[0].clone())
            act_l.append(int(traj_a[t]))
            _, r, _, _, info = env.step(verts[traj_a[t]][None],
                                        auto_reset=False)
            rew_l.append(float(r[0]))
            if bool(info['episode_done'][0]):
                break
        ret = np.zeros(len(rew_l), dtype=np.float32)
        acc = 0.0
        for t in range(len(rew_l) - 1, -1, -1):
            acc = rew_l[t] + 0.99 * acc
            ret[t] = acc
        n = len(ret)
        return (torch.stack(obs_l[:n]), act_l[:n], ret)

    @torch.no_grad()
    def eval_policy():
        env.line_dist = SingleTaskDistribution(spec1)
        env.reset()
        for t in range(env.max_steps):
            env.step(agent.actor_mean(env.current_obs()), auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        pf, _, _, _ = env.kin.tcp_fk_jac(env.q)
        return float(((pf[0] - env.p_start[0]) * env.line_dir[0]).sum())

    widths = [int(x) for x in a.exit_widths.split(',')]
    DOBS, DACT, DRET = [], [], []
    for rnd_i, W in enumerate(widths):
        t0 = time.time()
        sp, traj_a = prior_search(W, None if rnd_i == 0 else a.exit_gumbel)
        obs_r, act_r, ret_r = replay(traj_a)
        DOBS.append(obs_r); DACT.extend(act_r)
        DRET.append(torch.tensor(ret_r, device=dev))
        OBS = torch.cat(DOBS).float()
        ACT = torch.tensor(DACT, device=dev, dtype=torch.long)
        RET = torch.cat(DRET).float()
        opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
        for ep in range(a.exit_bc_epochs):
            logits = agent._logits_head(agent._actor_trunk(OBS))
            v = agent.critic(OBS).squeeze(-1)
            loss = (nn.functional.cross_entropy(logits, ACT)
                    + 0.5 * nn.functional.mse_loss(v, RET))
            opt.zero_grad(); loss.backward(); opt.step()
        pp = eval_policy()
        say(f"[exit] round {rnd_i}  W {W:>6}  search {sp:.4f} m  "
            f"policy {pp:.4f} m  demos {OBS.shape[0]}  "
            f"({time.time() - t0:.0f}s)")
        torch.save(agent.state_dict(), OUT / 'agent_exit.pt')

    # amortization ladder: how little search does the final policy need?
    for W in (256, 64, 16, 4, 1):
        sp, _ = prior_search(W, a.exit_gumbel)
        say(f"[exit] ladder: prior-guided search W {W:>4} -> {sp:.4f} m")
    say(f"[exit] done; myopic ref {myopic_ref:.4f}")
    log.close()


def stage_exit_multi(a, dev):
    """Generalization test for iterative search-and-distill: a stratified
    batch of tasks (by headroom gap L_hi - myopic), each run through
    (a) an UNGUIDED width sweep, (b) the policy-guided iteration of
    stage_exit, (c) the amortization ladder with the final policy.
    The question: does the policy prior let narrower search reach or beat
    wider unguided search beyond task 27?
    """
    import torch.nn as nn
    y, env = _env_and_yaml(1, dev)
    task, spec_sel = _load_task(dev, env.kin.dtype)
    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    log = open(OUT / 'exit_multi.log', 'w')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()

    # rebuild the standard pool (same recipe as select/select2)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=int(task['pool_seed']),
        env_cfg=env.cfg, feasibility_threshold_m=0.1, verbose=False)
    pidx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:1024]
    myo_all = task['all_myopic']
    lhi_all = task['all_L_hi']
    gap = lhi_all - myo_all
    order = np.argsort(-gap)
    ranks = [int(x) for x in a.emx_ranks.split(',')]
    tasks = [int(order[r]) for r in ranks]
    say(f"[exit_multi] tasks {tasks} (gap ranks {ranks})")

    widths = [int(x) for x in a.exit_widths.split(',')]
    ladder_ws = [256, 64, 16, 4, 1]
    results = {}

    for ti in tasks:
        q0_t = pool.q_pool[pidx[ti]].to(dev)
        dvec = pool.line_dir_pool[pidx[ti]].to(dev)
        nvec = pool.n_target_pool[pidx[ti]].to(dev)
        p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]
        spec1 = {'q0': q0_t[None], 'line_dir': dvec[None],
                 'n_target': nvec[None]}

        def featurize(qbatch, aprev):
            outs = []
            FB = 65536
            for i in range(0, qbatch.shape[0], FB):
                chunk = qbatch[i:i + FB]
                fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                                      'n_envs': chunk.shape[0]}), None, dev)
                fenv.line_dist = ScriptedLineDistribution(
                    {'q0': chunk,
                     'line_dir': dvec[None].expand(chunk.shape[0],
                                                   3).clone(),
                     'n_target': nvec[None].expand(chunk.shape[0],
                                                   3).clone()})
                fenv.reset()
                o = fenv.current_obs().clone()
                del fenv
                outs.append(o)
            obs = torch.cat(outs)
            obs[:, -4:] = aprev
            return obs

        agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                            hidden_dim=y['ppo']['hidden_dim']).to(dev)

        @torch.no_grad()
        def prior_search(W, gumbel):
            qq = q0_t[None].clone()
            score = torch.zeros(1, device=dev)
            aprev = torch.zeros(1, env.act_dim, device=dev)
            parents, actions, pools_l = [], [], [qq.cpu()]
            depth = 0
            while qq.shape[0] > 0 and depth < env.max_steps:
                P = qq.shape[0]
                if gumbel is not None:
                    obs = featurize(qq, aprev)
                    logp = torch.log_softmax(
                        agent._logits_head(
                            agent._actor_trunk(obs.float())), -1)
                else:
                    logp = torch.zeros(P, 16, device=dev)
                qe = qq.unsqueeze(1).expand(-1, 16, -1).reshape(P * 16, -1)
                ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
                CH = 32768
                qn = torch.cat([model.step(qe[i:i + CH],
                                           dvec.expand(min(CH, P*16-i), 3),
                                           nvec.expand(min(CH, P*16-i), 3),
                                           ae[i:i + CH])
                                for i in range(0, P * 16, CH)])
                m = torch.cat([model.margins(qn[i:i + CH],
                                             p0.expand(min(CH, P*16-i), 3),
                                             dvec.expand(min(CH, P*16-i), 3),
                                             nvec.expand(min(CH, P*16-i),
                                                         3))
                               for i in range(0, P * 16, CH)])
                alive = (m.amin(-1) > 0).nonzero(
                    as_tuple=False).squeeze(-1)
                if alive.numel() == 0:
                    break
                child_score = (score[:, None] + logp.to(score.dtype)
                               ).reshape(-1)[alive]
                qn = qn[alive]
                par = alive // 16
                act = alive % 16
                key = torch.round(qn / a.tree_dedupe).to(torch.int32)
                _, first = np.unique(key.cpu().numpy(), axis=0,
                                     return_index=True)
                keep = torch.as_tensor(np.sort(first), device=dev)
                if keep.numel() > W:
                    if gumbel is None:
                        keep = keep[torch.randperm(keep.numel(),
                                                   device=dev)[:W]]
                    else:
                        noisy = child_score[keep] + gumbel * (
                            -torch.log(-torch.log(
                                torch.rand(keep.numel(), device=dev,
                                           dtype=score.dtype))))
                        keep = keep[noisy.topk(W).indices]
                qq = qn[keep]
                score = child_score[keep]
                aprev = verts[act[keep]].to(env.kin.dtype)
                parents.append(par[keep].cpu())
                actions.append(act[keep].cpu())
                pools_l.append(qq.cpu())
                depth += 1
            qf = pools_l[depth].to(dev)
            pf, _, _, _ = env.kin.tcp_fk_jac(qf)
            prog = ((pf - p0) * dvec).sum(-1)
            best = int(prog.argmax())
            traj_a = []
            i = best
            for r in range(depth - 1, -1, -1):
                traj_a.append(int(actions[r][i]))
                i = int(parents[r][i])
            return float(prog.max()), np.array(traj_a[::-1],
                                               dtype=np.int64)

        @torch.no_grad()
        def replay(traj_a):
            env.line_dist = SingleTaskDistribution(spec1)
            env.reset()
            obs_l, act_l, rew_l = [], [], []
            for t in range(len(traj_a)):
                obs_l.append(env.current_obs()[0].clone())
                act_l.append(int(traj_a[t]))
                _, r, _, _, info = env.step(verts[traj_a[t]][None],
                                            auto_reset=False)
                rew_l.append(float(r[0]))
                if bool(info['episode_done'][0]):
                    break
            ret = np.zeros(len(rew_l), dtype=np.float32)
            acc = 0.0
            for t in range(len(rew_l) - 1, -1, -1):
                acc = rew_l[t] + 0.99 * acc
                ret[t] = acc
            n = len(ret)
            return (torch.stack(obs_l[:n]), act_l[:n], ret)

        @torch.no_grad()
        def eval_policy():
            env.line_dist = SingleTaskDistribution(spec1)
            env.reset()
            for t in range(env.max_steps):
                env.step(agent.actor_mean(env.current_obs()),
                         auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
            pf, _, _, _ = env.kin.tcp_fk_jac(env.q)
            return float(((pf[0] - env.p_start[0])
                          * env.line_dir[0]).sum())

        t0 = time.time()
        unguided = {}
        for W in widths:
            sp, _ = prior_search(W, None)
            unguided[W] = sp
        say(f"[exit_multi] task {ti}: unguided " +
            " ".join(f"W{W}={unguided[W]:.3f}" for W in widths))

        rounds = []
        DOBS, DACT, DRET = [], [], []
        for rnd_i, W in enumerate(widths):
            sp, traj_a = prior_search(
                W, None if rnd_i == 0 else a.exit_gumbel)
            obs_r, act_r, ret_r = replay(traj_a)
            DOBS.append(obs_r); DACT.extend(act_r)
            DRET.append(torch.tensor(ret_r, device=dev))
            OBS = torch.cat(DOBS).float()
            ACT = torch.tensor(DACT, device=dev, dtype=torch.long)
            RET = torch.cat(DRET).float()
            opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
            for ep in range(a.exit_bc_epochs):
                logits = agent._logits_head(agent._actor_trunk(OBS))
                v = agent.critic(OBS).squeeze(-1)
                loss = (nn.functional.cross_entropy(logits, ACT)
                        + 0.5 * nn.functional.mse_loss(v, RET))
                opt.zero_grad(); loss.backward(); opt.step()
            pp = eval_policy()
            rounds.append((W, sp, pp))
            say(f"[exit_multi] task {ti}: round {rnd_i} W {W:>6} "
                f"search {sp:.4f} policy {pp:.4f}")
        ladder = {}
        for W in ladder_ws:
            sp, _ = prior_search(W, a.exit_gumbel)
            ladder[W] = sp
        say(f"[exit_multi] task {ti}: ladder " +
            " ".join(f"W{W}={ladder[W]:.3f}" for W in ladder_ws) +
            f"  myopic {myo_all[ti]:.3f} L_hi {lhi_all[ti]:.3f} "
            f"({time.time() - t0:.0f}s)")
        results[ti] = {'unguided': unguided, 'rounds': rounds,
                       'ladder': ladder, 'myopic': float(myo_all[ti]),
                       'L_hi': float(lhi_all[ti])}
        np.savez(OUT / 'exit_multi.npz',
                 results=np.array([results], dtype=object),
                 widths=np.array(widths), ladder_ws=np.array(ladder_ws),
                 tasks=np.array(tasks))
    say("[exit_multi] done")
    log.close()


def stage_pool(a, dev):
    """Cross-task iterative search-and-distill under a wall-clock budget.

    ONE shared policy over a pool of tasks. Each round: take a fresh chunk of
    training tasks, run a POOLED tree search (all chunk tasks expanded in a
    single batch, per-task dedupe and width cap), replay the best sequence of
    each task in the real env, keep the best demo per task, and BC-train the
    shared policy on the aggregate. From round 1 on, thinning is guided by
    the shared policy — so search on a task the policy has never seen tests
    whether the prior transfers ACROSS tasks.

    Reported every --pool-eval-every rounds, on tasks never trained on and
    with NO search at deployment: policy progress relative to the classical
    law and to the one-step margin law, both measured in this protocol.
    """
    import torch.nn as nn
    y, env1 = _env_and_yaml(1, dev)
    model = hl.StraightModel(env1)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env1.act_dim, indexing='ij'),
                 -1).reshape(-1, env1.act_dim), dtype=torch.float32,
        device=dev)
    dtype = env1.kin.dtype
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / 'pool.log', 'a')

    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()

    # ---------------- task pool ----------------
    pool = LineDistribution.load_or_build(
        kin=env1.kin, collision=env1.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env1.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
    n_need = a.pool_train + a.pool_eval + a.pool_probe
    perm = torch.randperm(valid.numel(),
                          generator=torch.Generator().manual_seed(7))
    sel = valid[perm[:n_need]]
    Q0 = pool.q_pool[sel].to(dev)
    DIR = pool.line_dir_pool[sel].to(dev)
    NTG = pool.n_target_pool[sel].to(dev)
    P0 = env1.kin.tcp_fk_jac(Q0)[0]
    tr_ids = np.arange(a.pool_train)
    ev_ids = np.arange(a.pool_train, a.pool_train + a.pool_eval)
    pb_ids = np.arange(a.pool_train + a.pool_eval, n_need)
    say(f"[pool] {a.pool_train} train / {a.pool_eval} eval / "
        f"{a.pool_probe} probe tasks (pool seed {a.seed})")

    def spec_of(ids):
        ii = torch.as_tensor(ids, device=dev)
        return {'q0': Q0[ii].clone(), 'line_dir': DIR[ii].clone(),
                'n_target': NTG[ii].clone()}

    # ---------------- reference arms on every selected task -------------
    ref_path = OUT / 'pool_refs.npz'
    if ref_path.exists():
        rf = np.load(ref_path)
        REF_CL, REF_MY = rf['classical'], rf['myopic']
        say("[pool] loaded cached reference arms")
    else:
        _, env_ref = _env_and_yaml(n_need, dev)
        refs = {}
        for name, fn in _arms(env_ref).items():
            if name == 'zero':
                continue
            env_ref.line_dist = ScriptedLineDistribution(
                {k: v.clone() for k, v in spec_of(np.arange(n_need)).items()})
            st = rollout_first_episode(env_ref, fn)
            refs[name] = st['episode_progress'].cpu().numpy()
        REF_CL, REF_MY = refs['classical'], refs['myopic']
        np.savez(ref_path, classical=REF_CL, myopic=REF_MY)
        del env_ref
        say(f"[pool] reference arms: classical mean {REF_CL.mean():.3f} m, "
            f"myopic mean {REF_MY.mean():.3f} m, "
            f"myopic/classical {(REF_MY / np.maximum(REF_CL, 1e-6)).mean():.3f}")

    # ---------------- shared policy ----------------
    agent = VertexAgent(obs_dim=env1.obs_dim, act_dim=env1.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    if a.pool_resume:
        agent.load_state_dict(torch.load(REPO / a.pool_resume,
                                         map_location=dev))
        say(f"[pool] resumed policy from {a.pool_resume}")
    opt = torch.optim.Adam(agent.parameters(), lr=a.pool_lr)

    # ---------------- reusable featurizer env ----------------
    CAP = a.pool_chunk * max(int(x) for x in a.pool_widths.split(','))
    fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': CAP}), None, dev)

    @torch.no_grad()
    def featurize(q, d, n, aprev):
        outs = []
        for i in range(0, q.shape[0], CAP):
            qs, ds = q[i:i + CAP], d[i:i + CAP]
            ns = n[i:i + CAP]
            m = qs.shape[0]
            pad = CAP - m
            if pad > 0:
                qs = torch.cat([qs, qs[:1].expand(pad, 7)])
                ds = torch.cat([ds, ds[:1].expand(pad, 3)])
                ns = torch.cat([ns, ns[:1].expand(pad, 3)])
            fenv.line_dist = ScriptedLineDistribution(
                {'q0': qs, 'line_dir': ds, 'n_target': ns})
            fenv.reset()
            outs.append(fenv.current_obs()[:m].clone())
        o = torch.cat(outs)
        o[:, -4:] = aprev
        return o

    # ---------------- pooled tree search ----------------
    def cap_per_task(te, rank, W):
        """Indices keeping the top-W entries of each task by `rank`."""
        o1 = torch.argsort(rank, descending=True)
        order = o1[torch.argsort(te[o1], stable=True)]
        te_s = te[order]
        ar = torch.arange(te_s.shape[0], device=dev)
        bnd = torch.nn.functional.pad(
            (te_s[1:] != te_s[:-1]).long(), (1, 0), value=1)
        starts_ = torch.cummax(bnd * ar, 0).values
        return order[(ar - starts_) < W]

    @torch.no_grad()
    def pooled_search(ids, W, guided, starts=None, score_mode='logp'):
        """Search all `ids` entries in one batch (ids may repeat a task when
        searching from several of its states). Returns a list of
        (progress, action_seq) in the order of `ids`."""
        ii = torch.as_tensor(ids, device=dev)
        q = Q0[ii].clone() if starts is None else starts.clone()
        tid = torch.arange(len(ids), device=dev)
        score = torch.zeros(len(ids), device=dev, dtype=dtype)
        aprev = torch.zeros(len(ids), env1.act_dim, device=dev, dtype=dtype)
        d_task, n_task, p0_task = DIR[ii], NTG[ii], P0[ii]
        pools_q, pools_t, parents, actions = [q.cpu()], [tid.cpu()], [], []
        best = {int(t): (-1.0, 0, 0) for t in range(len(ids))}
        prog0 = ((env1.kin.tcp_fk_jac(q)[0] - p0_task)
                 * d_task).sum(-1)
        for t in range(len(ids)):
            best[t] = (float(prog0[t]), 0, t)
        depth = 0
        while q.shape[0] > 0 and depth < env1.max_steps:
            P = q.shape[0]
            if guided:
                obs = featurize(q, d_task[tid], n_task[tid], aprev)
                logp = torch.log_softmax(
                    agent._logits_head(agent._actor_trunk(obs.float())),
                    -1).to(dtype)
            else:
                logp = torch.zeros(P, 16, device=dev, dtype=dtype)
            qe = q.repeat_interleave(16, 0)
            te = tid.repeat_interleave(16)
            ae = verts.unsqueeze(0).expand(P, -1, -1).reshape(P * 16, -1)
            CH = 32768
            qn = torch.cat([model.step(qe[i:i + CH], d_task[te[i:i + CH]],
                                       n_task[te[i:i + CH]], ae[i:i + CH])
                            for i in range(0, P * 16, CH)])
            mg = torch.cat([model.margins(qn[i:i + CH], p0_task[te[i:i + CH]],
                                          d_task[te[i:i + CH]],
                                          n_task[te[i:i + CH]])
                            for i in range(0, P * 16, CH)])
            alive = (mg.amin(-1) > 0).nonzero(as_tuple=False).squeeze(-1)
            if alive.numel() == 0:
                break
            child = (score[:, None] + logp).reshape(-1)[alive]
            qn = qn[alive]
            par = (alive // 16).cpu()
            act = (alive % 16)
            te = te[alive]
            # per-task dedupe on the joint grid
            key = np.concatenate(
                [te.cpu().numpy()[:, None],
                 torch.round(qn / a.tree_dedupe).to(torch.int32).cpu().numpy()],
                axis=1)
            _, first = np.unique(key, axis=0, return_index=True)
            keep = torch.as_tensor(np.sort(first), device=dev)
            qn, te, act, child = qn[keep], te[keep], act[keep], child[keep]
            par = par[keep.cpu()]
            # per-task width cap: task-major, best score first
            if guided:
                rank = child + a.exit_gumbel * (-torch.log(-torch.log(
                    torch.rand(child.shape[0], device=dev, dtype=dtype))))
            else:
                rank = torch.rand(child.shape[0], device=dev, dtype=dtype)
            if score_mode == 'value' and guided:
                # the network ranks STATES (learnable) instead of scoring
                # ACTIONS (not learnable across tasks): cheap policy prefilter,
                # then the critic's remaining-return estimate decides
                pre = cap_per_task(te, rank, max(4 * W, 8))
                qn, te, act, child = qn[pre], te[pre], act[pre], child[pre]
                par = par[pre.cpu()]
                vobs = featurize(qn, d_task[te], n_task[te],
                                 verts[act].to(dtype))
                rank = agent.critic(vobs.float()).squeeze(-1).to(dtype)
            sel_o = cap_per_task(te, rank, W)
            sel_c = sel_o.cpu()
            q, tid, score = qn[sel_o], te[sel_o], child[sel_o]
            aprev = verts[act[sel_o]].to(dtype)
            parents.append(par[sel_c])
            actions.append(act[sel_o].cpu())
            pools_q.append(q.cpu()); pools_t.append(tid.cpu())
            depth += 1
            pf = env1.kin.tcp_fk_jac(q)[0]
            pr = ((pf - p0_task[tid]) * d_task[tid]).sum(-1)
            prc, tc = pr.cpu().numpy(), tid.cpu().numpy()
            for t in np.unique(tc):
                loc = np.nonzero(tc == t)[0]
                j = loc[int(prc[loc].argmax())]
                if prc[j] > best[int(t)][0]:
                    best[int(t)] = (float(prc[j]), depth, int(j))
        out = []
        for t in range(len(ids)):
            pr_, dp, idx = best[t]
            seq = []
            i = idx
            for r in range(dp - 1, -1, -1):
                seq.append(int(actions[r][i]))
                i = int(parents[r][i])
            out.append((pr_, np.array(seq[::-1], dtype=np.int64)))
        return out

    # ---------------- real-env replay of an action sequence ------------
    _, env_rp = _env_and_yaml(1, dev)

    @torch.no_grad()
    def replay(task_i, seq):
        env_rp.line_dist = ScriptedLineDistribution(spec_of([task_i]))
        env_rp.reset()
        obs_l, act_l, rew_l = [], [], []
        for t in range(len(seq)):
            obs_l.append(env_rp.current_obs()[0].clone())
            act_l.append(int(seq[t]))
            _, r, _, _, info = env_rp.step(verts[seq[t]][None],
                                           auto_reset=False)
            rew_l.append(float(r[0]))
            if bool(info['episode_done'][0]):
                break
        n_ = len(rew_l)
        ret = np.zeros(n_, dtype=np.float32)
        acc = 0.0
        for t in range(n_ - 1, -1, -1):
            acc = rew_l[t] + 0.99 * acc
            ret[t] = acc
        pf = env1.kin.tcp_fk_jac(env_rp.q)[0]
        prog = float(((pf[0] - env_rp.p_start[0])
                      * env_rp.line_dir[0]).sum())
        return (torch.stack(obs_l[:n_]), act_l[:n_], ret, prog)

    # ---------------- myopic demo seeding (the analytic expert) -------
    MB = 64
    _, env_my = _env_and_yaml(MB, dev)
    model_my = hl.StraightModel(env_my)
    model_my.terms = MYOPIC_TERMS
    myo_fn = hl.make_myopic(model_my)
    POW = torch.tensor([8.0, 4.0, 2.0, 1.0], device=dev)

    @torch.no_grad()
    def myopic_demos(ids):
        """Record the one-step margin law as vertex-index demos."""
        out = {}
        for base in range(0, len(ids), MB):
            grp = list(ids[base:base + MB])
            pad = MB - len(grp)
            grp_p = grp + [grp[0]] * pad
            env_my.line_dist = ScriptedLineDistribution(spec_of(grp_p))
            env_my.reset()
            O, A, R, AL = [], [], [], []
            for t in range(env_my.max_steps):
                alive = ~env_my.done_persistent
                if not bool(alive.any()):
                    break
                O.append(env_my.current_obs().clone())
                av = myo_fn(env_my, env_my.done_persistent)
                A.append(((av > 0).float() * POW).sum(-1).long())
                _, r, _, _, _ = env_my.step(av, auto_reset=False)
                R.append(r.clone())
                AL.append(alive.clone())
            if not O:
                continue
            O = torch.stack(O); A = torch.stack(A)
            R = torch.stack(R); AL = torch.stack(AL)
            pf = env_my.kin.tcp_fk_jac(env_my.q)[0]
            prog = ((pf - env_my.p_start) * env_my.line_dir).sum(-1)
            for k, ti in enumerate(grp):
                m = AL[:, k]
                n_ = int(m.sum())
                if n_ == 0:
                    continue
                rw = R[m, k].cpu().numpy()
                ret = np.zeros(n_, dtype=np.float32)
                acc = 0.0
                for t in range(n_ - 1, -1, -1):
                    acc = float(rw[t]) + 0.99 * acc
                    ret[t] = acc
                out[int(ti)] = (O[m, k].clone(),
                                [int(x) for x in A[m, k].cpu().numpy()],
                                ret, float(prog[k]))
        return out

    # ---------------- DAgger: expert labels on policy-visited states ---
    @torch.no_grad()
    def dagger_collect(ids64, use_search):
        """Roll the current policy on `ids64` tasks; at every visited state
        record the one-step margin law's action as the label (covariate-shift
        correction). Optionally upgrade a subset of labels to the first action
        of a search continuation started from that very state."""
        grp = list(ids64) + [ids64[0]] * (MB - len(ids64))
        env_my.line_dist = ScriptedLineDistribution(spec_of(grp))
        env_my.reset()
        loc = torch.arange(MB, device=dev)
        O, A, Q, T = [], [], [], []
        for t in range(env_my.max_steps):
            alive = ~env_my.done_persistent
            if not bool(alive.any()):
                break
            o = env_my.current_obs()
            lab = ((myo_fn(env_my, env_my.done_persistent) > 0).float()
                   * POW).sum(-1).long()
            O.append(o[alive].clone()); A.append(lab[alive].clone())
            Q.append(env_my.q[alive].clone()); T.append(loc[alive].clone())
            logits = agent._logits_head(agent._actor_trunk(o))
            ai = torch.distributions.Categorical(logits=logits).sample()
            env_my.step(verts[ai], auto_reset=False)
        if not O:
            return None, None, 0
        O = torch.cat(O); A = torch.cat(A)
        Q = torch.cat(Q); T = torch.cat(T)
        n_up = 0
        if use_search and O.shape[0] > 0:
            k = min(a.pool_dagger_search_k, O.shape[0])
            pick = torch.randperm(O.shape[0], device=dev)[:k]
            tids = [int(grp[int(x)]) for x in T[pick].cpu()]
            res = pooled_search(tids, a.pool_dagger_w, True, starts=Q[pick])
            for j, (_, seq) in enumerate(res):
                if len(seq) > 0:
                    A[pick[j]] = int(seq[0])
                    n_up += 1
        return O.float(), A, n_up

    # ---------------- held-out policy evaluation (no search) ----------
    _, env_ev = _env_and_yaml(len(ev_ids), dev)

    @torch.no_grad()
    def eval_heldout():
        env_ev.line_dist = ScriptedLineDistribution(spec_of(ev_ids))
        st = rollout_first_episode(
            env_ev, lambda e: agent.actor_mean(e.current_obs()))
        return st['episode_progress'].cpu().numpy()

    # ---------------- evaluation-only battery ----------------
    if a.pool_eval_only:
        agent.load_state_dict(torch.load(REPO / a.pool_eval_only,
                                         map_location=dev))
        agent.eval()
        pe = eval_heldout()
        cl, my = REF_CL[ev_ids], REF_MY[ev_ids]
        say(f"[pool-eval] policy alone (no search): "
            f"x{np.mean(pe / np.maximum(cl, 1e-6)):.3f} classical / "
            f"x{np.mean(pe / np.maximum(my, 1e-6)):.3f} margin law")
        say(f"[pool-eval] margin law itself: "
            f"x{np.mean(my / np.maximum(cl, 1e-6)):.3f} classical")
        for W in [int(x) for x in a.pool_eval_widths.split(',')]:
            for guided, sm in ((True, 'logp'), (True, 'value'),
                               (False, 'logp')):
                t0 = time.time()
                res = pooled_search(list(ev_ids), W, guided, score_mode=sm)
                pr = np.array([v[0] for v in res])
                tag = (f'guided ({sm:<5})' if guided else 'unguided     ')
                say(f"[pool-eval] search W{W:>4} {tag}: "
                    f"x{np.mean(pr / np.maximum(cl, 1e-6)):.3f} classical / "
                    f"x{np.mean(pr / np.maximum(my, 1e-6)):.3f} margin law  "
                    f"({time.time() - t0:.0f}s for {len(ev_ids)} tasks)")
        np.savez(OUT / 'pool_eval.npz', policy=pe, classical=cl, myopic=my)
        log.close()
        return

    # ---------------- training loop ----------------
    demos = {}
    OBS = ACT = RET = None
    DOBS = DACT = None               # DAgger buffer (obs, expert action)
    owner = {}                       # task -> 'myopic' | 'search'
    t_start = time.time()
    budget = a.pool_hours * 3600.0
    widths = [int(x) for x in a.pool_widths.split(',')]
    rnd = 0
    order_tr = np.random.default_rng(0).permutation(tr_ids)
    ptr = 0
    probe_base = None
    if a.pool_seed_myopic:
        t0 = time.time()
        seeded = myopic_demos(tr_ids)
        for k, v in seeded.items():
            demos[k] = v
            owner[k] = 'myopic'
        say(f"[pool] seeded {len(seeded)} train tasks with the one-step "
            f"margin law ({time.time() - t0:.0f}s); mean demo progress "
            f"{np.mean([v[3] for v in seeded.values()]):.3f} m")
    while time.time() - t_start < budget:
        W = widths[min(rnd, len(widths) - 1)]
        if ptr + a.pool_chunk > len(order_tr):
            order_tr = np.random.default_rng(rnd).permutation(tr_ids)
            ptr = 0
        chunk = order_tr[ptr:ptr + a.pool_chunk]
        ptr += a.pool_chunk
        t0 = time.time()
        try:
            found = dict(zip([int(c) for c in chunk],
                             pooled_search(chunk, W, guided=(rnd > 0))))
        except RuntimeError as e:
            say(f"[pool] r{rnd} search failed, skipping chunk: {e}")
            torch.cuda.empty_cache()
            rnd += 1
            continue
        t_srch = time.time() - t0
        n_new, n_imp = 0, 0
        for ti, (pr_, seq) in found.items():
            if len(seq) == 0:
                continue
            o, ac, rt, prog = replay(ti, seq)
            if ti not in demos:
                n_new += 1
            elif prog <= demos[ti][3] + 1e-4:
                continue
            else:
                n_imp += 1
            demos[ti] = (o, ac, rt, prog)
            owner[ti] = 'search'
        if demos:
            OBS = torch.cat([d[0] for d in demos.values()]).float()
            ACT = torch.tensor([x for d in demos.values() for x in d[1]],
                               device=dev, dtype=torch.long)
            RET = torch.tensor(np.concatenate([d[2] for d in demos.values()]),
                               device=dev, dtype=torch.float32)
        t1 = time.time()
        n_up = 0
        if a.pool_dagger_tasks > 0:
            dg_ids = list(np.random.default_rng(rnd).choice(
                tr_ids, min(MB, a.pool_dagger_tasks), replace=False))
            do, da, n_up = dagger_collect(
                dg_ids, use_search=(a.pool_dagger_search
                                    and rnd % a.pool_dagger_search_every == 0))
            if do is not None:
                DOBS = do if DOBS is None else torch.cat([DOBS, do])
                DACT = da if DACT is None else torch.cat([DACT, da])
                if DOBS.shape[0] > a.pool_dagger_cap:
                    DOBS = DOBS[-a.pool_dagger_cap:]
                    DACT = DACT[-a.pool_dagger_cap:]
        t_dg = time.time() - t1
        t1 = time.time()
        M = OBS.shape[0]
        MD = 0 if DOBS is None else DOBS.shape[0]
        for _ in range(a.pool_bc_steps):
            idx = torch.randint(0, M, (min(4096, M),), device=dev)
            logits = agent._logits_head(agent._actor_trunk(OBS[idx]))
            v = agent.critic(OBS[idx]).squeeze(-1)
            loss = (nn.functional.cross_entropy(logits, ACT[idx])
                    + 0.5 * nn.functional.mse_loss(v, RET[idx]))
            if MD > 0:
                di = torch.randint(0, MD, (min(4096, MD),), device=dev)
                dl = agent._logits_head(agent._actor_trunk(DOBS[di]))
                loss = loss + a.pool_dagger_coef * nn.functional.cross_entropy(
                    dl, DACT[di])
            opt.zero_grad(); loss.backward(); opt.step()
        t_bc = time.time() - t1
        srch_m = float(np.mean([v[0] for v in found.values()]))
        ratio_s = float(np.mean([found[t][0] / max(REF_MY[t], 1e-6)
                                 for t in found]))
        n_srch_owned = sum(1 for v in owner.values() if v == 'search')
        demo_ratio = float(np.mean([demos[k][3] / max(REF_MY[k], 1e-6)
                                    for k in demos]))
        say(f"[pool] r{rnd:>4} W{W:>5} {len(chunk)} tasks  search {srch_m:.3f} m "
            f"(x{ratio_s:.2f} myopic)  demos {len(demos)} (+{n_new}/^{n_imp}) "
            f"search-owned {n_srch_owned}  demo/myopic x{demo_ratio:.3f}  "
            f"buf {M}+{MD}d(^{n_up})  {t_srch:.0f}s search {t_dg:.0f}s dag "
            f"{t_bc:.0f}s bc  "
            f"elapsed {(time.time() - t_start) / 3600:.2f} h")

        if rnd % a.pool_eval_every == 0:
            pe = eval_heldout()
            rc = float(np.mean(pe / np.maximum(REF_CL[ev_ids], 1e-6)))
            rm = float(np.mean(pe / np.maximum(REF_MY[ev_ids], 1e-6)))
            mc = float(np.mean(REF_MY[ev_ids]
                               / np.maximum(REF_CL[ev_ids], 1e-6)))
            # probe: does the prior help search on NEVER-TRAINED tasks?
            if probe_base is None:
                pb = pooled_search(pb_ids, a.pool_probe_w, guided=False)
                probe_base = float(np.mean([v[0] for v in pb]))
            pg = pooled_search(pb_ids, a.pool_probe_w, guided=True)
            probe_g = float(np.mean([v[0] for v in pg]))
            say(f"[pool] EVAL r{rnd}: held-out policy (no search) "
                f"x{rc:.3f} classical / x{rm:.3f} myopic  "
                f"[myopic itself x{mc:.3f} classical]  | probe search W"
                f"{a.pool_probe_w}: unguided {probe_base:.3f} -> guided "
                f"{probe_g:.3f} m")
            torch.save(agent.state_dict(), OUT / 'agent_pool.pt')
            np.savez(OUT / 'pool_state.npz',
                     eval_progress=pe, ev_ids=ev_ids,
                     ref_cl=REF_CL, ref_my=REF_MY,
                     demo_tasks=np.array(sorted(demos.keys())),
                     demo_progress=np.array([demos[k][3]
                                             for k in sorted(demos)]),
                     round=rnd, hours=(time.time() - t_start) / 3600)
        rnd += 1
    torch.save(agent.state_dict(), OUT / 'agent_pool.pt')
    say(f"[pool] budget spent: {rnd} rounds, {len(demos)} tasks with demos, "
        f"{(time.time() - t_start) / 3600:.2f} h")
    log.close()


def stage_vstariter(a, dev):
    """Fitted value iteration to DE-SATURATE Vhat_search: on a fixed ~20k
    state set (bank + GE archive + fresh policy rollouts), precompute all 16
    model successors and their reset-style features once, then iterate
    Vhat <- max_a alive_a * (1 + gamma * Vhat(s'_a)) with a refit per sweep.
    Each sweep extends the effective horizon by ~1 step beyond the 40-step
    probe base; ~120 sweeps reach the full task depth."""
    import torch.nn as nn
    y, env = _env_and_yaml(128, dev)
    task, spec1 = _load_task(dev, env.kin.dtype)
    model = hl.StraightModel(env)
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
                 -1).reshape(-1, env.act_dim), dtype=torch.float32,
        device=dev)
    env.line_dist = SingleTaskDistribution(spec1)
    env.reset()
    dvec = env.line_dir[0].clone(); nvec = env.n_target[0].clone()
    p0 = env.p_start[0].clone()

    # ---- fixed state set ----
    Ss = []
    rb = np.load(OUT / 'reachtree_bank.npz')
    Ss.append(torch.tensor(rb['q'], device=dev, dtype=env.kin.dtype))
    ga = np.load(OUT / 'goexplore_archive.npz')
    gi = np.random.default_rng(0).choice(len(ga['q']), 6000, replace=False)
    Ss.append(torch.tensor(ga['q'][gi], device=dev, dtype=env.kin.dtype))
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(OUT / 'agent.pt', map_location=dev))
    agent.eval()
    with torch.no_grad():
        for t in range(70):
            logits = agent._logits_head(agent._actor_trunk(env.current_obs()))
            ai = torch.distributions.Categorical(logits=logits).sample()
            env.step(agent.vertices[ai])
            if t % 3 == 0:
                Ss.append(env.q.clone())
    S = torch.cat(Ss)
    S = S[torch.randperm(S.shape[0], device=dev)[:20000]]
    M = S.shape[0]
    print(f"[vstariter] state set {M}")

    # ---- precompute successors + features + own features ----
    with torch.no_grad():
        qe = S.unsqueeze(1).expand(-1, 16, -1).reshape(M * 16, -1)
        ae = verts.unsqueeze(0).expand(M, -1, -1).reshape(M * 16, -1)
        CH = 32768
        qn = torch.cat([model.step(qe[i:i+CH],
                                   dvec.expand(min(CH, M*16-i), 3),
                                   nvec.expand(min(CH, M*16-i), 3),
                                   ae[i:i+CH])
                        for i in range(0, M*16, CH)])
        mg = torch.cat([model.margins(qn[i:i+CH],
                                      p0.expand(min(CH, M*16-i), 3),
                                      dvec.expand(min(CH, M*16-i), 3),
                                      nvec.expand(min(CH, M*16-i), 3))
                       for i in range(0, M*16, CH)])
        alive = (mg.amin(-1) > 0).float().reshape(M, 16)

        def featurize(qbatch):
            outs = []
            FB = 65536
            for i in range(0, qbatch.shape[0], FB):
                chunk = qbatch[i:i+FB]
                fenv = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                                      'n_envs': chunk.shape[0]}), None, dev)
                fenv.line_dist = ScriptedLineDistribution(
                    {'q0': chunk,
                     'line_dir': dvec[None].expand(chunk.shape[0], 3).clone(),
                     'n_target': nvec[None].expand(chunk.shape[0], 3).clone()})
                fenv.reset()
                outs.append(fenv.current_obs().clone())
                del fenv
            return torch.cat(outs)
        F_next = featurize(qn)                      # (M*16, obs)
        F_own = featurize(S)                        # (M, obs)
    print("[vstariter] features ready")

    # twin networks + clipped (min) targets: plain max-backup VI on a fitted
    # net inflates optimistic bubbles that a greedy policy then chases into
    # walls (vguide v2 died confidently at 0.30)
    nets, opts = [], []
    for k in range(2):
        vk = _vstar_net(env.obs_dim, dev)
        vk.load_state_dict(torch.load(REPO / a.opt_value, map_location=dev))
        nets.append(vk)
        opts.append(torch.optim.Adam(vk.parameters(), lr=5e-4))
    for sweep in range(a.vi_sweeps):
        with torch.no_grad():
            vmins = []
            for i in range(0, M*16, 65536):
                v1 = nets[0](F_next[i:i+65536]).squeeze(-1)
                v2 = nets[1](F_next[i:i+65536]).squeeze(-1)
                vmins.append(torch.minimum(v1, v2))
            vn = torch.cat(vmins).reshape(M, 16)
            target = (alive * (1.0 + 0.99 * vn)).max(-1).values
        for k in range(2):
            for ep in range(60):
                idx = torch.randint(0, M, (1024,), device=dev)
                loss = torch.nn.functional.smooth_l1_loss(
                    nets[k](F_own[idx]).squeeze(-1), target[idx])
                opts[k].zero_grad(); loss.backward(); opts[k].step()
        if sweep % 20 == 0:
            print(f"  vi sweep {sweep:>4}  target mean {float(target.mean()):.1f} "
                  f"max {float(target.max()):.1f}  loss {float(loss):.3f}",
                  flush=True)
    torch.save(nets[0].state_dict(), OUT / 'vstar_net_vi.pt')
    print(f"wrote {OUT / 'vstar_net_vi.pt'}")


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
                             'ceiling', 'traj', 'reachtree',
                             'reachtree_bundle', 'goexplore',
                             'goexplore_env', 'selfimitate', 'fqi',
                             'vstarfit', 'vguide', 'vstariter', 'vdagger',
                             'vmask', 'exit', 'exit_multi', 'pool'])
    ap.add_argument('--pool-train', type=int, default=512)
    ap.add_argument('--pool-eval', type=int, default=128)
    ap.add_argument('--pool-probe', type=int, default=24)
    ap.add_argument('--pool-probe-w', type=int, default=64)
    ap.add_argument('--pool-chunk', type=int, default=16)
    ap.add_argument('--pool-widths', default='512,384,256,192,128,96,64')
    ap.add_argument('--pool-bc-steps', type=int, default=400)
    ap.add_argument('--pool-lr', type=float, default=3e-4)
    ap.add_argument('--pool-hours', type=float, default=10.0)
    ap.add_argument('--pool-eval-every', type=int, default=10)
    ap.add_argument('--pool-resume', default=None)
    ap.add_argument('--pool-eval-only', default=None,
                    help='checkpoint: run the held-out battery and exit')
    ap.add_argument('--pool-eval-widths', default='4,16,64')
    ap.add_argument('--pool-dagger-tasks', type=int, default=64)
    ap.add_argument('--pool-dagger-cap', type=int, default=600_000)
    ap.add_argument('--pool-dagger-coef', type=float, default=1.0)
    ap.add_argument('--pool-dagger-search', type=int, default=1)
    ap.add_argument('--pool-dagger-search-k', type=int, default=96)
    ap.add_argument('--pool-dagger-search-every', type=int, default=4)
    ap.add_argument('--pool-dagger-w', type=int, default=96)
    ap.add_argument('--pool-seed-myopic', type=int, default=1,
                    help='seed every train task demo with the one-step '
                         'margin law; search must strictly beat it')
    ap.add_argument('--emx-ranks', default='0,3,10,30,50,100,200,300,700,1000',
                    help='gap-sorted ranks of tasks for exit_multi')
    ap.add_argument('--exit-widths', default='16384,4096,1024,256,64')
    ap.add_argument('--exit-bc-epochs', type=int, default=4000)
    ap.add_argument('--exit-gumbel', type=float, default=0.3)
    ap.add_argument('--mask-value', default=None,
                    help='Vhat net for doomed-action masking (vmask stage)')
    ap.add_argument('--mask-alpha', type=float, default=0.7)
    ap.add_argument('--vd-rounds', type=int, default=4)
    ap.add_argument('--vd-tube', type=int, default=64,
                    help='n_envs of the eps-greedy collection tube')
    ap.add_argument('--vd-eps', type=float, default=0.1)
    ap.add_argument('--vd-states', type=int, default=50_000,
                    help='tree-label state budget')
    ap.add_argument('--vd-fit-steps', type=int, default=25_000)
    ap.add_argument('--vd-probe-budget', type=int, default=600,
                    help='max probe labels per DAgger round')
    ap.add_argument('--vd-probe-h', type=int, default=70)
    ap.add_argument('--bundle-min', type=float, default=0.70,
                    help='keep leaf trajectories with progress above this')
    ap.add_argument('--bundle-max-save', type=int, default=4000)
    ap.add_argument('--vi-sweeps', type=int, default=140)
    ap.add_argument('--vguide-updates', type=int, default=1500)
    ap.add_argument('--ent-coef2', type=float, default=0.01,
                    help='entropy bonus in the vguide objective')
    ap.add_argument('--opt-value', default=None,
                    help='vstar_net.pt: actor advantages become '
                         'r + gamma*Vhat*(s\') - Vhat*(s)')
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
    ap.add_argument('--rnd-beta', type=float, default=None,
                    help='RND intrinsic bonus weight (method-5 OFU proxy)')
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
     'reachtree_bundle': stage_reachtree_bundle,
     'goexplore': stage_goexplore, 'goexplore_env': stage_goexplore_env,
     'selfimitate': stage_selfimitate, 'fqi': stage_fqi,
     'vstarfit': stage_vstarfit, 'vguide': stage_vguide,
     'vstariter': stage_vstariter, 'vdagger': stage_vdagger,
     'vmask': stage_vmask, 'exit': stage_exit,
     'exit_multi': stage_exit_multi, 'pool': stage_pool}[a.stage](a, dev)


if __name__ == '__main__':
    main()
