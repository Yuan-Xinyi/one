"""PPO training entry for the Nova2 drag task.

Reuses the IJRR stage2_traj PPO implementation (read-only import):
Agent (tanh-squashed Gaussian, squashed-entropy option) and the generic
train() loop; only the env and the config file are Qling's.

Usage (the REAL entry point -- smoke tests must go through this):
    cd /home/lqin/one/Yuan/Qling
    /home/lqin/miniconda3/envs/one/bin/python -m drag.train_drag \
        --config drag/config_drag_v0.yaml --out drag/runs/v0_30M
    # smoke: add  --total-steps 65536 --eval-every 16384

Outputs in --out:
    agent.pt            checkpoint (saved every ckpt_every_n_updates)
    train_log.csv       one row per PPO update
    eval_log.csv        one row per eval pass
    config_resolved.yaml  the exact config this run used
"""
import matplotlib  # noqa: F401  must precede torch on this box (CXXABI)

import argparse
import csv
import os
import time

import torch
import yaml

from .drag_env import DragEnvConfig, Nova2DragEnv
from .container_env import ContainerDragEnv
from .container_scenario import ContainerScenario
from .regrasp_env import RegraspContainerEnv
from .regrasp_free_env import RegraspFreeEnv
from .bottle_env import BottleCompareEnv
from .bottle_slope_env import BottleSlopeEnv
from .bottle_hill_env import BottleHillEnv
from .ijrr_root import add_ijrr_path
add_ijrr_path()
from Yuan.IJRR.stage2_traj.ppo import PPOConfig, train  # noqa: E402

ENV_KINDS = {'drag': Nova2DragEnv, 'container': ContainerDragEnv,
             'regrasp': RegraspContainerEnv,
             'regrasp_free': RegraspFreeEnv,
             'bottle': BottleCompareEnv,
             'bottle_slope': BottleSlopeEnv,
             'bottle_hill': BottleHillEnv}


def build_env(kind, cfg_obj, sc_kwargs):
    cls = ENV_KINDS[kind]
    if kind in ('drag', 'bottle', 'bottle_slope', 'bottle_hill'):
        return cls(cfg_obj)
    sc = ContainerScenario(**sc_kwargs) if sc_kwargs else None
    return cls(cfg_obj, scenario=sc)


class CsvLogger:
    """Union-of-keys CSV writer; splits train and eval rows by marker key."""

    def __init__(self, path):
        self.path = path
        self.fieldnames = None
        self.f = None

    def __call__(self, row: dict):
        if self.f is None:
            self.fieldnames = list(row.keys())
            self.f = open(self.path, 'w', newline='')
            self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames,
                                    extrasaction='ignore')
            self.w.writeheader()
        self.w.writerow({k: row.get(k, '') for k in self.fieldnames})
        self.f.flush()


def make_eval_fn(env_kwargs, n_envs, seed, device, kind='drag',
                 sc_kwargs=None):
    kw = dict(env_kwargs)
    # eval always from the true task start, never mid-curriculum
    kw.update(n_envs=n_envs, seed=seed, start_mode='wp0')
    eval_env = build_env(kind, DragEnvConfig(**kw), sc_kwargs)

    def eval_fn(agent):
        # identical tasks every pass: re-seed the env's sampler
        if hasattr(eval_env, 'set_value_fn'):
            def _vfn(o):
                with torch.no_grad():
                    return agent.get_value(o)
            eval_env.set_value_fn(_vfn)
        eval_env.gen.manual_seed(seed)
        obs = eval_env.reset()
        L0 = eval_env.L0.clone()
        deaths = dict(jl=0, coll=0, ws=0, drift=0)
        success_ever = torch.zeros(n_envs, dtype=torch.bool,
                                   device=eval_env.device)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(eval_env.cfg.max_steps):
                a = agent.actor_mean(obs)
                obs, r, term, trunc, info = eval_env.step(
                    a, auto_reset=False)
                success_ever |= (info['success']
                                 & ~eval_env.done_persistent)
                for k, key in (('jl', 'died_jl'), ('coll', 'died_coll'),
                               ('ws', 'died_ws'), ('drift', 'died_drift')):
                    deaths[k] += int(info[key].sum())
                if eval_env.done_persistent.all():
                    break
        p, R, _, _ = eval_env._frames(eval_env.q)
        d_end = (eval_env.goal_xy
                 - eval_env._object_xy(p, R)).norm(dim=1)
        prog = ((L0 - d_end) / L0)
        placed = (d_end <= eval_env.cfg.goal_eps).float()
        stats = {
            'eval/success': float(success_ever.float().mean()),
            'eval/placed': float(placed.mean()),
            'eval/progress_mean': float(prog.mean()),
            'eval/progress_median': float(prog.median()),
            'eval/died_jl': deaths['jl'], 'eval/died_coll': deaths['coll'],
            'eval/died_ws': deaths['ws'], 'eval/died_drift': deaths['drift'],
            'eval/seconds': round(time.time() - t0, 1),
        }
        print('[eval] ' + '  '.join(f'{k.split("/")[1]}={v}'
                                    for k, v in stats.items()))
        return stats

    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='drag/config_drag_v0.yaml')
    ap.add_argument('--out', default='drag/runs/dev')
    ap.add_argument('--total-steps', type=int, default=None)
    ap.add_argument('--eval-every', type=int, default=None)
    ap.add_argument('--resume', default=None,
                    help='checkpoint to resume policy weights from')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.total_steps is not None:
        cfg['ppo']['total_timesteps'] = args.total_steps
    if args.eval_every is not None:
        cfg['train']['eval_every'] = args.eval_every

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config_resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)

    device = torch.device(cfg['env'].get('device', 'cpu'))
    kind = cfg['env'].pop('kind', 'drag')
    sc_kwargs = cfg['env'].pop('scenario', None)
    env = build_env(kind, DragEnvConfig(**cfg['env']), sc_kwargs)
    ppo_cfg = PPOConfig(**cfg['ppo'])

    # critic-in-the-loop envs (free regrasp): build the agent up front
    # and hand its value head to the env as the grasp selector
    agent = None
    if hasattr(env, 'set_value_fn'):
        from Yuan.IJRR.stage2_traj.ppo import Agent
        agent = Agent(env.obs_dim, env.act_dim,
                      hidden_dim=ppo_cfg.hidden_dim,
                      init_log_std=ppo_cfg.init_log_std,
                      squashed_entropy=ppo_cfg.squashed_entropy).to(device)

        def _train_vfn(o):
            with torch.no_grad():
                return agent.get_value(o)
        env.set_value_fn(_train_vfn)

    train_log = CsvLogger(os.path.join(args.out, 'train_log.csv'))
    eval_log = CsvLogger(os.path.join(args.out, 'eval_log.csv'))
    t_start = time.time()

    def log_fn(row: dict):
        if 'eval_at_step' in row:
            eval_log(row)
            return
        train_log(row)
        if row['update'] % 10 == 0:
            sps = row['global_step'] / max(time.time() - t_start, 1e-9)
            print(f"upd {row['update']:>5d}  step {row['global_step']:>9d}  "
                  f"ep_rew {row.get('episode/reward_mean', float('nan')):.3f}  "
                  f"ep_len {row.get('episode/length_mean', float('nan')):.1f}  "
                  f"kl {row['train/approx_kl']:.4f}  "
                  f"sigma {row['train/sigma_mean']:.2f}  "
                  f"{sps:,.0f} sps")

    eval_fn = make_eval_fn(cfg['env'], cfg['train']['eval_n_envs'],
                           cfg['train']['eval_seed'], device,
                           kind=kind, sc_kwargs=sc_kwargs)

    train(ppo_cfg, env, device,
          eval_fn=eval_fn, eval_every=cfg['train']['eval_every'],
          log_fn=log_fn,
          ckpt_path=os.path.join(args.out, 'agent.pt'),
          ckpt_every_n_updates=cfg['train']['ckpt_every_n_updates'],
          resume_from_ckpt=args.resume, agent=agent)
    print(f'done in {(time.time() - t_start)/60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
