"""Train one morphology-general null-space policy over several arms.

    python -m Yuan.IJRR.morph.train_morph --config Yuan/IJRR/morph/config_morph_all.yaml \
        --out-dir Yuan/IJRR/runs/morph_all

The config lists arm groups (a base arm and how many random variants of it,
with the environment count per arm); every arm gets its own task pool
(cached under runs/_pool_cache by the arm's name), the arms are stepped
together by MultiArmEnv and the TokenAgent is trained with the paper's PPO.
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if __name__ == "__main__" and _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.morph.chain_specs import BASE, with_spheres, perturb_spec, register_specs
from Yuan.IJRR.morph.token_env import TokenEnv, MultiArmEnv, env_config_for
from Yuan.IJRR.morph.token_agent import TokenAgent
from Yuan.IJRR.stage2_traj.ppo import PPOConfig, train as ppo_train


def build_specs(cfg_arms: list[dict]) -> list[tuple[dict, int]]:
    """[(spec, n_envs)] from the config's arm groups."""
    out = []
    for g in cfg_arms:
        base = with_spheres(copy.deepcopy(BASE[g['base']]))
        if int(g.get('variants', 0)) == 0:
            out.append((base, int(g['n_envs'])))
        else:
            rng = np.random.default_rng(int(g.get('seed', 0)))
            for k in range(int(g['variants'])):
                v = perturb_spec(base, rng, f"{g['base']}_v{int(g.get('seed', 0))}_{k}",
                                 len_scale=tuple(g.get('len_scale', (0.75, 1.25))),
                                 qd_scale=tuple(g.get('qd_scale', (0.7, 1.3))))
                out.append((v, int(g['n_envs_each'])))
    return out


def make_token_env(spec, n_envs, env_cfg_dict, line_cfg, seed, device, verbose=True):
    cfg = env_config_for(spec, n_envs, env_cfg_dict)
    tenv = TokenEnv(spec, cfg, None, device)
    thr = float(line_cfg['feasibility_threshold_m']) if line_cfg.get('feasibility_filter', False) else None
    n_pool = int(line_cfg.get('n_pool_variant', 20000)) if 'len_scale' in spec else int(line_cfg['n_pool'])
    tenv.env.line_dist = LineDistribution.load_or_build(
        kin=tenv.env.kin, collision=tenv.env.collision, n_pool=n_pool,
        n_target_noise_deg=line_cfg['n_target_noise_deg'], seed=seed, env_cfg=cfg,
        feasibility_threshold_m=thr, verbose=verbose)
    return tenv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out-dir', default='Yuan/IJRR/runs/morph')
    ap.add_argument('--device', default=None)
    ap.add_argument('--resume-from-ckpt', default=None)
    args = ap.parse_args()
    y = yaml.safe_load(open(args.config))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    specs = build_specs(y['arms'])
    register_specs([s for s, _ in specs])
    json.dump([{k: v for k, v in s.items() if k != 'spheres'} | {'n_envs': n} for s, n in specs],
              open(out / 'arms.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    line_cfg = y['line_distribution']; eval_cfg = y['eval']
    print(f'[morph] {len(specs)} arms, {sum(n for _, n in specs)} envs total', flush=True)
    t0 = time.time()
    train_envs = [make_token_env(s, n, y['env'], line_cfg, int(line_cfg['train_seed']), device) for s, n in specs]
    print(f'[morph] train pools ready ({time.time()-t0:.0f}s); holdout envs...', flush=True)
    hold = [make_token_env(s, int(eval_cfg['n_holdout']), y['env'], line_cfg, int(eval_cfg['holdout_seed']), device, verbose=False)
            for s, _ in specs]
    env = MultiArmEnv(train_envs)
    ppo_cfg = PPOConfig(**y['ppo'])
    ag = y.get('agent', {})
    agent = TokenAgent(hidden_dim=ppo_cfg.hidden_dim, init_log_std=ppo_cfg.init_log_std,
                       squashed_entropy=ppo_cfg.squashed_entropy, d_model=int(ag.get('d_model', 192)),
                       nhead=int(ag.get('nhead', 4)), n_layers=int(ag.get('n_layers', 3))).to(device)
    print(f'[morph] agent params {sum(p.numel() for p in agent.parameters())/1e6:.2f} M', flush=True)

    @torch.no_grad()
    def eval_fn(agent):
        res = {}
        for te in hold:
            obs = te.reset()
            for _ in range(te.env.max_steps):
                obs, _, _, _, _ = te.step(agent.actor_mean(obs), auto_reset=False)
                if bool(te.done_persistent.all()):
                    break
            res[f'eval/{te.spec["name"]}_mean_progress_m'] = float(te.arc_progress.float().mean().item())
        res['eval/mean_progress_m'] = float(np.mean(list(res.values())))
        return res

    log_file = open(out / 'train.log', 'w'); t0 = time.time()

    def log_fn(d):
        d2 = {'wall_s': time.time() - t0, **d}
        log_file.write(repr(d2) + '\n'); log_file.flush()
        if 'update' in d:
            print(f"upd {d['update']:>4}  step {d['global_step']:>9}  r/prog {d.get('reward/progress', 0):+.3f}  "
                  f"v_loss {d.get('train/v_loss', 0):.4f}  entropy {d.get('train/entropy', 0):.2f}", flush=True)
        elif 'eval_at_step' in d:
            per = ', '.join(f'{k.split("/")[1].replace("_mean_progress_m", "")} {v:.3f}' for k, v in d.items() if k.startswith('eval/') and k != 'eval/mean_progress_m')
            print(f"  eval @ {d['eval_at_step']:>9}  mean {d.get('eval/mean_progress_m', 0):.3f} m  [{per}]", flush=True)

    ppo_train(ppo_cfg, env, device=device, agent=agent, eval_fn=eval_fn,
              eval_every=int(y['train']['eval_every']), log_fn=log_fn,
              ckpt_path=str(out / 'agent.pt'),
              ckpt_every_n_updates=int(y['train'].get('ckpt_every_n_updates', 10)),
              resume_from_ckpt=args.resume_from_ckpt)
    log_file.close()
    print('[morph] done, ckpt ->', out / 'agent.pt', flush=True)


if __name__ == '__main__':
    main()
