"""Continue the search-free expert iteration beyond round 6 (user request).

Exactly Algorithm 1 of the paper: expert = BuildExpert(current student)
(interior = student's own action, shell = classical, soft band), DAgger
aggregation on top of the lineage dataset, warm-started regression, 10k
evaluation per round. Stop signal: no improvement in pure AND no decline in
the switch-benefit fraction.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch, yaml

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, build_task_aligned_basis
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.RL_controller.self_improve.distill import fit_actor, TARGET_CLAMP
from Yuan.RL_controller.self_improve.loop import eval_ckpt_on_10k

RUNS = Path('Yuan/RL_controller/runs')
START = RUNS / 'distill_simple_exit_final'     # round-6 student (avg r5,r6)
OUT = RUNS / 'exit_rounds7plus'
OUT.mkdir(exist_ok=True)
TAU_MAP, BAND = 0.98, 0.02                     # expert map (lineage convention)
TAU_RUN = (0.985, 0.96)                        # runtime thresholds for eval
N_TASKS = 12288
ROUNDS = range(7, 11)

dev = torch.device('cuda')
student, cfg_yaml = load_agent(START, dev)
env_kw = load_env_kw(cfg_yaml)
line_cfg = cfg_yaml['line_distribution']
threshold_m = (float(line_cfg['feasibility_threshold_m'])
               if line_cfg.get('feasibility_filter', False) else None)
proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': 1}),
                       line_dist=None, device=dev)
pool = LineDistribution.load_or_build(
    kin=proxy.kin, collision=proxy.collision, n_pool=line_cfg['n_pool'],
    n_target_noise_deg=line_cfg['n_target_noise_deg'],
    seed=line_cfg['train_seed'], env_cfg=EnvConfig(**{**env_kw, 'n_envs': 1}),
    feasibility_threshold_m=threshold_m)
del proxy

# lineage aggregate D (2.16M rows from rounds 1-6)
d6 = np.load(RUNS / 'distill_r6_soft/distill_dataset.npz')
D_obs = [torch.from_numpy(d6['obs']).float()]
D_act = [torch.from_numpy(d6['act']).float()]


@torch.no_grad()
def classical_action(env, ctrl):
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping)
    q_dot = ctrl.q_dot_null(env.q, env.line_dir, env.n_target)
    a = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    return (a / env.a_max).clamp(-1.0, 1.0)


@torch.no_grad()
def collect_and_label(behavior, n_tasks, seed):
    """Roll behavior on fresh tasks; label EVERY visited state with
    BuildExpert(behavior): interior -> behavior action, shell -> classical,
    band -> blend. Returns (obs, act) tensors."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    tasks = pool.sample(n_tasks, generator=gen)
    obs_l, act_l = [], []
    for s in range(0, n_tasks, 4096):
        e = min(s + 4096, n_tasks)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': e - s}),
                             line_dist=None, device=dev)
        env.line_dist = ScriptedLineDistribution(
            {k: v[s:e].clone() for k, v in tasks.items()})
        env.reset()
        ctrl = ClassicalNullspaceController(env.kin)
        for _ in range(env.max_steps + 1):
            active = ~env.done_persistent
            if bool(active.any()):
                obs = env.current_obs()
                a_stu = behavior.actor_mean(obs).clamp(-1, 1)
                a_cls = classical_action(env, ctrl)
                qn = ((env.q - env.q_mid).abs() / env.q_half).max(-1).values
                lam = ((qn - (TAU_MAP - BAND)) / BAND).clamp(0, 1).unsqueeze(-1)
                a_exp = (1 - lam) * a_stu + lam * a_cls.to(a_stu.dtype)
                obs_l.append(obs[active].float().cpu())
                act_l.append(a_exp[active].float().cpu())
            env.step(behavior.actor_mean(env.current_obs()).clamp(-1, 1),
                     auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        del env
        torch.cuda.empty_cache()
    return torch.cat(obs_l), torch.cat(act_l)


prev = START
for rnd in ROUNDS:
    rd = OUT / f'round{rnd}'
    rd.mkdir(exist_ok=True)
    if (rd / 'eval_10k.npz').exists():
        student, _ = load_agent(rd, dev)
        d = np.load(rd / 'dataset.npz')
        D_obs.append(torch.from_numpy(d['obs']).float())
        D_act.append(torch.from_numpy(d['act']).float())
        prev = rd
        print(f'[r{rnd}] cached, skip', flush=True)
        continue
    obs_i, act_i = collect_and_label(student, N_TASKS, 13000 + rnd)
    np.savez_compressed(rd / 'dataset.npz', obs=obs_i.numpy(), act=act_i.numpy())
    D_obs.append(obs_i)
    D_act.append(act_i)
    obs_all = torch.cat(D_obs)
    act_all = torch.cat(D_act).clamp(-TARGET_CLAMP, TARGET_CLAMP)
    print(f'[r{rnd}] dataset {obs_all.shape[0]} rows '
          f'({obs_i.shape[0]} new)', flush=True)
    torch.manual_seed(9000 + rnd)
    val = fit_actor(student, obs_all, act_all, dev, epochs=60)
    torch.save(student.state_dict(), rd / 'agent.pt')
    cfg = dict(cfg_yaml)
    cfg['distill'] = {'note': f'search-free expert iteration round {rnd}, '
                              f'warm-start round {rnd-1}', 'val_mse': float(val)}
    with open(rd / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    eval_ckpt_on_10k(rd, rd / 'eval_10k.npz',
                     tau_enter=TAU_RUN[0], tau_exit=TAU_RUN[1], device=dev)
    prev = rd

# weight average of the last two rounds
import copy
sd_a = torch.load(OUT / f'round{ROUNDS[-2] if len(list(ROUNDS))>1 else ROUNDS[-1]}/agent.pt', map_location='cpu', weights_only=False)
sd_b = torch.load(OUT / f'round{ROUNDS[-1]}/agent.pt', map_location='cpu', weights_only=False)
avg = {k: (sd_a[k].double() + sd_b[k].double()) / 2 for k in sd_a}
fd = OUT / 'final_avg'
fd.mkdir(exist_ok=True)
torch.save({k: v.float() for k, v in avg.items()}, fd / 'agent.pt')
with open(fd / 'config.yaml', 'w') as f:
    yaml.safe_dump(dict(cfg_yaml), f, sort_keys=False)
eval_ckpt_on_10k(fd, fd / 'eval_10k.npz',
                 tau_enter=TAU_RUN[0], tau_exit=TAU_RUN[1], device=dev)
print('[rounds7plus] done', flush=True)
