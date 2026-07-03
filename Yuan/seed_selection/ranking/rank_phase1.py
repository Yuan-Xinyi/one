"""Seed-ranking Phase 1: training data on TRAIN-distribution tasks.

20480 pool tasks (seed 9200, disjoint from everything) x K=8 DP candidates;
exact L labels with the adopted controller; initial obs (31-d) per candidate
as ranker features. All cached under seed_selection/runs/rank_train/.
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
from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.system_eval.seed_sources import diffusion_seeds

OUT = Path('Yuan/seed_selection/runs/rank_train')
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K = 8
N_TASKS = 20480
CHUNK = 4096

dev = torch.device('cuda')
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dc = cfg['diffusion']

# ---- tasks from the train pool ----
_, cfg_yaml = load_agent(CKPT, dev)
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
gen = torch.Generator(device=dev).manual_seed(9200)
tasks = pool.sample(N_TASKS, generator=gen)
q0_pilot = tasks['q0'].cpu().numpy().astype(np.float32)
ld = tasks['line_dir'].cpu().numpy().astype(np.float32)
nt = tasks['n_target'].cpu().numpy().astype(np.float32)
p0t, _, _, _ = proxy.kin.tcp_fk_jac(tasks['q0'].to(proxy.kin.dtype))
p0 = p0t.cpu().numpy().astype(np.float32)
del proxy

env = build_env(CKPT / 'config.yaml', CHUNK, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)

# ---- candidates ----
cand_npz = OUT / 'candidates_K8.npz'
if cand_npz.exists():
    d = np.load(cand_npz)
    seeds, ik_ok = d['seeds'], d['ik_ok']
    print(f'[p1] candidates cached: IK ok {100*ik_ok.mean():.1f}%', flush=True)
else:
    eval_like = {'cs_p0': p0, 'cs_line_dir': ld, 'cs_n_target': nt}
    seeds, ik_ok = diffusion_seeds(
        eval_like, dc['ckpt'], n_samples=K, ddim_steps=int(dc['ddim_steps']),
        cfg_w=1.5, sample_seed=9400, kin=env.kin, device=dev,
        use_ema=bool(dc['use_ema']))
    np.savez_compressed(cand_npz, seeds=seeds, ik_ok=ik_ok,
                        p0=p0, line_dir=ld, n_target=nt, q0_pilot=q0_pilot)
    print(f'[p1] candidates saved: IK ok {100*ik_ok.mean():.1f}%', flush=True)

# ---- labels per slot ----
L_slots = np.full((N_TASKS, K), np.nan, dtype=np.float32)
for si in range(K):
    slot_npz = OUT / f'L_slot{si}.npz'
    if slot_npz.exists():
        L_slots[:, si] = np.load(slot_npz)['L']
        continue
    r = rollout_seeds_batched(
        seeds[:, si].astype(np.float32), p0, ld, nt, env=env,
        controller='hybrid_variantB', classical=classical, agent=agent,
        tau_enter=TAU[0], tau_exit=TAU[1], progress_prefix=f'p1-slot{si} ')
    L_slots[:, si] = r['L']
    np.savez_compressed(slot_npz, L=r['L'], term_reason=r['term_reason'])
    print(f'[p1] slot {si} labeled', flush=True)

# ---- initial obs per candidate ----
obs_npz = OUT / 'obs0_K8.npz'
if not obs_npz.exists():
    dtype = env.kin.dtype
    obs_all = np.zeros((N_TASKS, K, 31), dtype=np.float32)
    for si in range(K):
        outs = []
        for s in range(0, N_TASKS, CHUNK):
            e = min(s + CHUNK, N_TASKS)
            pad = CHUNK - (e - s)
            def _t(x, w):
                t = torch.as_tensor(x[s:e], device=dev, dtype=dtype)
                return torch.cat([t, t[-1:].expand(pad, w)]) if pad else t
            env.line_dist = ScriptedLineDistribution(
                {'q0': _t(seeds[:, si], 7), 'line_dir': _t(ld, 3),
                 'n_target': _t(nt, 3)})
            env.reset()
            env.p_start[:] = _t(p0, 3)
            outs.append(env.current_obs()[:e - s].float().cpu())
        obs_all[:, si] = torch.cat(outs).numpy()
        print(f'[p1] obs slot {si} done', flush=True)
    np.savez_compressed(obs_npz, obs0=obs_all)
print('[p1] all done', flush=True)
