"""Eval-side K=16 extension: 8 MORE DP candidates (slots 8-15) on the 10k
eval set + full-slot rollouts + v2 features. Ranker scores per-candidate, so
no retraining is needed to exploit the bigger choice set.
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
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.system_eval.seed_sources import diffusion_seeds
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import obs_and_manip   # reuse feature builder

P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K_EXT = 8

dev = torch.device('cuda')
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dc = cfg['diffusion']
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
eval_set = {k: z[k] for k in z.keys()}

env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)

ext_npz = P0DIR / 'candidates_extw1.npz'
if ext_npz.exists():
    d = np.load(ext_npz)
    seeds, ik_ok = d['seeds'], d['ik_ok']
    print(f'[k16] ext candidates cached: IK ok {100*ik_ok.mean():.1f}%', flush=True)
else:
    seeds, ik_ok = diffusion_seeds(
        eval_set, dc['ckpt'], n_samples=K_EXT, ddim_steps=int(dc['ddim_steps']),
        cfg_w=1.0, sample_seed=9800, kin=env.kin, device=dev,
        use_ema=bool(dc['use_ema']))
    np.savez_compressed(ext_npz, seeds=seeds, ik_ok=ik_ok)
    print(f'[k16] ext candidates saved: IK ok {100*ik_ok.mean():.1f}%', flush=True)

p0s = eval_set['cs_p0'].astype(np.float32)
lds = eval_set['cs_line_dir'].astype(np.float32)
nts = eval_set['cs_n_target'].astype(np.float32)
for si in range(K_EXT):
    slot_npz = P0DIR / f'L_slot{16 + si}.npz'
    if slot_npz.exists():
        continue
    r = rollout_seeds_batched(
        seeds[:, si].astype(np.float32), p0s, lds, nts, env=env,
        controller='hybrid_variantB', classical=classical, agent=agent,
        tau_enter=TAU[0], tau_exit=TAU[1], progress_prefix=f'ext-slot{si} ')
    np.savez_compressed(slot_npz, L=r['L'], term_reason=r['term_reason'])
    print(f'[k16] slot {16+si} done', flush=True)

feat_npz = P0DIR / 'feat_v2_extw1.npz'
if not feat_npz.exists():
    n10 = seeds.shape[0]
    obs_s = np.zeros((n10, K_EXT, 31), np.float32)
    mu_s = np.zeros((n10, K_EXT), np.float32)
    for si in range(K_EXT):
        obs_s[:, si], mu_s[:, si] = obs_and_manip(
            env, seeds[:, si], p0s, lds, nts)
    np.savez_compressed(feat_npz, obs_slots=obs_s, mu_slots=mu_s)
    print('[k16] ext features cached', flush=True)
print('[k16] all done', flush=True)
