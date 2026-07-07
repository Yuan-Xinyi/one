"""B: autopsy of the Easy-bucket catastrophic tail (paper metric pct<50%).

For the tail tasks (DP seeds):
  1. rerun r12m-hybrid from the DP seed with full recording (term reason etc.)
  2. rerun r12m-hybrid from the classical-optimal label seed (max_label_q)
Attribution: if the same controller reaches ~oracle' from the label seed but
fails from the DP seed, the bleed is seed quality, not controller skill.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import TERM_NAMES

SW = Path('Yuan/system_eval/runs/eval_10k_systematic/sweeps')
OUT = Path('/home/lqin/one/Yuan/RL_controller/runs/exit_rounds7plus/final_avg/tail_autopsy.npz')

d = np.load(SW / 'main_final+switch0.985-0.96.npz')
pct = d['pct']
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'][:len(pct)] * 1.5
seeds_dp = np.load(SW / 'cfg_only_w1.5.npz')['seeds'].astype(np.float32)
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))

fin = np.isfinite(pct)
tail = fin & (bucket == 'Easy') & (pct < 50)
idx = np.nonzero(tail)[0]
print(f"[tail] {len(idx)} Easy-bucket tasks with pct<50% "
      f"(mean pct {pct[tail].mean():.1f}%)", flush=True)

p0 = z['cs_p0'][idx].astype(np.float32)
ld = z['cs_line_dir'][idx].astype(np.float32)
nt = z['cs_n_target'][idx].astype(np.float32)
q_dp = seeds_dp[idx]
q_lab = z['max_label_q'][idx].astype(np.float32)

dev = torch.device('cuda')
ckpt = Path('Yuan/RL_controller/runs/exit_rounds7plus/final_avg')
env = build_env(ckpt / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(ckpt, env, dev)

res = {}
for name, qs in (('dp', q_dp), ('label', q_lab)):
    r = rollout_seeds_batched(qs, p0, ld, nt, env=env,
                              controller='hybrid_variantB', classical=classical,
                              agent=agent, tau_enter=0.985, tau_exit=0.96,
                              progress_prefix=f'tail-{name} ')
    res[name] = r
    L = r['L'] * 1.5
    print(f"[tail-{name}] mean pct {100*(L/np.maximum(oh[idx],1e-9)).mean():.1f}%",
          flush=True)

np.savez_compressed(
    OUT, idx=idx, oh=oh[idx], pct_orig=pct[idx],
    **{f"{k}_{n}": res[n][k] for n in ('dp', 'label')
       for k in ('L', 'episode_len', 'term_reason', 'switch_count', 'init_max_qn')})

Ldp, Llab = res['dp']['L'] * 1.5, res['label']['L'] * 1.5
pdp, plab = Ldp / np.maximum(oh[idx], 1e-9), Llab / np.maximum(oh[idx], 1e-9)
print("\n==== ATTRIBUTION ====")
print(f"from DP seed   : mean {100*pdp.mean():.1f}%   >=90%: {100*(pdp>=0.9).mean():.1f}%")
print(f"from label seed: mean {100*plab.mean():.1f}%   >=90%: {100*(plab>=0.9).mean():.1f}%")
seed_fix = plab >= 0.9
print(f"seed-fixable (label-seed run >=90% oracle'): {100*seed_fix.mean():.1f}%")
print(f"controller-stuck (label-seed run <50%):      {100*(plab<0.5).mean():.1f}%")
print("\nterm reasons from DP seed:")
tr = res['dp']['term_reason']
for t in np.unique(tr):
    name = TERM_NAMES[int(t)] if 0 <= int(t) < len(TERM_NAMES) else str(t)
    print(f"  {name:12s} {100*(tr==t).mean():.1f}%")
print("\ninit_max_qn: DP seed mean "
      f"{res['dp']['init_max_qn'].mean():.3f}  label seed "
      f"{res['label']['init_max_qn'].mean():.3f}")
print("[tail] done", flush=True)
