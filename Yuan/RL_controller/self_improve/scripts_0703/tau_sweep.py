"""Fine tau sweep for hybrid(student, classical) on the 10k eval set.

Per-pair per-task caches under <ckpt>/tau_sweep/te{te}_tx{tx}.npz — skip if
present, so the sweep is resumable and follow-up analysis slices from cache.
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
from Yuan.system_eval.rollout_controllers import (
    build_env, load_rl_agent, rollout_seeds_batched)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

EVAL_SET = "/home/lqin/one/Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
RUNS = Path("/home/lqin/one/Yuan/RL_controller/runs")

CKPTS = ["distill_r11_belt0.965", "distill_soup2", "distill_r8_merged"]
PAIRS = [(0.990, 0.990), (0.985, 0.985), (0.980, 0.980), (0.975, 0.975),
         (0.985, 0.960), (0.975, 0.950),
         (0.990, 0.960), (0.985, 0.950), (0.985, 0.965), (0.990, 0.970)]

d = np.load(EVAL_SET)
qs, p0 = d["q0_seed"], d["cs_p0"]
ld, nt = d["cs_line_dir"], d["cs_n_target"]
L_oracle = d["max_label_L"]
valid = L_oracle > 1e-6

device = torch.device("cuda")
for name in CKPTS:
    ckpt = RUNS / name
    out_dir = ckpt / "tau_sweep"
    out_dir.mkdir(exist_ok=True)
    env = build_env(ckpt / "config.yaml", 4096, device)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(ckpt, env, device)
    for te, tx in PAIRS:
        out = out_dir / f"te{te:.3f}_tx{tx:.3f}.npz"
        if out.exists():
            r = np.load(out)
            print(f"[cached] {name} te={te} tx={tx}: "
                  f"hyb/oracle {float(r['metric_ratio_hyb_vs_oracle_mean']):.4f}",
                  flush=True)
            continue
        hyb = rollout_seeds_batched(
            qs, p0, ld, nt, env=env, controller="hybrid_variantB",
            classical=classical, agent=agent,
            tau_enter=te, tau_exit=tx, progress_prefix=f"{name[:12]} {te}/{tx} ")
        r_hyb = hyb["L"][valid] / L_oracle[valid]
        np.savez_compressed(
            out, L_hyb=hyb["L"], L_oracle=L_oracle,
            episode_len_hyb=hyb["episode_len"], term_hyb=hyb["term_reason"],
            switch_count=hyb["switch_count"],
            tau_enter=np.float64(te), tau_exit=np.float64(tx),
            ckpt_dir=np.str_(str(ckpt)),
            metric_ratio_hyb_vs_oracle_mean=np.float64(r_hyb.mean()),
            metric_ratio_hyb_vs_oracle_median=np.float64(np.median(r_hyb)),
            metric_mean_switches=np.float64(hyb["switch_count"].mean()))
        print(f"[done] {name} te={te} tx={tx}: hyb/oracle {r_hyb.mean():.4f} "
              f"(med {np.median(r_hyb):.4f})  sw/ep {hyb['switch_count'].mean():.2f}",
              flush=True)
    del env, agent
    torch.cuda.empty_cache()
print("[sweep] all done", flush=True)
