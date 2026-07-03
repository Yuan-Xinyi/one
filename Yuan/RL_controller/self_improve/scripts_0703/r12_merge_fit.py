"""r12 merges: r6 clean labels + fresh K=32/hold16 MPC labels at two
boundaries, two warm-starts. Eval pure + hybrid @0.985/0.96 (current best tau).

Also evals distill_r12_mpc32's own agent (pi0-safe + MPC composition).
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
from Yuan.RL_controller.self_improve.collect import load_agent
from Yuan.RL_controller.self_improve.distill import fit_actor, TARGET_CLAMP
from Yuan.RL_controller.self_improve.loop import eval_ckpt_on_10k

RUNS = Path("/home/lqin/one/Yuan/RL_controller/runs")
R12 = RUNS / "distill_r12_mpc32"
TAU = (0.985, 0.96)
device = torch.device("cuda")

# r12's own agent first (already fitted by mpc_distill).
if not (R12 / "eval_10k.npz").exists() and (R12 / "agent.pt").exists():
    eval_ckpt_on_10k(R12, R12 / "eval_10k.npz",
                     tau_enter=TAU[0], tau_exit=TAU[1], device=device)

d6 = np.load(RUNS / "distill_r6_soft/distill_dataset.npz")
obs6, act6 = d6["obs"], d6["act"]
qn6 = np.abs(obs6[:, :7]).max(1)
parts = [np.load(R12 / f"dataset_round{r}.npz")
         for r in (0, 1) if (R12 / f"dataset_round{r}.npz").exists()]
obs12 = np.concatenate([p["obs"] for p in parts])
act12 = np.concatenate([p["act"] for p in parts])
qn12 = np.abs(obs12[:, :7]).max(1)
print(f"[r12m] r12 rows {len(obs12)} (qn>=0.975: {(qn12>=0.975).sum()}, "
      f">=0.965: {(qn12>=0.965).sum()})", flush=True)

for tb in (0.975, 0.965):
    for warm in ("distill_r5_warmstart", "distill_soup2"):
        wtag = "r5" if "r5" in warm else "soup2"
        out_dir = RUNS / f"distill_r12m_b{tb:.3f}_{wtag}"
        if (out_dir / "eval_10k.npz").exists():
            print(f"[r12m] {out_dir.name} done, skip", flush=True)
            continue
        out_dir.mkdir(exist_ok=True)
        keep6, keep12 = qn6 < tb, qn12 >= tb
        obs = np.concatenate([obs6[keep6], obs12[keep12]])
        act = np.concatenate([act6[keep6], act12[keep12]])
        print(f"[r12m] {out_dir.name}: {keep6.sum()} clean + {keep12.sum()} "
              f"MPC32 rows", flush=True)
        student, cfg_yaml = load_agent(RUNS / warm, device)
        torch.manual_seed(8810)
        val = fit_actor(student,
                        torch.from_numpy(obs).float(),
                        torch.from_numpy(act).float().clamp(-TARGET_CLAMP,
                                                            TARGET_CLAMP),
                        device, epochs=80)
        torch.save(student.state_dict(), out_dir / "agent.pt")
        cfg = dict(cfg_yaml)
        cfg["distill"] = {"note": f"r12 merge: r6 clean (qn<{tb}) + r12 MPC "
                                  f"K=32/hold16 (qn>={tb}), warm-start {wtag}",
                          "val_mse": float(val)}
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        del student
        torch.cuda.empty_cache()
        eval_ckpt_on_10k(out_dir, out_dir / "eval_10k.npz",
                         tau_enter=TAU[0], tau_exit=TAU[1], device=device)
print("[r12m] all done", flush=True)
