"""r11-a: replicate the r8 merge recipe with a DEEPER MPC boundary.

r8 = fit(r6 clean rows qn<0.975  +  r7 MPC rows qn>=0.975, init=r5)
r11-a variants move that single knob: boundary 0.965 and 0.955.
No new MPC labeling — r7's dataset already carries MPC-blend labels down
to qn>=0.955 (pure MPC above 0.975, pi0->MPC linear blend in between).
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
from Yuan.RL_controller.self_improve.collect import load_agent
from Yuan.RL_controller.self_improve.distill import fit_actor, TARGET_CLAMP
from Yuan.RL_controller.self_improve.loop import eval_ckpt_on_10k

RUNS = Path("/home/lqin/one/Yuan/RL_controller/runs")
WARM = RUNS / "distill_r5_warmstart"
device = torch.device("cuda")

d6 = np.load(RUNS / "distill_r6_soft/distill_dataset.npz")
obs6, act6 = d6["obs"], d6["act"]
qn6 = np.abs(obs6[:, :7]).max(1)
parts7 = [np.load(RUNS / f"distill_r7_mpc/dataset_round{r}.npz") for r in (0, 1)]
obs7 = np.concatenate([p["obs"] for p in parts7])
act7 = np.concatenate([p["act"] for p in parts7])
qn7 = np.abs(obs7[:, :7]).max(1)

for tb in (0.965, 0.955):
    out_dir = RUNS / f"distill_r11_belt{tb:.3f}"
    if (out_dir / "eval_10k.npz").exists():
        print(f"[r11] {out_dir.name} already done, skip", flush=True)
        continue
    out_dir.mkdir(exist_ok=True)
    keep6 = qn6 < tb
    keep7 = qn7 >= tb
    obs = np.concatenate([obs6[keep6], obs7[keep7]])
    act = np.concatenate([act6[keep6], act7[keep7]])
    print(f"[r11] boundary {tb}: {keep6.sum()} clean + {keep7.sum()} MPC "
          f"= {len(obs)} rows", flush=True)
    student, cfg_yaml = load_agent(WARM, device)
    torch.manual_seed(8800)
    val = fit_actor(student,
                    torch.from_numpy(obs).float(),
                    torch.from_numpy(act).float().clamp(-TARGET_CLAMP, TARGET_CLAMP),
                    device, epochs=80)
    torch.save(student.state_dict(), out_dir / "agent.pt")
    cfg = dict(cfg_yaml)
    cfg["distill"] = {"note": f"r11-a: r6 clean (qn<{tb}) + r7 MPC (qn>={tb}), "
                              f"warm-start r5 (one-knob change from r8)",
                      "val_mse": float(val)}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    del student
    torch.cuda.empty_cache()
    os.chdir("/home/lqin/one")
    eval_ckpt_on_10k(out_dir, out_dir / "eval_10k.npz",
                     tau_enter=0.98, tau_exit=0.98, device=device)
print("[r11] all done", flush=True)
