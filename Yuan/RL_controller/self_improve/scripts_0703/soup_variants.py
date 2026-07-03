"""Model-soup variants over the warm-start-r5 basin (r8, r9, r11a, soup2).

Weight-average actor+critic state dicts, save as standard ckpt_dirs, then
eval each on the 10k set (pure + hybrid @0.98/0.98).
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import shutil
import torch, yaml

sys.path.insert(0, "/home/lqin/one")
from Yuan.RL_controller.self_improve.loop import eval_ckpt_on_10k

RUNS = Path("/home/lqin/one/Yuan/RL_controller/runs")

RECIPES = {
    # soup2-basin soups over the r12 merge students (both warm-started from
    # soup2) — the last lever for the single-net record.
    "soup3_s2_b975":    [("distill_soup2", 0.5), ("distill_r12m_b0.975_soup2", 0.5)],
    "soup3_s2_b975_b965": [("distill_soup2", 1/3), ("distill_r12m_b0.975_soup2", 1/3),
                           ("distill_r12m_b0.965_soup2", 1/3)],
    "soup3_b975_b965":  [("distill_r12m_b0.975_soup2", 0.5),
                         ("distill_r12m_b0.965_soup2", 0.5)],
    "soup3_s2w3_b975w7": [("distill_soup2", 0.3), ("distill_r12m_b0.975_soup2", 0.7)],
}

for name, members in RECIPES.items():
    out_dir = RUNS / f"distill_{name}"
    if (out_dir / "eval_10k.npz").exists():
        print(f"[soup] {name} already evaluated, skip", flush=True)
        continue
    out_dir.mkdir(exist_ok=True)
    avg = None
    for member, w in members:
        sd = torch.load(RUNS / member / "agent.pt", map_location="cpu",
                        weights_only=False)
        if avg is None:
            avg = {k: v.double() * w for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].double() * w
    torch.save({k: v.float() for k, v in avg.items()}, out_dir / "agent.pt")
    cfg = yaml.safe_load(open(RUNS / members[0][0] / "config.yaml"))
    cfg["distill"] = {"note": "soup: " + " + ".join(f"{w:.2f}*{m}" for m, w in members)}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[soup] {name} built from {members}", flush=True)
    os.chdir("/home/lqin/one")
    eval_ckpt_on_10k(out_dir, out_dir / "eval_10k.npz",
                     tau_enter=0.985, tau_exit=0.965)
print("[soup] all done", flush=True)
