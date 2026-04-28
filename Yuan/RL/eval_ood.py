"""Out-of-distribution and in-distribution evaluation suite for v3.

Sweeps a trained policy across several distribution shifts and reports
mean rollout length, success rate, and the home-seed and oracle baselines
on each split.

Usage:
    python -m Yuan.RL.eval_ood [--ckpt path] [--n 256] [--oracle-k 16]
"""
from __future__ import annotations
import argparse, glob, os, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.rollout import rollout
from Yuan.RL.policy import make_policy


# ---------------- splits ----------------
def _box(lo, hi):
    return (np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32))


def make_envs(seed_base: int = 12345):
    """Create one env per distribution split. Each env uses non-randomized
    defaults overridden by the split's specific kwargs."""
    common = dict(use_collision=cfg.USE_COLLISION_CHECK)
    # base FR3 + collider can be shared, but we keep them separate to keep
    # the random streams independent.
    splits = {
        "in_dist":     dict(seed=seed_base + 0, randomize=False, **common),
        "ood_tilt":    dict(seed=seed_base + 1, randomize=False,
                            n_tilt_range=(np.deg2rad(45.0),
                                          np.deg2rad(60.0)),
                            **common),
        "ood_box_far": dict(seed=seed_base + 2, randomize=False,
                            p0_box=_box([0.50, -0.30, 0.20],
                                        [0.75,  0.30, 0.55]),
                            **common),
        "ood_box_low": dict(seed=seed_base + 3, randomize=False,
                            p0_box=_box([0.30, -0.30, 0.10],
                                        [0.60,  0.30, 0.25]),
                            **common),
    }
    return splits


def latest_ckpt() -> str | None:
    paths = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "ckpt_*.pt")))
    return paths[-1] if paths else None


def load_policy(path: str, env: FarsightedSeedEnv,
                device: torch.device):
    qmid  = torch.as_tensor(env.action_mid,  dtype=torch.float32, device=device)
    qhalf = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(path, map_location=device, weights_only=False)
    pi = make_policy(cfg.STATE_DIM, env.action_dim, qmid, qhalf,
                     policy_type=state.get("policy_type", "gaussian")).to(device)
    pi.load_state_dict(state["policy"])
    pi.eval()
    return pi


# ---------------- per-split eval ----------------
def eval_split(policy, env: FarsightedSeedEnv, n: int,
               oracle_k: int, device: torch.device,
               rng_for_oracle: np.random.Generator,
               best_components: bool = False):
    home = env.arm.home_qs.astype(np.float32)
    pol_lens  = np.empty(n, dtype=np.int32)
    home_lens = np.empty(n, dtype=np.int32)
    orc_lens  = np.empty(n, dtype=np.int32)
    Ts        = np.empty(n, dtype=np.int32)
    for i in range(n):
        s = env.reset()
        task = env._cur
        c = task["c"]
        T = task["T"]
        Ts[i] = T
        # policy
        st = torch.as_tensor(s[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            if best_components and hasattr(policy, "component_actions"):
                cand = policy.component_actions(st).squeeze(0)
            else:
                a, _ = policy.act(st, deterministic=True)
                cand = a
        best_pol = -1
        for a_i in cand.reshape(-1, env.action_dim):
            a_np = a_i.cpu().numpy().astype(np.float32)
            info_p = rollout(env.arm, a_np, c[:3], c[3:6], c[6:9],
                             mjc=env.mjc, max_steps=T,
                             v_path=task["v_path"], eps_p=task["eps_p"])
            best_pol = max(best_pol, info_p["length"])
        pol_lens[i] = best_pol
        # home
        info_h = rollout(env.arm, home, c[:3], c[3:6], c[6:9],
                         mjc=env.mjc, max_steps=T,
                         v_path=task["v_path"], eps_p=task["eps_p"],
                         action_mode="joint_seed")
        home_lens[i] = info_h["length"]
        # oracle: best of K random + home
        best = info_h["length"]
        if oracle_k > 0:
            for _ in range(oracle_k):
                q = rng_for_oracle.uniform(env.lmt_lo,
                                           env.lmt_up).astype(np.float32)
                L = rollout(env.arm, q, c[:3], c[3:6], c[6:9],
                            mjc=env.mjc, max_steps=T,
                            v_path=task["v_path"], eps_p=task["eps_p"],
                            action_mode="joint_seed")["length"]
                if L > best:
                    best = L
        orc_lens[i] = best
        env._cur = None
    succ_p = (pol_lens >= Ts).mean()
    succ_h = (home_lens >= Ts).mean()
    succ_o = (orc_lens >= Ts).mean()
    rec_p  = (pol_lens / Ts).mean()
    rec_o  = (orc_lens / Ts).mean()
    pol_over_orc = float(rec_p / max(rec_o, 1e-6))
    return {
        "n": n,
        "policy":   {"len": float(pol_lens.mean()), "succ": float(succ_p),
                     "rec": float(rec_p)},
        "home":     {"len": float(home_lens.mean()), "succ": float(succ_h)},
        "oracle":   {"len": float(orc_lens.mean()), "succ": float(succ_o),
                     "rec": float(rec_o)} if oracle_k > 0 else None,
        "policy_vs_oracle": pol_over_orc if oracle_k > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--oracle-k", type=int, default=16)
    ap.add_argument("--best-components", action="store_true",
                    help="For mixture policies, rollout all component means and keep the best.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = make_envs()

    ckpt = args.ckpt or latest_ckpt()
    if ckpt is None:
        raise SystemExit(f"No checkpoint in {cfg.CKPT_DIR}/.")
    print(f"loading {ckpt}")

    # build one env first to make the policy with the right state dim
    first_env = FarsightedSeedEnv(**splits["in_dist"])
    policy = load_policy(ckpt, first_env, device)
    print(f"state_dim={cfg.STATE_DIM}  ndof={first_env.ndof}")

    rng_orc = np.random.default_rng(7)
    results = {}
    t0 = time.time()
    for name, kwargs in splits.items():
        if name == "in_dist":
            env = first_env
        else:
            env = FarsightedSeedEnv(**kwargs)
        out = eval_split(policy, env, args.n, args.oracle_k, device, rng_orc,
                         best_components=args.best_components)
        results[name] = out
        po = out["policy"]; ho = out["home"]; oc = out["oracle"]
        print(f"\n[{name:12s}]  n={out['n']}")
        print(f"  policy : len={po['len']:5.2f}  succ={po['succ']:.3f}  "
              f"rec={po['rec']:.3f}")
        print(f"  home   : len={ho['len']:5.2f}  succ={ho['succ']:.3f}")
        if oc is not None:
            print(f"  oracle : len={oc['len']:5.2f}  succ={oc['succ']:.3f}  "
                  f"rec={oc['rec']:.3f}")
            print(f"  pol/orc rec ratio: {out['policy_vs_oracle']:.3f}")
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
