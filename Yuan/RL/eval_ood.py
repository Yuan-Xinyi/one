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
from Yuan.RL.policy import make_policy
from Yuan.RL.batched_rollout import batched_rollout


def _branch_grid_actions(K_phi: int, K_psi: int) -> np.ndarray:
    """Deterministic (cos phi, sin phi, cos psi, sin psi) grid in 4D
    branch-descriptor space. Used as a tight Monte-Carlo oracle ceiling
    in the same action space as the policy."""
    phis = np.linspace(0.0, 2.0 * np.pi, int(K_phi), endpoint=False)
    psis = np.linspace(0.0, 2.0 * np.pi, int(K_psi), endpoint=False)
    actions = np.array(
        [[np.cos(p), np.sin(p), np.cos(s), np.sin(s)]
         for p in phis for s in psis], dtype=np.float32)
    return actions


def _batched_oracle(contexts: np.ndarray,
                    Ts: np.ndarray,
                    v_paths: np.ndarray,
                    eps_ps: np.ndarray,
                    K_phi: int = 32,
                    K_psi: int = 16,
                    chunk: int = 1024) -> np.ndarray:
    """Compute per-task oracle as max L over a (K_phi x K_psi) action grid,
    using GPU-batched rollouts.

    contexts: (n, 9) [p0, d, n] per task
    Ts:       (n,)
    Returns:  (n,) int — best L per task across all grid actions.
    """
    actions = _branch_grid_actions(K_phi, K_psi)            # (K, 4)
    K = actions.shape[0]
    n = contexts.shape[0]

    # Cartesian-product all (task, action) pairs
    c_all = np.repeat(contexts, K, axis=0)                  # (n*K, 9)
    a_all = np.tile(actions,    (n, 1))                     # (n*K, 4)
    v_all = np.repeat(v_paths,  K).astype(np.float32)
    e_all = np.repeat(eps_ps,   K).astype(np.float32)
    T_all = np.repeat(Ts,       K).astype(np.int32)

    L_flat = np.empty(n * K, dtype=np.int32)
    for start in range(0, n * K, chunk):
        end = min(start + chunk, n * K)
        out = batched_rollout(a_all[start:end], c_all[start:end],
                              v_all[start:end], e_all[start:end],
                              T_all[start:end])
        L_flat[start:end] = np.asarray(out["lengths"], dtype=np.int32)
    L = L_flat.reshape(n, K)
    return L.max(axis=1)


# ---------------- splits ----------------
def _box(lo, hi):
    return (np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32))


def make_envs(seed_base: int = 12345):
    """Create one env per distribution split. Each env uses non-randomized
    defaults; specific path-length splits override the env's max_steps via
    the env's `eval_T` knob (see FarsightedSeedEnv).

    Splits cover:
      - distribution shifts: tilt, box_far, box_low
      - path-length shifts: T=80 (0.4 m, backward-compat), T=160 (0.8 m),
        T=240 (1.2 m, full reach)
    """
    common = dict(use_collision=cfg.USE_COLLISION_CHECK)
    splits = {
        # backward-compat: short path, in distribution
        "in_dist_T80":  dict(seed=seed_base + 0, randomize=False,
                             eval_T=80, **common),
        # path-length shifts (in-dist context, but longer T)
        "long_T160":    dict(seed=seed_base + 1, randomize=False,
                             eval_T=160, **common),
        "long_T240":    dict(seed=seed_base + 2, randomize=False,
                             eval_T=240, **common),
        # distribution shifts (T fixed at 80 to compare vs v6/v7 numbers)
        "ood_tilt":     dict(seed=seed_base + 3, randomize=False,
                             eval_T=80,
                             n_tilt_range=(np.deg2rad(45.0),
                                           np.deg2rad(60.0)),
                             **common),
        "ood_box_far":  dict(seed=seed_base + 4, randomize=False,
                             eval_T=80,
                             p0_box=_box([0.50, -0.30, 0.20],
                                         [0.75,  0.30, 0.55]),
                             **common),
        "ood_box_low":  dict(seed=seed_base + 5, randomize=False,
                             eval_T=80,
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
               best_components: bool = False,
               oracle_K_phi: int = 32,
               oracle_K_psi: int = 16,
               oracle_chunk: int = 1024):
    Ts        = np.empty(n, dtype=np.int32)

    # ---- collect tasks first so we can batch the oracle in one pass ----
    states    = np.empty((n, cfg.STATE_DIM), dtype=np.float32)
    contexts  = np.empty((n, 9), dtype=np.float32)
    v_paths   = np.empty(n, dtype=np.float32)
    eps_ps    = np.empty(n, dtype=np.float32)

    for i in range(n):
        s = env.reset()
        task = env._cur
        c = task["c"]
        T = task["T"]
        states[i] = s
        Ts[i] = T
        contexts[i] = c
        v_paths[i] = task["v_path"]
        eps_ps[i] = task["eps_p"]
        env._cur = None

    # ---- policy: evaluate all policy candidates in one GPU batch ----
    st = torch.as_tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        if best_components and hasattr(policy, "component_actions"):
            cand = policy.component_actions(st)
            if cand.ndim == 2:
                cand = cand[:, None, :]
        else:
            a, _ = policy.act(st, deterministic=True)
            cand = a[:, None, :]
    cand_np = cand.detach().cpu().numpy().astype(np.float32)
    n_cand = cand_np.shape[1]
    pol_out = batched_rollout(
        cand_np.reshape(n * n_cand, env.action_dim),
        np.repeat(contexts, n_cand, axis=0),
        np.repeat(v_paths, n_cand),
        np.repeat(eps_ps, n_cand),
        np.repeat(Ts, n_cand),
        action_mode=cfg.ACTION_MODE,
    )
    pol_lens = np.asarray(pol_out["lengths"], dtype=np.int32).reshape(n, n_cand).max(axis=1)

    # ---- home: joint-seed baseline in one GPU batch ----
    home = env.arm.home_qs.astype(np.float32)
    home_actions = np.repeat(home[None, :], n, axis=0).astype(np.float32)
    home_out = batched_rollout(
        home_actions, contexts, v_paths, eps_ps, Ts,
        action_mode="joint_seed",
    )
    home_lens = np.asarray(home_out["lengths"], dtype=np.int32)

    # ---- oracle: GPU-batched grid search over (phi, psi) in 4D action space ----
    # K_phi=32, K_psi=16 -> 512 actions per task; in same action space as policy
    if oracle_K_phi * oracle_K_psi > 0:
        orc_lens = _batched_oracle(contexts, Ts, v_paths, eps_ps,
                                   K_phi=oracle_K_phi, K_psi=oracle_K_psi,
                                   chunk=oracle_chunk)
        # also include home as a fallback (in case grid misses)
        orc_lens = np.maximum(orc_lens, home_lens)
    else:
        orc_lens = home_lens.copy()
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
    ap.add_argument("--oracle-k", type=int, default=16,
                    help="(legacy, unused) old random-seed oracle K")
    ap.add_argument("--oracle-Kphi", type=int, default=32,
                    help="grid points along phi for batched 4D oracle")
    ap.add_argument("--oracle-Kpsi", type=int, default=16,
                    help="grid points along psi for batched 4D oracle")
    ap.add_argument("--oracle-chunk", type=int, default=1024,
                    help="batch size for batched_rollout chunks during oracle")
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
    first_name = next(iter(splits.keys()))
    first_env = FarsightedSeedEnv(**splits[first_name])
    policy = load_policy(ckpt, first_env, device)
    print(f"state_dim={cfg.STATE_DIM}  ndof={first_env.ndof}")

    rng_orc = np.random.default_rng(7)
    results = {}
    t0 = time.time()
    for name, kwargs in splits.items():
        if name == first_name:
            env = first_env
        else:
            env = FarsightedSeedEnv(**kwargs)
        out = eval_split(policy, env, args.n, args.oracle_k, device, rng_orc,
                         best_components=args.best_components,
                         oracle_K_phi=args.oracle_Kphi,
                         oracle_K_psi=args.oracle_Kpsi,
                         oracle_chunk=args.oracle_chunk)
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
