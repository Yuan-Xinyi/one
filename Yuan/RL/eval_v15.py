"""v15 evaluation: residual-corrected bandit vs all baselines.

Loads a v15 checkpoint (policy + ResidualNet) and compares deploy strategies:

  baselines (no learning at deploy):
    random_K1                    : 1 uniform random (φ, ψ) → real rollout
    phantom_select_uniform_K=N   : N uniform candidates, phantom-argmax → 1 real
    phantom_select_policy_K=N    : N policy samples, phantom-argmax → 1 real
    uniform_oracle_K=N           : N uniform real rollouts, take max (ceiling)
  v15 (this paper):
    residual_aug_policy_K=N      : N policy samples, argmax(phantom+R) → 1 real
    residual_aug_uniform_K=N     : N uniform candidates, argmax(phantom+R) → 1 real

Real rollout uses cfg.USE_CONTACT_MODE if True, else geo. Phantom is always
geometric (phantom_rollout in batched_rollout.py).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.qnet import ResidualNet
from Yuan.RL.batched_rollout import (
    batched_rollout, batched_rollout_contact, phantom_rollout,
)


def _load_v15(ckpt_path: str, env, device):
    qmid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    qhalf = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, qmid, qhalf,
                         policy_type=state.get("policy_type", "flow")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    rnet = ResidualNet(cfg.STATE_DIM, env.action_dim).to(device)
    rnet.load_state_dict(state["rnet"])
    rnet.eval()
    use_contact = bool(state.get("use_contact_mode", False))
    return policy, rnet, use_contact


def _real_rollout(actions_np, c_np, v_np, e_np, T_np, use_contact):
    fn = batched_rollout_contact if use_contact else batched_rollout
    out = fn(actions_np, c_np, v_np, e_np, T_np)
    return np.asarray(out["lengths"], dtype=np.int32)


def _phantom_lengths(actions_np, c_np, v_np, e_np, T_np):
    out = phantom_rollout(actions_np, c_np, v_np, e_np, T_np)
    return np.asarray(out["lengths"], dtype=np.int32)


def _normalize_L(L, T):
    return np.clip(L.astype(np.float32) /
                   np.maximum(T.astype(np.float32), 1.0), 0.0, 1.0)


def _stats(L_chosen, L_top, mask, name):
    """Print mean/std/quantile/L=0 stats over (L_chosen / L_top)[mask]."""
    safe_top = np.where(L_top > 0, L_top, 1).astype(np.float64)
    r = (L_chosen.astype(np.float64) / safe_top)[mask]
    n0 = int((L_chosen[mask] == 0).sum())
    nlo = int((r < 0.3).sum())
    print(f"  {name:>34}  mean={r.mean():.4f}  std={r.std():.4f}  "
          f"min={r.min():.3f}  p10={np.percentile(r,10):.3f}  "
          f"p25={np.percentile(r,25):.3f}  p50={np.percentile(r,50):.3f}  "
          f"p75={np.percentile(r,75):.3f}  L=0:{n0:3d}  r<0.3:{nlo:3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--K-oracle", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip K=1000 oracle (faster, use K=K_oracle as ratio base)")
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env (need it to construct policy w/ correct action_mid/half)
    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    policy, rnet, use_contact = _load_v15(args.ckpt, env, device)
    print(f"loaded {args.ckpt}  use_contact_mode={use_contact}")

    tasks = env._sample_tasks(args.n_tasks)
    c_np = np.stack([t['c'] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t['v_path'] for t in tasks], dtype=np.float32)
    e_np = np.array([t['eps_p']  for t in tasks], dtype=np.float32)
    T_np = np.array([t['T']      for t in tasks], dtype=np.int32)
    states_np = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    states_t = torch.as_tensor(states_np, device=device, dtype=torch.float32)

    K = args.K
    arange = np.arange(args.n_tasks)

    # ----- 1. Sample K UNIFORM candidates per task -----
    rng = np.random.default_rng(args.seed)
    phi_u = rng.uniform(0, 2*np.pi, size=(K, args.n_tasks)).astype(np.float32)
    psi_u = rng.uniform(0, 2*np.pi, size=(K, args.n_tasks)).astype(np.float32)
    a_unif_KB = np.stack([np.cos(phi_u), np.sin(phi_u),
                          np.cos(psi_u), np.sin(psi_u)], axis=-1)  # (K, B, 4)
    a_unif_flat = a_unif_KB.reshape(K * args.n_tasks, 4).astype(np.float32)

    # ----- 2. Sample K POLICY candidates per task -----
    a_pol_KB = []
    with torch.no_grad():
        for _ in range(K):
            a, _ = policy.act(states_t, deterministic=False)
            a_pol_KB.append(a)
    a_pol_KB_t = torch.stack(a_pol_KB, dim=0)                       # (K, B, 4)
    a_pol_flat = a_pol_KB_t.reshape(K * args.n_tasks, 4).cpu().numpy().astype(np.float32)

    rep_c = np.tile(c_np, (K, 1))
    rep_v = np.tile(v_np, K)
    rep_e = np.tile(e_np, K)
    rep_T = np.tile(T_np, K)

    # ----- 3. Phantom on both candidate sets -----
    print("\n[phantom on K candidates]")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_ph_unif = _phantom_lengths(a_unif_flat, rep_c, rep_v, rep_e, rep_T).reshape(K, args.n_tasks)
    L_ph_pol  = _phantom_lengths(a_pol_flat,  rep_c, rep_v, rep_e, rep_T).reshape(K, args.n_tasks)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- 4. Residual on policy candidates (= aug = phantom + R) -----
    print("\n[residual on policy candidates]")
    states_KB_flat = states_t.unsqueeze(0).expand(K, args.n_tasks, -1).reshape(K * args.n_tasks, -1)
    a_pol_KB_flat = a_pol_KB_t.reshape(K * args.n_tasks, -1)
    with torch.no_grad():
        R_pol = rnet(states_KB_flat, a_pol_KB_flat).cpu().numpy().reshape(K, args.n_tasks)
    aug_pol = _normalize_L(L_ph_pol, np.tile(T_np, (K, 1))) + R_pol
    # for uniform also; R could in principle help even on uniform candidates
    a_unif_t = torch.as_tensor(a_unif_flat, device=device, dtype=torch.float32)
    with torch.no_grad():
        R_unif = rnet(states_KB_flat, a_unif_t).cpu().numpy().reshape(K, args.n_tasks)
    aug_unif = _normalize_L(L_ph_unif, np.tile(T_np, (K, 1))) + R_unif

    # ----- 5. Real rollouts on K candidates (for analysis only — deploy needs only 1) -----
    print(f"\n[real K={K} rollouts on both candidate sets — for ground-truth analysis]")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_rl_unif = _real_rollout(a_unif_flat, rep_c, rep_v, rep_e, rep_T, use_contact).reshape(K, args.n_tasks)
    L_rl_pol  = _real_rollout(a_pol_flat,  rep_c, rep_v, rep_e, rep_T, use_contact).reshape(K, args.n_tasks)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- 6. K=1000 uniform oracle (ratio base) -----
    if args.no_oracle:
        L_top = L_rl_unif.max(axis=0)
        print(f"\n[oracle skipped] using K={K} uniform max as ratio base")
    else:
        print(f"\n[oracle] K={args.K_oracle} uniform real rollouts (ratio base)")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        Ko = args.K_oracle
        phi_o = rng.uniform(0, 2*np.pi, size=(Ko, args.n_tasks)).astype(np.float32)
        psi_o = rng.uniform(0, 2*np.pi, size=(Ko, args.n_tasks)).astype(np.float32)
        a_orc = np.stack([np.cos(phi_o), np.sin(phi_o),
                          np.cos(psi_o), np.sin(psi_o)], axis=-1)
        a_orc_flat = a_orc.reshape(Ko * args.n_tasks, 4).astype(np.float32)
        rep_co = np.tile(c_np, (Ko, 1))
        rep_vo = np.tile(v_np, Ko)
        rep_eo = np.tile(e_np, Ko)
        rep_To = np.tile(T_np, Ko)
        L_orc_flat = np.empty(Ko * args.n_tasks, dtype=np.int32)
        chunk = 4096
        for s in range(0, len(a_orc_flat), chunk):
            e = min(s + chunk, len(a_orc_flat))
            L_orc_flat[s:e] = _real_rollout(
                a_orc_flat[s:e], rep_co[s:e], rep_vo[s:e],
                rep_eo[s:e], rep_To[s:e], use_contact)
        L_top = L_orc_flat.reshape(Ko, args.n_tasks).max(axis=0)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- 7. Strategy lookups -----
    # all "_select" strategies pick argmax_score over K, look up REAL L
    L_random          = L_rl_unif[0]                                    # K=1 random
    L_ph_unif_select  = L_rl_unif[L_ph_unif.argmax(axis=0), arange]     # uniform + phantom
    L_ph_pol_select   = L_rl_pol[L_ph_pol.argmax(axis=0),  arange]      # policy + phantom
    L_unif_orc        = L_rl_unif.max(axis=0)                            # uniform ceiling at K
    L_pol_orc         = L_rl_pol.max(axis=0)                             # policy ceiling at K
    L_aug_pol_select  = L_rl_pol[aug_pol.argmax(axis=0), arange]         # OURS: policy + phantom + R
    L_aug_unif_select = L_rl_unif[aug_unif.argmax(axis=0), arange]       # ablation: uniform + phantom + R

    # ----- 8. Report -----
    p0 = c_np[:, :3].astype(np.float64)
    base_dist = np.linalg.norm(p0, axis=-1)
    well = (base_dist >= args.min_base_dist) & (L_top > 0)
    n_well = int(well.sum())
    print(f"\n=== n_well = {n_well} / {args.n_tasks} ===")

    print("\n=== A. Bandit baselines (no learning at deploy) ===")
    _stats(L_random,         L_top, well, f"random_K=1")
    _stats(L_ph_unif_select, L_top, well, f"phantom_select_uniform_K={K}")
    _stats(L_ph_pol_select,  L_top, well, f"phantom_select_policy_K={K}")

    print("\n=== B. Ceilings ===")
    _stats(L_unif_orc, L_top, well, f"uniform_oracle_K={K}")
    _stats(L_pol_orc,  L_top, well, f"policy_oracle_K={K}")
    _stats(L_top,      L_top, well, f"oracle_K={args.K_oracle}")

    print("\n=== C. v15 (OURS) ===")
    _stats(L_aug_pol_select,  L_top, well, f"residual_aug_policy_K={K}")
    _stats(L_aug_unif_select, L_top, well, f"residual_aug_uniform_K={K}")

    # diagnostics
    print(f"\n[diag] residual on POLICY candidates:")
    R_well = R_pol[:, well].reshape(-1)
    print(f"  R range over (K*N) = ({R_well.min():+.3f}, {R_well.max():+.3f})  "
          f"mean={R_well.mean():+.3f}  std={R_well.std():.3f}")
    real_minus_ph = (_normalize_L(L_rl_pol, np.tile(T_np, (K, 1)))
                     - _normalize_L(L_ph_pol, np.tile(T_np, (K, 1))))
    print(f"  TRUE (real - phantom) on policy: "
          f"mean={real_minus_ph.mean():+.3f}  std={real_minus_ph.std():.3f}")
    # residual prediction quality
    R_err = (R_pol - real_minus_ph)[:, well]
    print(f"  R prediction error (pred - true): "
          f"mean={R_err.mean():+.3f}  mae={np.abs(R_err).mean():.3f}  "
          f"std={R_err.std():.3f}")

    # how often does aug pick agree with oracle (best per task)
    oracle_choice = L_rl_pol.argmax(axis=0)
    aug_choice    = aug_pol.argmax(axis=0)
    ph_choice     = L_ph_pol.argmax(axis=0)
    print(f"\n[choice agreement] over {n_well} well-defined tasks:")
    print(f"  aug == oracle: {(aug_choice[well] == oracle_choice[well]).mean():.3f}")
    print(f"  ph  == oracle: {(ph_choice[well]  == oracle_choice[well]).mean():.3f}")
    print(f"  aug == ph:     {(aug_choice[well] == ph_choice[well]).mean():.3f}")


if __name__ == "__main__":
    main()
