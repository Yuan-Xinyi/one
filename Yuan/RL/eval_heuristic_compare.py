"""Compare policy vs deterministic heuristic seed-selection baselines.

All metrics are RATIOS of recovered length against the K=1000 uniform
upper bound. Absolute step counts and L>=T success indicators are not
reported (many tasks are intrinsically infeasible to complete fully).

Strategies compared
-------------------
  policy_det           : policy.act(deterministic=True), rollout once
  uniform_oracle K=N   : sample N uniform (phi, psi); rollout all N;
                         pick the (phi, psi) whose rollout went furthest.
  manip_select K=N     : sample N uniform (phi, psi); IK each candidate;
                         pick the one with highest directional manipulability
                         m_d = ||J_v^T d|| at q_init; rollout ONLY that 1.
  jlmargin_select K=N  : sample N uniform (phi, psi); IK each candidate;
                         pick the one with the largest min normalized
                         joint-limit margin at q_init; rollout ONLY that 1.

Both heuristics see the same K candidates as uniform_oracle; the diff is
that the oracle has god-mode (rolls out all K to learn which is best),
while the heuristics commit to a choice from a CHEAP scalar score and
then pay only one rollout. The wall-time breakdown shows what that buys.

Usage
-----
    python -m Yuan.RL.eval_heuristic_compare
    python -m Yuan.RL.eval_heuristic_compare --n-tasks 32 --k-list 8,128,1000
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.qnet import QNet
from Yuan.RL.batched_rollout import (
    batched_rollout, branch_project_multistart, build_branch_rotmat_batch,
    _device_from_cfg,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timer:
    """Accumulate wall-time per named stage."""
    def __init__(self):
        self.totals: dict[str, float] = {}

    def __call__(self, name: str):
        return _StageCtx(self, name)

    def add(self, name: str, dt: float):
        self.totals[name] = self.totals.get(name, 0.0) + dt


class _StageCtx:
    def __init__(self, owner: _Timer, name: str):
        self.owner = owner
        self.name = name
    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        _sync()
        self.owner.add(self.name, time.perf_counter() - self.t0)


def _load_policy_and_q(ckpt_path, env, device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()

    qnet = QNet(cfg.STATE_DIM, env.action_dim).to(device)
    qnet.load_state_dict(state["qnet"])
    qnet.eval()
    return policy, qnet


def _load_policy(ckpt_path, env, device):
    return _load_policy_and_q(ckpt_path, env, device)[0]


def _rollout_chunked(actions_np, c_np, v_np, e_np, T_np, chunk=4096):
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out["lengths"], dtype=np.int32)
    return L


# ---------------------------------------------------------------------------
# heuristic scores at the IK-projected initial configuration
# ---------------------------------------------------------------------------
def _directional_manip(kin: BatchedFR3Kinematics,
                       q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """m_d = || J_v^T d || at q. Higher = better TCP speed capability along d."""
    _, _, J, _ = kin.tcp_fk_jac(q)        # J: (B, 6, 7)
    Jv = J[:, 0:3, :]                      # translational part
    # J_v^T d  shape (B, 7)
    Jt_d = (Jv.transpose(1, 2) @ d.unsqueeze(-1)).squeeze(-1)
    return Jt_d.norm(dim=-1)               # (B,)


def _joint_limit_margin(kin: BatchedFR3Kinematics,
                        q: torch.Tensor) -> torch.Tensor:
    """min over joints of normalized distance to nearest limit, in [0, 0.5]."""
    span = (kin.lmt_up - kin.lmt_lo).clamp_min(1e-6)             # (7,)
    lo_d = (q - kin.lmt_lo) / span                                # (B, 7)
    up_d = (kin.lmt_up - q) / span                                # (B, 7)
    per_joint = torch.minimum(lo_d, up_d)                         # (B, 7)
    return per_joint.min(dim=-1).values                           # (B,)


# ---------------------------------------------------------------------------
# core evaluation: precompute everything once, derive all strategies
# ---------------------------------------------------------------------------
def precompute(policy, qnet, env, n_tasks: int, K_max: int, seed: int, timer: _Timer):
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed)

    with timer("01_sample_tasks"):
        tasks = env._sample_tasks(n_tasks)
        states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
        c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
        v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
        e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
        T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)

    with timer("02_policy_inference_det"):
        with torch.no_grad():
            a_det, _ = policy.act(states_t, deterministic=True)
        a_det_np = a_det.cpu().numpy().astype(np.float32)

    with timer("03_policy_rollout"):
        L_det = _rollout_chunked(a_det_np, c_np, v_np, e_np, T_np)

    with timer("04_uniform_sample_K"):
        phi = rng.uniform(0.0, 2 * np.pi, size=(K_max, n_tasks)).astype(np.float32)
        psi = rng.uniform(0.0, 2 * np.pi, size=(K_max, n_tasks)).astype(np.float32)
        a_unif = np.stack([np.cos(phi), np.sin(phi),
                           np.cos(psi), np.sin(psi)], axis=-1)    # (K, N, 4)
        a_unif_flat = a_unif.reshape(K_max * n_tasks, 4).astype(np.float32)
        rep_c = np.tile(c_np, (K_max, 1))
        rep_v = np.tile(v_np, K_max)
        rep_e = np.tile(e_np, K_max)
        rep_T = np.tile(T_np, K_max)

    with timer("05_uniform_rollout_full_KxN"):
        L_flat = _rollout_chunked(a_unif_flat, rep_c, rep_v, rep_e, rep_T)
        L_unif = L_flat.reshape(K_max, n_tasks)

    # IK + heuristic scores at the K_max*N initial configurations.
    # Heuristics CAN use these scores to pick a candidate without rolling
    # out anything beyond the chosen one.
    with timer("06_ik_project_KxN"):
        device_kin = _device_from_cfg()
        kin = BatchedFR3Kinematics(device=device_kin)
        c_t = torch.as_tensor(rep_c, device=device_kin, dtype=torch.float32)
        a_t = torch.as_tensor(a_unif_flat, device=device_kin, dtype=torch.float32)
        p0_t = c_t[:, :3]
        d_t  = c_t[:, 3:6]
        n_t  = c_t[:, 6:9]
        R_tgt = build_branch_rotmat_batch(d_t, n_t, a_t)
        q_all, ik_ok, _ = branch_project_multistart(kin, p0_t, R_tgt, a_t)
        # q_all: (K*N, 7),  ik_ok: (K*N,) bool

    with timer("07_score_manipulability"):
        m_d = _directional_manip(kin, q_all, d_t)                 # (K*N,)

    with timer("08_score_joint_margin"):
        margin = _joint_limit_margin(kin, q_all)                  # (K*N,)

    # reshape scores to (K, N), mask out IK failures
    score_manip = m_d.view(K_max, n_tasks).cpu().numpy()
    score_jlm   = margin.view(K_max, n_tasks).cpu().numpy()
    ok_mat      = ik_ok.view(K_max, n_tasks).cpu().numpy()
    score_manip = np.where(ok_mat, score_manip, -np.inf)
    score_jlm   = np.where(ok_mat, score_jlm,   -np.inf)

    # ----- Q-ranked: K policy stochastic samples + Q score per (s, a) -----
    # Sample K policy actions per task (replicate state K times along
    # the sample dim).
    with timer("09_policy_K_samples"):
        rep_states_t = states_t.repeat(K_max, 1)               # (K*N, S)
        with torch.no_grad():
            a_pol_K, _ = policy.act(rep_states_t, deterministic=False)
        a_pol_K_np = a_pol_K.cpu().numpy().astype(np.float32)   # (K*N, 4)

    with timer("10_q_score_KxN"):
        with torch.no_grad():
            q_scores = qnet(rep_states_t, a_pol_K)              # (K*N,)
        q_scores_np = q_scores.cpu().numpy().reshape(K_max, n_tasks)

    with timer("11_rollout_pol_K"):
        L_pol_flat = _rollout_chunked(a_pol_K_np, rep_c, rep_v, rep_e, rep_T)
        L_pol_K = L_pol_flat.reshape(K_max, n_tasks)

    return {
        "tasks":   tasks,
        "T":       T_np,
        "L_det":   L_det,
        "L_unif":  L_unif,            # (K_max, N) — uniform random samples
        "L_pol":   L_pol_K,           # (K_max, N) — policy stochastic samples
        "q_pol":   q_scores_np,       # (K_max, N) — Q values for those policy samples
        "score_manip": score_manip,   # (K_max, N) — heuristic score on UNIFORM samples
        "score_jlm":   score_jlm,     # (K_max, N) — heuristic score on UNIFORM samples
    }


def derive_strategy_lengths(out, K: int):
    """For a given K, compute L for each strategy by looking up rollouts.

    Returns dict with keys:
      uniform_oracle: K uniform samples, pick best L (god-mode)
      manip_select:   K uniform samples, IK + manip score, pick argmax, look up L
      jlm_select:     K uniform samples, IK + joint-margin score, pick argmax
      q_ranked:       K policy stochastic samples, Q-net score, pick argmax  ← OURS
      pol_stoch_orc:  K policy stochastic samples, pick best L (q_ranked ceiling)
    """
    L_unif = out["L_unif"][:K]
    L_pol  = out["L_pol"][:K]
    Qs     = out["q_pol"][:K]
    sm     = out["score_manip"][:K]
    sj     = out["score_jlm"][:K]
    n      = L_unif.shape[1]
    arange = np.arange(n)

    # uniform-based god-mode
    L_oracle = L_unif.max(axis=0)

    # uniform + heuristic score (mask IK-failed rows)
    no_valid_m = ~np.isfinite(sm.max(axis=0))
    no_valid_j = ~np.isfinite(sj.max(axis=0))
    L_manip = L_unif[sm.argmax(axis=0), arange]
    L_jlm   = L_unif[sj.argmax(axis=0), arange]
    L_manip = np.where(no_valid_m, 0, L_manip)
    L_jlm   = np.where(no_valid_j, 0, L_jlm)

    # policy stochastic + Q score
    L_q     = L_pol[Qs.argmax(axis=0), arange]
    # ceiling: if Q ranked perfectly, q_ranked == pol_stoch_orc
    L_pol_orc = L_pol.max(axis=0)

    return {
        "uniform_oracle": L_oracle,
        "manip_select":   L_manip,
        "jlm_select":     L_jlm,
        "q_ranked":       L_q,
        "pol_stoch_orc":  L_pol_orc,
    }


# ---------------------------------------------------------------------------
# realistic deployed-time measurement (not precomputed)
# ---------------------------------------------------------------------------
def time_strategy_deploy(strategy: str,
                         policy, qnet, env, n_tasks: int, K: int, seed: int):
    """Run one strategy in the way it would be deployed (sample what it
    needs, do the smallest amount of rollout it needs). Returns wall_time."""
    device = next(policy.parameters()).device
    device_kin = _device_from_cfg()
    rng = np.random.default_rng(seed + 7919)         # different seed for this run
    env.rng = np.random.default_rng(seed + 7919)
    tasks = env._sample_tasks(n_tasks)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)

    if strategy == "policy_det":
        _sync(); t0 = time.perf_counter()
        st = torch.as_tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            a, _ = policy.act(st, deterministic=True)
        a_np = a.cpu().numpy().astype(np.float32)
        _ = _rollout_chunked(a_np, c_np, v_np, e_np, T_np)
        _sync()
        return time.perf_counter() - t0

    if strategy == "uniform_oracle":
        _sync(); t0 = time.perf_counter()
        phi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        psi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        a = np.stack([np.cos(phi), np.sin(phi),
                      np.cos(psi), np.sin(psi)], axis=-1)
        a_flat = a.reshape(K * n_tasks, 4).astype(np.float32)
        rep_c = np.tile(c_np, (K, 1)); rep_v = np.tile(v_np, K)
        rep_e = np.tile(e_np, K); rep_T = np.tile(T_np, K)
        L = _rollout_chunked(a_flat, rep_c, rep_v, rep_e, rep_T)
        _ = L.reshape(K, n_tasks).max(axis=0)
        _sync()
        return time.perf_counter() - t0

    if strategy == "q_ranked":
        # 1) sample K policy actions per task
        # 2) Q-net score each
        # 3) argmax → pick 1 per task
        # 4) rollout that 1
        _sync(); t0 = time.perf_counter()
        st = torch.as_tensor(states, dtype=torch.float32, device=device)
        st_rep = st.repeat(K, 1)
        with torch.no_grad():
            a, _ = policy.act(st_rep, deterministic=False)        # (K*N, 4)
            q = qnet(st_rep, a)                                   # (K*N,)
        a = a.view(K, n_tasks, 4)
        q = q.view(K, n_tasks)
        idx = q.argmax(dim=0).cpu().numpy()
        a_chosen = a[idx, torch.arange(n_tasks, device=device)].cpu().numpy().astype(np.float32)
        _ = _rollout_chunked(a_chosen, c_np, v_np, e_np, T_np)
        _sync()
        return time.perf_counter() - t0

    if strategy in ("manip_select", "jlmargin_select"):
        _sync(); t0 = time.perf_counter()
        phi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        psi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        a = np.stack([np.cos(phi), np.sin(phi),
                      np.cos(psi), np.sin(psi)], axis=-1)
        a_flat = a.reshape(K * n_tasks, 4).astype(np.float32)
        rep_c = np.tile(c_np, (K, 1)); rep_v = np.tile(v_np, K)
        rep_e = np.tile(e_np, K); rep_T = np.tile(T_np, K)

        kin = BatchedFR3Kinematics(device=device_kin)
        c_t = torch.as_tensor(rep_c, device=device_kin, dtype=torch.float32)
        a_t = torch.as_tensor(a_flat, device=device_kin, dtype=torch.float32)
        p0_t = c_t[:, :3]; d_t = c_t[:, 3:6]; n_t = c_t[:, 6:9]
        R_tgt = build_branch_rotmat_batch(d_t, n_t, a_t)
        q_all, ik_ok, _ = branch_project_multistart(kin, p0_t, R_tgt, a_t)

        if strategy == "manip_select":
            score = _directional_manip(kin, q_all, d_t)
        else:
            score = _joint_limit_margin(kin, q_all)
        score = torch.where(ik_ok, score, torch.full_like(score, float('-inf')))
        score = score.view(K, n_tasks)
        idx = score.argmax(dim=0).cpu().numpy()                    # (N,)

        # gather chosen action per task and rollout 1 each
        a_grid = a.reshape(K, n_tasks, 4)
        a_chosen = a_grid[idx, np.arange(n_tasks)].astype(np.float32)
        _ = _rollout_chunked(a_chosen, c_np, v_np, e_np, T_np)
        _sync()
        return time.perf_counter() - t0

    raise ValueError(strategy)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _ratio_mean(num: np.ndarray, den: np.ndarray, mask=None) -> float:
    base = (den > 0) if mask is None else (mask & (den > 0))
    if not base.any():
        return float('nan')
    return float((num[base] / den[base]).mean())


def report(out, K_list, deploy_times, min_oracle_dist: float):
    T = out["T"]
    L_det = out["L_det"]
    L_unif = out["L_unif"]
    K_max = L_unif.shape[0]
    L_top = L_unif.max(axis=0)              # K=K_max baseline
    v_path = np.array([t["v_path"] for t in out["tasks"]], dtype=np.float64)
    oracle_dist = L_top.astype(np.float64) * float(cfg.DT) * v_path
    well = oracle_dist >= min_oracle_dist

    print(f"\n=== {len(T)} tasks; K_max={K_max} ===")
    print(f"feasible     (L_top > 0)                       : {int((L_top > 0).sum())}/{len(T)}")
    print(f"well-defined (oracle TCP distance >= {min_oracle_dist*100:.0f} cm)  : "
          f"{int(well.sum())}/{len(T)}     ← used for stats below")

    # main table: strategy x K -> ratio vs K_max upper bound
    print("\nratio of recovered length vs uniform oracle K=K_max  (avg over well-defined):")
    header = (f"  {'K':>5}  {'policy_det':>11}  {'unif_oracle':>12}  "
              f"{'manip_K':>9}  {'jlm_K':>9}  {'q_ranked_K':>11}  "
              f"{'pol_stoch_orc':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for K in K_list:
        if K > K_max: continue
        d = derive_strategy_lengths(out, K)
        r_pd  = _ratio_mean(L_det,                L_top, mask=well)
        r_orc = _ratio_mean(d["uniform_oracle"],  L_top, mask=well)
        r_m   = _ratio_mean(d["manip_select"],    L_top, mask=well)
        r_j   = _ratio_mean(d["jlm_select"],      L_top, mask=well)
        r_q   = _ratio_mean(d["q_ranked"],        L_top, mask=well)
        r_pso = _ratio_mean(d["pol_stoch_orc"],   L_top, mask=well)
        print(f"  {K:>5d}  {r_pd:>11.4f}  {r_orc:>12.4f}  "
              f"{r_m:>9.4f}  {r_j:>9.4f}  {r_q:>11.4f}  {r_pso:>14.4f}")
    print("\n  policy_det is K-independent (shown for reference).")
    print("  q_ranked = OURS: K policy samples + Q-net argmax + 1 rollout. "
          "No K-fold rollouts.")
    print("  pol_stoch_orc is the ceiling for q_ranked: if Q ranked perfectly, "
          "q_ranked = pol_stoch_orc.")

    print("\n  K=K_max corresponds to the true oracle; ratios for "
          "uniform_oracle approach 1.0 there by definition.")

    # per-task percentile rank of each method in the K_max distribution
    p50 = np.percentile(L_unif, 50, axis=0)
    p90 = np.percentile(L_unif, 90, axis=0)
    pmax = L_unif.max(axis=0)
    print("\nfraction of well-defined tasks where each method >= the "
          "K_max-distribution percentile:")
    print(f"  {'method':>24}  {'>= median':>10}  {'>= p90':>8}  {'== max':>8}")
    d_max = derive_strategy_lengths(out, K_max)
    for name, L in [("policy_det",                  L_det),
                    ("uniform_oracle K_max",        d_max["uniform_oracle"]),
                    ("manip_select K_max",          d_max["manip_select"]),
                    ("jlm_select K_max",            d_max["jlm_select"]),
                    ("q_ranked K_max (OURS)",       d_max["q_ranked"]),
                    ("pol_stoch_orc K_max",         d_max["pol_stoch_orc"])]:
        f = well
        print(f"  {name:>24}  "
              f"{(L[f] >= p50[f]).mean():>10.3f}  "
              f"{(L[f] >= p90[f]).mean():>8.3f}  "
              f"{(L[f] >= pmax[f]).mean():>8.3f}")

    # deploy timings
    print("\ndeployed wall-time per strategy (for the timing batch only):")
    print(f"  {'strategy':>22}  {'wall (s)':>10}")
    for k_label, secs in deploy_times.items():
        print(f"  {k_label:>22}  {secs:>10.3f}")


def report_pipeline(timer: _Timer):
    print("\npipeline stage timings (single precompute pass):")
    total = sum(timer.totals.values())
    print(f"  {'stage':>32}  {'wall (s)':>10}  {'frac':>6}")
    for name, dt in sorted(timer.totals.items()):
        frac = dt / total if total > 0 else 0.0
        print(f"  {name:>32}  {dt:>10.3f}  {frac:>6.1%}")
    print(f"  {'TOTAL':>32}  {total:>10.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--n-tasks", type=int, default=32)
    ap.add_argument("--k-list", type=str, default="8,32,128,1000",
                    help="comma-separated K values to evaluate")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-deploy-timing", action="store_true",
                    help="skip the per-strategy deploy timing pass")
    ap.add_argument("--min-oracle-distance", type=float, default=0.20,
                    help="drop tasks whose K_max oracle TCP path is shorter "
                         "than this many METERS (intrinsically infeasible). "
                         "default 0.20 m (20 cm).")
    args = ap.parse_args()

    K_list = [int(k) for k in args.k_list.split(",") if k.strip()]
    K_max = max(K_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    print(f"loading {args.ckpt}")
    policy, qnet = _load_policy_and_q(args.ckpt, env, device)

    timer = _Timer()
    out = precompute(policy, qnet, env, args.n_tasks, K_max, args.seed, timer)

    deploy_times: dict[str, float] = {}
    if not args.no_deploy_timing:
        K_for_deploy = K_max
        for strat in ("policy_det", "q_ranked",
                      "uniform_oracle", "manip_select", "jlmargin_select"):
            label = strat if strat == "policy_det" else f"{strat} K={K_for_deploy}"
            deploy_times[label] = time_strategy_deploy(
                strat, policy, qnet, env, args.n_tasks, K_for_deploy, args.seed)

    report(out, K_list, deploy_times, min_oracle_dist=args.min_oracle_distance)
    report_pipeline(timer)


if __name__ == "__main__":
    main()
