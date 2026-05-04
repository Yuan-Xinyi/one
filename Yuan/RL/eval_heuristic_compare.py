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

    # Auto-detect Q hidden dim from checkpoint so older ckpts (h=256)
    # load even when cfg.Q_HIDDEN_DIM has been bumped (e.g., to 512 for v13).
    q_hidden = state["qnet"]["net.0.weight"].shape[0]
    qnet = QNet(cfg.STATE_DIM, env.action_dim, hidden=q_hidden).to(device)
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
def _cache_path(ckpt_path: str, n_tasks: int, K_max: int, seed: int) -> str:
    """Per-task data cache next to the ckpt, keyed by eval hyperparams."""
    base = os.path.dirname(os.path.abspath(ckpt_path))
    fname = (f"eval_cache_"
             f"{os.path.basename(ckpt_path).replace('.pt','')}_"
             f"n{n_tasks}_K{K_max}_s{seed}.npz")
    return os.path.join(base, fname)


def precompute(policy, qnet, env, n_tasks: int, K_max: int, seed: int,
               timer: _Timer, cache_path: str | None = None):
    if cache_path and os.path.exists(cache_path):
        print(f"[cache hit] loading {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        # backwards-compat: log_p_pol was added later. If missing, recompute it
        # from cached actions + policy.
        log_p_pol = d["log_p_pol"] if "log_p_pol" in d.files else None
        out = {
            "tasks": list(d["tasks"]),
            "T": d["T"], "L_det": d["L_det"],
            "L_unif": d["L_unif"], "L_pol": d["L_pol"],
            "q_pol": d["q_pol"],
            "score_manip": d["score_manip"], "score_jlm": d["score_jlm"],
            "score_manip_pol": d["score_manip_pol"],
            "score_jlm_pol":   d["score_jlm_pol"],
        }
        if log_p_pol is not None:
            out["log_p_pol"] = log_p_pol
        # backwards-compat: q_robust scores added later
        if "q_robust_min" in d.files:
            out["q_robust_min"] = d["q_robust_min"]
            out["q_robust_mean"] = d["q_robust_mean"]
        return out
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

    # ----- Policy stochastic K samples + Q score + IK + heuristic scores -----
    with timer("09_policy_K_samples"):
        rep_states_t = states_t.repeat(K_max, 1)               # (K*N, S)
        with torch.no_grad():
            a_pol_K, _ = policy.act(rep_states_t, deterministic=False)
        a_pol_K_np = a_pol_K.cpu().numpy().astype(np.float32)   # (K*N, 4)

    with timer("10_q_score_KxN"):
        with torch.no_grad():
            q_scores = qnet(rep_states_t, a_pol_K)              # (K*N,)
        q_scores_np = q_scores.cpu().numpy().reshape(K_max, n_tasks)

    # ----- log p_pol(a|s) for mode-seeking deployment -----
    # For multimodal flow policies, deterministic z=0 lands BETWEEN modes.
    # Mode-seeking via argmax log p(a|s) finds the actual peak instead.
    with timer("10b_logp_KxN"):
        with torch.no_grad():
            if hasattr(policy, "log_prob_action"):
                log_p_pol = policy.log_prob_action(rep_states_t, a_pol_K)
            else:
                log_p_pol = torch.zeros(rep_states_t.shape[0], device=device)
        log_p_pol_np = log_p_pol.cpu().numpy().reshape(K_max, n_tasks)

    # ----- Q-robust score: worst-case Q in a small (φ, ψ) ball -----
    # For each candidate, generate J angular perturbations of (φ, ψ),
    # re-encode to (cos, sin) pairs, query Q on all neighbors, then take
    # the MIN as a robustness score. Picks candidates whose neighborhood
    # is also high-Q, avoiding narrow fail-strips inside otherwise good
    # regions.
    J_robust = 8
    delta_rad = 0.15
    with timer("10c_q_robust_KxN"):
        # decode current (φ, ψ) per sample
        phi = torch.atan2(a_pol_K[:, 1], a_pol_K[:, 0])              # (KN,)
        psi = torch.atan2(a_pol_K[:, 3], a_pol_K[:, 2])              # (KN,)
        # angular perturbations
        d_phi = torch.randn(a_pol_K.shape[0], J_robust, device=device) * delta_rad
        d_psi = torch.randn(a_pol_K.shape[0], J_robust, device=device) * delta_rad
        phi_p = phi[:, None] + d_phi
        psi_p = psi[:, None] + d_psi
        a_neigh = torch.stack([torch.cos(phi_p), torch.sin(phi_p),
                                torch.cos(psi_p), torch.sin(psi_p)],
                              dim=-1)                                # (KN, J, 4)
        # batch query Q on all (state, neighbor) pairs
        st_rep = rep_states_t[:, None, :].expand(-1, J_robust, -1).reshape(-1, rep_states_t.shape[-1])
        a_flat = a_neigh.reshape(-1, 4)
        with torch.no_grad():
            q_neigh = qnet(st_rep, a_flat).view(a_pol_K.shape[0], J_robust)
        # worst-case Q over the J neighbors → (KN,)
        q_robust_min = q_neigh.min(dim=-1).values
        q_robust_mean = q_neigh.mean(dim=-1)
        q_robust_min_np = q_robust_min.cpu().numpy().reshape(K_max, n_tasks)
        q_robust_mean_np = q_robust_mean.cpu().numpy().reshape(K_max, n_tasks)

    with timer("11_rollout_pol_K"):
        L_pol_flat = _rollout_chunked(a_pol_K_np, rep_c, rep_v, rep_e, rep_T)
        L_pol_K = L_pol_flat.reshape(K_max, n_tasks)

    # IK + heuristic scores on the POLICY samples (was previously only on
    # uniform samples). Lets us score policy candidates with manip/jlm,
    # which is OURS-style: policy filters to good region, heuristic picks
    # the safest one within that region.
    with timer("12_ik_project_pol_KxN"):
        a_pol_t = torch.as_tensor(a_pol_K_np, device=device_kin, dtype=torch.float32)
        # rep_c/d/n already computed for uniform samples; reuse them since
        # both are (K*N, 3) and the task ordering matches.
        R_tgt_pol = build_branch_rotmat_batch(d_t, n_t, a_pol_t)
        q_pol_all, ik_ok_pol, _ = branch_project_multistart(
            kin, p0_t, R_tgt_pol, a_pol_t)

    with timer("13_score_manip_on_pol"):
        m_d_pol = _directional_manip(kin, q_pol_all, d_t)

    with timer("14_score_jlm_on_pol"):
        margin_pol = _joint_limit_margin(kin, q_pol_all)

    score_manip_pol = m_d_pol.view(K_max, n_tasks).cpu().numpy()
    score_jlm_pol   = margin_pol.view(K_max, n_tasks).cpu().numpy()
    ok_mat_pol      = ik_ok_pol.view(K_max, n_tasks).cpu().numpy()
    score_manip_pol = np.where(ok_mat_pol, score_manip_pol, -np.inf)
    score_jlm_pol   = np.where(ok_mat_pol, score_jlm_pol,   -np.inf)

    out = {
        "tasks":   tasks,
        "T":       T_np,
        "L_det":   L_det,
        "L_unif":  L_unif,                  # (K_max, N) uniform samples
        "L_pol":   L_pol_K,                 # (K_max, N) policy samples
        "q_pol":   q_scores_np,             # (K_max, N) Q on policy samples
        "log_p_pol": log_p_pol_np,          # (K_max, N) log p(a|s) on policy samples
        "q_robust_min":  q_robust_min_np,   # (K_max, N) worst-case Q in ball  ← NEW
        "q_robust_mean": q_robust_mean_np,  # (K_max, N) mean Q in ball        ← NEW
        "score_manip":     score_manip,
        "score_jlm":       score_jlm,
        "score_manip_pol": score_manip_pol,
        "score_jlm_pol":   score_jlm_pol,
    }
    if cache_path is not None:
        try:
            np.savez_compressed(cache_path,
                                tasks=np.asarray(tasks, dtype=object),
                                T=T_np, L_det=L_det, L_unif=L_unif,
                                L_pol=L_pol_K, q_pol=q_scores_np,
                                log_p_pol=log_p_pol_np,
                                q_robust_min=q_robust_min_np,
                                q_robust_mean=q_robust_mean_np,
                                score_manip=score_manip, score_jlm=score_jlm,
                                score_manip_pol=score_manip_pol,
                                score_jlm_pol=score_jlm_pol)
            print(f"[cache write] saved {cache_path}")
        except Exception as e:
            print(f"[cache write] failed: {e}")
    return out


def derive_strategy_lengths(out, K: int):
    """For a given K, compute L for each strategy by looking up rollouts.

    All "policy_*" strategies sample K POLICY actions then score them with
    a cheap selector and rollout only the chosen one — same deploy cost
    as q_ranked, just different scorers.
    """
    L_unif = out["L_unif"][:K]
    L_pol  = out["L_pol"][:K]
    Qs     = out["q_pol"][:K]
    sm_u   = out["score_manip"][:K]      # manip on UNIFORM
    sj_u   = out["score_jlm"][:K]        # jlm on UNIFORM
    sm_p   = out["score_manip_pol"][:K]  # manip on POLICY
    sj_p   = out["score_jlm_pol"][:K]    # jlm on POLICY
    log_p  = out.get("log_p_pol", None)
    if log_p is not None:
        log_p = log_p[:K]
    qr_min  = out.get("q_robust_min", None)
    qr_mean = out.get("q_robust_mean", None)
    if qr_min is not None:
        qr_min  = qr_min[:K]
        qr_mean = qr_mean[:K]
    n      = L_unif.shape[1]
    arange = np.arange(n)

    # uniform-based god-mode
    L_oracle = L_unif.max(axis=0)

    # ---- uniform-sample baselines ----
    no_valid_mu = ~np.isfinite(sm_u.max(axis=0))
    no_valid_ju = ~np.isfinite(sj_u.max(axis=0))
    L_manip = np.where(no_valid_mu, 0, L_unif[sm_u.argmax(axis=0), arange])
    L_jlm   = np.where(no_valid_ju, 0, L_unif[sj_u.argmax(axis=0), arange])

    # ---- policy-sample selectors ----
    L_q          = L_pol[Qs.argmax(axis=0), arange]
    no_valid_mp  = ~np.isfinite(sm_p.max(axis=0))
    no_valid_jp  = ~np.isfinite(sj_p.max(axis=0))
    L_pol_manip  = np.where(no_valid_mp, 0, L_pol[sm_p.argmax(axis=0), arange])
    L_pol_jlm    = np.where(no_valid_jp, 0, L_pol[sj_p.argmax(axis=0), arange])

    # combined: normalize each score to [0, 1] across samples-of-this-task,
    # then sum with equal weights. Q first standardized to ~[0,1].
    def _norm01(arr):
        # arr is (K, N); normalize per task to [0,1] using min/max
        a_min = arr.min(axis=0, keepdims=True)
        a_max = arr.max(axis=0, keepdims=True)
        rng = np.where(a_max - a_min > 1e-8, a_max - a_min, 1.0)
        return (arr - a_min) / rng
    Qn = _norm01(np.where(np.isfinite(Qs), Qs, -1e9))
    Mn = _norm01(np.where(np.isfinite(sm_p), sm_p, -1e9))
    Jn = _norm01(np.where(np.isfinite(sj_p), sj_p, -1e9))
    combined = Qn + Mn + Jn
    # mask IK fails (anywhere in policy samples — use any of the three)
    valid_combined = np.isfinite(sm_p) & np.isfinite(sj_p)
    combined = np.where(valid_combined, combined, -np.inf)
    no_valid_c = ~np.isfinite(combined.max(axis=0))
    L_pol_combined = np.where(no_valid_c, 0,
                              L_pol[combined.argmax(axis=0), arange])

    # ---- Mode-seeking via log_p argmax (NEW: extracts safety from policy
    # geometry alone, no external IK info needed). For multimodal flow
    # policies, deterministic z=0 lands BETWEEN modes; argmax log_p
    # finds the actual mode in the K samples.
    if log_p is not None:
        L_mode_logp = L_pol[log_p.argmax(axis=0), arange]
        # combined log_p + Q (both normalized per task to [0, 1])
        Qn = _norm01(np.where(np.isfinite(Qs), Qs, -1e9))
        Pn = _norm01(np.where(np.isfinite(log_p), log_p, -1e9))
        logp_q = Qn + Pn
        L_logp_q = L_pol[logp_q.argmax(axis=0), arange]
    else:
        L_mode_logp = L_pol[0]   # fallback (won't be used)
        L_logp_q   = L_pol[0]

    # ---- Q-robust selection (NEW): pick action whose ε-ball has high
    # worst-case (or mean) Q. Avoids "policy mean lands in narrow
    # fail-strip surrounded by good region" failure mode.
    if qr_min is not None:
        L_q_robust_min  = L_pol[qr_min.argmax(axis=0),  arange]
        L_q_robust_mean = L_pol[qr_mean.argmax(axis=0), arange]
    else:
        L_q_robust_min  = L_pol[0]
        L_q_robust_mean = L_pol[0]

    # ceiling: if Q ranked perfectly, q_ranked == pol_stoch_orc
    L_pol_orc = L_pol.max(axis=0)

    return {
        "uniform_oracle":  L_oracle,
        "manip_select":    L_manip,
        "jlm_select":      L_jlm,
        "q_ranked":        L_q,
        "policy_manip":    L_pol_manip,
        "policy_jlm":      L_pol_jlm,
        "policy_combined": L_pol_combined,
        "policy_mode_logp": L_mode_logp,
        "policy_logp_q":   L_logp_q,
        "policy_q_robust_min":  L_q_robust_min,    # ← NEW: argmax min Q in ball
        "policy_q_robust_mean": L_q_robust_mean,   # ← NEW: argmax mean Q in ball
        "pol_stoch_orc":   L_pol_orc,
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


def report(out, K_list, deploy_times,
           min_oracle_dist: float, min_base_dist: float):
    T = out["T"]
    L_det = out["L_det"]
    L_unif = out["L_unif"]
    K_max = L_unif.shape[0]
    L_top = L_unif.max(axis=0)              # K=K_max baseline
    v_path = np.array([t["v_path"] for t in out["tasks"]], dtype=np.float64)
    oracle_dist = L_top.astype(np.float64) * float(cfg.DT) * v_path
    p0 = np.stack([t["c"][:3] for t in out["tasks"]]).astype(np.float64)
    base_dist = np.linalg.norm(p0, axis=-1)
    cond_oracle = oracle_dist >= min_oracle_dist
    cond_base   = base_dist  >= min_base_dist
    well = cond_oracle & cond_base

    print(f"\n=== {len(T)} tasks; K_max={K_max} ===")
    print(f"feasible     (L_top > 0)                                : "
          f"{int((L_top > 0).sum())}/{len(T)}")
    print(f"well-defined (oracle TCP >= {min_oracle_dist*100:.0f}cm AND ||p0|| "
          f">= {min_base_dist*100:.0f}cm) : {int(well.sum())}/{len(T)}     ← used for stats")

    # main table: uniform-sample baselines vs OURS (policy-sample selectors)
    print("\n=== A. Uniform-sample baselines (random + cheap selector) ===")
    header_a = (f"  {'K':>5}  {'unif_orc':>9}  {'manip_u':>9}  {'jlm_u':>9}")
    print(header_a)
    print("  " + "-" * (len(header_a) - 2))
    for K in K_list:
        if K > K_max: continue
        d = derive_strategy_lengths(out, K)
        r_orc = _ratio_mean(d["uniform_oracle"],  L_top, mask=well)
        r_m   = _ratio_mean(d["manip_select"],    L_top, mask=well)
        r_j   = _ratio_mean(d["jlm_select"],      L_top, mask=well)
        print(f"  {K:>5d}  {r_orc:>9.4f}  {r_m:>9.4f}  {r_j:>9.4f}")

    print("\n=== B. Policy-sample selectors (OURS, K policy samples + cheap score + 1 rollout) ===")
    r_pd = _ratio_mean(L_det, L_top, mask=well)
    print(f"  reference: policy_det = {r_pd:.4f}  (K-independent)")
    header_b = (f"  {'K':>5}  {'q_ranked':>9}  {'q_robmin':>9}  "
                f"{'q_robmen':>9}  {'pol_jlm':>9}  {'pol_combd':>10}  "
                f"{'pol_orc(ceil)':>14}")
    print(header_b)
    print("  " + "-" * (len(header_b) - 2))
    for K in K_list:
        if K > K_max: continue
        d = derive_strategy_lengths(out, K)
        r_q    = _ratio_mean(d["q_ranked"],              L_top, mask=well)
        r_qrmn = _ratio_mean(d["policy_q_robust_min"],   L_top, mask=well)
        r_qrme = _ratio_mean(d["policy_q_robust_mean"],  L_top, mask=well)
        r_pj   = _ratio_mean(d["policy_jlm"],            L_top, mask=well)
        r_pc   = _ratio_mean(d["policy_combined"],       L_top, mask=well)
        r_pso  = _ratio_mean(d["pol_stoch_orc"],         L_top, mask=well)
        print(f"  {K:>5d}  {r_q:>9.4f}  {r_qrmn:>9.4f}  "
              f"{r_qrme:>9.4f}  {r_pj:>9.4f}  {r_pc:>10.4f}  {r_pso:>14.4f}")
    print("\n  q_robmin = argmax_{k} min_{δ in ball} Q(s, a_k+δ)  worst-case neighborhood Q")
    print("  q_robmen = argmax_{k} mean_{δ in ball} Q(s, a_k+δ)  expected neighborhood Q")
    print("\n  pol_orc is the ceiling: if scorer ranked perfectly, the chosen "
          "L equals pol_orc.")
    print("  pol_combd = normalized(Q) + normalized(manip) + normalized(jlm), "
          "argmax per task.")

    print("\n  K=K_max corresponds to the true oracle; ratios for "
          "uniform_oracle approach 1.0 there by definition.")

    # per-task percentile rank of each method in the K_max distribution
    p50 = np.percentile(L_unif, 50, axis=0)
    p90 = np.percentile(L_unif, 90, axis=0)
    pmax = L_unif.max(axis=0)
    # ---- LOWER-BOUND focus: per-strategy ratio distribution ----
    # For paper writing, the WORST cases matter more than the mean.
    # Show per-strategy: mean, std, min, p10, p25, # of L=0 catastrophic
    # failures, # of ratio<0.3 bad cases. All at K=K_max.
    d_max = derive_strategy_lengths(out, K_max)
    print("\n=== C. Per-strategy LOWER-BOUND stats at K=K_max  (well-defined n)===")
    print("    'L=0' = catastrophic policy failure | 'r<0.3' = bad recovery")
    print(f"  {'method':>26}  {'mean':>6}  {'std':>6}  {'min':>5}  "
          f"{'p10':>5}  {'p25':>5}  {'L=0':>4}  {'r<0.3':>5}")
    safe_top = np.where(L_top > 0, L_top, 1).astype(np.float64)
    for name, L in [("policy_det",                    L_det),
                    ("uniform_oracle K_max",          d_max["uniform_oracle"]),
                    ("manip_select K_max",            d_max["manip_select"]),
                    ("jlm_select K_max",              d_max["jlm_select"]),
                    ("q_ranked K_max",                d_max["q_ranked"]),
                    ("policy_manip K_max",            d_max["policy_manip"]),
                    ("policy_jlm K_max",              d_max["policy_jlm"]),
                    ("policy_combined K_max",         d_max["policy_combined"]),
                    ("policy_mode_logp K_max",        d_max["policy_mode_logp"]),
                    ("policy_logp_q K_max",           d_max["policy_logp_q"]),
                    ("policy_q_robust_min K_max",     d_max["policy_q_robust_min"]),
                    ("policy_q_robust_mean K_max",    d_max["policy_q_robust_mean"]),
                    ("pol_stoch_orc K_max (ceiling)", d_max["pol_stoch_orc"])]:
        L_w = L[well].astype(np.float64)
        ratios = L_w / safe_top[well]
        n_zero  = int((L_w == 0).sum())
        n_bad   = int((ratios < 0.3).sum())
        print(f"  {name:>26}  {ratios.mean():>6.3f}  {ratios.std():>6.3f}  "
              f"{ratios.min():>5.3f}  {np.percentile(ratios, 10):>5.3f}  "
              f"{np.percentile(ratios, 25):>5.3f}  {n_zero:>4d}  {n_bad:>5d}")

    print("\nfraction of well-defined tasks where each method >= the "
          "K_max-distribution percentile:")
    print(f"  {'method':>26}  {'>= median':>10}  {'>= p90':>8}  {'== max':>8}")
    # d_max already computed above for section C
    for name, L in [("policy_det",                    L_det),
                    ("uniform_oracle K_max",          d_max["uniform_oracle"]),
                    ("manip_select K_max",            d_max["manip_select"]),
                    ("jlm_select K_max",              d_max["jlm_select"]),
                    ("q_ranked K_max",                d_max["q_ranked"]),
                    ("policy_manip K_max  (OURS)",    d_max["policy_manip"]),
                    ("policy_jlm K_max    (OURS)",    d_max["policy_jlm"]),
                    ("policy_combined K_max (OURS)",  d_max["policy_combined"]),
                    ("pol_stoch_orc K_max  (ceiling)", d_max["pol_stoch_orc"])]:
        f = well
        print(f"  {name:>26}  "
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
    ap.add_argument("--min-base-dist", type=float, default=0.30,
                    help="drop tasks whose start point p0 is closer than "
                         "this many meters to the FR3 base origin (no "
                         "swivel freedom inside near-base shell). default 0.30 m.")
    args = ap.parse_args()

    K_list = [int(k) for k in args.k_list.split(",") if k.strip()]
    K_max = max(K_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    print(f"loading {args.ckpt}")
    policy, qnet = _load_policy_and_q(args.ckpt, env, device)

    timer = _Timer()
    cache = _cache_path(args.ckpt, args.n_tasks, K_max, args.seed)
    out = precompute(policy, qnet, env, args.n_tasks, K_max, args.seed, timer,
                     cache_path=cache)

    deploy_times: dict[str, float] = {}
    if not args.no_deploy_timing:
        K_for_deploy = K_max
        for strat in ("policy_det", "q_ranked",
                      "uniform_oracle", "manip_select", "jlmargin_select"):
            label = strat if strat == "policy_det" else f"{strat} K={K_for_deploy}"
            deploy_times[label] = time_strategy_deploy(
                strat, policy, qnet, env, args.n_tasks, K_for_deploy, args.seed)

    report(out, K_list, deploy_times,
           min_oracle_dist=args.min_oracle_distance,
           min_base_dist=args.min_base_dist)
    report_pipeline(timer)


if __name__ == "__main__":
    main()
