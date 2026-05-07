"""Find a task in the "21% alive but needs branch switching" category and
trace the actual q-sequence chosen by Backward DP.

Algorithm
---------
For each task we ALSO compute backward DP at two thresholds:
  thresh_loose = 10.0  → naive forward feasibility (no specific target)
  thresh_tight = 2.0   → "branch matching" feasibility

Tasks of interest:
  ∃ q₀ such that  in_S0_loose[q₀] = True  AND  in_S0_tight[q₀] = True
  AND the loose-DP would have chosen a different chain than tight-DP
    (= controller naturally drifts to one branch family but tight requires another)

For one such task, extract the q-sequence backward DP picked (under tight)
and identify branch transitions via sign changes in joint 1 / 4 / 6.

Branch identification
---------------------
J1 (shoulder yaw): + → arm pointing one side, − → other
J4 (elbow):        + → up,    − → down (FR3 has J4 typically negative when pointing forward)
J6 (wrist flex):   + → normal, − → flipped
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    batched_rollout_segment, build_branch_rotmat_batch,
    _batched_ik_project, _device_from_cfg, _load_fr3_sphere_collision_cls,
    phantom_rollout,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _dense_ik_at(kin, p_target, R_target, branch_action, M, rng):
    device = kin.device
    lo = kin.lmt_lo.cpu().numpy()
    hi = kin.lmt_up.cpu().numpy()
    seeds_np = rng.uniform(lo[None, :], hi[None, :],
                           size=(M, 7)).astype(np.float32)
    q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
    p_rep = p_target.unsqueeze(0).expand(M, 3)
    R_rep = R_target.unsqueeze(0).expand(M, 3, 3)
    a_rep = branch_action.unsqueeze(0).expand(M, 4)
    q, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=a_rep)
    return q[ok]


def _check_transitions(kin, q_curr, q_target, R_tgt, a_t, p0, d_dir,
                       v_path_s, eps_p_s, T_s, start_step, end_step,
                       sphere_cc, q_dist_thresh, chunk=4096):
    device = kin.device
    Mc = q_curr.shape[0]
    Mn = q_target.shape[0]
    if Mc == 0 or Mn == 0:
        return torch.zeros(Mc, Mn, device=device, dtype=torch.bool)
    j_idx = torch.arange(Mc, device=device).view(Mc, 1).expand(Mc, Mn).reshape(-1)
    k_idx = torch.arange(Mn, device=device).view(1, Mn).expand(Mc, Mn).reshape(-1)
    q_init_flat = q_curr[j_idx]
    q_targ_flat = q_target[k_idx]
    n_pairs = Mc * Mn
    R_flat   = R_tgt.unsqueeze(0).expand(n_pairs, 3, 3)
    a_flat   = a_t.unsqueeze(0).expand(n_pairs, 4)
    p0_flat  = p0.unsqueeze(0).expand(n_pairs, 3)
    d_flat   = d_dir.unsqueeze(0).expand(n_pairs, 3)
    v_flat   = torch.full((n_pairs,), float(v_path_s), device=device, dtype=torch.float32)
    eps_flat = torch.full((n_pairs,), float(eps_p_s),  device=device, dtype=torch.float32)
    T_flat   = torch.full((n_pairs,), int(T_s),         device=device, dtype=torch.long)
    success_flat = torch.zeros(n_pairs, device=device, dtype=torch.bool)
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        out = batched_rollout_segment(
            q_init_flat[s:e], R_flat[s:e], a_flat[s:e],
            p0_flat[s:e], d_flat[s:e],
            v_flat[s:e], eps_flat[s:e], T_flat[s:e],
            start_step=start_step, end_step=end_step,
            sphere_cc=sphere_cc, kin=kin, is_phantom=False,
            q_ref=q_targ_flat[s:e])
        q_final = out['q_final']
        alive = out['alive_out']
        q_diff = (q_final - q_targ_flat[s:e]).norm(dim=-1)
        success_flat[s:e] = alive & (q_diff < q_dist_thresh)
    return success_flat.view(Mc, Mn)


def backward_dp(kin, sphere_cc, task, args, rng, q_dist_thresh):
    """Returns (Q, in_S, success_mats) for analysis & tracing."""
    device = kin.device
    c = task['c']
    p0 = torch.as_tensor(c[:3], device=device, dtype=torch.float32)
    d_dir = torch.as_tensor(c[3:6], device=device, dtype=torch.float32)
    n_out = torch.as_tensor(c[6:9], device=device, dtype=torch.float32)
    v_path = float(task['v_path'])
    eps_p  = float(task['eps_p'])
    T_eff  = int(args.T_eval) if args.T_eval is not None else int(task['T'])
    N = int(args.n_segments)
    M_over = int(args.M_oversample)

    # phantom-best (φ, ψ)
    K_pp = 8
    phi = rng.uniform(0, 2*np.pi, size=K_pp).astype(np.float32)
    psi = rng.uniform(0, 2*np.pi, size=K_pp).astype(np.float32)
    a_cands = np.stack([np.cos(phi), np.sin(phi),
                        np.cos(psi), np.sin(psi)], axis=-1)
    a_cands_flat = a_cands.astype(np.float32)
    c_rep = np.tile(c[None, :], (K_pp, 1)).astype(np.float32)
    v_rep = np.full(K_pp, v_path, dtype=np.float32)
    e_rep = np.full(K_pp, eps_p,  dtype=np.float32)
    T_rep = np.full(K_pp, T_eff,  dtype=np.int32)
    ph_out = phantom_rollout(a_cands_flat, c_rep, v_rep, e_rep, T_rep)
    L_ph = np.asarray(ph_out['lengths'], dtype=np.int32)
    best_pp = int(L_ph.argmax())
    a_t = torch.as_tensor(a_cands[best_pp], device=device, dtype=torch.float32)
    R_tgt = build_branch_rotmat_batch(d_dir.unsqueeze(0), n_out.unsqueeze(0),
                                      a_t.unsqueeze(0)).squeeze(0)

    # dense IK at each checkpoint
    Q = []
    for i in range(N + 1):
        t_step = i * T_eff // N
        x_i = p0 + (t_step * float(cfg.DT)) * v_path * d_dir
        Q.append(_dense_ik_at(kin, x_i, R_tgt, a_t, M_over, rng))

    # if any checkpoint has 0 IK, abort
    sizes = [q.shape[0] for q in Q]
    if min(sizes) == 0:
        return None

    # transition matrices per segment + DP
    success_mats = []
    in_S = [None] * (N + 1)
    in_S[N] = torch.ones(Q[N].shape[0], device=device, dtype=torch.bool)
    for i in range(N):
        t_start = i * T_eff // N
        t_end   = (i + 1) * T_eff // N
        succ = _check_transitions(
            kin, Q[i], Q[i + 1], R_tgt, a_t, p0, d_dir,
            v_path, eps_p, T_eff,
            start_step=t_start, end_step=t_end,
            sphere_cc=sphere_cc, q_dist_thresh=q_dist_thresh,
            chunk=args.chunk_size)
        success_mats.append(succ)
    # backward
    for i in range(N - 1, -1, -1):
        succ_i = success_mats[i]
        v_next = in_S[i + 1].unsqueeze(0).expand(Q[i].shape[0], -1)
        in_S[i] = (succ_i & v_next).any(dim=1)

    return {
        'Q': Q, 'in_S': in_S, 'success': success_mats,
        'a_t': a_t, 'R_tgt': R_tgt, 'p0': p0, 'd_dir': d_dir,
        'v_path': v_path, 'eps_p': eps_p, 'T_eff': T_eff,
        'p_target_per_ckpt': [
            p0 + ((i * T_eff // N) * float(cfg.DT)) * v_path * d_dir
            for i in range(N + 1)
        ],
    }


def trace_path(result, j0):
    """From q[0]=Q[0][j0], greedily find a feasible chain through DP.
    Returns list of q indices [j0, j1, ..., jN]."""
    Q = result['Q']
    succ = result['success']
    in_S = result['in_S']
    N = len(succ)
    j = j0
    chain = [j]
    for i in range(N):
        viable = succ[i][j] & in_S[i + 1]
        idx = viable.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None
        cur_q = Q[i][j]
        next_qs = Q[i + 1][idx]
        dists = (next_qs - cur_q).norm(dim=-1)
        k = int(idx[dists.argmin()].item())
        chain.append(k)
        j = k
    return chain


def natural_forward_from(kin, sphere_cc, q0_init, result, args):
    """Simulate controller from q0_init with NO q_ref push (default
    K_NULL pulls back to start). Record q at each checkpoint and whether
    full path completes. Returns list of q's per checkpoint or None if dies."""
    device = kin.device
    a_t = result['a_t']; R_tgt = result['R_tgt']
    p0 = result['p0']; d_dir = result['d_dir']
    v_path = result['v_path']; eps_p = result['eps_p']; T_eff = result['T_eff']
    N = int(args.n_segments)
    q_per_ckpt = [q0_init.clone()]
    q = q0_init.unsqueeze(0)
    for i in range(N):
        t_start = i * T_eff // N
        t_end   = (i + 1) * T_eff // N
        out = batched_rollout_segment(
            q, R_tgt.unsqueeze(0), a_t.unsqueeze(0),
            p0.unsqueeze(0), d_dir.unsqueeze(0),
            torch.tensor([v_path], device=device, dtype=torch.float32),
            torch.tensor([eps_p], device=device, dtype=torch.float32),
            torch.tensor([T_eff], device=device, dtype=torch.long),
            start_step=t_start, end_step=t_end,
            sphere_cc=sphere_cc, kin=kin, is_phantom=False,
            q_ref=None)   # ← NO target push, natural drift
        alive = bool(out['alive_out'].item())
        q = out['q_final']
        q_per_ckpt.append(q.squeeze(0).clone())
        if not alive:
            return q_per_ckpt, False
    return q_per_ckpt, True


def branch_signature(q):
    """3-tuple of joint sign patterns. (J1, J4, J6) — these distinguish
    'shoulder side / elbow up-down / wrist flip' for FR3."""
    return (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--M-oversample", type=int, default=128)
    ap.add_argument("--T-eval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--thresh-loose", type=float, default=10.0)
    ap.add_argument("--thresh-tight", type=float, default=2.0)
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    sphere_cc = None
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cc = _load_fr3_sphere_collision_cls()(
            device=device, margin=cfg.BATCHED_COLLISION_MARGIN)

    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    tasks = env._sample_tasks(args.n_tasks)
    base_dist = np.linalg.norm(np.stack([t['c'][:3] for t in tasks]), axis=-1)

    # iterate tasks, look for one with the desired property:
    #   ∃ j s.t. in_S_loose[j] = True AND in_S_tight[j] = True
    #   AND the chain at tight has a branch switch (sign change in J1/J4/J6)
    found = False
    rng_master = np.random.default_rng(args.seed)
    for b, task in enumerate(tasks):
        if base_dist[b] < args.min_base_dist or base_dist[b] > 0.85:
            continue

        # use SAME rng seed for both runs so Q is identical
        rng_l = np.random.default_rng(args.seed + b * 1000)
        rng_t = np.random.default_rng(args.seed + b * 1000)

        res_loose = backward_dp(kin, sphere_cc, task, args, rng_l,
                                args.thresh_loose)
        if res_loose is None:
            continue
        res_tight = backward_dp(kin, sphere_cc, task, args, rng_t,
                                args.thresh_tight)
        if res_tight is None:
            continue

        # Q should be identical (same seeds). in_S_tight ⊆ in_S_loose typically.
        in_S0_l = res_loose['in_S'][0].cpu().numpy()
        in_S0_t = res_tight['in_S'][0].cpu().numpy()
        Q0      = res_tight['Q'][0]

        n_loose = int(in_S0_l.sum())
        n_tight = int(in_S0_t.sum())
        n_q0    = int(Q0.shape[0])
        print(f"\nTask {b:3d}  ‖p₀‖={base_dist[b]:.2f}  "
              f"|Q_0|={n_q0}  |S_0_loose|={n_loose}  |S_0_tight|={n_tight}  "
              f"need_switch={n_loose - n_tight}")

        # for each j in S_0_tight, compare TIGHT-DP chain vs NATURAL-FORWARD chain
        # interesting case: natural forward FAILS (or ends in different branch)
        # while tight DP succeeds via different chain
        switch_j = None
        switch_chain = None
        switch_segment = None
        switch_kind = None
        natural_qs = None
        natural_alive = None
        for j in np.where(in_S0_t)[0]:
            chain = trace_path(res_tight, int(j))
            if chain is None:
                continue
            tight_qs = [res_tight['Q'][i][chain[i]].cpu().numpy() for i in range(len(chain))]
            tight_sigs = [branch_signature(q) for q in tight_qs]

            # natural drift from same q_0
            q0_init = res_tight['Q'][0][int(j)]
            nat_qs, nat_ok = natural_forward_from(kin, sphere_cc, q0_init, res_tight, args)
            nat_qs_np = [q.detach().cpu().numpy() for q in nat_qs]
            nat_sigs = [branch_signature(q) for q in nat_qs_np]

            # case A: natural dies somewhere → DP saved
            if not nat_ok:
                switch_j = int(j); switch_chain = chain
                switch_segment = len(nat_qs) - 1
                switch_kind = ['DP saved (natural failed at this segment)']
                natural_qs = nat_qs_np
                natural_alive = False
                break
            # case B: natural alive but ends in DIFFERENT branch than tight DP
            if nat_sigs[-1] != tight_sigs[-1]:
                # find first segment where they diverge
                for i in range(1, len(nat_sigs)):
                    if nat_sigs[i] != tight_sigs[i]:
                        switch_j = int(j); switch_chain = chain
                        switch_segment = i
                        diff = tuple(int(tight_sigs[i][k] != nat_sigs[i][k]) for k in range(3))
                        names = ['J1 (shoulder)', 'J4 (elbow)', 'J6 (wrist)']
                        switch_kind = [n for n, d in zip(names, diff) if d]
                        natural_qs = nat_qs_np
                        natural_alive = True
                        break
                if switch_j is not None:
                    break

        if switch_j is None:
            print(f"  no branch-switch case in this task")
            continue

        # found one!
        print(f"\n{'='*78}")
        print(f"FOUND branch-divergence task: Task {b}, q₀ index {switch_j}")
        print(f"{'='*78}")
        print(f"  divergence type: {switch_kind}")
        print(f"  natural forward: {'ALIVE to end' if natural_alive else 'DIED at segment ' + str(switch_segment)}")
        print(f"  tight DP:        ALIVE via different branch chain")

        print(f"\n  --- Backward DP chain (with q_target push toward dp choice) ---")
        print(f"  {'ckpt':>4}  {'J1':>7} {'J2':>7} {'J3':>7} {'J4':>7} "
              f"{'J5':>7} {'J6':>7} {'J7':>7}   {'sig':>10}")
        prev_sig = None
        for i, k in enumerate(switch_chain):
            q = res_tight['Q'][i][k].cpu().numpy()
            sig = branch_signature(q)
            sig_str = f"({sig[0]:+d},{sig[1]:+d},{sig[2]:+d})"
            marker = "  ← SWITCH" if (prev_sig is not None and sig != prev_sig) else ""
            print(f"  {i:>2d}   " + "  ".join(f"{v:+6.2f}" for v in q)
                  + f"   {sig_str:>10}{marker}")
            prev_sig = sig

        print(f"\n  --- Natural forward chain (NO target push, controller drifts) ---")
        print(f"  {'ckpt':>4}  {'J1':>7} {'J2':>7} {'J3':>7} {'J4':>7} "
              f"{'J5':>7} {'J6':>7} {'J7':>7}   {'sig':>10}")
        prev_sig = None
        for i, q in enumerate(natural_qs):
            sig = branch_signature(q)
            sig_str = f"({sig[0]:+d},{sig[1]:+d},{sig[2]:+d})"
            marker = "  ← drifted" if (prev_sig is not None and sig != prev_sig) else ""
            if i == switch_segment and not natural_alive:
                marker = "  ← DIED HERE"
            print(f"  {i:>2d}   " + "  ".join(f"{v:+6.2f}" for v in q)
                  + f"   {sig_str:>10}{marker}")
            prev_sig = sig

        print(f"\n  → INTERPRETATION:")
        if not natural_alive:
            print(f"      Natural drift from q₀ leads to controller failure at ckpt {switch_segment}.")
            print(f"      Backward DP detected this in advance, picked different target q's, ")
            print(f"      successfully steered through. This is RL's value-add territory.")
        else:
            print(f"      Both alive but end in different branches. Backward DP saw that the ")
            print(f"      naive endpoint branch would not have a feasible chain to terminal,")
            print(f"      so it deliberately switched to a different branch at ckpt {switch_segment}.")

        found = True
        break

    if not found:
        print("\nNo branch-switching task found — try increasing --n-tasks "
              "or --M-oversample.")


if __name__ == "__main__":
    main()
