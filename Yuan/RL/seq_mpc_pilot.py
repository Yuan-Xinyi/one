"""Sequential MPC pilot — concept-validation experiment.

Hypothesis under test
---------------------
Bandit framework forces all decisions at t=0. Sequential framework lets the
controller switch nullspace preset every H steps based on phantom lookahead
from CURRENT state. If sequential MPC > phantom_select bandit, then the
intermediate-state info has real value and full RL is worth pursuing.

If sequential MPC ≈ bandit, abandon the sequential RL plan.

Method
------
1. For each task: get initial action a_0 from phantom_select K=8 (best bandit).
2. IK → q_init. Set R_tgt from a_0.
3. At each decision boundary t = H, 2H, ...:
   - Current q_t known, alive rows known.
   - Fork K_dec phantom rollouts from q_t, each with a different nullspace
     preset, run for the REMAINING horizon.
   - Pick preset with longest phantom lookahead per row.
   - Continue REAL rollout from q_t with chosen preset for next H steps.
4. Sum lengths.

Compare:
  - bandit: phantom_select K=8 (committed default nullspace) — current SOTA
  - seq_mpc: same a_0, but mid-rollout nullspace switching by phantom lookahead

If seq_mpc > bandit, sequential framework adds value beyond bandit ceiling.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    batched_rollout, batched_rollout_segment, phantom_rollout,
    branch_project_multistart, build_branch_rotmat_batch,
    _device_from_cfg, _load_fr3_sphere_collision_cls,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


# ----- Nullspace presets -----
# 4 distinct controller "personalities". Each row picks one per decision interval.
PRESETS = [
    # 0: manip-priority — try to stay in well-conditioned configurations
    dict(manip=1.5, jlm=0.05, angle_attract=0.0),
    # 1: jlm-priority — push toward joint-limit center
    dict(manip=0.0, jlm=1.0,  angle_attract=0.0),
    # 2: angle-attract — pull tightly toward target z-axis (less drift)
    dict(manip=0.3, jlm=0.10, angle_attract=0.6),
    # 3: cfg-default — current production behaviour
    dict(manip=float(cfg.NULL_MANIP_GAIN),
         jlm=float(cfg.NULL_JOINT_LIMIT_GAIN),
         angle_attract=float(cfg.NULL_ANGLE_ATTRACT_GAIN)),
]
N_PRESETS = len(PRESETS)


def _phantom_select_action(env_tasks, K_pick=8, seed=12345):
    """Return action_per_task (N, 4) chosen by phantom_select K=K_pick."""
    n = len(env_tasks)
    c_np = np.stack([t['c'] for t in env_tasks], axis=0).astype(np.float32)
    v_np = np.array([t['v_path'] for t in env_tasks], dtype=np.float32)
    e_np = np.array([t['eps_p']  for t in env_tasks], dtype=np.float32)
    T_np = np.array([t['T']      for t in env_tasks], dtype=np.int32)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2*np.pi, size=(K_pick, n)).astype(np.float32)
    psi = rng.uniform(0, 2*np.pi, size=(K_pick, n)).astype(np.float32)
    a = np.stack([np.cos(phi), np.sin(phi),
                  np.cos(psi), np.sin(psi)], axis=-1)            # (K, N, 4)
    a_flat = a.reshape(K_pick * n, 4).astype(np.float32)
    rep_c = np.tile(c_np, (K_pick, 1))
    rep_v = np.tile(v_np, K_pick)
    rep_e = np.tile(e_np, K_pick)
    rep_T = np.tile(T_np, K_pick)
    out = phantom_rollout(a_flat, rep_c, rep_v, rep_e, rep_T)
    L = out['lengths'].reshape(K_pick, n)
    pick = L.argmax(axis=0)
    return a[pick, np.arange(n)], c_np, v_np, e_np, T_np


def _gain_dict_for_preset_idx(preset_idx_t: torch.Tensor) -> dict:
    """preset_idx_t: (B,) long. Return dict of (B,) gain tensors."""
    device = preset_idx_t.device
    B = preset_idx_t.shape[0]
    g_manip = torch.empty(B, device=device, dtype=torch.float32)
    g_jlm = torch.empty_like(g_manip)
    g_a_att = torch.empty_like(g_manip)
    for k, p in enumerate(PRESETS):
        mask = (preset_idx_t == k)
        if mask.any():
            g_manip[mask] = float(p['manip'])
            g_jlm[mask]   = float(p['jlm'])
            g_a_att[mask] = float(p['angle_attract'])
    return {'manip': g_manip, 'jlm': g_jlm, 'angle_attract': g_a_att}


def seq_mpc_run(actions_init, c_np, v_np, e_np, T_np, H_decision=10):
    """Sequential MPC: phantom-evaluate K presets every H_decision steps,
    pick best per row, advance H_decision steps with chosen preset.
    Returns lengths (N,) and per-decision preset choices (n_decisions, N)."""
    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    sphere_cc = None
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cls = _load_fr3_sphere_collision_cls()
        sphere_cc = sphere_cls(device=device,
                               margin=cfg.BATCHED_COLLISION_MARGIN)

    a = torch.as_tensor(actions_init, device=device, dtype=torch.float32)
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_np, device=device, dtype=torch.float32)
    eps_p = torch.as_tensor(e_np, device=device, dtype=torch.float32)
    T = torch.as_tensor(T_np, device=device, dtype=torch.long)
    p0    = c[:, :3]
    d_dir = c[:, 3:6]
    n_out = c[:, 6:9]

    R_tgt = build_branch_rotmat_batch(d_dir, n_out, a)
    # initial IK
    q, ik_ok, _ = branch_project_multistart(kin, p0, R_tgt, a)
    alive = ik_ok.clone()
    B = q.shape[0]
    lengths = torch.zeros(B, device=device, dtype=torch.long)

    max_T = int(T.max().item())
    n_decisions = (max_T + H_decision - 1) // H_decision
    decisions_log = torch.zeros((n_decisions, B), device=device, dtype=torch.long)

    cur_step = 0
    for d_idx in range(n_decisions):
        if not alive.any():
            break
        # ---- evaluate K presets via phantom from current (q, cur_step) ----
        # Each preset: rollout REMAINING (T - cur_step) steps with that preset's
        # gains, no nullspace gradient (phantom). Return phantom_L per preset.
        phantom_L = torch.full((N_PRESETS, B), -1.0,
                               device=device, dtype=torch.float32)
        for k in range(N_PRESETS):
            preset_idx = torch.full((B,), k, device=device, dtype=torch.long)
            gains = _gain_dict_for_preset_idx(preset_idx)
            # phantom rollout from (q, cur_step) to T_max with preset k.
            # is_phantom=True drops nullspace -> our standard "phantom" definition.
            # Use cur_step as start, max_T as end.
            out_p = batched_rollout_segment(
                q, R_tgt, a, p0, d_dir, v_path, eps_p, T,
                start_step=cur_step, end_step=max_T,
                preset_gains=gains, alive_mask=alive,
                sphere_cc=sphere_cc, kin=kin, is_phantom=True)
            phantom_L[k] = out_p['lengths'].float()
        # pick per-row best preset
        chosen = phantom_L.argmax(dim=0)                          # (B,) long
        decisions_log[d_idx] = chosen

        # ---- run REAL rollout for H_decision steps with chosen preset ----
        end_step = min(cur_step + H_decision, max_T)
        gains_real = _gain_dict_for_preset_idx(chosen)
        prev_alive = alive.clone()                # rows alive at start of seg
        out_r = batched_rollout_segment(
            q, R_tgt, a, p0, d_dir, v_path, eps_p, T,
            start_step=cur_step, end_step=end_step,
            preset_gains=gains_real, alive_mask=alive,
            sphere_cc=sphere_cc, kin=kin, is_phantom=False)
        q = out_r['q_final']
        # Only extend length for rows that were alive going INTO this segment.
        # For rows already dead, segment's `lengths` is the (uninformative) init
        # value `start_step`, so we must not overwrite their true death step.
        new_lengths = out_r['lengths']
        lengths = torch.where(prev_alive, new_lengths, lengths)
        alive = out_r['alive_out']
        cur_step = end_step

    return lengths.cpu().numpy(), decisions_log.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--K-bandit", type=int, default=8,
                    help="K for phantom_select bandit baseline")
    ap.add_argument("--H-decision", type=int, default=10,
                    help="control steps per decision interval in seq MPC")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"K_bandit={args.K_bandit}  H_decision={args.H_decision}  "
          f"presets={N_PRESETS}")

    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    tasks = env._sample_tasks(args.n_tasks)
    c_np = np.stack([t['c'] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t['v_path'] for t in tasks], dtype=np.float32)
    e_np = np.array([t['eps_p']  for t in tasks], dtype=np.float32)
    T_np = np.array([t['T']      for t in tasks], dtype=np.int32)

    # ----- bandit baseline: K=K_bandit phantom_select, default nullspace, REAL rollout -----
    print("\n[bandit] phantom_select + cfg nullspace REAL rollout")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    a_pick, _, _, _, _ = _phantom_select_action(tasks, K_pick=args.K_bandit, seed=args.seed)
    out_bandit = batched_rollout(a_pick.astype(np.float32), c_np, v_np, e_np, T_np)
    L_bandit = np.asarray(out_bandit['lengths'], dtype=np.int32)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- seq MPC: same a_pick, but switch nullspace presets via phantom -----
    print("\n[seq_mpc] same a_init, mid-rollout preset switching every "
          f"H={args.H_decision} steps")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_seq, decisions = seq_mpc_run(
        a_pick.astype(np.float32), c_np, v_np, e_np, T_np,
        H_decision=args.H_decision)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- oracle: K=1000 uniform max -----
    print("\n[oracle] K=1000 uniform real rollouts (for ratio reference)")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    K_orc = 1000
    rng = np.random.default_rng(args.seed)
    phi = rng.uniform(0, 2*np.pi, size=(K_orc, args.n_tasks)).astype(np.float32)
    psi = rng.uniform(0, 2*np.pi, size=(K_orc, args.n_tasks)).astype(np.float32)
    a_o = np.stack([np.cos(phi), np.sin(phi),
                    np.cos(psi), np.sin(psi)], axis=-1)
    a_o_flat = a_o.reshape(K_orc * args.n_tasks, 4).astype(np.float32)
    rep_c = np.tile(c_np, (K_orc, 1))
    rep_v = np.tile(v_np, K_orc)
    rep_e = np.tile(e_np, K_orc)
    rep_T = np.tile(T_np, K_orc)
    # chunked
    L_orc_flat = np.empty(K_orc * args.n_tasks, dtype=np.int32)
    chunk = 4096
    for s in range(0, len(a_o_flat), chunk):
        e = min(s + chunk, len(a_o_flat))
        out_c = batched_rollout(a_o_flat[s:e], rep_c[s:e], rep_v[s:e],
                                 rep_e[s:e], rep_T[s:e])
        L_orc_flat[s:e] = np.asarray(out_c['lengths'], dtype=np.int32)
    L_orc = L_orc_flat.reshape(K_orc, args.n_tasks).max(axis=0)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ----- analysis -----
    p0 = c_np[:, :3].astype(np.float64)
    base_dist = np.linalg.norm(p0, axis=-1)
    well = (base_dist >= args.min_base_dist) & (L_orc > 0)
    n_well = int(well.sum())
    safe_top = np.where(L_orc > 0, L_orc, 1).astype(np.float64)

    def _stats(L, name):
        r = (L.astype(np.float64) / safe_top)[well]
        n0 = int((L[well] == 0).sum())
        nlo = int((r < 0.3).sum())
        print(f"  {name:>16}  mean={r.mean():.4f}  std={r.std():.4f}  "
              f"min={r.min():.3f}  p10={np.percentile(r,10):.3f}  "
              f"p25={np.percentile(r,25):.3f}  p50={np.percentile(r,50):.3f}  "
              f"L=0:{n0:3d}  r<0.3:{nlo:3d}")

    print(f"\n=== n_well = {n_well} (of {args.n_tasks}) ===")
    _stats(L_bandit, 'bandit')
    _stats(L_seq, 'seq_mpc')
    _stats(L_orc, 'oracle')

    diff = (L_seq.astype(np.float64) - L_bandit.astype(np.float64))[well] / safe_top[well]
    print(f"\nseq_mpc - bandit (well-defined):")
    print(f"  mean diff = {diff.mean():+.4f}")
    print(f"  >0:{int((diff>0).sum())}  =0:{int((diff==0).sum())}  "
          f"<0:{int((diff<0).sum())}")
    print(f"  big_win  (>+0.1):  {int((diff>0.1).sum())}")
    print(f"  big_loss (<-0.1):  {int((diff<-0.1).sum())}")

    # decision diversity
    print(f"\ndecision diversity (preset usage across {decisions.shape[0]} "
          "decision rounds, all rows):")
    for k in range(N_PRESETS):
        frac = (decisions == k).mean()
        print(f"  preset {k} ({list(PRESETS[k].keys())[0]}-led): {frac:.3f}")


if __name__ == "__main__":
    main()
