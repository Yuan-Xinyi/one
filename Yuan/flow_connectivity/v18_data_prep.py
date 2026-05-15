"""v18 data prep: backward-feasible (q_next → q_curr) transition tuples.

Per task, enumerates dense IK candidates at each path checkpoint, runs
backward DP to identify the feasible set S_i at each i, then emits training
tuples for a Conditional Flow Matching model that learns the BACKWARD
transition distribution:

    p(q_curr | q_next, x_i, x_{i+1}, c)

Each q_next may have MULTIPLE valid q_curr (different IK branches that
lead into it via the controller), so the conditional is genuinely
multimodal — exactly the case where CFM beats a deterministic regressor.

Output NPZ:
    cond:    (N_tuples, COND_DIM)   conditioning features
    target:  (N_tuples, 7)          q_curr  (samples from S_i)
    meta:    dict (task indexing)
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

import Yuan.flow_connectivity.config as cfg
from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import (
    batched_rollout_segment, _batched_ik_project,
    _device_from_cfg, _load_fr3_sphere_collision_cls,
)


# Conditioning vector layout:
#   [q_next (7), x_curr (3), x_next (3), plane_normal (3), direction (3)]
COND_DIM = 7 + 3 + 3 + 3 + 3   # = 19


def _dense_ik_at(kin, p_target, R_target, M, rng,
                 extra_seeds: np.ndarray | None = None,
                 mix_boundary: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-distribution IK seed sampling for better branch coverage:
        50% uniform (interior coverage)
        50% boundary-biased (1-2 random joints pushed within 3% of limit)
    Plus optional `extra_seeds` (e.g., dataset's q[t]) for guaranteed coverage."""
    device = kin.device
    lo = kin.lmt_lo.cpu().numpy(); hi = kin.lmt_up.cpu().numpy()
    span = hi - lo

    if mix_boundary and M >= 4:
        n_unif = M // 2
        n_edge = M - n_unif
        seeds_unif = rng.uniform(lo[None, :], hi[None, :],
                                 size=(n_unif, 7)).astype(np.float32)
        seeds_edge = rng.uniform(lo[None, :], hi[None, :],
                                 size=(n_edge, 7)).astype(np.float32)
        # for each edge seed, pick 1-2 random joints, push within 3% of limit
        for i in range(n_edge):
            n_extreme = int(rng.integers(1, 3))           # 1 or 2 joints extreme
            joints = rng.choice(7, size=n_extreme, replace=False)
            for j in joints:
                if int(rng.integers(0, 2)) == 0:
                    seeds_edge[i, j] = lo[j] + 0.03 * span[j]
                else:
                    seeds_edge[i, j] = hi[j] - 0.03 * span[j]
        seeds_np = np.concatenate([seeds_unif, seeds_edge], axis=0)
    else:
        seeds_np = rng.uniform(lo[None, :], hi[None, :],
                               size=(M, 7)).astype(np.float32)
    if extra_seeds is not None and extra_seeds.shape[0] > 0:
        extra = np.clip(extra_seeds.astype(np.float32),
                        lo[None, :] + 1e-3, hi[None, :] - 1e-3)
        seeds_np = np.concatenate([seeds_np, extra], axis=0)

    q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
    n_total = q_seed.shape[0]
    p_rep = p_target.unsqueeze(0).expand(n_total, 3)
    R_rep = R_target.unsqueeze(0).expand(n_total, 3, 3)
    q, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=None)
    return q[ok], ok


def _check_transitions_geometric(kin, q_curr_set: torch.Tensor,   # (Mc, 7)
                                 q_next_set: torch.Tensor,         # (Mn, 7)
                                 x_curr: torch.Tensor,             # (3,)
                                 x_next: torch.Tensor,             # (3,)
                                 n_check: int = 8,
                                 tol_pos: float = 0.05,
                                 chunk: int = 65536) -> torch.Tensor:
    """Geometric (kinematic) connectivity test.

    For each (j, k) pair, linearly interpolate q from q_curr[j] to q_next[k]
    in n_check steps. Compute FK at each step and check it stays within
    `tol_pos` of the LINEAR x-path from x_curr to x_next. Also check joint
    limits along the way.

    This does NOT depend on any controller dynamics. Two q's are 'connected'
    iff their joint-space line gives an FK trace that matches the desired
    x-path closely. Captures the user's "joint space topology" notion: q's
    in the same IK branch homotopy class are linearly connected; q's
    requiring a branch flip pass through near-singular configs and fail.
    """
    device = kin.device
    Mc = q_curr_set.shape[0]; Mn = q_next_set.shape[0]
    if Mc == 0 or Mn == 0:
        return torch.zeros(Mc, Mn, device=device, dtype=torch.bool)

    # interpolation alphas
    alphas = torch.linspace(0.0, 1.0, n_check, device=device, dtype=torch.float32)
    # q_interp: (n_check, Mc, Mn, 7)
    q_curr_b = q_curr_set.view(1, Mc, 1, 7)
    q_next_b = q_next_set.view(1, 1, Mn, 7)
    a = alphas.view(n_check, 1, 1, 1)
    q_interp = (1 - a) * q_curr_b + a * q_next_b
    # joint-limit check
    in_lim = ((q_interp >= kin.lmt_lo + 1e-3)
              & (q_interp <= kin.lmt_up - 1e-3)).all(dim=-1)         # (n_check, Mc, Mn)
    in_lim_all = in_lim.all(dim=0)                                   # (Mc, Mn)
    # FK on all interp points (chunked)
    n_total = n_check * Mc * Mn
    q_flat = q_interp.reshape(n_total, 7)
    p_actual_flat = torch.empty(n_total, 3, device=device, dtype=torch.float32)
    for s in range(0, n_total, chunk):
        e = min(s + chunk, n_total)
        p_b, _ = kin.fk_batch(q_flat[s:e])
        p_actual_flat[s:e] = p_b
    p_actual = p_actual_flat.view(n_check, Mc, Mn, 3)
    # expected x along path: x_curr + α (x_next - x_curr)
    x_expected = ((1 - a.view(n_check, 1, 1, 1)) * x_curr.view(1, 1, 1, 3)
                  + a.view(n_check, 1, 1, 1) * x_next.view(1, 1, 1, 3))
    err = (p_actual - x_expected).norm(dim=-1)                       # (n_check, Mc, Mn)
    on_path = (err <= tol_pos).all(dim=0)                            # (Mc, Mn)
    return in_lim_all & on_path


def _check_transitions_pairwise(kin,
                                 q_curr_set: torch.Tensor,    # (Mc, 7)
                                 q_next_set: torch.Tensor,    # (Mn, 7)
                                 R_tgt: torch.Tensor,         # (3, 3)
                                 a_dummy: torch.Tensor,       # (4,)
                                 p_curr: torch.Tensor,        # (3,)  start of segment
                                 d_dir: torch.Tensor,         # (3,)
                                 v_path: float, eps_p: float, T_total: int,
                                 start_step: int, end_step: int,
                                 sphere_cc, q_dist_thresh: float = 1.5,
                                 chunk: int = 4096) -> torch.Tensor:
    """Returns (Mc, Mn) bool: success[j, k] iff controller drives q_curr[j]
    → near q_next[k] over segment."""
    device = kin.device
    Mc = q_curr_set.shape[0]; Mn = q_next_set.shape[0]
    if Mc == 0 or Mn == 0:
        return torch.zeros(Mc, Mn, device=device, dtype=torch.bool)
    j = torch.arange(Mc, device=device).view(Mc, 1).expand(Mc, Mn).reshape(-1)
    k = torch.arange(Mn, device=device).view(1, Mn).expand(Mc, Mn).reshape(-1)
    n_pairs = Mc * Mn
    q_init = q_curr_set[j]; q_targ = q_next_set[k]
    R_flat = R_tgt.unsqueeze(0).expand(n_pairs, 3, 3)
    a_flat = a_dummy.unsqueeze(0).expand(n_pairs, 4)
    p_flat = p_curr.unsqueeze(0).expand(n_pairs, 3)
    d_flat = d_dir.unsqueeze(0).expand(n_pairs, 3)
    v_flat = torch.full((n_pairs,), float(v_path), device=device, dtype=torch.float32)
    e_flat = torch.full((n_pairs,), float(eps_p),  device=device, dtype=torch.float32)
    T_flat = torch.full((n_pairs,), int(T_total),   device=device, dtype=torch.long)
    # transition oracle: use STRONG K_NULL pull toward q_target so controller
    # actively switches branches within the segment. The dataset's controller
    # uses K_NULL=0.5 (gentle) for nominal tracking; here we want to test
    # FEASIBILITY ('does any q_target lie in the reachable set'), not faithful
    # imitation of the dataset's null-space dynamics.
    aggressive_gains = {
        'k_null':         20.0,
        'manip':          0.0,
        'jlm':            0.0,
        'angle_attract':  0.0,
        'angle_boundary': 0.0,
    }
    success_flat = torch.zeros(n_pairs, device=device, dtype=torch.bool)
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        out = batched_rollout_segment(
            q_init[s:e], R_flat[s:e], a_flat[s:e],
            p_flat[s:e], d_flat[s:e], v_flat[s:e], e_flat[s:e], T_flat[s:e],
            start_step=start_step, end_step=end_step,
            preset_gains=aggressive_gains,
            sphere_cc=sphere_cc, kin=kin, is_phantom=False,
            q_ref=q_targ[s:e])
        alive = out['alive_out']
        q_diff = (out['q_final'] - q_targ[s:e]).norm(dim=-1)
        success_flat[s:e] = alive & (q_diff < q_dist_thresh)
    return success_flat.view(Mc, Mn)


def _build_R_from_normal_direction(plane_normal: np.ndarray,
                                    direction: np.ndarray) -> np.ndarray:
    """R s.t. TCP_z = -plane_normal (pen INTO surface), TCP_x = direction
    projected onto surface plane. Used as R_target for arbitrary surface
    orientation tasks."""
    z = -plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    x = direction - z * (direction @ z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1).astype(np.float32)
    return R


def _sample_random_task(rng: np.random.Generator,
                        kin: BatchedFR3Kinematics,
                        max_tilt_deg: float = 60.0) -> dict | None:
    """Sample a random (plane_point, direction, plane_normal, L) task and
    verify IK feasibility at the start point. Returns None if no feasible
    task found after rejection sampling.

    plane_normal: uniform on cap of upper hemisphere within max_tilt_deg
    plane_point: in box [-0.4, -0.5, 0.05] to [0.7, 0.5, 0.6] within reach
    direction: random unit vector perp to plane_normal
    L: random in [0.20, 0.70]
    """
    cos_max = float(np.cos(np.deg2rad(max_tilt_deg)))
    for _ in range(50):
        # plane normal: upper hemisphere with tilt cap
        u = rng.normal(size=3)
        u = u / (np.linalg.norm(u) + 1e-12)
        if u[2] < cos_max:                              # too tilted
            continue
        plane_normal = u.astype(np.float32)

        # plane point: in reachable box
        p0_ok = False
        for _ in range(20):
            p0 = rng.uniform(np.array([-0.4, -0.5, 0.05]),
                             np.array([0.7, 0.5, 0.6])).astype(np.float32)
            d_to_base = float(np.linalg.norm(p0))
            if 0.30 < d_to_base < 0.80:
                p0_ok = True
                plane_point = p0
                break
        if not p0_ok:
            continue

        # direction: random in surface plane
        d_ok = False
        for _ in range(20):
            v = rng.normal(size=3)
            v = v - plane_normal * (v @ plane_normal)
            nv = float(np.linalg.norm(v))
            if nv > 0.1:
                direction = (v / nv).astype(np.float32)
                d_ok = True
                break
        if not d_ok:
            continue

        L = float(rng.uniform(0.20, 0.70))
        R_target = _build_R_from_normal_direction(plane_normal, direction)

        # quick IK feasibility test at start
        device = kin.device
        seeds_np = rng.uniform(kin.lmt_lo.cpu().numpy()[None, :],
                               kin.lmt_up.cpu().numpy()[None, :],
                               size=(16, 7)).astype(np.float32)
        q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
        p_t = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
        R_t = torch.as_tensor(R_target,    device=device, dtype=torch.float32)
        p_rep = p_t.unsqueeze(0).expand(16, 3)
        R_rep = R_t.unsqueeze(0).expand(16, 3, 3)
        _, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=None)
        if not bool(ok.any().item()):
            continue                                    # no IK at start, skip

        return dict(plane_point=plane_point, direction=direction,
                    plane_normal=plane_normal, L_max=L,
                    R_target=R_target)
    return None


def process_one_task(kin, sphere_cc, task_meta, M_oversample, N_segments,
                     q_dist_thresh, T_eff, rng, max_pairs_per_segment=2000):
    """Run backward DP, emit (cond, target) tuples for one task.

    T_eff is OVERRIDDEN here so that one full path of length task_meta['L_max']
    takes exactly T_eff steps at v_path=0.10 m/s, dt=0.02 s. Per-segment
    step count = T_eff / N_segments. This lets the controller actually traverse
    each segment within its allotted time budget.
    """
    pass
    device = kin.device
    plane_point = task_meta['plane_point']
    direction   = task_meta['direction']
    plane_normal = task_meta['plane_normal']
    L_total     = task_meta['L_max']                        # path length to discretize
    R_target    = task_meta['R_target']                      # (3, 3) — derived from plane geometry

    p0_t = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
    d_t  = torch.as_tensor(direction,   device=device, dtype=torch.float32)
    n_t  = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
    R_t  = torch.as_tensor(R_target,     device=device, dtype=torch.float32)
    a_dummy = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    v_path = 0.10                                            # match dataset speed
    eps_p  = 0.01                                            # generous segment tolerance
    dt = 0.02

    # T_eff = total steps to traverse full L at v_path · dt per step.
    # This is what the controller's path-target evolution expects.
    T_eff = max(1, int(L_total / (v_path * dt)))             # e.g. 0.6m → 300

    # checkpoints x_0 ... x_N
    checkpoints = [p0_t + (i / N_segments) * L_total * d_t for i in range(N_segments + 1)]

    # 1. dense IK at each checkpoint
    Q = []
    for x_i in checkpoints:
        q_set, _ = _dense_ik_at(kin, x_i, R_t, M_oversample, rng)
        Q.append(q_set)
    sizes = [q.shape[0] for q in Q]
    if min(sizes) < 2:
        return [], 0                                          # too few IKs to backward-DP
    if process_one_task._verbose:
        print(f"    Q sizes per ckpt: {sizes}", flush=True)

    # 2. backward DP using GEOMETRIC connectivity (joint-space linear interp
    # + FK on linear path tolerance). This tests joint-space topology, not
    # any specific controller's drift behavior.
    in_S = [None] * (N_segments + 1)
    in_S[N_segments] = torch.ones(Q[N_segments].shape[0], device=device, dtype=torch.bool)
    transition_masks = []
    for i in range(N_segments - 1, -1, -1):
        succ = _check_transitions_geometric(
            kin, Q[i], Q[i + 1],
            x_curr=checkpoints[i], x_next=checkpoints[i + 1],
            n_check=8, tol_pos=q_dist_thresh)        # repurpose q_dist_thresh as tol_pos
        transition_masks.append((i, succ))
        v_next = in_S[i + 1].unsqueeze(0).expand(Q[i].shape[0], -1)
        in_S[i] = (succ & v_next).any(dim=1)
        if process_one_task._verbose:
            tr_rate = float(succ.float().mean().item())
            print(f"    seg {i}: trans rate {tr_rate:.2f}, |S_{i}|={int(in_S[i].sum().item())}", flush=True)

    # 3. emit (cond, q_curr) tuples
    cond_list = []
    targ_list = []
    for (i, succ) in transition_masks:                       # i ∈ [0, N-1]
        # for each (q_curr ∈ S_i, q_next ∈ S_{i+1}) pair where transition succeeds
        feas_curr = in_S[i].nonzero(as_tuple=True)[0]
        feas_next = in_S[i + 1].nonzero(as_tuple=True)[0]
        if feas_curr.numel() == 0 or feas_next.numel() == 0:
            continue
        # restrict succ to feasible-feasible pairs
        succ_feas = succ[feas_curr][:, feas_next]            # (|S_i|, |S_{i+1}|)
        idxs = succ_feas.nonzero(as_tuple=False)             # (n_pairs, 2)
        if idxs.numel() == 0:
            continue
        # cap
        if idxs.shape[0] > max_pairs_per_segment:
            sel = torch.randperm(idxs.shape[0])[:max_pairs_per_segment]
            idxs = idxs[sel]
        for pair in idxs.cpu().numpy():
            j, k = int(pair[0]), int(pair[1])
            q_curr = Q[i][feas_curr[j]].cpu().numpy()
            q_next = Q[i + 1][feas_next[k]].cpu().numpy()
            x_curr = checkpoints[i].cpu().numpy()
            x_next = checkpoints[i + 1].cpu().numpy()
            cond = np.concatenate([q_next, x_curr, x_next, plane_normal, direction]).astype(np.float32)
            cond_list.append(cond)
            targ_list.append(q_curr.astype(np.float32))

    return cond_list, targ_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--out",  default="Yuan/flow_connectivity/data/v18_train.npz")
    ap.add_argument("--n-tasks", type=int, default=500)
    ap.add_argument("--M-oversample", type=int, default=32)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--q-dist-thresh", type=float, default=1.5)
    ap.add_argument("--T-eval", type=int, default=60)
    ap.add_argument("--max-pairs-per-segment", type=int, default=400)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--random-tasks", action="store_true",
                    help="generate random multi-orientation tasks (ignores hdf5)")
    ap.add_argument("--max-tilt-deg", type=float, default=60.0,
                    help="max plane normal tilt from +z when --random-tasks")
    args = ap.parse_args()

    import h5py
    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    sphere_cc = None
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cc = _load_fr3_sphere_collision_cls()(
            device=device, margin=cfg.BATCHED_COLLISION_MARGIN)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    process_one_task._verbose = True
    all_cond = []; all_targ = []
    n_diag = 0
    t0 = time.perf_counter()
    n_processed = 0
    n_skipped = 0

    if args.random_tasks:
        # ------- Multi-orientation random task generation -------
        print(f"random task gen: max_tilt={args.max_tilt_deg}°  "
              f"target {args.n_tasks} tasks", flush=True)
        n_attempts = 0
        while n_processed < args.n_tasks:
            n_attempts += 1
            task_meta = _sample_random_task(rng, kin, max_tilt_deg=args.max_tilt_deg)
            if task_meta is None:
                continue                                # rejection
            t_task = time.perf_counter()
            n_diag += 1
            process_one_task._verbose = (n_diag <= 5)
            if process_one_task._verbose:
                print(f"\n  [diag {n_diag}] "
                      f"L={task_meta['L_max']:.2f}  "
                      f"n={task_meta['plane_normal']}  "
                      f"d={task_meta['direction']}", flush=True)
            cond_list, targ_list = process_one_task(
                kin, sphere_cc, task_meta,
                M_oversample=args.M_oversample,
                N_segments=args.n_segments,
                q_dist_thresh=args.q_dist_thresh,
                T_eff=args.T_eval, rng=rng,
                max_pairs_per_segment=args.max_pairs_per_segment)
            wall_task = time.perf_counter() - t_task
            if not cond_list:
                n_skipped += 1
                if n_skipped % 100 == 0:
                    print(f"  [skip {n_skipped}]  total attempts={n_attempts}  "
                          f"({wall_task:.1f}s)", flush=True)
                continue
            all_cond.extend(cond_list)
            all_targ.extend(targ_list)
            n_processed += 1
            if n_processed <= 10 or n_processed % 25 == 0:
                print(f"  task {n_processed}/{args.n_tasks}  "
                      f"+{len(cond_list):>4d} tuples  total={len(all_cond):>8d}  "
                      f"attempts={n_attempts}  skipped={n_skipped}  "
                      f"task_wall={wall_task:.1f}s  "
                      f"total={time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print(f"loading {args.hdf5}")
        with h5py.File(args.hdf5, 'r') as f:
            keys = sorted(f.keys())
            idx_pool = list(range(len(keys)))
            rng.shuffle(idx_pool)
            print(f"  pool size: {len(idx_pool)}", flush=True)
            for ti, traj_idx in enumerate(idx_pool):
                if n_processed >= args.n_tasks:
                    break
                g = f[keys[traj_idx]]
                plane_point = np.asarray(g['plane_point'])
                base_dist = float(np.linalg.norm(plane_point))
                if base_dist < 0.30 or base_dist > 0.85:
                    continue
                direction = np.asarray(g['direction'])
                plane_normal = np.asarray(g['plane_normal'])
                L_max = float(g.attrs['total_projected_length'])
                if L_max < 0.20:
                    continue
                start_q = torch.as_tensor(g['q'][0], device=device, dtype=torch.float32)
                _, R_at_start = kin.fk_batch(start_q.unsqueeze(0))
                R_target = R_at_start.squeeze(0).cpu().numpy()
                task_meta = dict(
                    plane_point=plane_point, direction=direction,
                    plane_normal=plane_normal, L_max=L_max, R_target=R_target,
                )
                t_task = time.perf_counter()
                n_diag += 1
                process_one_task._verbose = (n_diag <= 5)
                if process_one_task._verbose:
                    print(f"\n  [diag {n_diag}, traj_idx={traj_idx}] "
                          f"L={task_meta['L_max']:.2f}  base_d={base_dist:.2f}",
                          flush=True)
                cond_list, targ_list = process_one_task(
                    kin, sphere_cc, task_meta,
                    M_oversample=args.M_oversample,
                    N_segments=args.n_segments,
                    q_dist_thresh=args.q_dist_thresh,
                    T_eff=args.T_eval, rng=rng,
                    max_pairs_per_segment=args.max_pairs_per_segment)
                wall_task = time.perf_counter() - t_task
                if not cond_list:
                    n_skipped += 1
                    if n_skipped <= 3 or n_skipped % 100 == 0:
                        print(f"  [skip {n_skipped}]  ({wall_task:.1f}s)", flush=True)
                    continue
                all_cond.extend(cond_list)
                all_targ.extend(targ_list)
                n_processed += 1
                if n_processed <= 10 or n_processed % 25 == 0:
                    print(f"  task {n_processed}/{args.n_tasks}  "
                          f"+{len(cond_list):>4d} tuples  total={len(all_cond):>8d}  "
                          f"skipped={n_skipped}  task_wall={wall_task:.1f}s  "
                          f"total={time.perf_counter()-t0:.1f}s", flush=True)

    cond_arr = np.stack(all_cond, axis=0).astype(np.float32)
    targ_arr = np.stack(all_targ, axis=0).astype(np.float32)
    print(f"\nTotal: {n_processed} tasks → {cond_arr.shape[0]:,} (cond, q_curr) tuples")
    print(f"  cond shape: {cond_arr.shape}  (= [q_next 7, x_curr 3, x_next 3, n 3, d 3] = {COND_DIM})")
    print(f"  targ shape: {targ_arr.shape}")
    print(f"  ||q_curr - q_next|| stats:")
    diffs = np.linalg.norm(cond_arr[:, :7] - targ_arr, axis=1)
    print(f"    mean={diffs.mean():.4f}  median={np.median(diffs):.4f}  "
          f"max={diffs.max():.4f}  p90={np.percentile(diffs, 90):.4f}")
    np.savez_compressed(args.out, cond=cond_arr, targ=targ_arr,
                        n_tasks=n_processed)
    print(f"saved {args.out}  size={os.path.getsize(args.out)/1e6:.1f} MB  "
          f"({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
