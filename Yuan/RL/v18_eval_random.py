"""v18 multi-orientation eval on FRESH random tasks (matching training convention).

Generates random (plane_point, direction, plane_normal) tasks via the same
_sample_random_task() used in v18_data_prep, then runs:
  1. multi-branch sample diversity (K IK branches at goal → unique q_0 branches)
  2. max-reach via linear scan
  3. oracle ceiling for branch coverage comparison
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample
from Yuan.RL.v18_data_prep import (
    _sample_random_task, _build_R_from_normal_direction,
    _check_transitions_geometric, _dense_ik_at,
)


def branch_signature(q):
    return (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))


def enumerate_ik_at_goal(kin, goal_pos, R_target, M_oversample=64,
                          rng=None):
    """Random + boundary-mixed IK enumeration."""
    if rng is None:
        rng = np.random.default_rng()
    q_set, _ = _dense_ik_at(kin, goal_pos, R_target, M_oversample, rng)
    return q_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_multi/best.pt")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--K-goal", type=int, default=8,
                    help="number of IK branches sampled at each goal")
    ap.add_argument("--n-checkpoints", type=int, default=5)
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--snap-iters", type=int, default=8)
    ap.add_argument("--max-tilt-deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--with-oracle", action="store_true")
    ap.add_argument("--oracle-M", type=int, default=128)
    ap.add_argument("--save-data", type=str, default=None,
                    help="save per-task task spec + q_traj samples to npz for viz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = CFMFlowModel(q_dim=7, cond_dim=COND_DIM,
                         hidden=margs.get("hidden", 512),
                         depth=margs.get("depth", 6)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt}, epoch {state.get('epoch', '?')}")

    rng = np.random.default_rng(args.seed)

    print(f"\n{'task':>5}  {'tilt':>5}  {'L':>5}  {'IK@goal':>8}  {'q0_uniq':>8}  "
          f"{'oracle':>7}  {'cov%':>5}  {'tcp_err':>8}  {'ms':>5}")
    print('-' * 80)

    rows = []
    saved_tasks = []                                    # for viz npz
    n_tried = 0
    while len(rows) < args.n_tasks and n_tried < args.n_tasks * 10:
        n_tried += 1
        ti = len(rows)
        task = _sample_random_task(rng, kin, max_tilt_deg=args.max_tilt_deg)
        if task is None:
            continue
        plane_point  = task['plane_point']
        direction    = task['direction']
        plane_normal = task['plane_normal']
        L            = task['L_max']
        R_target     = torch.as_tensor(task['R_target'], device=device, dtype=torch.float32)
        sp = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
        dr = torch.as_tensor(direction,   device=device, dtype=torch.float32)
        pn = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
        tilt_deg = float(np.degrees(np.arccos(np.clip(plane_normal[2], -1, 1))))

        # path discretization (ckpt 0 = start)
        T_co = args.n_checkpoints + 1
        path_pts = torch.stack([
            sp + (k / args.n_checkpoints) * L * dr
            for k in range(T_co)
        ], dim=0)

        # K IK at goal (use random+boundary mix from data prep)
        goal_pos = path_pts[-1]
        q_goals = enumerate_ik_at_goal(kin, goal_pos, R_target,
                                        M_oversample=args.K_goal * 8, rng=rng)
        if q_goals.shape[0] == 0:
            # this task counts toward n_tried but not n_processed
            continue
        # take up to K
        if q_goals.shape[0] > args.K_goal:
            idx = rng.permutation(q_goals.shape[0])[:args.K_goal]
            q_goals = q_goals[idx]
        n_ik = q_goals.shape[0]

        # for each goal-IK, sample q_traj
        t0 = time.perf_counter()
        sigs_at_start = []
        tcp_err_max = 0.0
        all_q_trajs = []                                 # for viz
        all_tcp_errs = []
        for k in range(n_ik):
            q_traj = backward_sample(model, kin, q_goals[k], path_pts, pn, dr,
                                      n_ode_steps=args.n_ode_steps,
                                      snap_iters=args.snap_iters)
            q_t = q_traj.cpu().numpy()
            sigs_at_start.append(branch_signature(q_t[0]))
            p_pred, _ = kin.fk_batch(q_traj)
            err = float((p_pred - path_pts).norm(dim=-1).max().item())
            tcp_err_max = max(tcp_err_max, err)
            all_q_trajs.append(q_t)
            all_tcp_errs.append(err)
        wall_ms = (time.perf_counter() - t0) * 1000.0 / n_ik
        unique_sigs = len(set(sigs_at_start))

        # oracle ceiling
        oracle_uniq = "—"
        cov_str = "—"
        if args.with_oracle:
            from Yuan.RL.v18_eval_multibranch import oracle_max_branches_at_start
            _, sigs = oracle_max_branches_at_start(
                kin, plane_point, direction, R_target.cpu().numpy(), L,
                n_seg=args.n_checkpoints,
                M_oversample=args.oracle_M, rng=rng)
            oracle_uniq = len(sigs)
            cov_str = f"{int(unique_sigs / max(oracle_uniq, 1) * 100)}%"

        print(f"  {ti:>3d}  {tilt_deg:>4.0f}°  {L:.2f}  {n_ik:>8d}  "
              f"{unique_sigs:>8d}  {oracle_uniq!s:>7}  {cov_str:>5}  "
              f"{tcp_err_max:>8.4f}  {wall_ms:>5.1f}")
        rows.append((tilt_deg, L, n_ik, unique_sigs, oracle_uniq, tcp_err_max, wall_ms))

        if args.save_data is not None:
            saved_tasks.append({
                'plane_point':  np.asarray(plane_point),
                'direction':    np.asarray(direction),
                'plane_normal': np.asarray(plane_normal),
                'L':            float(L),
                'tilt_deg':     float(tilt_deg),
                'R_target':     R_target.cpu().numpy(),
                'path_pts':     path_pts.cpu().numpy(),       # (N+1, 3)
                'q_goals':      q_goals.cpu().numpy(),         # (n_ik, 7)
                'q_trajs':      np.stack(all_q_trajs, axis=0),  # (n_ik, N+1, 7)
                'sigs_at_start': sigs_at_start,
                'tcp_err_per_sample': np.array(all_tcp_errs, dtype=np.float32),
                'unique_q0_branches': int(unique_sigs),
                'oracle_branches': int(oracle_uniq) if isinstance(oracle_uniq, int) else -1,
            })

    if rows:
        arr = np.array([(r[0], r[1], r[2], r[3], r[4] if isinstance(r[4], int) else 0,
                         r[5], r[6]) for r in rows], dtype=np.float64)
        tilt, L, n_ik, uniq, oracle, tcp, ms = arr.T
        print("\n" + "=" * 80)
        print(f"AGGREGATE over {len(rows)} tasks:")
        print(f"  tilt distribution:  mean={tilt.mean():.1f}°  max={tilt.max():.1f}°")
        print(f"  L distribution:     mean={L.mean():.2f}m   max={L.max():.2f}m")
        print(f"  IK@goal mean:       {n_ik.mean():.1f}")
        print(f"  v18 unique q_0 br:  {uniq.mean():.2f}")
        print(f"  oracle q_0 br:      {oracle.mean():.2f}")
        cov = uniq / np.maximum(oracle, 1)
        print(f"  v18 / oracle:       {cov.mean()*100:.0f}%  (median {np.median(cov)*100:.0f}%)")
        print(f"  TCP err: median={np.median(tcp):.4f}  p90={np.percentile(tcp,90):.4f}  max={tcp.max():.4f}")
        print(f"  mean wall:          {ms.mean():.1f} ms / sample")

        # tilt buckets
        print(f"\n  By tilt:")
        for lo_t, hi_t in [(0, 5), (5, 15), (15, 25), (25, 30)]:
            mask = (tilt >= lo_t) & (tilt < hi_t)
            if mask.sum() == 0: continue
            print(f"    {lo_t:>2d}°-{hi_t:>2d}°: n={int(mask.sum())}  "
                  f"v18 uniq={uniq[mask].mean():.2f}  oracle={oracle[mask].mean():.2f}  "
                  f"cov={cov[mask].mean()*100:.0f}%")

        if args.save_data is not None and saved_tasks:
            import pickle
            with open(args.save_data, 'wb') as f:
                pickle.dump(saved_tasks, f)
            print(f"\nsaved {len(saved_tasks)} tasks to {args.save_data}")


if __name__ == "__main__":
    main()
