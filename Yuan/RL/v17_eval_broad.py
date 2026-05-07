"""v17 broad evaluation: run backward inference on N trajectories, report
q_err statistics at multiple cfg_scale settings.

Compares:
  cfg=0.0  : pure J† task-space backward, no flow guidance (= baseline)
  cfg=1.0  : with trained flow guidance
  cfg=2.0  : with CFG amplification
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v17_flow_model import FlowModel
from Yuan.RL.v17_inference import backward_inference


def eval_one(model, kin, q_full, tcp_pos, direction, plane_normal, plane_point,
             cfg_scale: float, v_scale: float, lam: float = 1e-3,
             snap_iters: int = 2):
    device = kin.device
    T = q_full.shape[0]
    q_goal = torch.as_tensor(q_full[T - 1], device=device, dtype=torch.float32)
    path_pts = torch.as_tensor(tcp_pos, device=device, dtype=torch.float32)
    c = torch.as_tensor(np.concatenate([direction, plane_normal, plane_point]),
                         device=device, dtype=torch.float32)
    _, R_at_goal = kin.fk_batch(q_goal.unsqueeze(0))
    R_target = R_at_goal.squeeze(0).contiguous()

    q_traj_pred = backward_inference(model, kin, q_goal, path_pts, R_target, c,
                                     cfg_scale=cfg_scale, lam=lam,
                                     snap_iters=snap_iters, v_scale=v_scale)
    q_gt = torch.as_tensor(q_full, device=device, dtype=torch.float32)
    q_diff = (q_traj_pred - q_gt).norm(dim=-1)            # (T,)
    p_pred, _ = kin.fk_batch(q_traj_pred)
    tcp_err = (p_pred - path_pts).norm(dim=-1)
    return {
        'T': T,
        'q_err_mean': float(q_diff.mean().item()),
        'q_err_max':  float(q_diff.max().item()),
        'q_err_at_start': float(q_diff[0].item()),
        'tcp_err_mean': float(tcp_err.mean().item()),
        'tcp_err_max':  float(tcp_err.max().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="Yuan/RL/checkpoints_v17_pos/best.pt")
    ap.add_argument("--hdf5",  default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--n-trajs", type=int, default=20)
    ap.add_argument("--cfg-list", type=str, default="0.0,1.0,2.0")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    import h5py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = FlowModel(hidden=margs.get("hidden", 256),
                      depth=margs.get("depth", 4)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    v_scale = float(state.get("v_scale", 1.0))
    print(f"loaded {args.ckpt}, epoch {state.get('epoch', '?')}, v_scale={v_scale}")

    cfg_list = [float(x) for x in args.cfg_list.split(",")]
    rng = np.random.default_rng(args.seed)

    # pick random trajectories, also stratify by termination type
    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        # categorize by termination
        by_reason = {'angle_violation': [], 'joint_margin': [],
                     'self_collision': [], 'max_steps': []}
        for i, k in enumerate(keys[:5000]):
            r = str(f[k].attrs['termination_reason'])
            if r in by_reason:
                by_reason[r].append(i)
        print(f"\ntermination reason counts (in first 5000):")
        for r, lst in by_reason.items():
            print(f"  {r}: {len(lst)}")

        # sample n_trajs/4 from each category if available
        n_per = max(1, args.n_trajs // 4)
        chosen = []
        for r, lst in by_reason.items():
            if not lst: continue
            picks = rng.choice(lst, size=min(n_per, len(lst)), replace=False)
            chosen.extend([(int(i), r) for i in picks])
        chosen = chosen[:args.n_trajs]

        print(f"\nrunning eval on {len(chosen)} trajectories at cfg_scales={cfg_list}")
        all_results = {cfg: [] for cfg in cfg_list}
        t0 = time.perf_counter()
        for traj_idx, reason in chosen:
            g = f[keys[traj_idx]]
            q_full = np.asarray(g['q'])
            tcp_pos = np.asarray(g['tcp_pos'])
            direction = np.asarray(g['direction'])
            plane_normal = np.asarray(g['plane_normal'])
            plane_point = np.asarray(g['plane_point'])
            for cfg in cfg_list:
                res = eval_one(model, kin, q_full, tcp_pos, direction,
                                plane_normal, plane_point,
                                cfg_scale=cfg, v_scale=v_scale)
                all_results[cfg].append((traj_idx, reason, res))
        print(f"  wall = {time.perf_counter()-t0:.1f}s")

    # ---------- per-traj table ----------
    print(f"\n{'='*92}")
    print(f"{'traj_idx':>9}  {'reason':>17}  " +
          "  ".join(f"q_err_mean@cfg{c}" for c in cfg_list))
    print('-' * 92)
    for j, (traj_idx, reason, _) in enumerate(all_results[cfg_list[0]]):
        q_errs = [all_results[cfg][j][2]['q_err_mean'] for cfg in cfg_list]
        print(f"  {traj_idx:>7d}  {reason:>17}  " +
              "  ".join(f"{e:>14.4f}" for e in q_errs))

    # ---------- aggregate ----------
    print(f"\n{'='*92}")
    print(f"AGGREGATE STATS over {len(chosen)} trajectories")
    print(f"{'='*92}")
    print(f"\n{'cfg':>5}  {'q_err mean':>11}  {'q_err median':>13}  "
          f"{'q_err max':>11}  {'tcp_err mean':>13}  {'tcp_err max':>11}")
    print('-' * 75)
    for cfg in cfg_list:
        results = [r[2] for r in all_results[cfg]]
        q_means = np.array([r['q_err_mean'] for r in results])
        q_maxs = np.array([r['q_err_max'] for r in results])
        tcp_means = np.array([r['tcp_err_mean'] for r in results])
        tcp_maxs = np.array([r['tcp_err_max'] for r in results])
        print(f"  {cfg:>3}    {q_means.mean():>11.4f}  "
              f"{np.median(q_means):>13.4f}  {q_maxs.max():>11.4f}  "
              f"{tcp_means.mean():>13.6f}  {tcp_maxs.max():>11.6f}")

    # by termination reason
    print(f"\nBY TERMINATION REASON:")
    for cfg in cfg_list:
        print(f"\n  cfg={cfg}:")
        per_reason = {}
        for traj_idx, reason, res in all_results[cfg]:
            per_reason.setdefault(reason, []).append(res['q_err_mean'])
        for r, lst in per_reason.items():
            arr = np.array(lst)
            print(f"    {r:>17}  n={len(arr):>3d}  "
                  f"q_err mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"max={arr.max():.4f}")


if __name__ == "__main__":
    main()
