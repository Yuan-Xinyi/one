"""Deep-dive on a single task to understand v18 max-reach failures.

Try one task at multiple L's. At each L, enumerate K IK branches at goal,
draw N samples per branch, count feasible. Show:
  - feasibility rate per L (fraction of K*N samples that pass)
  - which K_branch / N_sample gave the best q_0
  - q_0 distribution at the L the dataset reached but v18 said fail
  - dataset's actual q_0 vs v18 samples — distance / branch alignment
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project, _branch_seed_bank
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample
from Yuan.RL.v18_max_reach import is_feasible, joint_margin_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_50k/best.pt")
    ap.add_argument("--hdf5", default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--task", type=int, default=14774,
                    help="task index where v18 underperformed")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N-samples", type=int, default=16,
                    help="samples per IK branch (1 = standard inference)")
    ap.add_argument("--n-checkpoints", type=int, default=5)
    args = ap.parse_args()

    import h5py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = CFMFlowModel(q_dim=7, cond_dim=COND_DIM,
                         hidden=margs.get("hidden", 512),
                         depth=margs.get("depth", 6)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        g = f[keys[args.task]]
        plane_point = np.asarray(g['plane_point'])
        direction   = np.asarray(g['direction'])
        plane_normal = np.asarray(g['plane_normal'])
        L_dataset   = float(g.attrs['total_projected_length'])
        q_dataset   = np.asarray(g['q'])               # full GT q-traj
        start_q_data = q_dataset[0]
        end_q_data   = q_dataset[-1]
        T_data = q_dataset.shape[0]

    sp = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
    dr = torch.as_tensor(direction,   device=device, dtype=torch.float32)
    pn = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
    sq = torch.as_tensor(start_q_data, device=device, dtype=torch.float32)
    eq = torch.as_tensor(end_q_data,   device=device, dtype=torch.float32)
    _, R_at_start = kin.fk_batch(sq.unsqueeze(0))
    R_target = R_at_start.squeeze(0).contiguous()
    a_dummy = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    print(f"\n=== Task {args.task}: L_dataset = {L_dataset:.3f} m ===")
    print(f"  start_q (dataset GT): {np.round(start_q_data, 2)}")
    print(f"  end_q   (dataset GT): {np.round(end_q_data, 2)}")
    print(f"  ‖end_q - start_q‖ = {np.linalg.norm(end_q_data - start_q_data):.3f} rad")

    # scan a range of L values
    L_scan = np.linspace(0.1, L_dataset * 1.2, 8)

    print(f"\n{'L':>6}  {'IK@goal':>8}  {'feasible/total':>15}  "
          f"{'best margin':>11}  {'best q_0 ‖q-q_data‖':>20}")
    print('-' * 72)

    for L in L_scan:
        L = float(L)
        goal_pos = sp + L * dr
        path_pts = torch.stack([
            sp + (k / args.n_checkpoints) * L * dr
            for k in range(args.n_checkpoints + 1)
        ], dim=0)

        # K IK at goal
        seeds = _branch_seed_bank(kin)[:args.K]
        p_rep = goal_pos.unsqueeze(0).expand(args.K, 3)
        R_rep = R_target.unsqueeze(0).expand(args.K, 3, 3)
        a_rep = a_dummy.unsqueeze(0).expand(args.K, 4)
        q_goal_K, ok_K, _ = _batched_ik_project(kin, seeds, p_rep, R_rep,
                                                branch_action=a_rep)
        n_ik = int(ok_K.sum().item())

        # N samples per IK
        n_feas = 0
        n_total = 0
        best_margin = -1.0
        best_q0 = None
        for k in range(args.K):
            if not bool(ok_K[k].item()): continue
            for _ in range(args.N_samples):
                q_traj = backward_sample(model, kin, q_goal_K[k], path_pts,
                                          pn, dr, n_ode_steps=16, snap_iters=8)
                n_total += 1
                if is_feasible(q_traj, kin.lmt_lo, kin.lmt_up, margin=0.02):
                    n_feas += 1
                    margin = joint_margin_score(q_traj, kin.lmt_lo, kin.lmt_up)
                    if margin > best_margin:
                        best_margin = margin
                        best_q0 = q_traj[0]
        if best_q0 is None:
            best_q0_str = "—"
            margin_str = "—"
            dist_str = "—"
        else:
            margin_str = f"{best_margin:.3f}"
            dist = float((best_q0 - sq).norm().item())
            dist_str = f"{dist:.3f}"
        print(f"  {L:>6.3f}  {n_ik:>8d}  {n_feas:>4d}/{n_total:>4d} "
              f"({100.0*n_feas/max(n_total,1):>3.0f}%)  "
              f"{margin_str:>11}  {dist_str:>20}")


if __name__ == "__main__":
    main()
