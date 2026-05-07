"""v17 training: supervised regression of FlowModel on null-space velocity targets.

Loss: L = ||v_θ(q, t, c) - v*||²    (standard CFM-style, MSE on velocity field)

The training set comes from v17_data_prep.py — every transition in successful
plane-tracking trajectories contributes one (q, t, c, v*) tuple.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Yuan.RL.v17_flow_model import FlowModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     default="Yuan/RL/data/v17_train.npz")
    ap.add_argument("--ckpt-dir", default="Yuan/RL/checkpoints_v17")
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--hidden",   type=int, default=256)
    ap.add_argument("--depth",    type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--v-scale",  type=float, default=100.0,
                    help="multiply v* by this before training (= 1/dt for "
                         "dataset dt=0.01). Brings target magnitude to ~rad/s.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"loading {args.data}")
    d = np.load(args.data)
    q = torch.as_tensor(d["q"], dtype=torch.float32)
    t = torch.as_tensor(d["t"], dtype=torch.float32)
    c = torch.as_tensor(d["c"], dtype=torch.float32)
    v = torch.as_tensor(d["v"], dtype=torch.float32)
    N = q.shape[0]
    print(f"  total pairs: {N:,}  q={q.shape}  c={c.shape}  v*={v.shape}")
    print(f"  ||v*|| (raw) stats: mean={v.norm(dim=-1).mean():.4f}  "
          f"max={v.norm(dim=-1).max():.4f}")
    # scale v* to ~rad/s units so targets are O(1) and gradients are clean
    v = v * float(args.v_scale)
    print(f"  ||v*|| (×{args.v_scale}) stats: mean={v.norm(dim=-1).mean():.4f}  "
          f"max={v.norm(dim=-1).max():.4f}")

    # train / val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_val = int(args.val_frac * N)
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    ds_tr = TensorDataset(q[tr_idx], t[tr_idx], c[tr_idx], v[tr_idx])
    ds_va = TensorDataset(q[val_idx], t[val_idx], c[val_idx], v[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    model = FlowModel(hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"\nFlowModel: hidden={args.hidden} depth={args.depth}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    log_path = os.path.join(args.ckpt_dir, "train_log_v17.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_mse,val_mse,wall\n")

    t0 = time.perf_counter()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for q_b, t_b, c_b, v_b in dl_tr:
            q_b = q_b.to(device); t_b = t_b.to(device)
            c_b = c_b.to(device); v_b = v_b.to(device)
            v_pred = model(q_b, t_b, c_b)
            loss = ((v_pred - v_b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * q_b.shape[0]
            train_n += q_b.shape[0]
        train_mse = train_loss / train_n

        model.eval()
        val_loss = 0.0
        val_n = 0
        cos_sim_sum = 0.0
        rel_err_sum = 0.0
        norm_pred_sum = 0.0
        norm_gt_sum   = 0.0
        with torch.no_grad():
            for q_b, t_b, c_b, v_b in dl_va:
                q_b = q_b.to(device); t_b = t_b.to(device)
                c_b = c_b.to(device); v_b = v_b.to(device)
                v_pred = model(q_b, t_b, c_b)
                loss = ((v_pred - v_b) ** 2).mean()
                val_loss += float(loss.item()) * q_b.shape[0]
                val_n += q_b.shape[0]
                eps = 1e-8
                pn = v_pred.norm(dim=-1)
                gn = v_b.norm(dim=-1)
                cos = (v_pred * v_b).sum(dim=-1) / (pn * gn).clamp_min(eps)
                cos_sim_sum += float(cos.sum().item())
                rel_err_sum += float(((v_pred - v_b).norm(dim=-1) /
                                       gn.clamp_min(eps)).sum().item())
                norm_pred_sum += float(pn.sum().item())
                norm_gt_sum   += float(gn.sum().item())
        val_mse = val_loss / val_n
        val_cos = cos_sim_sum / val_n
        val_rel = rel_err_sum / val_n
        mean_pred = norm_pred_sum / val_n
        mean_gt   = norm_gt_sum   / val_n

        wall = time.perf_counter() - t0
        print(f"[epoch {epoch:>3d}/{args.epochs}]  mse(train/val)="
              f"{train_mse:.4f}/{val_mse:.4f}  cos={val_cos:+.3f}  "
              f"rel_err={val_rel:.3f}  |pred|={mean_pred:.4f}  "
              f"|gt|={mean_gt:.4f}  ({wall:.1f}s)")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_mse},{val_mse},{wall}\n")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "args": vars(args),
                "v_scale": float(args.v_scale),
            }, os.path.join(args.ckpt_dir, "best.pt"))

    # also save final
    torch.save({"epoch": args.epochs, "model": model.state_dict(),
                "args": vars(args), "v_scale": float(args.v_scale)},
               os.path.join(args.ckpt_dir, f"ckpt_{args.epochs:03d}.pt"))
    print(f"\nbest val_mse = {best_val:.5f}")


if __name__ == "__main__":
    main()
