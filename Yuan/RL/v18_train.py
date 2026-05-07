"""v18 CFM training: learn p(q_curr | q_next, x_curr, x_next, c).

Standard CFM loss on linear-path interpolation:
  v_τ = (1-τ) z + τ q_curr;  target = q_curr - z;  loss = ||v_θ - target||²

Validation reports sample-based metric: ODE-sample q_curr, compare to ground
truth via cosine sim AND distance to GT. With multimodal targets, sampled
q_curr might be far from any specific GT (different mode), so we also
monitor diversity via std across multiple samples for same condition.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     default="Yuan/RL/data/v18_train.npz")
    ap.add_argument("--ckpt-dir", default="Yuan/RL/checkpoints_v18")
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--hidden",   type=int, default=512)
    ap.add_argument("--depth",    type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"loading {args.data}")
    d = np.load(args.data)
    cond = torch.as_tensor(d["cond"], dtype=torch.float32)
    targ = torch.as_tensor(d["targ"], dtype=torch.float32)
    N = cond.shape[0]
    print(f"  N={N:,}  cond={cond.shape}  targ={targ.shape}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_val = int(args.val_frac * N)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    ds_tr = TensorDataset(cond[tr_idx], targ[tr_idx])
    ds_va = TensorDataset(cond[val_idx], targ[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    model = CFMFlowModel(q_dim=7, cond_dim=COND_DIM,
                         hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"\nCFM model: hidden={args.hidden} depth={args.depth}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    log_path = os.path.join(args.ckpt_dir, "train_log_v18.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,sample_dist,sample_div,wall\n")

    t0 = time.perf_counter()
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0; train_n = 0
        for cond_b, targ_b in dl_tr:
            cond_b = cond_b.to(device); targ_b = targ_b.to(device)
            B = cond_b.shape[0]
            tau = torch.rand(B, 1, device=device, dtype=torch.float32)
            z = torch.randn_like(targ_b)
            v_tau = (1 - tau) * z + tau * targ_b
            target = targ_b - z
            pred = model(v_tau, tau.squeeze(-1), cond_b)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * B; train_n += B
        train_l = train_loss / train_n

        model.eval()
        val_loss = 0.0; val_n = 0
        sample_dist_sum = 0.0
        sample_div_sum = 0.0
        n_sampled_batches = 0
        with torch.no_grad():
            for cond_b, targ_b in dl_va:
                cond_b = cond_b.to(device); targ_b = targ_b.to(device)
                B = cond_b.shape[0]
                tau = torch.rand(B, 1, device=device, dtype=torch.float32)
                z = torch.randn_like(targ_b)
                v_tau = (1 - tau) * z + tau * targ_b
                target = targ_b - z
                pred = model(v_tau, tau.squeeze(-1), cond_b)
                loss = ((pred - target) ** 2).mean()
                val_loss += float(loss.item()) * B; val_n += B
                # sample q_curr and compare to GT
                if n_sampled_batches < 3:
                    s1 = model.sample(cond_b, n_steps=args.n_ode_steps)
                    s2 = model.sample(cond_b, n_steps=args.n_ode_steps)
                    sample_dist_sum += float((s1 - targ_b).norm(dim=-1).mean().item())
                    sample_div_sum  += float((s1 - s2).norm(dim=-1).mean().item())
                    n_sampled_batches += 1
        val_l = val_loss / val_n
        s_dist = sample_dist_sum / max(n_sampled_batches, 1)
        s_div  = sample_div_sum  / max(n_sampled_batches, 1)
        wall = time.perf_counter() - t0
        print(f"[epoch {epoch:>3d}/{args.epochs}]  train={train_l:.4f}  "
              f"val={val_l:.4f}  sample-to-GT={s_dist:.4f}  "
              f"sample-diversity={s_div:.4f}  ({wall:.1f}s)")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_l},{val_l},{s_dist},{s_div},{wall}\n")

        if val_l < best_val:
            best_val = val_l
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "args": vars(args)},
                       os.path.join(args.ckpt_dir, "best.pt"))
    torch.save({"epoch": args.epochs, "model": model.state_dict(),
                "args": vars(args)},
               os.path.join(args.ckpt_dir, f"ckpt_{args.epochs:03d}.pt"))
    print(f"\nbest val={best_val:.4f}")


if __name__ == "__main__":
    main()
