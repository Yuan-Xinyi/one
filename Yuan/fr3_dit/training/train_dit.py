#!/usr/bin/env python3
"""Train the task-conditioned DiT on composite FR3 trajectories.

Data: composite-task HDF5 produced by fr3_dit/stitching/.
Target: joint-space trajectory q(t) ∈ R^7 resampled to fixed length (default 512).
Condition: variable-length task-token sequence (padded to max_tokens).
Noise model: DDPM with cosine β-schedule.

Produces:
  - checkpoints every `--ckpt-every` steps under `ckpt-dir`
  - `train.log` with per-step loss + per-epoch timing
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Yuan.fr3_dit.training.task_cond_dit import (
    DDPMCosineSchedule,
    DiTConfig,
    TaskCondDiT,
    q_sample,
)


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k.hdf5"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_ckpts"


class CompositeQDataset(Dataset):
    """Streams task-token conditioning + fixed-length joint-trajectory targets.

    Uses HDF5 random access; reopens the file lazily per worker.
    """

    def __init__(
        self,
        h5_path: Path,
        target_qsteps: int = 512,
        max_tokens: int = 11,
        task_indices: np.ndarray | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.target_qsteps = int(target_qsteps)
        self.max_tokens = int(max_tokens)
        with h5py.File(self.h5_path, "r") as f:
            ts = f["tasks"]
            self.token_offset = np.asarray(ts["token_offset"][()], dtype=np.int64)
            self.qtraj_offset = np.asarray(ts["qtraj_offset"][()], dtype=np.int64)
            self.token_dim = int(f["meta"].attrs["token_dim"])
            self.num_tasks_total = int(self.token_offset.shape[0] - 1)
        if task_indices is None:
            self.task_indices = np.arange(self.num_tasks_total, dtype=np.int64)
        else:
            self.task_indices = np.asarray(task_indices, dtype=np.int64)
        self._f: h5py.File | None = None

    def _ensure_open(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r", swmr=True)
        return self._f

    def __len__(self) -> int:
        return int(self.task_indices.shape[0])

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __getstate__(self):
        s = self.__dict__.copy(); s["_f"] = None; return s

    def __setstate__(self, s):
        self.__dict__.update(s)

    def _resample_qtraj(self, q: np.ndarray) -> np.ndarray:
        T = q.shape[0]
        if T == self.target_qsteps:
            return q.astype(np.float32)
        src = np.linspace(0.0, T - 1, self.target_qsteps, dtype=np.float32)
        lo = np.clip(np.floor(src).astype(np.int64), 0, T - 1)
        hi = np.clip(lo + 1, 0, T - 1)
        frac = (src - lo.astype(np.float32))[:, None]
        out = (1 - frac) * q[lo] + frac * q[hi]
        return out.astype(np.float32)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        task_idx = int(self.task_indices[i])
        f = self._ensure_open()
        ts = f["tasks"]
        t_lo, t_hi = int(self.token_offset[task_idx]), int(self.token_offset[task_idx + 1])
        q_lo, q_hi = int(self.qtraj_offset[task_idx]), int(self.qtraj_offset[task_idx + 1])
        n_tok = t_hi - t_lo
        if n_tok > self.max_tokens:
            raise ValueError(f"task {task_idx}: tokens {n_tok} > max_tokens {self.max_tokens}")

        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_mask[:n_tok] = 1.0

        q = np.asarray(ts["qtraj_flat"][q_lo:q_hi], dtype=np.float32)  # (T_raw, 7)
        q_target = self._resample_qtraj(q)

        return {
            "tokens": torch.from_numpy(tokens),
            "token_mask": torch.from_numpy(token_mask),
            "qtraj": torch.from_numpy(q_target),  # (target_qsteps, 7)
            "qtraj_mask": torch.ones(self.target_qsteps, dtype=torch.float32),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def make_splits(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1024, int(n * val_frac))
    val = perm[:n_val]
    train = perm[n_val:]
    return train, val


def save_ckpt(path: Path, model: TaskCondDiT, optimizer, ema, step: int, cfg: DiTConfig, schedule: DDPMCosineSchedule, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema,
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "T": schedule.T,
        "args": vars(args),
    }, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--target-qsteps", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=11)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def update_ema(ema_params: dict, model: TaskCondDiT, decay: float) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            ema_params[name].mul_(decay).add_(p.detach(), alpha=1 - decay)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset + splits
    print(f"[data] loading {args.data}")
    full = CompositeQDataset(args.data, target_qsteps=args.target_qsteps, max_tokens=args.max_tokens)
    train_idx, val_idx = make_splits(full.num_tasks_total, args.val_frac, args.seed)
    train_ds = CompositeQDataset(args.data, args.target_qsteps, args.max_tokens, train_idx)
    val_ds = CompositeQDataset(args.data, args.target_qsteps, args.max_tokens, val_idx)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} token_dim={full.token_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
        collate_fn=collate,
    )

    # Model + schedule
    cfg = DiTConfig(
        act_dim=7, token_dim=full.token_dim,
        d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers,
        dropout=args.dropout, max_qsteps=args.target_qsteps, max_tokens=args.max_tokens,
        diffusion_steps=args.diffusion_steps,
    )
    model = TaskCondDiT(cfg).to(device)
    schedule = DDPMCosineSchedule(T=args.diffusion_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] TaskCondDiT d_model={args.d_model} layers={args.n_layers} heads={args.n_head} params={n_params:.2f}M")

    # EMA
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Logging
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.ckpt_dir / "train.log"
    with open(log_path, "w") as f:
        f.write(f"# cfg {json.dumps(cfg.__dict__)}\n# args {json.dumps(vars(args), default=str)}\n")

    # Training loop
    model.train()
    t0 = time.time()
    step = 0
    train_iter = iter(train_loader)
    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x0 = batch["qtraj"].to(device, non_blocking=True)            # (B, T_q, 7)
        tokens = batch["tokens"].to(device, non_blocking=True)       # (B, T_tok, D)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        qtraj_mask = batch["qtraj_mask"].to(device, non_blocking=True)

        t = torch.randint(0, schedule.T, (x0.shape[0],), device=device)
        xt, eps = q_sample(x0, t, schedule)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(xt, t, tokens, token_mask, qtraj_mask)
            # Predict noise ε (standard DDPM). Mask padded q-steps (here all-ones, so no-op).
            loss = F.mse_loss(pred, eps, reduction="none")
            loss = (loss * qtraj_mask.unsqueeze(-1)).sum() / (qtraj_mask.sum() * pred.shape[-1] + 1e-8)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update()
        sched_lr.step()

        update_ema(ema, model, args.ema_decay)
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            steps_per_s = step / max(dt, 1e-6)
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"[step {step:6d}/{args.num_steps}] loss={loss.item():.4f} "
                f"gnorm={gnorm.item():.3f} lr={lr_now:.2e} sps={steps_per_s:.2f}"
            )
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        if step % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    x0v = vb["qtraj"].to(device)
                    tv = torch.randint(0, schedule.T, (x0v.shape[0],), device=device)
                    xtv, epsv = q_sample(x0v, tv, schedule)
                    predv = model(
                        xtv, tv,
                        vb["tokens"].to(device), vb["token_mask"].to(device), vb["qtraj_mask"].to(device),
                    )
                    val_losses.append(F.mse_loss(predv, epsv).item())
                    if len(val_losses) >= 20:
                        break
            v = float(np.mean(val_losses))
            print(f"[val   step {step:6d}] val_loss={v:.4f}")
            with open(log_path, "a") as f:
                f.write(f"[val step {step}] val_loss={v:.4f}\n")
            model.train()

        if step % args.ckpt_every == 0 or step == args.num_steps:
            ckpt = args.ckpt_dir / f"step_{step:06d}.pt"
            save_ckpt(ckpt, model, optimizer, ema, step, cfg, schedule, args)
            print(f"[ckpt] saved {ckpt}")

    # Final ckpt
    save_ckpt(args.ckpt_dir / "final.pt", model, optimizer, ema, step, cfg, schedule, args)
    print(f"[done] final checkpoint → {args.ckpt_dir/'final.pt'}")


if __name__ == "__main__":
    main()
