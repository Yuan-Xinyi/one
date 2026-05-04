#!/usr/bin/env python3
"""Train the q₀-predicting DiT.

For each composite task in the filtered HDF5 we only need:
  - the variable-length token sequence (conditioning)
  - the scalar ``start_q`` (7-D initial joint config, target)

v-prediction + CFG dropout (p=0.1) + FR3 joint-limit normalization to keep
everything in [-1, 1].
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    DiTq0Config,
    TaskCondDiTq0,
    denormalize_q,
    normalize_q,
    q_sample,
    v_target_from,
)
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3GPU


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_q0_v3_ckpts"

# Token slot offsets inside the 32-D token vector
DIR_LOCAL_OFFSET = 3   # right after the 3-dim kind_onehot

# Mirror augmentation: which joints flip sign when we reflect the desk across xz plane.
# Standard Franka kinematics convention — verified at runtime via FK.
MIRROR_JOINT_FLIPS = (0, 2, 4, 6)
# Boolean mask form for vectorized multiply
_FLIP_MULT = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float32)


class StartQDataset(Dataset):
    """Streams (tokens, token_mask, start_q_normalized, tcp_target) per composite task.

    Optional mirror augmentation (reflect across desk's xz plane): with prob ``mirror_prob``
    we negate y-component of the spatial-anchor token, sign-flip joints {0,2,4,6}, and
    flip y of the GT TCP target. Tokens otherwise are invariant (they live in a per-task
    local frame that flips with the world).
    """

    def __init__(
        self,
        h5_path: Path,
        max_tokens: int = 11,
        task_indices: np.ndarray | None = None,
        mirror_prob: float = 0.0,
    ):
        self.h5_path = Path(h5_path)
        self.max_tokens = int(max_tokens)
        self.mirror_prob = float(mirror_prob)
        with h5py.File(self.h5_path, "r") as f:
            ts = f["tasks"]
            self.token_offset = np.asarray(ts["token_offset"][()], dtype=np.int64)
            self.token_dim = int(f["meta"].attrs["token_dim"])
            # start_q is small — load fully into RAM.
            self.start_q = np.asarray(ts["start_q"][()], dtype=np.float32)  # (M, 7), raw rad
            self.tcp_target = np.asarray(ts["local_origin"][()], dtype=np.float32)  # (M, 3) world
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

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        task_idx = int(self.task_indices[i])
        f = self._ensure_open()
        ts = f["tasks"]
        t_lo, t_hi = int(self.token_offset[task_idx]), int(self.token_offset[task_idx + 1])
        n_tok = t_hi - t_lo
        if n_tok > self.max_tokens:
            raise ValueError(f"task {task_idx}: tokens {n_tok} > max_tokens {self.max_tokens}")

        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        tokens[:n_tok] = ts["token_flat"][t_lo:t_hi]
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_mask[:n_tok] = 1.0

        q0_raw = self.start_q[task_idx].copy()
        tcp_gt = self.tcp_target[task_idx].copy()  # (3,) world

        # Canonicalize q7 (pen self-rotation): the original sampled q7 was task-irrelevant
        # noise. Snapping it to 0 turns "predict random q7" into the well-defined function
        # "predict q7 = 0", which the model can actually fit and which automatically
        # eliminates the ~20% out-of-limit rate observed in v3.
        q0_raw[6] = 0.0

        # Mirror augmentation across desk's xz plane (y → -y).
        if self.mirror_prob > 0.0 and np.random.rand() < self.mirror_prob:
            q0_raw = q0_raw * _FLIP_MULT
            tcp_gt[1] = -tcp_gt[1]
            tokens[0, DIR_LOCAL_OFFSET + 1] = -tokens[0, DIR_LOCAL_OFFSET + 1]
            # Tokens describe shape in a per-task local frame that mirrors with the world,
            # so segment dir_local / corner axis_local entries are invariant under mirror.

        q0_norm = normalize_q(q0_raw).astype(np.float32)

        return {
            "tokens": torch.from_numpy(tokens),
            "token_mask": torch.from_numpy(token_mask),
            "q0": torch.from_numpy(q0_norm),
            "tcp_gt": torch.from_numpy(tcp_gt.astype(np.float32)),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def make_splits(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(2048, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def update_ema(ema: dict, model: TaskCondDiTq0, decay: float) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            ema[n].mul_(decay).add_(p.detach(), alpha=1 - decay)


def save_ckpt(path: Path, model, optimizer, ema, step, cfg, schedule, args) -> None:
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
    p.add_argument("--max-tokens", type=int, default=11)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-enc-layers", type=int, default=4)
    p.add_argument("--n-dec-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    # Training
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=40000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1,
                   help="Probability of replacing the condition with the null token during training.")
    p.add_argument("--mirror-prob", type=float, default=0.5,
                   help="Probability of mirror-augmenting each sample across desk xz plane.")
    p.add_argument("--lambda-tcp", type=float, default=5.0,
                   help="Weight on the TCP-position auxiliary loss.")
    p.add_argument("--lambda-orient", type=float, default=2.0,
                   help="Weight on the TCP_z direction (pen-into-desk) loss.")
    p.add_argument("--mask-q7", action="store_true", default=False,
                   help="Zero-out q7 in the v-loss. Default off because data now canonicalizes q7=0 "
                        "and the model can learn that explicitly.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[data] loading {args.data}")
    seed_ds = StartQDataset(args.data, args.max_tokens)
    train_idx, val_idx = make_splits(seed_ds.num_tasks_total, args.val_frac, args.seed)
    train_ds = StartQDataset(args.data, args.max_tokens, train_idx, mirror_prob=args.mirror_prob)
    val_ds = StartQDataset(args.data, args.max_tokens, val_idx, mirror_prob=0.0)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} token_dim={seed_ds.token_dim} mirror_prob={args.mirror_prob}")
    print(f"[data] start_q range (raw): min={seed_ds.start_q.min(axis=0)} max={seed_ds.start_q.max(axis=0)}")

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

    cfg = DiTq0Config(
        act_dim=7, token_dim=seed_ds.token_dim, max_tokens=args.max_tokens,
        d_model=args.d_model, n_head=args.n_head,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        dropout=args.dropout, diffusion_steps=args.diffusion_steps, pred_type="v",
    )
    model = TaskCondDiTq0(cfg).to(device)
    schedule = DDPMCosineSchedule(T=args.diffusion_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[model] TaskCondDiTq0 d_model={args.d_model} enc={args.n_enc_layers} dec={args.n_dec_layers} "
        f"params={n_params:.2f}M  pred=v  cfg_drop={args.cfg_drop_prob}"
    )

    # Differentiable FR3 FK on GPU (used for the TCP-space auxiliary loss).
    fr3 = PenFrankaResearch3GPU(device)
    # Per-joint loss weight: zero out q7 if requested.
    q_weight = torch.ones(7, device=device)
    if args.mask_q7:
        q_weight[6] = 0.0
        print(f"[loss] mask_q7=on → v-loss weights={q_weight.tolist()}")
    # TCP_z target: pen points opposite the desk's outward normal.
    with h5py.File(args.data, "r") as fh:
        desk_normal_np = np.asarray(fh["meta"].attrs["source_desk_normal"], dtype=np.float32)
    desk_normal_np = desk_normal_np / max(float(np.linalg.norm(desk_normal_np)), 1e-12)
    tcp_z_target = torch.tensor(-desk_normal_np, device=device, dtype=torch.float32)
    print(f"[loss] lambda_tcp={args.lambda_tcp}  lambda_orient={args.lambda_orient}  "
          f"tcp_z_target={tcp_z_target.cpu().tolist()}")

    ema = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.ckpt_dir / "train.log"
    with open(log_path, "w") as f:
        f.write(f"# cfg {json.dumps(cfg.__dict__)}\n# args {json.dumps(vars(args), default=str)}\n")

    model.train()
    t0 = time.time()
    step = 0
    train_iter = iter(train_loader)
    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)

        q0 = batch["q0"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        tcp_gt = batch["tcp_gt"].to(device, non_blocking=True)  # (B, 3) world
        B = q0.shape[0]

        # Diffusion forward + v-target
        t = torch.randint(0, schedule.T, (B,), device=device)
        xt, eps = q_sample(q0, t, schedule)
        v_gt = v_target_from(q0, eps, t, schedule)

        # CFG dropout: flip each sample to unconditional with prob p
        uncond_mask = torch.rand(B, device=device) < args.cfg_drop_prob

        with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
            v_pred = model(xt, t, tokens, token_mask, uncond_mask=uncond_mask)

            # v-loss with optional q7 masking
            v_sq = (v_pred - v_gt) ** 2                          # (B, 7)
            loss_v = (v_sq * q_weight).sum(dim=-1).mean() / q_weight.sum()

            # TCP auxiliary loss: recover x0_hat from v_pred, denormalize, FK, MSE on
            # both TCP position and TCP_z direction (pen-into-desk).
            bar = schedule.alphas_cumprod.gather(0, t).view(-1, 1)
            alpha_t = bar.sqrt()
            sigma_t = (1 - bar).sqrt()
            x0_hat = alpha_t * xt - sigma_t * v_pred             # (B, 7) normalized
            q0_pred_raw = denormalize_q(x0_hat)
            tcp_pred, tcp_rot_pred = fr3.robot.fk_batch(q0_pred_raw)
            tcp_z_pred = tcp_rot_pred[:, :, 2]                   # (B, 3) world

            tcp_sq = ((tcp_pred - tcp_gt) ** 2).sum(dim=-1)
            orient_sq = ((tcp_z_pred - tcp_z_target.unsqueeze(0)) ** 2).sum(dim=-1)
            w = alpha_t.squeeze(-1) ** 2
            loss_tcp = (w * tcp_sq).mean()
            loss_orient = (w * orient_sq).mean()

            loss = loss_v + args.lambda_tcp * loss_tcp + args.lambda_orient * loss_orient

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update()
        sched_lr.step()
        update_ema(ema, model, args.ema_decay)
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0; sps = step / max(dt, 1e-6)
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"[step {step:6d}/{args.num_steps}] loss={loss.item():.4f} "
                f"v={loss_v.item():.4f} tcp={loss_tcp.item():.4f} ori={loss_orient.item():.4f} "
                f"gnorm={gnorm.item():.3f} lr={lr_now:.2e} sps={sps:.2f}"
            )
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        if step % args.val_every == 0:
            model.eval()
            v_losses, tcp_losses, ori_losses = [], [], []
            with torch.no_grad():
                for vb in val_loader:
                    x0v = vb["q0"].to(device)
                    tcp_gt_v = vb["tcp_gt"].to(device)
                    tv = torch.randint(0, schedule.T, (x0v.shape[0],), device=device)
                    xtv, epsv = q_sample(x0v, tv, schedule)
                    vgt = v_target_from(x0v, epsv, tv, schedule)
                    um = torch.rand(x0v.shape[0], device=device) < args.cfg_drop_prob
                    vp = model(xtv, tv,
                               vb["tokens"].to(device),
                               vb["token_mask"].to(device),
                               uncond_mask=um)
                    vsq = ((vp - vgt) ** 2 * q_weight).sum(dim=-1).mean() / q_weight.sum()
                    bar = schedule.alphas_cumprod.gather(0, tv).view(-1, 1)
                    a_v, s_v = bar.sqrt(), (1 - bar).sqrt()
                    x0_hat_v = a_v * xtv - s_v * vp
                    tcp_pred_v, tcp_rot_pred_v = fr3.robot.fk_batch(denormalize_q(x0_hat_v))
                    tcp_z_pred_v = tcp_rot_pred_v[:, :, 2]
                    tcp_sq = ((tcp_pred_v - tcp_gt_v) ** 2).sum(dim=-1)
                    ori_sq = ((tcp_z_pred_v - tcp_z_target.unsqueeze(0)) ** 2).sum(dim=-1)
                    w_v = a_v.squeeze(-1) ** 2
                    v_losses.append(vsq.item())
                    tcp_losses.append((w_v * tcp_sq).mean().item())
                    ori_losses.append((w_v * ori_sq).mean().item())
                    if len(v_losses) >= 20:
                        break
            v_avg = float(np.mean(v_losses))
            tcp_avg = float(np.mean(tcp_losses))
            ori_avg = float(np.mean(ori_losses))
            print(f"[val   step {step:6d}] val_v={v_avg:.4f}  val_tcp={tcp_avg:.4f}  val_ori={ori_avg:.4f}")
            with open(log_path, "a") as f:
                f.write(f"[val step {step}] val_v={v_avg:.4f} val_tcp={tcp_avg:.4f} val_ori={ori_avg:.4f}\n")
            model.train()

        if step % args.ckpt_every == 0 or step == args.num_steps:
            p = args.ckpt_dir / f"step_{step:06d}.pt"
            save_ckpt(p, model, optimizer, ema, step, cfg, schedule, args)
            print(f"[ckpt] saved {p}")

    save_ckpt(args.ckpt_dir / "final.pt", model, optimizer, ema, step, cfg, schedule, args)
    print(f"[done] final → {args.ckpt_dir/'final.pt'}")


if __name__ == "__main__":
    main()
