"""Train v6: token-aligned per-keypoint q prediction with 5-loss objective.

Loss = v_pred (per keypoint, masked)
     + λ_tcp     · TCP closure (FK(q_i_pred) → vertex_i_world)
     + λ_orient  · TCP_z direction (each keypoint's pen ≈ -desk_normal)
     + λ_smooth  · |q_{i+1} - q_i|² between adjacent valid keypoints
     + λ_margin  · max(0, |q_i_j - center_j|/span_j - 0.85)² (per joint, per keypoint)

Mirror augmentation flips ALL keypoints (not just q0). q7 canonicalization
applied per keypoint (set q7=0 throughout the trajectory).
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

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3GPU
from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    Q_CENTER,
    Q_HALF,
    denormalize_q,
    normalize_q,
    q_sample,
    v_target_from,
)
from Yuan.fr3_dit.training.task_cond_dit_q0_v6 import DiTq0Config_v6, TaskCondDiTq0_v6


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs" / "dit_q0_v6_ckpts"

DIR_LOCAL_OFFSET = 3
_FLIP_MULT = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float32)


class KeypointDataset(Dataset):
    """Streams (tokens, mask, keypoint_q_normalized, keypoint_mask, vertex_world,
    desk_normal_world) per task.

    keypoint_q_norm  : (max_tokens, 7)   normalized to joint-limit space.
                                          Position i holds q at the vertex this token
                                          maps to:
                                            kind 0 (START)   → vertex 0
                                            kind 1 (SEGMENT) → vertex i (end of seg)
                                            kind 2 (CORNER)  → same vertex as adjacent SEG
                                          Pad positions: zero (masked anyway).
    keypoint_mask    : (max_tokens,)     1 for any non-pad token (all kinds).
    vertex_world     : (max_tokens, 3)   world XYZ of the vertex this token maps to.
    """

    def __init__(self, h5_path, max_tokens=11, task_indices=None, mirror_prob=0.0):
        self.h5_path = Path(h5_path)
        self.max_tokens = int(max_tokens)
        self.mirror_prob = float(mirror_prob)
        # Preload all gzip-compressed datasets we'll iterate over into RAM. Total memory
        # for the 50k-task HDF5 is ≈ 200 MB (token_flat ~150 MB + qtraj_flat ~40 MB + small).
        # This kills the per-batch gzip-decompression bottleneck and should ~2x sps.
        print(f"[data] preloading HDF5 into RAM ({h5_path})...")
        with h5py.File(self.h5_path, "r") as f:
            ts = f["tasks"]
            self.token_offset = np.asarray(ts["token_offset"][()], dtype=np.int64)
            self.qtraj_offset = np.asarray(ts["qtraj_offset"][()], dtype=np.int64)
            self.subseg_offset = np.asarray(ts["subseg_offset"][()], dtype=np.int64)
            self.seg_step_counts_flat = np.asarray(ts["seg_step_counts_flat"][()], dtype=np.int32)
            self.local_origin = np.asarray(ts["local_origin"][()], dtype=np.float32)
            self.local_frame_all = np.asarray(ts["local_frame"][()], dtype=np.float32)
            self.token_flat = np.asarray(ts["token_flat"][()], dtype=np.float32)
            self.token_kind_flat = np.asarray(ts["token_kind"][()], dtype=np.uint8)
            self.qtraj_flat = np.asarray(ts["qtraj_flat"][()], dtype=np.float32)
            self.token_dim = int(f["meta"].attrs["token_dim"])
            self.length_ref = float(f["meta"].attrs["length_ref"])
            self.num_tasks_total = int(self.token_offset.shape[0] - 1)
        total_mb = (self.token_flat.nbytes + self.qtraj_flat.nbytes
                    + self.token_kind_flat.nbytes) / 1e6
        print(f"[data] preloaded: token_flat={self.token_flat.shape} qtraj_flat={self.qtraj_flat.shape}  "
              f"≈ {total_mb:.0f}MB RAM")
        if task_indices is None:
            self.task_indices = np.arange(self.num_tasks_total, dtype=np.int64)
        else:
            self.task_indices = np.asarray(task_indices, dtype=np.int64)

    def _ensure_open(self):
        # Kept for API compatibility — no longer needed since we preloaded.
        return None

    def __len__(self):
        return int(self.task_indices.shape[0])

    def close(self):
        if self._f is not None:
            self._f.close(); self._f = None

    def __getitem__(self, i):
        task_idx = int(self.task_indices[i])
        t_lo, t_hi = int(self.token_offset[task_idx]), int(self.token_offset[task_idx + 1])
        n_tok = t_hi - t_lo
        if n_tok > self.max_tokens:
            raise ValueError(f"task {task_idx}: tokens {n_tok} > max_tokens {self.max_tokens}")
        tokens = np.zeros((self.max_tokens, self.token_dim), dtype=np.float32)
        tokens[:n_tok] = self.token_flat[t_lo:t_hi]
        token_kind = np.zeros((self.max_tokens,), dtype=np.uint8)
        token_kind[:n_tok] = self.token_kind_flat[t_lo:t_hi]
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_mask[:n_tok] = 1.0

        # Per-task q-trajectory and step counts.
        q_lo, q_hi = int(self.qtraj_offset[task_idx]), int(self.qtraj_offset[task_idx + 1])
        s_lo, s_hi = int(self.subseg_offset[task_idx]), int(self.subseg_offset[task_idx + 1])
        step_counts = self.seg_step_counts_flat[s_lo:s_hi].astype(np.int64)   # (K,)
        n_segs = int(step_counts.shape[0])
        cum = np.concatenate([[0], np.cumsum(step_counts)])      # (K+1,)
        n_frames_task = q_hi - q_lo
        vertex_indices = np.minimum(cum, n_frames_task - 1)       # (K+1,)
        gt_kp_q = self.qtraj_flat[q_lo:q_hi][vertex_indices, :].astype(np.float32)

        # Compute world-frame vertex positions from polyline geometry.
        local_origin = self.local_origin[task_idx]                # (3,)
        local_frame = self.local_frame_all[task_idx]              # (3, 3)
        # Walk: vertex_0 = local_origin; vertex_i = vertex_{i-1} + local_frame @ dir_local_i * len_i
        seg_token_positions = np.where(token_kind == 1)[0]        # in token sequence
        vertices_world = np.zeros((n_segs + 1, 3), dtype=np.float32)
        vertices_world[0] = local_origin
        for k in range(n_segs):
            tok_pos = int(seg_token_positions[k])
            dir_local = tokens[tok_pos, DIR_LOCAL_OFFSET:DIR_LOCAL_OFFSET + 3].astype(np.float32)
            length_m = float(tokens[tok_pos, 6]) * float(self.length_ref)
            dir_world = local_frame @ dir_local
            n = float(np.linalg.norm(dir_world))
            if n < 1e-9:
                vertices_world[k + 1] = vertices_world[k]
            else:
                vertices_world[k + 1] = vertices_world[k] + (dir_world / n) * length_m

        # Build per-token aligned arrays of length max_tokens.
        # token kind 0 (START) → vertex 0
        # token kind 1 (SEGMENT) → vertex i (the i-th seg → end vertex i+1 in vertex list)
        # token kind 2 (CORNER) → same vertex as previous segment
        # We walk the token sequence and assign appropriately.
        kp_q = np.zeros((self.max_tokens, 7), dtype=np.float32)
        vertex_world = np.zeros((self.max_tokens, 3), dtype=np.float32)
        kp_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        seg_count_so_far = 0
        last_assigned_vidx = 0
        for j in range(self.max_tokens):
            if token_mask[j] < 0.5:
                continue
            kind = int(token_kind[j])
            if kind == 0:        # START
                v_idx = 0
            elif kind == 1:      # SEGMENT
                seg_count_so_far += 1
                v_idx = seg_count_so_far     # 1, 2, ..., n_segs
            elif kind == 2:      # CORNER
                v_idx = last_assigned_vidx   # repeat previous SEGMENT's vertex
            else:
                continue
            v_idx = min(v_idx, gt_kp_q.shape[0] - 1)
            kp_q[j] = gt_kp_q[v_idx]
            vertex_world[j] = vertices_world[min(v_idx, vertices_world.shape[0] - 1)]
            kp_mask[j] = 1.0
            last_assigned_vidx = v_idx

        # Canonicalize q7 (pen self-rotation) → 0 across the full keypoint sequence.
        kp_q[:, 6] = 0.0

        # Mirror augmentation across desk's xz plane.
        if self.mirror_prob > 0.0 and np.random.rand() < self.mirror_prob:
            kp_q = kp_q * _FLIP_MULT[None, :]
            vertex_world[:, 1] = -vertex_world[:, 1]
            tokens[0, DIR_LOCAL_OFFSET + 1] = -tokens[0, DIR_LOCAL_OFFSET + 1]

        kp_q_norm = (kp_q - Q_CENTER) / Q_HALF      # (max_tokens, 7) normalized
        return {
            "tokens": torch.from_numpy(tokens),
            "token_mask": torch.from_numpy(token_mask),
            "kp_q_norm": torch.from_numpy(kp_q_norm.astype(np.float32)),
            "kp_mask": torch.from_numpy(kp_mask),
            "vertex_world": torch.from_numpy(vertex_world),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def make_splits(n, val_frac, seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    n_val = max(2048, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def update_ema(ema, model, decay):
    with torch.no_grad():
        for n, p in model.named_parameters():
            ema[n].mul_(decay).add_(p.detach(), alpha=1 - decay)


def save_ckpt(path, model, optimizer, ema, step, cfg, args, T):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "model": model.state_dict(), "ema": ema,
        "optimizer": optimizer.state_dict(), "cfg": cfg.__dict__,
        "T": T, "args": vars(args),
    }, path)


def parse_args():
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
    p.add_argument("--num-steps", type=int, default=60000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)
    p.add_argument("--mirror-prob", type=float, default=0.5)
    p.add_argument("--lambda-tcp", type=float, default=5.0)
    p.add_argument("--lambda-orient", type=float, default=2.0)
    p.add_argument("--lambda-smooth", type=float, default=1.0)
    p.add_argument("--lambda-margin", type=float, default=0.5)
    p.add_argument("--margin-threshold", type=float, default=0.85,
                   help="|q-center|/span above which the joint-margin loss kicks in.")
    p.add_argument("--orient-loss", type=str, default="l2", choices=["l2", "hinge"],
                   help="l2: ||tcp_z - (0,0,-1)||² (v6 default). "
                        "hinge: max(0, cos(30°) - cos_theta)² — only penalize out-of-cone.")
    p.add_argument("--theta-max-deg", type=float, default=30.0,
                   help="Cone half-angle for hinge orient loss.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[data] loading {args.data}")
    seed_ds = KeypointDataset(args.data, args.max_tokens)
    train_idx, val_idx = make_splits(seed_ds.num_tasks_total, args.val_frac, args.seed)
    train_ds = KeypointDataset(args.data, args.max_tokens, train_idx, mirror_prob=args.mirror_prob)
    val_ds = KeypointDataset(args.data, args.max_tokens, val_idx, mirror_prob=0.0)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} mirror_prob={args.mirror_prob}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, collate_fn=collate,
    )

    cfg = DiTq0Config_v6(
        token_dim=seed_ds.token_dim, max_tokens=args.max_tokens,
        d_model=args.d_model, n_head=args.n_head,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        dropout=args.dropout, diffusion_steps=args.diffusion_steps,
    )
    model = TaskCondDiTq0_v6(cfg).to(device)
    schedule = DDPMCosineSchedule(T=cfg.diffusion_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] TaskCondDiTq0_v6 d_model={args.d_model} enc={args.n_enc_layers} "
          f"dec={args.n_dec_layers} params={n_params:.2f}M  cfg_drop={args.cfg_drop_prob}")

    fr3 = PenFrankaResearch3GPU(device)
    with h5py.File(args.data, "r") as fh:
        desk_normal_np = np.asarray(fh["meta"].attrs["source_desk_normal"], dtype=np.float32)
    desk_normal_np /= max(float(np.linalg.norm(desk_normal_np)), 1e-12)
    tcp_z_target = torch.tensor(-desk_normal_np, device=device, dtype=torch.float32)
    print(f"[loss] λ_tcp={args.lambda_tcp} λ_orient={args.lambda_orient} "
          f"λ_smooth={args.lambda_smooth} λ_margin={args.lambda_margin}")

    ema = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, betas=(0.9, 0.99))
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.ckpt_dir / "train.log"
    with open(log_path, "w") as f:
        f.write(f"# cfg {json.dumps(cfg.__dict__)}\n# args {json.dumps(vars(args), default=str)}\n")

    Q_CENTER_T = torch.as_tensor(Q_CENTER, device=device, dtype=torch.float32)
    Q_HALF_T = torch.as_tensor(Q_HALF, device=device, dtype=torch.float32)
    margin_threshold = float(args.margin_threshold)
    use_hinge_orient = (args.orient_loss == "hinge")
    cos_theta_max = float(np.cos(np.deg2rad(args.theta_max_deg)))
    print(f"[loss] orient_loss={args.orient_loss}  theta_max={args.theta_max_deg}°  "
          f"margin_threshold={margin_threshold}")

    def compute_losses(x0, xt, v_pred, kp_mask, vertex_world, t):
        """x0: GT normalized keypoint q (B, T, 7).  v_pred: (B, T, 7).
        kp_mask: (B, T) float, vertex_world: (B, T, 3).  t: (B,)."""
        # Recover x0_hat from v_pred (for the auxiliary geometric losses).
        ba_t = schedule.alphas_cumprod.gather(0, t).view(-1, 1, 1)         # (B,1,1)
        alpha_t = ba_t.sqrt(); sigma_t = (1 - ba_t).sqrt()
        x0_hat = alpha_t * xt - sigma_t * v_pred                             # (B, T, 7) normalized

        # v = α·ε − σ·x0 ; recover ε from (xt = α·x0 + σ·ε).
        eps = (xt - alpha_t * x0) / sigma_t.clamp_min(1e-6)
        v_target = alpha_t * eps - sigma_t * x0

        # Masked v-pred loss.
        v_sq = (v_pred - v_target) ** 2                                       # (B, T, 7)
        m = kp_mask.unsqueeze(-1)
        denom = m.sum() * 7 + 1e-6
        loss_v = (v_sq * m).sum() / denom

        # x0_hat → q_pred_raw via denormalize, FK for TCP closure & orient.
        q_pred_raw = x0_hat * Q_HALF_T + Q_CENTER_T                           # (B, T, 7)
        # Time weighting α² makes losses focus on clean regime (matches v3/v5).
        w = (alpha_t ** 2).squeeze(-1)                                        # (B, 1)
        # FK in batched mode — flatten (B*T, 7).
        q_flat = q_pred_raw.reshape(-1, 7)
        tcp_pred, tcp_rot_pred = fr3.robot.fk_batch(q_flat)                    # (B*T, 3), (B*T, 3, 3)
        tcp_pred = tcp_pred.reshape(*q_pred_raw.shape[:2], 3)                  # (B, T, 3)
        tcp_z_pred = tcp_rot_pred[:, :, 2].reshape(*q_pred_raw.shape[:2], 3)   # (B, T, 3)

        tcp_sq = ((tcp_pred - vertex_world) ** 2).sum(dim=-1)                  # (B, T)
        loss_tcp = (w * (tcp_sq * kp_mask).sum(dim=-1) / kp_mask.sum(dim=-1).clamp_min(1)).mean()

        if use_hinge_orient:
            # Hinge: only penalize out-of-cone predictions. cos_theta = (tcp_z · -desk_n).
            cos_theta = (tcp_z_pred * tcp_z_target.view(1, 1, 3)).sum(dim=-1)         # (B, T)
            orient_sq = F.relu(cos_theta_max - cos_theta) ** 2                          # (B, T)
        else:
            orient_sq = ((tcp_z_pred - tcp_z_target.view(1, 1, 3)) ** 2).sum(dim=-1)    # (B, T)
        loss_orient = (w * (orient_sq * kp_mask).sum(dim=-1) / kp_mask.sum(dim=-1).clamp_min(1)).mean()

        # Smoothness: sum |q_pred[i+1] - q_pred[i]|² where both i and i+1 are valid kp.
        q_diff = q_pred_raw[:, 1:] - q_pred_raw[:, :-1]                        # (B, T-1, 7)
        adj_mask = kp_mask[:, 1:] * kp_mask[:, :-1]                            # (B, T-1)
        smooth_sq = (q_diff ** 2).sum(dim=-1)                                  # (B, T-1)
        loss_smooth = (smooth_sq * adj_mask).sum() / (adj_mask.sum() + 1e-6)

        # Joint margin: penalize predictions where any joint > margin_threshold * span from center.
        norm_pos = torch.abs((q_pred_raw - Q_CENTER_T) / Q_HALF_T)             # (B, T, 7) ∈ [0, ~1]
        excess = (norm_pos - margin_threshold).clamp_min(0.0)                  # (B, T, 7)
        margin_sq = (excess ** 2).sum(dim=-1)                                  # (B, T)
        loss_margin = (margin_sq * kp_mask).sum() / kp_mask.sum().clamp_min(1)

        loss = (loss_v
                + args.lambda_tcp * loss_tcp
                + args.lambda_orient * loss_orient
                + args.lambda_smooth * loss_smooth
                + args.lambda_margin * loss_margin)
        return loss, loss_v, loss_tcp, loss_orient, loss_smooth, loss_margin

    model.train()
    t0 = time.time(); step = 0
    train_iter = iter(train_loader)
    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)

        kp_q_norm = batch["kp_q_norm"].to(device, non_blocking=True)              # (B, T, 7)
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_mask = batch["token_mask"].to(device, non_blocking=True)
        kp_mask = batch["kp_mask"].to(device, non_blocking=True)
        vertex_world = batch["vertex_world"].to(device, non_blocking=True)
        B, T, _ = kp_q_norm.shape

        # Sample t and noise per batch item; same t broadcasts across keypoints.
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        ba_t = schedule.alphas_cumprod.gather(0, t).view(-1, 1, 1)
        alpha_t = ba_t.sqrt(); sigma_t = (1 - ba_t).sqrt()
        eps = torch.randn_like(kp_q_norm)
        xt = alpha_t * kp_q_norm + sigma_t * eps

        uncond_mask = torch.rand(B, device=device) < args.cfg_drop_prob

        with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
            v_pred = model(xt, t, tokens, token_mask, uncond_mask=uncond_mask)
            loss, l_v, l_tcp, l_ori, l_sm, l_mg = compute_losses(
                kp_q_norm, xt, v_pred, kp_mask, vertex_world, t,
            )

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
            msg = (f"[step {step:6d}/{args.num_steps}] loss={loss.item():.4f} "
                   f"v={l_v.item():.4f} tcp={l_tcp.item():.4f} ori={l_ori.item():.4f} "
                   f"sm={l_sm.item():.4f} mg={l_mg.item():.4f} "
                   f"gnorm={gnorm.item():.3f} lr={lr_now:.2e} sps={sps:.2f}")
            print(msg)
            with open(log_path, "a") as f: f.write(msg + "\n")

        if step % args.val_every == 0:
            model.eval()
            v_losses, tcp_losses, ori_losses, sm_losses, mg_losses = [], [], [], [], []
            with torch.no_grad():
                for vb in val_loader:
                    x0v = vb["kp_q_norm"].to(device); kpmv = vb["kp_mask"].to(device)
                    tkv = vb["tokens"].to(device); tmv = vb["token_mask"].to(device)
                    vw = vb["vertex_world"].to(device)
                    Bv = x0v.shape[0]
                    tv = torch.randint(0, schedule.T, (Bv,), device=device, dtype=torch.long)
                    ba_v = schedule.alphas_cumprod.gather(0, tv).view(-1, 1, 1)
                    av = ba_v.sqrt(); sv = (1 - ba_v).sqrt()
                    epsv = torch.randn_like(x0v); xtv = av * x0v + sv * epsv
                    umv = torch.rand(Bv, device=device) < args.cfg_drop_prob
                    vp = model(xtv, tv, tkv, tmv, uncond_mask=umv)
                    _, lv, lt, lo, lsm, lmg = compute_losses(x0v, xtv, vp, kpmv, vw, tv)
                    v_losses.append(lv.item()); tcp_losses.append(lt.item())
                    ori_losses.append(lo.item()); sm_losses.append(lsm.item())
                    mg_losses.append(lmg.item())
                    if len(v_losses) >= 20: break
            print(f"[val   step {step:6d}] val_v={np.mean(v_losses):.4f} "
                  f"val_tcp={np.mean(tcp_losses):.4f} val_ori={np.mean(ori_losses):.4f} "
                  f"val_sm={np.mean(sm_losses):.4f} val_mg={np.mean(mg_losses):.4f}")
            with open(log_path, "a") as f:
                f.write(f"[val step {step}] val_v={np.mean(v_losses):.4f} "
                        f"val_tcp={np.mean(tcp_losses):.4f} val_ori={np.mean(ori_losses):.4f} "
                        f"val_sm={np.mean(sm_losses):.4f} val_mg={np.mean(mg_losses):.4f}\n")
            model.train()

        if step % args.ckpt_every == 0 or step == args.num_steps:
            p = args.ckpt_dir / f"step_{step:06d}.pt"
            save_ckpt(p, model, optimizer, ema, step, cfg, args, schedule.T)
            print(f"[ckpt] saved {p}")

    save_ckpt(args.ckpt_dir / "final.pt", model, optimizer, ema, step, cfg, args, schedule.T)
    print(f"[done] final → {args.ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
