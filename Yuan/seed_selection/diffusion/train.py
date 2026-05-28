"""Minimal pilot training for c → q0 diffusion.

Loads SeedSelectionDataset, trains SeedQ0DiT with v-prediction + DDPM cosine
schedule + EMA. Joint-limit normalization is the same as fr3_dit's q0-DiT.

Adds train/val split: a fixed-seed random split saved as `split.json` in the
ckpt dir. Same split is reused on resume and by `eval/eval_joint_distance.py`.

Usage:
    python -m Yuan.seed_selection.diffusion.train
    python -m Yuan.seed_selection.diffusion.train --resume .../step_20000.pt --num-steps 50000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    normalize_q,
    q_sample,
    v_target_from,
)
from Yuan.seed_selection.diffusion.dataset import SeedSelectionDataset
from Yuan.seed_selection.diffusion.model import SeedQ0Config, SeedQ0DiT


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz'
DEFAULT_CKPT_DIR = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/q0_20k_cfg_mirror_ckpts'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DEFAULT_NPZ)
    p.add_argument('--ckpt-dir', type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument('--resume', type=Path, default=None,
                   help='ckpt to load model/ema/optimizer/step from before continuing.')
    # Split
    p.add_argument('--val-n', type=int, default=100,
                   help='# tasks held out for validation (random with --val-seed).')
    p.add_argument('--val-seed', type=int, default=0)
    # Model (ignored on resume: cfg is read from ckpt)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--diffusion-steps', type=int, default=1000)
    # Training
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-steps', type=int, default=20000,
                   help='ABSOLUTE training step to stop at (incl. resumed steps).')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--ema-decay', type=float, default=0.9995)
    p.add_argument('--cfg-drop-prob', type=float, default=0.0,
                   help='probability of replacing c with null condition each step '
                        '(classifier-free guidance dropout). 0 = no CFG training.')
    p.add_argument('--mirror-prob', type=float, default=0.0,
                   help='probability of mirror-flipping a sample across the robot\'s '
                        'xz plane (y → -y in c; sign-flip joints 0/2/4/6 in q0). '
                        'Applied only to the TRAIN set; val never mirrors.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--val-every', type=int, default=500,
                   help='compute and log val loss every N steps.')
    p.add_argument('--val-batches', type=int, default=8,
                   help='# diffusion-step batches to average val loss over (per --val-every).')
    p.add_argument('--ckpt-every', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    # WandB
    p.add_argument('--wandb-project', default='seed_selection_q0',
                   help='WandB project name. Pass empty string or --no-wandb to disable.')
    p.add_argument('--wandb-entity', default=None,
                   help='WandB entity/team. Default: account default.')
    p.add_argument('--wandb-name', default=None,
                   help='WandB run name. Default: derived from --ckpt-dir.')
    p.add_argument('--no-wandb', action='store_true', help='disable WandB entirely')
    return p.parse_args()


def make_or_load_split(ckpt_dir: Path, n_total: int, val_n: int, val_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Persisted random split. Saves split.json with train_idx, val_idx (rel. to filtered ds)."""
    split_path = ckpt_dir / 'split.json'
    if split_path.exists():
        s = json.loads(split_path.read_text())
        if int(s.get('n_total', -1)) != n_total or int(s.get('val_n', -1)) != val_n or int(s.get('val_seed', -1)) != val_seed:
            raise RuntimeError(
                f'Existing {split_path} has different params '
                f'(n_total={s.get("n_total")} val_n={s.get("val_n")} val_seed={s.get("val_seed")}). '
                f'Delete it or use matching args.'
            )
        return np.array(s['train_idx'], dtype=np.int64), np.array(s['val_idx'], dtype=np.int64)
    rng = np.random.default_rng(val_seed)
    perm = rng.permutation(n_total)
    val_idx = np.sort(perm[:val_n])
    train_idx = np.sort(perm[val_n:])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        'n_total': int(n_total),
        'val_n': int(val_n),
        'val_seed': int(val_seed),
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
    }))
    print(f'[train] wrote split → {split_path} (train={len(train_idx)} val={len(val_idx)})')
    return train_idx, val_idx


@torch.no_grad()
def val_loss(model, schedule, val_loader, device, diffusion_steps, n_batches):
    """Mean v-prediction MSE over `n_batches` random diffusion-step batches on val set."""
    model.eval()
    losses = []
    it = iter(val_loader)
    count = 0
    while count < n_batches:
        try:
            c, q0_raw = next(it)
        except StopIteration:
            it = iter(val_loader)
            c, q0_raw = next(it)
        c = c.to(device); q0_raw = q0_raw.to(device)
        q0 = normalize_q(q0_raw)
        t = torch.randint(0, diffusion_steps, (q0.shape[0],), device=device)
        xt, eps = q_sample(q0, t, schedule)
        v_tgt = v_target_from(q0, eps, t, schedule)
        v_pred = model(xt, t, c)
        losses.append(float(((v_pred - v_tgt) ** 2).mean().item()))
        count += 1
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[train] device={device}  data={args.data}')

    # Train side may use mirror aug; val never mirrors so metrics are comparable.
    full_ds_train = SeedSelectionDataset(args.data, mirror_prob=args.mirror_prob)
    full_ds_val   = SeedSelectionDataset(args.data, mirror_prob=0.0)
    print(f'[train] full dataset: {len(full_ds_train)} entries  '
          f'(train mirror_prob={args.mirror_prob}, val mirror_prob=0.0)')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx = make_or_load_split(args.ckpt_dir, len(full_ds_train), args.val_n, args.val_seed)
    print(f'[train] split: train={len(train_idx)}  val={len(val_idx)}')
    train_ds = Subset(full_ds_train, train_idx.tolist())
    val_ds   = Subset(full_ds_val,   val_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Build model (cfg from args, or overridden by resume).
    cfg = SeedQ0Config(
        d_model=args.d_model, n_layers=args.n_layers,
        dropout=args.dropout, diffusion_steps=args.diffusion_steps,
    )
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = SeedQ0Config(**ck['cfg'])
        print(f'[train] resuming from {args.resume} (step={ck["step"]})  cfg from ckpt')
    model = SeedQ0DiT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[train] model params: {n_params:.3f}M')

    schedule = DDPMCosineSchedule(T=cfg.diffusion_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}

    start_step = 0
    if args.resume is not None:
        model.load_state_dict(ck['model'])
        # ema state — keys may equal model parameter names
        for n, p in ema.items():
            if n in ck['ema']:
                p.data.copy_(ck['ema'][n])
        if 'optimizer' in ck:
            try:
                opt.load_state_dict(ck['optimizer'])
            except Exception as e:
                print(f'[train] warning: optimizer state not restored ({e}); continuing with fresh AdamW')
        start_step = int(ck['step'])

    if start_step >= args.num_steps:
        print(f'[train] start_step={start_step} >= num_steps={args.num_steps}; nothing to do.')
        return

    # WandB
    use_wandb = (not args.no_wandb) and bool(args.wandb_project)
    if use_wandb:
        import wandb
        run_name = args.wandb_name or args.ckpt_dir.name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            resume='allow',
        )
        wandb.config.update({
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx)),
            'n_params_M': float(n_params),
            'cfg': cfg.__dict__,
        }, allow_val_change=True)
        print(f'[train] wandb: {wandb.run.url}', flush=True)
    else:
        wandb = None  # type: ignore

    with open(args.ckpt_dir / 'train_args.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    print(f'[train] training from step {start_step+1} to {args.num_steps}')
    it = iter(train_loader)
    losses = []
    t0 = time.time()
    for step in range(start_step + 1, args.num_steps + 1):
        try:
            c, q0_raw = next(it)
        except StopIteration:
            it = iter(train_loader)
            c, q0_raw = next(it)
        c = c.to(device); q0_raw = q0_raw.to(device)
        q0 = normalize_q(q0_raw)

        t = torch.randint(0, cfg.diffusion_steps, (q0.shape[0],), device=device)
        xt, eps = q_sample(q0, t, schedule)
        v_tgt = v_target_from(q0, eps, t, schedule)
        if args.cfg_drop_prob > 0.0:
            uncond_mask = (torch.rand(q0.shape[0], device=device) < args.cfg_drop_prob)
            v_pred = model(xt, t, c, uncond_mask=uncond_mask)
        else:
            v_pred = model(xt, t, c)
        loss = ((v_pred - v_tgt) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        with torch.no_grad():
            for n, p in model.named_parameters():
                ema[n].mul_(args.ema_decay).add_(p.detach(), alpha=1 - args.ema_decay)

        losses.append(float(loss.item()))
        if use_wandb:
            wandb.log({'train/loss_step': float(loss.item())}, step=step)
        if step % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            dt = time.time() - t0
            sps = (step - start_step) / max(dt, 1e-6)
            print(f'[train] step={step:>6d}/{args.num_steps} loss={recent:.4f} sps={sps:.1f}', flush=True)
            if use_wandb:
                wandb.log({'train/loss_avg': recent, 'train/sps': sps}, step=step)
        if step % args.val_every == 0:
            v_l = val_loss(model, schedule, val_loader, device, cfg.diffusion_steps, args.val_batches)
            print(f'[train] step={step:>6d} VAL loss={v_l:.4f}', flush=True)
            if use_wandb:
                # train_loss reference for the gap plot.
                t_l = float(np.mean(losses[-args.log_every:])) if losses else float('nan')
                wandb.log({'val/loss': v_l,
                           'val/train_loss_concurrent': t_l,
                           'val/gap': v_l - t_l}, step=step)
        if step % args.ckpt_every == 0 or step == args.num_steps:
            ckpt_path = args.ckpt_dir / f'step_{step}.pt'
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'ema': ema,
                'optimizer': opt.state_dict(),
                'cfg': cfg.__dict__,
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }, ckpt_path)
            print(f'[train] saved {ckpt_path}', flush=True)
            if use_wandb:
                wandb.log({'ckpt_saved_step': step}, step=step)

    print(f'[train] DONE total_steps={args.num_steps} elapsed={time.time()-t0:.1f}s')
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
