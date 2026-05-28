"""Per-cell seed builders.

A cell's seed is the q0 fed to the controller at episode start. The cells
differ in *where* that q0 comes from:

  baseline_seeds(eval_set)        shape (n_tasks, 1, 7)  — cls_cls, cls_hyb
  oracle_seeds(eval_set)          shape (n_tasks, 1, 7)  — oracle_cls
  diffusion_seeds(eval_set, ...)  shape (n_tasks, N, 7)  — diff_cls, diff_hyb
                                  also returns an `ik_ok` mask

(oracle_hyb is the controller-aware oracle and is built in run_oracle_prime
directly from the SMM top-K' pool — not via this dispatcher.)

For the diffusion path we sample N q0 from the trained DiT (with mirror aug
DISABLED at eval — the model handles symmetry itself), then refine each
sample onto the task's start manifold via Newton IK (newton_project), as
in eval_rollout.py. Samples whose IK didn't converge are marked invalid in
`ik_ok`; the cell runner translates that to L = 0 in best-of-N selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from Yuan.seed_selection.diffusion import ddim_sample_q0, denormalize_q, load_ckpt
from Yuan.seed_selection.smm import _build_R_target_strict, newton_project


def baseline_seeds(eval_set: dict) -> np.ndarray:
    """Pilot q0_seed (classical-derived). Shape (n_tasks, 1, 7)."""
    return eval_set['q0_seed'][:, None, :].astype(np.float32).copy()


def oracle_seeds(eval_set: dict) -> np.ndarray:
    """SMM classical-label oracle: labels_q0[argmax(labels_L_clean)].

    Shape (n_tasks, 1, 7). Note: this is optimal under the classical
    controller (which generated the labels) but not necessarily under
    the hybrid deployment controller — see oracle_hyb in run_oracle_prime.
    """
    return eval_set['max_label_q'][:, None, :].astype(np.float32).copy()


def diffusion_seeds(
    eval_set: dict,
    ckpt_path: str | Path,
    *,
    n_samples: int,
    ddim_steps: int,
    cfg_w: float,
    sample_seed: int,
    kin,
    device: torch.device,
    use_ema: bool = True,
    chunk_tasks: int = 64,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion + Newton-IK seeds for cells B and D.

    Returns:
        seeds (n_tasks, n_samples, 7) float32
        ik_ok (n_tasks, n_samples)    bool   — True if newton_project converged
    """
    p0_all       = eval_set['cs_p0'].astype(np.float32)
    line_dir_all = eval_set['cs_line_dir'].astype(np.float32)
    n_target_all = eval_set['cs_n_target'].astype(np.float32)
    n_tasks = p0_all.shape[0]

    torch.manual_seed(int(sample_seed))
    np.random.seed(int(sample_seed))

    model, schedule, _model_cfg, step = load_ckpt(
        Path(ckpt_path), device, use_ema=use_ema)
    if verbose:
        print(f'[diffusion_seeds] ckpt={ckpt_path} step={step} '
              f'N={n_samples} ddim={ddim_steps} cfg_w={cfg_w} mirror=OFF')

    lo_np = kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = kin.lmt_up.detach().cpu().numpy().astype(np.float32)

    seeds = np.zeros((n_tasks, n_samples, 7), dtype=np.float32)
    ik_ok = np.zeros((n_tasks, n_samples), dtype=bool)

    M = int(n_samples)
    for start in range(0, n_tasks, chunk_tasks):
        end = min(start + chunk_tasks, n_tasks)
        Bt = end - start
        c_np = np.concatenate(
            [p0_all[start:end], line_dir_all[start:end], n_target_all[start:end]],
            axis=1).astype(np.float32)                    # (Bt, 9)
        c_t = torch.from_numpy(c_np).to(device)
        c_rep = c_t.repeat_interleave(M, dim=0)           # (Bt*M, 9)

        q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                                num_steps=ddim_steps, cfg_w=cfg_w)
        q_raw = denormalize_q(q_norm).cpu().numpy().astype(np.float32)  # (Bt*M, 7)

        # Newton refine — CPU loop (newton_project is per-task numeric).
        for bi in range(Bt):
            ti = start + bi
            p0 = p0_all[ti]
            d  = line_dir_all[ti]
            n  = n_target_all[ti]
            R_tgt = _build_R_target_strict(n, d)
            for si in range(M):
                q_seed = q_raw[bi * M + si]
                q_ref, ok, _err = newton_project(kin, q_seed, p0, R_tgt, lo_np, hi_np)
                seeds[ti, si] = q_ref
                ik_ok[ti, si] = bool(ok)

        if verbose and ((start // chunk_tasks) % 5 == 0):
            ok_rate = 100.0 * ik_ok[:end].mean()
            print(f'  [diffusion_seeds] {end}/{n_tasks} tasks  IK ok so far: {ok_rate:.1f}%',
                  flush=True)

    if verbose:
        print(f'[diffusion_seeds] overall IK convergence: {100*ik_ok.mean():.1f}%')

    return seeds, ik_ok


def build_seeds_for_cell(
    cell: str,
    eval_set: dict,
    *,
    diffusion_cfg: dict | None = None,
    kin=None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch on cell name.

    Returns:
        seeds:  (n_tasks, n_samples, 7) float32
        ik_ok:  (n_tasks, n_samples) bool OR None for non-diffusion cells
    """
    if cell in ('cls_cls', 'cls_hyb'):
        return baseline_seeds(eval_set), None
    if cell == 'oracle_cls':
        return oracle_seeds(eval_set), None
    if cell in ('diff_cls', 'diff_hyb'):
        assert diffusion_cfg is not None and kin is not None and device is not None
        seeds, ik_ok = diffusion_seeds(
            eval_set,
            ckpt_path=diffusion_cfg['ckpt'],
            n_samples=int(diffusion_cfg['n_samples']),
            ddim_steps=int(diffusion_cfg['ddim_steps']),
            cfg_w=float(diffusion_cfg['cfg_w']),
            sample_seed=int(diffusion_cfg['sample_seed']),
            use_ema=bool(diffusion_cfg.get('use_ema', True)),
            kin=kin,
            device=device,
        )
        return seeds, ik_ok
    raise ValueError(f'unknown cell: {cell!r}')
