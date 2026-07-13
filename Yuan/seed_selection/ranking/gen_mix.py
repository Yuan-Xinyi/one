"""Shared: mixed-guidance diffusion candidate generation + hybrid rollout +
features. 20k-consistent; only candidate diversity is scaled (mixed cfg_w).

Projection: z-axis-mode batched IK (_batched_ik_project) — same projector the
SMM oracle_hyb reference uses. Cone-relaxed (z within THETA_MAX, FREE tool spin),
NOT the over-constrained _build_R_target_strict/newton_project used by the
stock diffusion_seeds. Higher IK convergence + preserves spin diversity (higher
best-of-N ceiling) + fully batched on GPU (fast).

All rollouts flatten (task x candidate) into one array and run through
rollout_seeds_batched, chunked by env.n_envs (set large for GPU efficiency).
"""
from pathlib import Path
import numpy as np, torch
from Yuan.system_eval.rollout_controllers import rollout_seeds_batched
from Yuan.seed_selection.diffusion import ddim_sample_q0, denormalize_q, load_ckpt
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z

WEIGHTS = [1.0, 1.5, 2.0, 2.5]        # mixed guidance (3.0 dropped: over-guided, 79% IK)
N_PER_W = 8                            # -> 32 candidates/task
TDM = 1.5
TAU = (0.985, 0.96)
_MODEL = {}


def gen_candidates(p0, ld, nt, diff_ckpt, kin, device, seed0=1000):
    """Mixed-guidance diffusion candidates, cone-relaxed z-axis projection
    (free spin). Returns seeds (T,32,7) float32, ok (T,32) bool."""
    if diff_ckpt not in _MODEL:
        _MODEL[diff_ckpt] = load_ckpt(Path(diff_ckpt), device, use_ema=True)
    model, schedule, _cfg, _step = _MODEL[diff_ckpt]
    dt = kin.dtype
    T = p0.shape[0]
    p0t = torch.as_tensor(p0, device=device, dtype=dt)
    ldt = torch.as_tensor(ld, device=device, dtype=dt)
    ntt = torch.as_tensor(nt, device=device, dtype=dt)
    hint = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dt)  # unused in z_axis mode
    p0_rep = p0t.repeat_interleave(N_PER_W, 0)
    z_rep = ntt.repeat_interleave(N_PER_W, 0)
    R_tgt = _build_R_with_z(z_rep, hint)                 # (T*N,3,3), z col = n_target
    c = torch.cat([p0t, ldt, ntt], 1)                    # (T,9)
    c_rep = c.repeat_interleave(N_PER_W, 0)              # (T*N,9)

    seeds_l, ok_l = [], []
    for wi, w in enumerate(WEIGHTS):
        torch.manual_seed(seed0 + 137 * wi)
        q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                                num_steps=50, cfg_w=w)
        q_raw = denormalize_q(q_norm).to(dt)             # (T*N,7)
        q_out, ok, _ = _batched_ik_project(kin, q_raw, p0_rep, R_tgt, branch_action=None)
        seeds_l.append(q_out.detach().cpu().numpy().reshape(T, N_PER_W, 7).astype(np.float32))
        ok_l.append(ok.detach().cpu().numpy().reshape(T, N_PER_W))
        print(f'  [gen_mix] cfg_w={w}: IK ok {100*ok.float().mean().item():.1f}% (z-axis, free spin)', flush=True)
    return np.concatenate(seeds_l, 1), np.concatenate(ok_l, 1)


def roll_hybrid(seeds3d, ok3d, p0, ld, nt, env, classical, agent):
    """(T,C,7) -> l (T,C) metres under hybrid, invalid=0. Fully flattened;
    rollout_seeds_batched chunks by env.n_envs (large for GPU efficiency)."""
    T, C = seeds3d.shape[:2]
    out = np.zeros((T, C), np.float32)
    flat = np.nonzero(ok3d.reshape(-1))[0]; tof = np.repeat(np.arange(T), C)
    qv = seeds3d.reshape(T * C, 7)[flat].astype(np.float32)
    r = rollout_seeds_batched(qv, p0[tof][flat], ld[tof][flat], nt[tof][flat],
                              env=env, controller='hybrid_variantB', classical=classical,
                              agent=agent, tau_enter=TAU[0], tau_exit=TAU[1], progress_prefix='  ')
    out.reshape(-1)[flat] = r['L'].astype(np.float32) * TDM
    return out


def features(seeds3d, ok3d, p0, ld, nt, env, obs_and_manip):
    T, C = seeds3d.shape[:2]
    flat = np.nonzero(ok3d.reshape(-1))[0]; tof = np.repeat(np.arange(T), C)
    qv = seeds3d.reshape(T * C, 7)[flat].astype(np.float32)
    obf, muf = obs_and_manip(env, qv, p0[tof][flat], ld[tof][flat], nt[tof][flat])
    obs = np.zeros((T, C, 31), np.float32); mu = np.zeros((T, C), np.float32)
    obs.reshape(-1, 31)[flat] = obf; mu.reshape(-1)[flat] = muf
    return np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1).astype(np.float32)
