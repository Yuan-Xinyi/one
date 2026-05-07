"""v15: residual-corrected bandit RL with phantom-augmented selection.

Framework
---------
At deploy: policy proposes K candidates → augmented score is computed for
each as `phantom_L(s, a) + R(s, a)` where phantom is a cheap analytic
forward simulator and R is a learned residual predictor → argmax → 1
real rollout. Single-step bandit framing preserved.

What's learned
--------------
1. **Residual R(s, a)**: supervised regression on
   target = real_L_normalised - phantom_L_normalised   (~ ±0.1 to ±0.3)
   over per-iteration K samples. No Bellman bootstrap.
2. **Policy π(a|s)**: REINFORCE on chosen-candidate's real_L. The "chosen"
   candidate is selected by argmax(phantom + R). Policy gets baseline
   = mean real_L over the K samples (variance reduction).

Contact mode
------------
If `cfg.USE_CONTACT_MODE = True`, "real" rollouts use
`batched_rollout_contact` (linear-spring contact + force failure modes).
Phantom remains kinematic-only. Residual then learns BOTH geometric
phantom-bias AND phantom's blindness to force failures. The latter is
substantial (16-31pp on contact tasks per the smoke test), giving the
residual lots of signal to learn from.
"""
from __future__ import annotations
import argparse, os, time, collections
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.qnet import ResidualNet
from Yuan.RL.batched_rollout import (
    batched_rollout, batched_rollout_contact, phantom_rollout,
)


def _maybe_init_wandb():
    if not cfg.WANDB_ENABLE:
        return None
    import wandb
    config = {n: v for n, v in vars(cfg).items()
              if n.isupper()
              and isinstance(v, (int, float, str, bool, tuple, list, type(None)))}
    return wandb.init(project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY,
                      name=cfg.WANDB_RUN_NAME, config=config)


def _real_rollout(actions_np, c_np, v_np, e_np, T_np):
    """Dispatch to contact or geo rollout based on cfg.USE_CONTACT_MODE."""
    if bool(getattr(cfg, "USE_CONTACT_MODE", False)):
        return batched_rollout_contact(actions_np, c_np, v_np, e_np, T_np)
    return batched_rollout(actions_np, c_np, v_np, e_np, T_np)


def _normalize_L(L_np: np.ndarray, T_np: np.ndarray) -> np.ndarray:
    """L / T, clipped to [0, 1]."""
    return np.clip(L_np.astype(np.float32) /
                   np.maximum(T_np.astype(np.float32), 1.0),
                   0.0, 1.0)


def _sample_training_batch(env: FarsightedSeedEnv, batch_size: int):
    tasks = env._sample_tasks(batch_size)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0)
    return states.astype(np.float32), tasks


def _sample_K_actions(policy, states_t: torch.Tensor, K: int) -> torch.Tensor:
    """Returns (K, B, A)."""
    out = []
    for _ in range(int(K)):
        with torch.no_grad():
            a, _ = policy.act(states_t, deterministic=False)
        out.append(a)
    return torch.stack(out, dim=0)


def _gather_chosen(tensor_KB, chosen_idx_B):
    """tensor_KB: (K, B, *), chosen_idx_B: (B,) long. Returns (B, *)."""
    K, B = tensor_KB.shape[:2]
    arange = torch.arange(B, device=tensor_KB.device)
    return tensor_KB[chosen_idx_B, arange]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters",  type=int, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--contact",  action="store_true",
                    help="force USE_CONTACT_MODE=True for this run")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--K-samples", type=int, default=None,
                    help="override SAC_ACTION_SAMPLES_PER_TASK")
    args = ap.parse_args()
    if args.K_samples is not None: cfg.SAC_ACTION_SAMPLES_PER_TASK = int(args.K_samples)
    if args.n_iters  is not None: cfg.N_ITERS = int(args.n_iters)
    if args.ckpt_dir is not None: cfg.CKPT_DIR = args.ckpt_dir
    if args.run_name is not None: cfg.WANDB_RUN_NAME = args.run_name
    if args.contact:              cfg.USE_CONTACT_MODE = True
    if args.no_wandb:             cfg.WANDB_ENABLE = False

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = _maybe_init_wandb()

    use_contact = bool(getattr(cfg, "USE_CONTACT_MODE", False))
    print(f"[config] USE_CONTACT_MODE={use_contact}  "
          f"K_per_task={cfg.SAC_ACTION_SAMPLES_PER_TASK}  "
          f"BATCH_SIZE={cfg.BATCH_SIZE}  N_ITERS={cfg.N_ITERS}")

    env = FarsightedSeedEnv(seed=cfg.SEED, randomize=True,
                            use_collision=False,
                            contact_mode=use_contact)
    state_dim = cfg.STATE_DIM
    action_dim = env.action_dim
    K = int(cfg.SAC_ACTION_SAMPLES_PER_TASK)
    B = int(cfg.BATCH_SIZE)

    qmid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    qhalf = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    policy = make_policy(state_dim, action_dim, qmid, qhalf).to(device)
    rnet = ResidualNet(state_dim, action_dim).to(device)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.SAC_LR_PI)
    opt_r  = torch.optim.Adam(rnet.parameters(),   lr=cfg.SAC_LR_Q)

    log_alpha = torch.tensor(np.log(cfg.SAC_ALPHA_INIT), device=device,
                             dtype=torch.float32, requires_grad=cfg.SAC_AUTO_ALPHA)
    opt_alpha = (torch.optim.Adam([log_alpha], lr=cfg.SAC_LR_ALPHA)
                 if cfg.SAC_AUTO_ALPHA else None)
    target_h = float(cfg.SAC_TARGET_H)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.CKPT_DIR, "train_log_v15.csv")
    with open(log_path, "w") as f:
        f.write("iter,mean_real_R,mean_chosen_real_R,mean_unif_oracle_R,"
                "mean_phantom_R,res_loss,res_mae,pi_loss,alpha,entropy,"
                "frac_chosen_eq_phantom_argmax,wall\n")
    t_start = time.time()

    # ---------- main loop ----------
    for it in range(1, cfg.N_ITERS + 1):
        # ----- sample tasks + K policy actions per task -----
        states_np, tasks = _sample_training_batch(env, B)
        states_t = torch.as_tensor(states_np, device=device, dtype=torch.float32)
        a_KB = _sample_K_actions(policy, states_t, K)              # (K, B, A)

        # task params (broadcast to K)
        c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
        v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
        e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
        T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)
        a_flat_np = a_KB.reshape(K * B, action_dim).cpu().numpy().astype(np.float32)
        c_rep = np.tile(c_np, (K, 1))
        v_rep = np.tile(v_np, K)
        e_rep = np.tile(e_np, K)
        T_rep = np.tile(T_np, K)

        # ----- phantom + real rollouts (parallel batched) -----
        ph_out  = phantom_rollout(a_flat_np, c_rep, v_rep, e_rep, T_rep)
        rl_out  = _real_rollout(a_flat_np, c_rep, v_rep, e_rep, T_rep)
        L_ph_flat = np.asarray(ph_out["lengths"], dtype=np.float32)
        L_rl_flat = np.asarray(rl_out["lengths"], dtype=np.float32)
        # normalise to ratios in [0, 1]
        r_ph = _normalize_L(L_ph_flat, T_rep)                       # (K*B,)
        r_rl = _normalize_L(L_rl_flat, T_rep)                       # (K*B,)
        residual_target = (r_rl - r_ph).astype(np.float32)           # (K*B,)

        # tensors: (K, B) per-task views
        s_KB = states_t.unsqueeze(0).expand(K, B, -1)
        s_flat = s_KB.reshape(K * B, -1)
        a_flat = a_KB.reshape(K * B, -1)
        r_ph_t  = torch.as_tensor(r_ph,            device=device, dtype=torch.float32)
        r_rl_t  = torch.as_tensor(r_rl,            device=device, dtype=torch.float32)
        res_tgt = torch.as_tensor(residual_target, device=device, dtype=torch.float32)

        # ----- residual update (supervised) -----
        # Multiple grad steps per env step (analogous to SAC_K_Q).
        res_loss_val = 0.0
        res_mae_val  = 0.0
        for _ in range(cfg.SAC_K_Q):
            r_pred = rnet(s_flat, a_flat)
            res_loss = ((r_pred - res_tgt) ** 2).mean()
            opt_r.zero_grad()
            res_loss.backward()
            torch.nn.utils.clip_grad_norm_(rnet.parameters(), cfg.GRAD_CLIP)
            opt_r.step()
            res_loss_val = float(res_loss.item())
            res_mae_val  = float((r_pred - res_tgt).abs().mean().item())

        # ----- compute deploy choice per task: argmax(phantom + R) -----
        with torch.no_grad():
            r_pred_chosen = rnet(s_flat, a_flat)                        # (K*B,)
            aug_score = (r_ph_t + r_pred_chosen).view(K, B)             # (K, B)
            chosen_k = aug_score.argmax(dim=0)                           # (B,)
            phantom_argmax_k = r_ph_t.view(K, B).argmax(dim=0)
            frac_aug_eq_phantom = float(
                (chosen_k == phantom_argmax_k).float().mean().item())

        # ----- policy update: BC on K-oracle's best + entropy regularisation -----
        # Per task, find the K-sample whose REAL L was highest. Push policy
        # log_prob of that action up. Entropy term (auto-α) prevents
        # mode collapse — policy has to keep K-sample diversity so that the
        # "best of K" objective stays achievable.
        r_rl_KB = r_rl_t.view(K, B)                                     # (K, B)
        best_k = r_rl_KB.argmax(dim=0)                                  # (B,)
        best_actions = _gather_chosen(a_KB, best_k)                     # (B, A)
        if not hasattr(policy, "log_prob_action"):
            raise RuntimeError("policy needs log_prob_action; check policy.py")
        log_p_best = policy.log_prob_action(states_t, best_actions)     # (B,)

        # entropy estimate over the full K-sample population
        states_KB_flat = states_t.unsqueeze(0).expand(K, B, -1).reshape(-1, state_dim)
        actions_KB_flat = a_KB.reshape(K * B, action_dim)
        log_p_all = policy.log_prob_action(states_KB_flat, actions_KB_flat
                                           ).view(K, B)
        ent = -log_p_all.mean()
        alpha = log_alpha.exp().detach()

        pi_loss = -log_p_best.mean() - alpha * ent
        opt_pi.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.GRAD_CLIP)
        opt_pi.step()
        pi_loss_val = float(pi_loss.item())
        ent_val = float(ent.item())

        if cfg.SAC_AUTO_ALPHA:
            # standard SAC auto-α: push entropy toward target_h
            alpha_loss = -(log_alpha * (-log_p_all.detach().mean() - target_h))
            opt_alpha.zero_grad()
            alpha_loss.backward()
            opt_alpha.step()

        # ----- log -----
        if it % cfg.LOG_EVERY == 0 or it == 1:
            # what the deploy procedure (argmax ph + R) would have picked
            with torch.no_grad():
                aug_chosen_real = _gather_chosen(r_rl_KB, chosen_k)     # (B,)
            mean_real_R   = float(r_rl_t.mean().item())
            mean_chosen_R = float(aug_chosen_real.mean().item())
            mean_unif_orc = float(r_rl_KB.max(dim=0).values.mean().item())
            mean_ph_R     = float(r_ph_t.mean().item())
            wall = time.time() - t_start
            alpha_now = float(log_alpha.exp().item())
            print(f"[{it:5d}/{cfg.N_ITERS}] "
                  f"r_chosen={mean_chosen_R:.3f}  r_orcK={mean_unif_orc:.3f}  "
                  f"r_real={mean_real_R:.3f}  r_ph={mean_ph_R:.3f}  "
                  f"r_loss={res_loss_val:.4f}  r_mae={res_mae_val:.4f}  "
                  f"pi={pi_loss_val:+.3f}  a={alpha_now:.3f}  H={ent_val:+.2f}  "
                  f"chosen=ph_argmax:{frac_aug_eq_phantom:.2f}  "
                  f"({wall:.1f}s)")
            with open(log_path, "a") as f:
                f.write(f"{it},{mean_real_R},{mean_chosen_R},{mean_unif_orc},"
                        f"{mean_ph_R},{res_loss_val},{res_mae_val},"
                        f"{pi_loss_val},{alpha_now},{ent_val},"
                        f"{frac_aug_eq_phantom},{wall}\n")
            if wandb_run is not None:
                wandb_run.log({
                    "train/r_chosen":    mean_chosen_R,
                    "train/r_oracle_K":  mean_unif_orc,
                    "train/r_real_mean": mean_real_R,
                    "train/r_phantom_mean": mean_ph_R,
                    "train/policy_lift_over_random":
                        mean_chosen_R - mean_real_R,
                    "train/policy_lift_over_oracle_K":
                        mean_chosen_R - mean_unif_orc,    # ≤ 0 by definition
                    "loss/residual_mse": res_loss_val,
                    "loss/residual_mae": res_mae_val,
                    "loss/pi": pi_loss_val,
                    "policy/alpha": alpha_now,
                    "policy/entropy": ent_val,
                    "policy/chosen_eq_phantom_argmax": frac_aug_eq_phantom,
                    "time/wall_sec": wall,
                }, step=it)

        if it % cfg.CKPT_EVERY == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"ckpt_{it:06d}.pt")
            torch.save({
                "iter": it,
                "policy_type": cfg.POLICY_TYPE,
                "action_mode": cfg.ACTION_MODE,
                "policy": policy.state_dict(),
                "rnet": rnet.state_dict(),
                "log_alpha": float(log_alpha.detach().cpu().item()),
                "state_dim": cfg.STATE_DIM,
                "use_contact_mode": use_contact,
            }, ckpt_path)
            if wandb_run is not None:
                wandb_run.save(ckpt_path, policy="now")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
