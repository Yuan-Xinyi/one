"""v16: DPO-trained policy + phantom-select bandit deploy.

Why this differs from v15
-------------------------
v15 tried to learn a residual R(s, a) → real_L − phantom_L on top of phantom-
greedy selection. Empirically failed: per-(s,a) phantom error on contact
tasks is stochastic-noise dominated, so R could only learn the global mean
bias and ended up *worsening* selection by misranking K candidates.

v16 attacks the problem differently: instead of trying to correct phantom's
per-action errors, train the POLICY DISTRIBUTION so that K samples from it
concentrate on actions that actually work. Phantom selection within that
better distribution wins by inheritance.

Method
------
1. π_ref = frozen copy of π_θ at init. Provides KL anchor → no entropy
   collapse (the killer of v15's BC objective).
2. Per iter:
     a. Sample K candidates per task from π_θ.
     b. Real-rollout all K → real_L_k.
     c. winner = argmax_k real_L_k per task. Losers = the other K−1.
     d. DPO pairwise loss over (winner, loser):
        loss = −log σ( β · [ log π_θ(w|s) − log π_ref(w|s)
                            −log π_θ(l|s) + log π_ref(l|s) ] )
        averaged over (B, K−1) pairs.
     e. Adam step on π_θ.
3. Deploy: phantom_select on K=8 policy samples (same single-step bandit
   structure as before, NOT sequential).

Why DPO over BC + entropy:
- BC loss can be arbitrarily large (log_prob is unbounded).
- Auto-α SAC alpha-tuning is too slow (v15 saw H collapse 2.71 → 0.06).
- DPO loss is bounded by −log σ(0) = log 2 per pair.
- π_ref's role is exactly anti-collapse without ad hoc α schedules.

Contact mode: same as v15 — `cfg.USE_CONTACT_MODE` switches "real" rollout
between geo and contact-spring. Phantom is always kinematic.
"""
from __future__ import annotations
import argparse, copy, os, time
import numpy as np
import torch
import torch.nn.functional as F

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
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
    if bool(getattr(cfg, "USE_CONTACT_MODE", False)):
        return batched_rollout_contact(actions_np, c_np, v_np, e_np, T_np)
    return batched_rollout(actions_np, c_np, v_np, e_np, T_np)


def _normalize_L(L_np, T_np):
    return np.clip(L_np.astype(np.float32) /
                   np.maximum(T_np.astype(np.float32), 1.0),
                   0.0, 1.0)


def _sample_training_batch(env, B):
    tasks = env._sample_tasks(B)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0)
    return states.astype(np.float32), tasks


def _sample_K_actions(policy, states_t, K):
    out = []
    for _ in range(int(K)):
        with torch.no_grad():
            a, _ = policy.act(states_t, deterministic=False)
        out.append(a)
    return torch.stack(out, dim=0)                                    # (K, B, A)


def _gather_chosen(tensor_KB, idx_B):
    K, B = tensor_KB.shape[:2]
    arange = torch.arange(B, device=tensor_KB.device)
    return tensor_KB[idx_B, arange]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iters",  type=int, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--contact",  action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--K-samples", type=int, default=None)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="DPO temperature; controls how aggressively policy "
                         "moves away from π_ref toward winners")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="only train on (w, l) pairs where real_L_w - real_L_l > margin")
    args = ap.parse_args()
    if args.n_iters  is not None: cfg.N_ITERS = int(args.n_iters)
    if args.ckpt_dir is not None: cfg.CKPT_DIR = args.ckpt_dir
    if args.run_name is not None: cfg.WANDB_RUN_NAME = args.run_name
    if args.contact:              cfg.USE_CONTACT_MODE = True
    if args.no_wandb:             cfg.WANDB_ENABLE = False
    if args.K_samples is not None: cfg.SAC_ACTION_SAMPLES_PER_TASK = int(args.K_samples)
    beta = float(args.beta)
    margin = float(args.margin)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = _maybe_init_wandb()

    use_contact = bool(getattr(cfg, "USE_CONTACT_MODE", False))
    print(f"[config] USE_CONTACT_MODE={use_contact}  K={cfg.SAC_ACTION_SAMPLES_PER_TASK}  "
          f"B={cfg.BATCH_SIZE}  N_ITERS={cfg.N_ITERS}  β={beta}  margin={margin}")

    env = FarsightedSeedEnv(seed=cfg.SEED, randomize=True,
                            use_collision=False, contact_mode=use_contact)
    state_dim, action_dim = cfg.STATE_DIM, env.action_dim
    K = int(cfg.SAC_ACTION_SAMPLES_PER_TASK)
    B = int(cfg.BATCH_SIZE)

    qmid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    qhalf = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    policy = make_policy(state_dim, action_dim, qmid, qhalf).to(device)

    # frozen reference policy = init copy
    policy_ref = make_policy(state_dim, action_dim, qmid, qhalf).to(device)
    policy_ref.load_state_dict(policy.state_dict())
    for p in policy_ref.parameters():
        p.requires_grad_(False)
    policy_ref.eval()

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.SAC_LR_PI)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.CKPT_DIR, "train_log_v16.csv")
    with open(log_path, "w") as f:
        f.write("iter,r_real_mean,r_winner_mean,r_phsel_pol,r_orcK,"
                "dpo_loss,dpo_acc,frac_pairs_used,kl_to_ref,entropy,wall\n")
    t_start = time.time()

    for it in range(1, cfg.N_ITERS + 1):
        # ----- sample tasks + K policy candidates -----
        states_np, tasks = _sample_training_batch(env, B)
        states_t = torch.as_tensor(states_np, device=device, dtype=torch.float32)
        a_KB = _sample_K_actions(policy, states_t, K)               # (K, B, A)

        # task params
        c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
        v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
        e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
        T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)
        a_flat_np = a_KB.reshape(K * B, action_dim).cpu().numpy().astype(np.float32)
        rep_c = np.tile(c_np, (K, 1))
        rep_v = np.tile(v_np, K)
        rep_e = np.tile(e_np, K)
        rep_T = np.tile(T_np, K)

        # ----- real rollouts -----
        rl_out = _real_rollout(a_flat_np, rep_c, rep_v, rep_e, rep_T)
        L_rl_flat = np.asarray(rl_out["lengths"], dtype=np.float32)
        r_rl = _normalize_L(L_rl_flat, rep_T)
        r_rl_KB = torch.as_tensor(r_rl, device=device, dtype=torch.float32).view(K, B)

        # ----- identify winners + build pair mask -----
        winner_k = r_rl_KB.argmax(dim=0)                              # (B,)
        # for each task, winner_k[b] vs every other k → (K, B) mask
        # pair (winner, k) is "valid" iff k != winner AND r_w - r_k > margin
        idx_K = torch.arange(K, device=device)[:, None].expand(K, B)
        is_loser = (idx_K != winner_k[None, :])                       # (K, B)
        r_winner = r_rl_KB.gather(0, winner_k[None, :].expand(K, B))  # (K, B)
        gap = r_winner - r_rl_KB                                       # (K, B)
        valid = is_loser & (gap > margin)                              # (K, B)
        n_valid = int(valid.sum().item())
        frac_pairs = float(valid.float().mean().item())                # ~0..1

        # ----- log_probs of all K samples under π_θ and π_ref -----
        states_KB_flat = states_t.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
        a_KB_flat = a_KB.reshape(K * B, action_dim)
        log_p_theta = policy.log_prob_action(states_KB_flat, a_KB_flat).view(K, B)
        with torch.no_grad():
            log_p_ref = policy_ref.log_prob_action(states_KB_flat, a_KB_flat).view(K, B)

        # winner log probs (same for all k slots in each task column)
        log_p_theta_w = log_p_theta.gather(0, winner_k[None, :].expand(K, B))
        log_p_ref_w   = log_p_ref.gather(  0, winner_k[None, :].expand(K, B))

        # DPO logits per (k, b) with k as the "loser" candidate
        # logit = β * [(log π_θ(w) - log π_ref(w)) - (log π_θ(l) - log π_ref(l))]
        logit = beta * ((log_p_theta_w - log_p_ref_w)
                        - (log_p_theta - log_p_ref))
        # masked DPO loss
        if n_valid > 0:
            dpo_loss_per = -F.logsigmoid(logit) * valid.float()
            dpo_loss = dpo_loss_per.sum() / max(n_valid, 1)
            dpo_acc = float(((logit > 0) & valid).float().sum().item() / max(n_valid, 1))
        else:
            dpo_loss = torch.tensor(0.0, device=device)
            dpo_acc = 0.0

        opt_pi.zero_grad()
        dpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.GRAD_CLIP)
        opt_pi.step()

        # ----- diagnostics -----
        if it % cfg.LOG_EVERY == 0 or it == 1:
            with torch.no_grad():
                # phantom_select_policy ratio at K (for the same K samples this iter)
                ph_out = phantom_rollout(a_flat_np, rep_c, rep_v, rep_e, rep_T)
                L_ph = np.asarray(ph_out["lengths"], dtype=np.float32).reshape(K, B)
                r_ph = _normalize_L(L_ph, np.tile(T_np, (K, 1)))
                ph_argmax = r_ph.argmax(axis=0)
                arange_b = np.arange(B)
                r_phsel = r_rl_KB.cpu().numpy()[ph_argmax, arange_b]
                kl = float((log_p_theta - log_p_ref).mean().item())
                ent = float((-log_p_theta.mean()).item())
            r_real_mean   = float(r_rl_KB.mean().item())
            r_winner_mean = float(r_winner[0].mean().item())   # winner same across k
            r_phsel_mean  = float(r_phsel.mean())
            r_orcK_mean   = r_winner_mean                        # they're the same
            wall = time.time() - t_start
            print(f"[{it:5d}/{cfg.N_ITERS}] "
                  f"r_phsel={r_phsel_mean:.3f}  r_orcK={r_orcK_mean:.3f}  "
                  f"r_real_mean={r_real_mean:.3f}  "
                  f"dpo={float(dpo_loss.item()):.4f}  acc={dpo_acc:.3f}  "
                  f"pairs/task={frac_pairs * (K - 1):.1f}  "
                  f"KL(θ‖ref)={kl:+.3f}  H={ent:+.2f}  ({wall:.1f}s)")
            with open(log_path, "a") as f:
                f.write(f"{it},{r_real_mean},{r_winner_mean},{r_phsel_mean},"
                        f"{r_orcK_mean},{float(dpo_loss.item())},{dpo_acc},"
                        f"{frac_pairs},{kl},{ent},{wall}\n")
            if wandb_run is not None:
                wandb_run.log({
                    "train/r_phantom_select": r_phsel_mean,
                    "train/r_oracle_K":       r_orcK_mean,
                    "train/r_real_mean":      r_real_mean,
                    "train/r_phsel_lift_over_random": r_phsel_mean - r_real_mean,
                    "loss/dpo":               float(dpo_loss.item()),
                    "loss/dpo_acc":           dpo_acc,
                    "loss/frac_pairs_used":   frac_pairs,
                    "policy/KL_theta_ref":    kl,
                    "policy/entropy":         ent,
                    "time/wall_sec":          wall,
                }, step=it)

        if it % cfg.CKPT_EVERY == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"ckpt_{it:06d}.pt")
            torch.save({
                "iter": it,
                "policy_type": cfg.POLICY_TYPE,
                "action_mode": cfg.ACTION_MODE,
                "policy": policy.state_dict(),
                "policy_ref": policy_ref.state_dict(),
                "state_dim": cfg.STATE_DIM,
                "use_contact_mode": use_contact,
                "beta": beta,
                "margin": margin,
            }, ckpt_path)
            if wandb_run is not None:
                wandb_run.save(ckpt_path, policy="now")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
