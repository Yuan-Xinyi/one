"""PPO training for the farsighted-seed problem.

Algorithm
---------
Single-step contextual bandit -> no GAE needed; advantage = R - V(s).

Per iteration:
  1. Collect a batch by sampling states, actions ~ pi_theta_old, rollouts.
  2. Cache log pi_theta_old(a|s) at the data-collection moment (no grad).
  3. For E PPO epochs over minibatches:
        ratio  = exp(log pi_new - log pi_old)
        L_clip = -E[ min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) ]
        L_V    =  E[ (V_phi(s) - R)^2 ]
        L_ent  = -ENT_COEF * H(pi)
        Stop early if approx_KL(old || new) > target.
"""
from __future__ import annotations
import os, time, collections
import numpy as np
import torch
import torch.nn.functional as F

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import GaussianPolicy, ValueNet


def main():
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FarsightedSeedEnv(seed=cfg.SEED)
    q_mid  = torch.as_tensor(env.q_mid,  dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.q_half, dtype=torch.float32, device=device)

    policy = GaussianPolicy(cfg.STATE_DIM, env.ndof, q_mid, q_half).to(device)
    value  = ValueNet(cfg.STATE_DIM).to(device)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.LR_PI)
    opt_v  = torch.optim.Adam(value.parameters(),  lr=cfg.LR_V)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.CKPT_DIR, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("iter,mean_R,mean_len,success_rate,pi_loss,v_loss,"
                "entropy,kl,clip_frac,wall,reason_breakdown\n")

    def sample_fn(states_np: np.ndarray):
        s = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        a, u = policy.act(s, deterministic=False)
        return a.cpu().numpy().astype(np.float32), u.detach().cpu().numpy()

    t_start = time.time()

    for it in range(1, cfg.N_ITERS + 1):
        # -------- entropy coefficient annealing --------
        progress = min(1.0, (it - 1) / float(cfg.ENT_ANNEAL_END))
        ent_coef = (cfg.ENT_COEF_INIT * (1.0 - progress)
                    + cfg.ENT_COEF_FINAL * progress)

        # -------- collect batch (sequential rollouts) --------
        states_np, actions_np, rewards_np, lengths_np, Ts_np, u_np, reasons = \
            env.collect_batch(sample_fn, cfg.BATCH_SIZE)

        s = torch.as_tensor(states_np,  dtype=torch.float32, device=device)
        u = torch.as_tensor(u_np,       dtype=torch.float32, device=device)
        R = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            log_prob_old = policy.log_prob(s, u)
            V_old = value(s)
        adv = R - V_old
        if adv.numel() > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------- PPO epochs --------
        idx = np.arange(cfg.BATCH_SIZE)
        clip_fracs = []
        last_kl = 0.0
        early_stopped = False
        for epoch in range(cfg.PPO_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, cfg.BATCH_SIZE, cfg.MINIBATCH):
                mb = idx[start:start + cfg.MINIBATCH]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)
                s_b = s[mb_t]; u_b = u[mb_t]; R_b = R[mb_t]
                adv_b = adv[mb_t]; logp_old_b = log_prob_old[mb_t]

                logp_new = policy.log_prob(s_b, u_b)
                ratio = torch.exp(logp_new - logp_old_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - cfg.PPO_CLIP,
                                    1.0 + cfg.PPO_CLIP) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()
                ent = policy.entropy(s_b).mean()
                pi_loss = pi_loss - ent_coef * ent

                opt_pi.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               cfg.GRAD_CLIP)
                opt_pi.step()

                V_b = value(s_b)
                v_loss = F.mse_loss(V_b, R_b)
                opt_v.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(),
                                               cfg.GRAD_CLIP)
                opt_v.step()

                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > cfg.PPO_CLIP).float().mean().item())

            # --- approx KL on full batch, early-stop if too large ---
            with torch.no_grad():
                logp_now = policy.log_prob(s, u)
                last_kl = float((log_prob_old - logp_now).mean().item())
            if last_kl > cfg.PPO_TARGET_KL:
                early_stopped = True
                break

        # -------- log --------
        if it % cfg.LOG_EVERY == 0 or it == 1:
            mean_R   = float(R.mean().item())
            mean_len = float(lengths_np.mean())
            mean_T   = float(Ts_np.mean())
            # succ: rolled all the way to that episode's per-task T
            succ     = float((lengths_np >= Ts_np).mean())
            ent_now  = float(policy.entropy(s).mean().item())
            cf       = float(np.mean(clip_fracs)) if clip_fracs else 0.0
            wall     = time.time() - t_start
            ctr = collections.Counter(reasons)
            reason_str = "|".join(f"{k}:{v}" for k, v in ctr.most_common(4))
            es_tag = "*" if early_stopped else " "
            print(f"[{it:5d}/{cfg.N_ITERS}] "
                  f"R={mean_R:.3f} len={mean_len:5.1f}/{mean_T:4.1f} "
                  f"succ={succ:.2f} pi={pi_loss.item():+.3f} V={v_loss.item():.3f} "
                  f"H={ent_now:+.2f} ec={ent_coef:.3f} "
                  f"KL={last_kl:.3f}{es_tag} cf={cf:.2f} "
                  f"({wall:.1f}s) [{reason_str}]")
            with open(log_path, "a") as f:
                f.write(f"{it},{mean_R},{mean_len},{succ},"
                        f"{pi_loss.item()},{v_loss.item()},{ent_now},"
                        f"{last_kl},{cf},{wall},{reason_str}\n")

        if it % cfg.CKPT_EVERY == 0:
            torch.save({
                "iter": it,
                "policy": policy.state_dict(),
                "value":  value.state_dict(),
                "log_std": policy.log_std.detach().cpu().numpy(),
            }, os.path.join(cfg.CKPT_DIR, f"ckpt_{it:06d}.pt"))


if __name__ == "__main__":
    main()
