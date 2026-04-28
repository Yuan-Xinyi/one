"""Evaluate a trained farsighted-seed policy.

Usage (from repo root):
    python -m Yuan.RL.eval                         # numeric eval, 256 contexts
    python -m Yuan.RL.eval --viz                   # show one rollout in viewer
    python -m Yuan.RL.eval --ckpt path/to/ckpt.pt
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.rollout import rollout, build_target_rotmat
from Yuan.RL.policy import make_policy


def latest_ckpt() -> str | None:
    paths = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "ckpt_*.pt")))
    return paths[-1] if paths else None


def load_policy(ckpt_path: str, env: FarsightedSeedEnv, device: torch.device
                ):
    q_mid  = torch.as_tensor(env.q_mid,  dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.q_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.ndof, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


def evaluate(policy, env: FarsightedSeedEnv,
             n_contexts: int = 256, deterministic: bool = True,
             best_components: bool = False) -> dict:
    device = next(policy.parameters()).device
    lengths = np.empty(n_contexts, dtype=np.int32)
    home_lens = np.empty(n_contexts, dtype=np.int32)
    Ts = np.empty(n_contexts, dtype=np.int32)
    home = env.arm.home_qs.astype(np.float32)
    for i in range(n_contexts):
        s = env.reset()
        task = env._cur                                  # full task dict
        st = torch.as_tensor(s[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            if best_components and hasattr(policy, "component_actions"):
                cand = policy.component_actions(st).squeeze(0)
            else:
                a, _ = policy.act(st, deterministic=deterministic)
                cand = a
        best = -1
        for a_i in cand.reshape(-1, env.ndof):
            env._cur = task
            _, _, _, info = env.step(a_i.cpu().numpy().astype(np.float32))
            best = max(best, info["length"])
        lengths[i] = best
        Ts[i] = task["T"]
        # reference: home seed under the same task params (fair)
        c = task["c"]
        info_h = rollout(env.arm, home, c[:3], c[3:6], c[6:9],
                         mjc=env.mjc, max_steps=task["T"],
                         v_path=task["v_path"], eps_p=task["eps_p"])
        home_lens[i] = info_h["length"]
    return {"policy_mean_len": float(lengths.mean()),
            "policy_succ":     float((lengths >= Ts).mean()),
            "home_mean_len":   float(home_lens.mean()),
            "home_succ":       float((home_lens >= Ts).mean()),
            "policy_recovery": float((lengths / Ts).mean()),
            "home_recovery":   float((home_lens / Ts).mean()),
            "lengths":         lengths,
            "Ts":              Ts,
            "home_lengths":    home_lens}


def _pick_demo_task(policy, env, max_tries: int = 30, min_gap: int = 20):
    """Resample contexts until we find one where the policy meaningfully
    outperforms the home seed (gap >= min_gap), so the viz tells a story.
    Returns (state, policy_info, home_info)."""
    device = next(policy.parameters()).device
    home = env.arm.home_qs.astype(np.float32)
    best = None  # (gap, s, info_pi, info_home)
    for _ in range(max_tries):
        s = env.reset()
        p0, d, n = s[:3], s[3:6], s[6:9]
        st = torch.as_tensor(s[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            a, _ = policy.act(st, deterministic=True)
        a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
        info_pi = rollout(env.arm, a_np, p0, d, n, mjc=env.mjc)
        info_h  = rollout(env.arm, home, p0, d, n, mjc=env.mjc)
        gap = info_pi["length"] - info_h["length"]
        if best is None or gap > best[0]:
            best = (gap, s, info_pi, info_h)
        if gap >= min_gap and info_pi["length"] >= cfg.MAX_STEPS // 2:
            break
    return best[1], best[2], best[3]


def visualize(policy, env: FarsightedSeedEnv,
              animate: bool = True):
    import builtins, time
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from one.robots.manipulators.franka.fr3.fr3 import fr3_with_hand

    s, info_pi, info_h = _pick_demo_task(policy, env)
    p0, d, n = s[:3], s[3:6], s[6:9]
    print(f"context  p0={p0} d={d} n={n}")
    print(f"policy   length={info_pi['length']}/{cfg.MAX_STEPS}  reason={info_pi['reason']}")
    print(f"home seed length={info_h['length']}/{cfg.MAX_STEPS}  reason={info_h['reason']}")

    base = ovw.World(cam_pos=[1.5, 1.0, 0.8],
                     cam_lookat_pos=(p0 + 0.2 * d).tolist())
    arm, _ = fr3_with_hand()
    builtins.base, builtins.arm = base, arm
    arm.attach_to(base.scene)
    ossop.frame().attach_to(base.scene)

    # full path: green for what the policy reached, gray for unreached prefix,
    # red dot at the failure point (if any)
    for t in range(0, cfg.MAX_STEPS + 1, 2):
        p = p0 + t * cfg.PATH_STEP * d
        if t <= info_pi["length"]:
            rgb = (0.2, 0.85, 0.2)
        else:
            rgb = (0.85, 0.2, 0.2)
        sp = ossop.sphere(pos=tuple(p), radius=0.005, rgb=rgb)
        sp.attach_to(base.scene)

    # show the achieved q at p0 (or home if init IK failed)
    q_traj = info_pi["q_traj"]
    arm.fk(q_traj[0] if q_traj else env.arm.home_qs)

    if animate and len(q_traj) > 1:
        # play the joint trajectory in a viewer task: ~30 fps
        idx = [0]
        def _step(_dt):
            i = idx[0]
            if i < len(q_traj):
                arm.fk(q_traj[i])
                idx[0] = i + 1
            else:
                idx[0] = 0  # loop
        try:
            import pyglet
            pyglet.clock.schedule_interval(_step, 1.0 / 30.0)
        except Exception as e:
            print(f"animation skipped: {e}")
    base.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--best-components", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FarsightedSeedEnv(seed=12345)
    ckpt = args.ckpt or latest_ckpt()
    if ckpt is None:
        raise SystemExit(f"No checkpoint in {cfg.CKPT_DIR}/. Train first.")
    print(f"loading {ckpt}")
    policy = load_policy(ckpt, env, device)

    if args.viz:
        visualize(policy, env)
    else:
        out = evaluate(policy, env, n_contexts=args.n,
                       best_components=args.best_components)
        print(f"policy: mean_len={out['policy_mean']:.2f}  "
              f"succ={out['policy_succ']:.2f}")
        print(f"home  : mean_len={out['home_mean']:.2f}  "
              f"succ={out['home_succ']:.2f}")


if __name__ == "__main__":
    main()
