"""Does this method meet the conditions for being called a controller?

Emitting joint velocities is neither necessary nor sufficient. What separates a
controller from an open-loop generator is whether a loop is closed around a
state that can disagree with the model. Three operational tests:

  A. repeatability      Run the same task twice. Bit-identical output means
                        there is nothing in the loop that could disagree with
                        the model -- necessary for an open-loop generator,
                        and by itself proves nothing either way (a
                        deterministic controller on a deterministic plant is
                        also repeatable). Reported for completeness.

  B. disturbance rejection
                        Displace the TCP off the path by a known amount
                        mid-episode and watch the path error afterwards. A
                        closed loop pulls it back at the designed rate; a
                        feed-forward command carries the offset along
                        untouched. With the task-space gain k_lateral the
                        predicted per-step decay is exp(-k_lateral * dt).

  C. plant/model mismatch
                        Apply the commanded joint velocity through a plant
                        whose per-joint gain differs from the model,
                        qdot_actual = G qdot_cmd. Feedback should absorb it;
                        feed-forward should accumulate error.

B and C are run at k_lateral = 0 (the configuration the submitted results were
produced with) and at k_lateral = 5 (the configuration the curved-path
experiments use), because that single term is what decides the answer.

    python -m Yuan.system_eval.control_audit --n-tasks 128
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH.
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _e = dict(os.environ)
    _e["LD_LIBRARY_PATH"] = _conda_lib + ":" + _e.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _e)

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, build_task_aligned_basis, damped_pinv,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent

CAND = "Yuan/unified_rl/runs/iksel_final_n48/iksel_eval10k_candidates.npz"
EVALSET = "Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
ENV_YAML = "Yuan/RL_controller/config.yaml"


def make_env(n, k_lateral, device):
    with open(REPO / ENV_YAML) as f:
        y = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y["env"].items() if k in keys}
    return NSRLBatchedEnv(
        EnvConfig(**{**kw, "n_envs": n, "k_lateral": k_lateral}),
        line_dist=None, device=device)


def load_agent(env, device):
    with open(REPO / RL_CKPT / "config.yaml") as f:
        c = yaml.safe_load(f)
    a = Agent(env.obs_dim, env.act_dim, hidden_dim=c["ppo"]["hidden_dim"],
              init_log_std=c["ppo"]["init_log_std"]).to(device)
    a.load_state_dict(torch.load(REPO / RL_CKPT / "agent.pt", map_location=device))
    return a.eval()


@torch.no_grad()
def run(env, agent, classical, spec, *, kick_step=None, kick_m=0.0,
        gain_err=None, tau_enter=0.98, tau_exit=0.94, n_track=30):
    """Roll out; optionally kick the TCP off the path once, and/or push the
    commanded joint velocity through a mismatched plant.

    Returns arc length, termination code, and the path-error trace after the
    kick (n_track steps, per env).
    """
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    n = env.n_envs
    qm, qh = env.q_mid, env.q_half
    mx = lambda q: ((q - qm).abs() / qh).max(-1).values
    using = mx(env.q) < tau_enter
    term = np.full(n, -1)
    err_trace = np.full((n_track + 1, n), np.nan, np.float32)
    err_free = []

    for t in range(env.max_steps + 1):
        if kick_step is not None and t == kick_step:
            # Displace the TCP a known distance perpendicular to the path,
            # realised through the joint-space least-norm solution so the
            # configuration stays consistent.
            p, _, J, _ = env.kin.tcp_fk_jac(env.q)
            _, lat_vec, _ = env._path_frame(p)
            tang = env.line_dir
            side = torch.linalg.cross(tang, env.n_target, dim=-1)
            side = side / side.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            Jp = J[:, :3, :]
            Jplus, _ = damped_pinv(Jp, env.cfg.lambda_0, env.cfg.sigma_thr)
            dq = (Jplus @ (kick_m * side).unsqueeze(-1)).squeeze(-1)
            env.q = env.q + dq

        if kick_step is not None and kick_step <= t <= kick_step + n_track:
            p, _, _, _ = env.kin.tcp_fk_jac(env.q)
            _, _, e = env._path_frame(p)
            err_trace[t - kick_step] = torch.where(
                env.done_persistent, torch.full_like(e, float("nan")), e
            ).cpu().numpy()

        cq = mx(env.q)
        using = torch.where(using, cq < tau_enter, cq < tau_exit)
        obs = env.current_obs()
        rl = agent.actor_mean(obs).clamp(-1.0, 1.0)
        B, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        qd = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cl = ((B.transpose(-1, -2) @ qd.unsqueeze(-1)).squeeze(-1)
              / env.a_max).clamp(-1.0, 1.0)
        a = torch.where(using.unsqueeze(-1), rl, cl)

        q_before = env.q.clone()
        _, _, _, _, info = env.step(a, auto_reset=False)
        if gain_err is not None:
            # qdot_actual = G qdot_cmd: the plant realises a different joint
            # velocity than the one the resolution law asked for.
            env.q = torch.where(env.done_persistent.unsqueeze(-1), env.q,
                                q_before + gain_err * (env.q - q_before))
        nd = info["episode_done"]
        if bool(nd.any().item()):
            term[nd.cpu().numpy()] = info["term_reason"][nd].cpu().numpy()
        if bool(env.done_persistent.all().item()):
            break
    term[term < 0] = 5
    return env.arc_progress.cpu().numpy(), term, err_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=128)
    ap.add_argument("--kick-mm", type=float, default=5.0)
    ap.add_argument("--kick-step", type=int, default=20)
    ap.add_argument("--gain-err", type=float, default=0.10,
                    help="per-joint multiplicative plant gain error, uniform +-")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)

    es = np.load(REPO / EVALSET, allow_pickle=True)
    ca = np.load(REPO / CAND)
    rows = np.arange(args.n_tasks)
    q0 = np.stack([ca["seeds"][i, np.nonzero(ca["ik_ok"][i])[0][0]] for i in rows])
    N = args.n_tasks

    def spec_of(env):
        dt = env.kin.dtype
        t = lambda v: torch.as_tensor(v, device=dev, dtype=dt)
        return {"q0": t(q0), "line_dir": t(es["cs_line_dir"][rows]),
                "n_target": t(es["cs_n_target"][rows]),
                "p0": t(es["cs_p0"][rows]),
                "kappa": torch.zeros(N, device=dev, dtype=dt)}

    g = torch.Generator(device="cpu").manual_seed(0)
    gain = (1.0 + (torch.rand(N, 7, generator=g) * 2 - 1) * args.gain_err)

    print(f"# control audit — {N} tasks, straight-line tasks (kappa = 0), "
          f"hybrid controller\n")
    print("## A. repeatability")
    env = make_env(N, 0.0, dev)
    agent, cls = load_agent(env, dev), ClassicalNullspaceController(env.kin)
    a1, _, _ = run(env, agent, cls, spec_of(env))
    a2, _, _ = run(env, agent, cls, spec_of(env))
    print(f"   same task run twice: bit-identical arc length on "
          f"{100*np.mean(a1 == a2):.1f}% of tasks  (max |diff| "
          f"{np.abs(a1-a2).max()*1e6:.1f} um)")
    print("   -> nothing in the loop can disagree with the model. Necessary "
          "for an open-loop\n      generator; on its own it settles nothing.\n")

    dt_s = env.dt
    for kl in (0.0, 5.0):
        env = make_env(N, kl, dev)
        agent = load_agent(env, dev)
        cls = ClassicalNullspaceController(env.kin)
        base, tb, _ = run(env, agent, cls, spec_of(env))
        kick, tk, tr = run(env, agent, cls, spec_of(env),
                           kick_step=args.kick_step,
                           kick_m=args.kick_mm * 1e-3)
        gerr, tg, _ = run(env, agent, cls, spec_of(env),
                          gain_err=gain.to(dev, dtype=env.kin.dtype))

        e0 = np.nanmedian(tr[0])
        tail = [np.nanmedian(tr[k]) for k in (1, 3, 5, 10, 20, 30)]
        with np.errstate(invalid="ignore"):
            ratio = np.nanmedian(tr[5] / np.maximum(tr[0], 1e-9))
        decay = ratio ** (1.0 / 5.0)
        pred = float(np.exp(-kl * dt_s))

        print(f"## k_lateral = {kl:g}"
              + ("   <- the configuration the submitted results used"
                 if kl == 0 else "   <- the configuration used on curved paths"))
        print(f"   B. disturbance rejection: TCP displaced {args.kick_mm:.0f} mm "
              f"off the path at step {args.kick_step}")
        print(f"      path error after the kick [mm]: "
              f"t+0 {e0*1e3:5.2f} | " +
              " ".join(f"t+{k} {v*1e3:5.2f}" for k, v in
                       zip((1, 3, 5, 10, 20, 30), tail)))
        print(f"      measured per-step decay {decay:.3f}   "
              f"predicted exp(-k_lateral*dt) {pred:.3f}")
        print(f"      arc length vs undisturbed: median ratio "
              f"{np.median(kick/np.maximum(base,1e-9)):.4f}")
        print(f"   C. plant/model mismatch: qdot_actual = G qdot_cmd, "
              f"G within +-{args.gain_err*100:.0f}% per joint")
        print(f"      arc length vs matched plant: median ratio "
              f"{np.median(gerr/np.maximum(base,1e-9)):.4f}")
        term_shift = {TERM_NAMES.get(int(c), "?"):
                      f"{100*np.mean(tg==c):.1f}% (was {100*np.mean(tb==c):.1f}%)"
                      for c in (3, 4, 6)}
        print(f"      termination mix under mismatch: {term_shift}\n")
        del env
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
