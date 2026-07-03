"""Value-based switch: classical when V_cls(s) - V_pi(s) > margin.

Stage 1 (build): collect states on classical+student visitation, label each
state with BOTH exact returns (classical rollout and student rollout from the
re-rooted state) — paired labels give (a) symmetric fits for V_cls / V_pi and
(b) a ground-truth estimate of the switch headroom.
Stage 2 (eval): hybrid_value rollout on the 10k eval set, switch by net
comparison (optional qn gate + margin hysteresis), per-task cache.

Usage:
    python vswitch.py build --student <ckpt_dir>
    python vswitch.py eval  --student <ckpt_dir> [--qn-gate 0.9]
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lqin/one")
from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_TRUNCATED, build_task_aligned_basis)
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.RL_controller.self_improve.collect import (
    load_agent, load_env_kw, rl_action_fn)
from Yuan.RL_controller.self_improve.vcls import (
    VClsNet, _snapshot_rollout, _classical_discounted_return)

EVAL_SET = "/home/lqin/one/Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
GAMMA = 0.99


@torch.no_grad()
def _policy_discounted_return(env, agent, gamma):
    """Exact G_pi: roll the deterministic student to termination."""
    action_fn = rl_action_fn(agent)
    env.reset()
    G = torch.zeros(env.n_envs, dtype=torch.float64, device=env.device)
    disc = 1.0
    for _ in range(env.max_steps + 1):
        a = action_fn(env)
        _, r, _, _, _ = env.step(a, auto_reset=False)
        G += disc * r.double()
        disc *= gamma
        if bool(env.done_persistent.all().item()):
            break
    return G.float()


def build(student_dir: Path, out_dir: Path, *, n_tasks=8192, stride=4,
          chunk=4096, max_states=200_000, seed=8900, device=None):
    device = torch.device(device or "cuda")
    agent, cfg_yaml = load_agent(student_dir, device)
    env_kw = load_env_kw(cfg_yaml)
    line_cfg = cfg_yaml["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": 1}),
                           line_dist=None, device=device)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision, n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"], env_cfg=EnvConfig(**{**env_kw, "n_envs": 1}),
        feasibility_threshold_m=threshold_m)
    del proxy
    gen = torch.Generator(device=device).manual_seed(seed)
    tasks = pool.sample(n_tasks, generator=gen)

    parts = []
    for name in ("classical", "policy"):
        for s in range(0, n_tasks, chunk):
            e = min(s + chunk, n_tasks)
            env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": e - s}),
                                 line_dist=None, device=device)
            env.line_dist = ScriptedLineDistribution(
                {k: v[s:e].clone() for k, v in tasks.items()})
            fn = (cn_action_fn(ClassicalNullspaceController(env.kin))
                  if name == "classical" else rl_action_fn(agent))
            parts.append(_snapshot_rollout(env, fn, stride))
            del env; torch.cuda.empty_cache()
        print(f"[vswitch] {name} states: "
              f"{sum(p['obs'].shape[0] for p in parts)}", flush=True)
    states = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    n = states["obs"].shape[0]
    if n > max_states:
        pick = torch.randperm(n)[:max_states]
        states = {k: v[pick] for k, v in states.items()}
        n = max_states

    G = {"cls": torch.zeros(n), "pi": torch.zeros(n)}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        for which in ("cls", "pi"):
            env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": e - s}),
                                 line_dist=None, device=device)
            env.line_dist = ScriptedLineDistribution({
                "q0": states["q"][s:e].to(device, env.kin.dtype),
                "line_dir": states["line_dir"][s:e].to(device, env.kin.dtype),
                "n_target": states["n_target"][s:e].to(device, env.kin.dtype)})
            if which == "cls":
                G["cls"][s:e] = _classical_discounted_return(
                    env, ClassicalNullspaceController(env.kin), GAMMA).cpu()
            else:
                G["pi"][s:e] = _policy_discounted_return(env, agent, GAMMA).cpu()
            del env; torch.cuda.empty_cache()
        print(f"[vswitch] labeled {e}/{n} (paired)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "paired_dataset.npz",
                        obs=states["obs"].numpy(),
                        G_cls=G["cls"].numpy(), G_pi=G["pi"].numpy(),
                        gamma=np.float64(GAMMA))
    qn = states["obs"][:, :7].abs().max(1).values.numpy()
    adv = (G["cls"] - G["pi"]).numpy()
    print(f"[vswitch] dataset {n} states; frac(G_cls>G_pi) "
          f"{100*(adv>0).mean():.1f}% overall, "
          f"{100*(adv[qn>=0.95]>0).mean():.1f}% @qn>=0.95", flush=True)

    for key in ("cls", "pi"):
        tgt = G[key]
        m, sd = float(tgt.mean()), float(tgt.std().clamp(min=1e-6))
        net = VClsNet().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        obs_t = states["obs"].to(device)
        tgt_t = ((tgt - m) / sd).to(device)
        n_val = n // 20
        perm = torch.randperm(n, device=device)
        va, tr = perm[:n_val], perm[n_val:]
        for ep in range(40):
            order = tr[torch.randperm(len(tr), device=device)]
            for s in range(0, len(order), 4096):
                i = order[s:s + 4096]
                loss = ((net(obs_t[i]) - tgt_t[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rmse = float((((net(obs_t[va]) - tgt_t[va]) ** 2).mean().sqrt() * sd))
        torch.save({"state_dict": net.state_dict(), "target_mean": m,
                    "target_std": sd, "gamma": GAMMA, "margin": 0.0,
                    "val_rmse": rmse}, out_dir / f"v{key}_sym.pt")
        print(f"[vswitch] v{key}_sym fitted: val RMSE {rmse:.2f} raw "
              f"(labels {m:.0f}±{sd:.0f})", flush=True)


def _load_vnet(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = VClsNet().to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    m, sd = float(ckpt["target_mean"]), float(ckpt["target_std"])
    return lambda o: net(o) * sd + m


@torch.no_grad()
def _rollout_value_switch_chunk(qs, p0s, lds, nts, *, env, classical, agent,
                                vcls_fn, vpi_fn, m_enter, m_exit, qn_gate):
    spec = {"q0": qs, "line_dir": lds, "n_target": nts}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = p0s
    p_start, line_dir = env.p_start.clone(), env.line_dir.clone()
    n = env.n_envs
    q_mid, q_half = env.q_mid, env.q_half

    def _qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    progress = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    ep_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    switch_count = torch.zeros(n, dtype=torch.long, device=env.device)
    using_rl = torch.ones(n, dtype=torch.bool, device=env.device)

    for _ in range(env.max_steps + 1):
        obs = env.current_obs()
        dv = vcls_fn(obs.float()) - vpi_fn(obs.float())   # >0: classical better
        gate = _qn(env.q) >= qn_gate
        want_cls = torch.where(using_rl, dv > m_enter, dv > m_exit) & gate
        new_using_rl = ~want_cls
        switched = new_using_rl != using_rl
        switch_count += (switched & ~finished).long()
        using_rl = new_using_rl

        rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)
        B_basis, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
        cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)
        a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info["episode_done"]
        if bool(new_done.any().item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            prog = ((p_now - p_start) * line_dir).sum(-1)
            progress[new_done] = prog[new_done]
            ep_len[new_done] = env.t[new_done]
            term[new_done] = info["term_reason"][new_done]
            finished |= new_done
        if bool(env.done_persistent.all().item()):
            break
    if (~finished).any():
        nd = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p_now - p_start) * line_dir).sum(-1)
        progress[nd] = prog[nd]
        ep_len[nd] = env.t[nd]
        term[nd] = TERM_TRUNCATED
    return {"progress_m": progress, "episode_len": ep_len,
            "term_reason": term, "switch_count": switch_count}


def eval_10k(student_dir: Path, out_dir: Path, *, variants, chunk=4096,
             device=None):
    from Yuan.system_eval.rollout_controllers import build_env, load_rl_agent
    device = torch.device(device or "cuda")
    d = np.load(EVAL_SET)
    qs, p0 = d["q0_seed"], d["cs_p0"]
    ld, nt = d["cs_line_dir"], d["cs_n_target"]
    L_oracle = d["max_label_L"]
    valid = L_oracle > 1e-6

    env = build_env(student_dir / "config.yaml", chunk, device)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(student_dir, env, device)
    vcls_fn = _load_vnet(out_dir / "vcls_sym.pt", device)
    vpi_fn = _load_vnet(out_dir / "vpi_sym.pt", device)
    dtype = env.kin.dtype

    for m_enter, m_exit, qn_gate in variants:
        tag = f"me{m_enter:g}_mx{m_exit:g}_g{qn_gate:g}"
        out = out_dir / f"eval_{tag}.npz"
        if out.exists():
            r = np.load(out)
            print(f"[cached] {tag}: {float(r['metric_ratio_vs_oracle_mean']):.4f}",
                  flush=True)
            continue
        B = qs.shape[0]
        L = np.zeros(B, np.float32)
        elen = np.zeros(B, np.int64)
        trm = np.zeros(B, np.int32)
        sw = np.zeros(B, np.int32)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            pad = chunk - (e - s)
            def _t(x, w):
                t = torch.as_tensor(x[s:e], device=device, dtype=dtype)
                return (torch.cat([t, t[-1:].expand(pad, w)]) if pad else t)
            res = _rollout_value_switch_chunk(
                _t(qs, 7), _t(p0, 3), _t(ld, 3), _t(nt, 3),
                env=env, classical=classical, agent=agent,
                vcls_fn=vcls_fn, vpi_fn=vpi_fn,
                m_enter=m_enter, m_exit=m_exit, qn_gate=qn_gate)
            L[s:e] = (res["progress_m"][:e - s].cpu().numpy() / 1.5)
            elen[s:e] = res["episode_len"][:e - s].cpu().numpy()
            trm[s:e] = res["term_reason"][:e - s].cpu().numpy()
            sw[s:e] = res["switch_count"][:e - s].cpu().numpy()
            print(f"  vswitch {tag} {e}/{B}", flush=True)
        r_v = L[valid] / L_oracle[valid]
        np.savez_compressed(
            out, L_vswitch=L, L_oracle=L_oracle, episode_len=elen,
            term_reason=trm, switch_count=sw,
            m_enter=np.float64(m_enter), m_exit=np.float64(m_exit),
            qn_gate=np.float64(qn_gate), ckpt_dir=np.str_(str(student_dir)),
            metric_ratio_vs_oracle_mean=np.float64(r_v.mean()),
            metric_ratio_vs_oracle_median=np.float64(np.median(r_v)),
            metric_mean_switches=np.float64(sw.mean()))
        print(f"[done] {tag}: L/L_oracle {r_v.mean():.4f} "
              f"(med {np.median(r_v):.4f})  sw/ep {sw.mean():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--student", required=True)
    ap.add_argument("--qn-gate", type=float, default=0.90)
    args = ap.parse_args()
    student = Path(args.student)
    out_dir = student / "vswitch"
    os.chdir("/home/lqin/one")
    if args.cmd == "build":
        build(student, out_dir)
    else:
        # margins in raw reward units; scale sanity-checked against dataset std
        dd = np.load(out_dir / "paired_dataset.npz")
        sd = float(np.std(dd["G_cls"] - dd["G_pi"]))
        print(f"[vswitch] dv std = {sd:.2f} raw units", flush=True)
        variants = [(0.0, 0.0, args.qn_gate),
                    (0.0, -0.1 * sd, args.qn_gate),
                    (0.1 * sd, -0.1 * sd, args.qn_gate),
                    (0.0, 0.0, 0.0)]
        eval_10k(student, out_dir, variants=variants)
