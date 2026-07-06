"""Gate for r13: does hybrid-continuation scoring change/improve MPC labels?

For belt states from pi_D rollouts, score the SAME candidate set under two
continuations:
  G_cls: hold a for H steps, then classical to termination   (r12 protocol)
  G_hyb: hold a for H steps, then the DEPLOYED hybrid — classical while in
         the shell, hand back to the frozen student once rho < tau_exit
         (hysteresis (0.985, 0.96), student = distill_r12m_b0.965_soup2)

Metrics: argmax disagreement rate; value left on the table by the old labels,
measured in deployed-system units: E[G_hyb(a*_hyb) - G_hyb(a*_cls)].
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.RL_controller.self_improve.mpc_teacher import (
    make_candidates, _classical_action, GAMMA)

CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU_E, TAU_X = 0.985, 0.96
N_STATES = 2048
K, HOLD_H, CONT_CAP, SIGMA = 32, 16, 240, 0.2
CHUNK = 768
OUT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2/hybcont_gate.npz')

dev = torch.device('cuda')
student, cfg_yaml = load_agent(CKPT, dev)
env_kw = load_env_kw(cfg_yaml)
line_cfg = cfg_yaml['line_distribution']
threshold_m = (float(line_cfg['feasibility_threshold_m'])
               if line_cfg.get('feasibility_filter', False) else None)


@torch.no_grad()
def collect_belt_states(n_want, qn_lo=0.955):
    proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': 1}),
                           line_dist=None, device=dev)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision, n_pool=line_cfg['n_pool'],
        n_target_noise_deg=line_cfg['n_target_noise_deg'],
        seed=line_cfg['train_seed'], env_cfg=EnvConfig(**{**env_kw, 'n_envs': 1}),
        feasibility_threshold_m=threshold_m)
    del proxy
    gen = torch.Generator(device=dev).manual_seed(9900)
    out = {'q': [], 'ld': [], 'nt': []}
    got = 0
    while got < n_want:
        tasks = pool.sample(4096, generator=gen)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': 4096}),
                             line_dist=None, device=dev)
        env.line_dist = ScriptedLineDistribution(tasks)
        env.reset()
        for _ in range(env.max_steps + 1):
            active = ~env.done_persistent
            qn = ((env.q - env.q_mid).abs() / env.q_half).max(-1).values
            sel = active & (qn >= qn_lo)
            if bool(sel.any()):
                out['q'].append(env.q[sel].float().cpu())
                out['ld'].append(env.line_dir[sel].float().cpu())
                out['nt'].append(env.n_target[sel].float().cpu())
                got += int(sel.sum())
            a = student.actor_mean(env.current_obs()).clamp(-1, 1)
            env.step(a, auto_reset=False)
            if bool(env.done_persistent.all().item()):
                break
        del env
        torch.cuda.empty_cache()
        print(f'[gate] belt states so far: {got}', flush=True)
    q = torch.cat(out['q'])[:n_want]
    ld = torch.cat(out['ld'])[:n_want]
    nt = torch.cat(out['nt'])[:n_want]
    return q, ld, nt


@torch.no_grad()
def score(q0, ld, nt, continuation):
    """G (B,K) under 'cls' or 'hyb' continuation; same candidate seed."""
    B = q0.shape[0]
    gen = torch.Generator(device=dev).manual_seed(1234)
    probe = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': B}),
                           line_dist=None, device=dev)
    spec = {'q0': q0.to(dev, probe.kin.dtype),
            'line_dir': ld.to(dev, probe.kin.dtype),
            'n_target': nt.to(dev, probe.kin.dtype)}
    probe.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
    probe.reset()
    a_cls = _classical_action(probe, ClassicalNullspaceController(probe.kin)).float()
    a_pol = student.actor_mean(probe.current_obs()).clamp(-1, 1).float()
    del probe
    torch.cuda.empty_cache()
    cands = make_candidates(a_cls, a_pol, K, gen, SIGMA)     # (B,K,4)

    env = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': B * K}),
                         line_dist=None, device=dev)
    tiled = {k: v.repeat_interleave(K, dim=0) for k, v in spec.items()}
    env.line_dist = ScriptedLineDistribution(tiled)
    env.reset()
    ctrl = ClassicalNullspaceController(env.kin)
    flat = cands.reshape(B * K, -1).to(env.kin.dtype)
    using_rl = torch.zeros(B * K, dtype=torch.bool, device=dev)  # start: shell
    G = torch.zeros(B * K, dtype=torch.float64, device=dev)
    disc = 1.0
    for t in range(HOLD_H + CONT_CAP):
        if bool(env.done_persistent.all().item()):
            break
        if t < HOLD_H:
            a = flat
        elif continuation == 'cls':
            a = _classical_action(env, ctrl)
        else:
            qn = ((env.q - env.q_mid).abs() / env.q_half).max(-1).values
            using_rl = torch.where(using_rl, qn < TAU_E, qn < TAU_X)
            a_c = _classical_action(env, ctrl)
            a_s = student.actor_mean(env.current_obs()).clamp(-1, 1).to(env.kin.dtype)
            a = torch.where(using_rl.unsqueeze(-1), a_s, a_c)
        _, r, _, _, _ = env.step(a, auto_reset=False)
        G += disc * r.double()
        disc *= GAMMA
    del env
    torch.cuda.empty_cache()
    return G.float().view(B, K).cpu(), cands.cpu()


q0, ld, nt = collect_belt_states(N_STATES)
print(f'[gate] scoring {N_STATES} states x {K} candidates, two continuations',
      flush=True)
Gc_l, Gh_l = [], []
for s in range(0, N_STATES, CHUNK):
    e = min(s + CHUNK, N_STATES)
    gc, _ = score(q0[s:e], ld[s:e], nt[s:e], 'cls')
    gh, _ = score(q0[s:e], ld[s:e], nt[s:e], 'hyb')
    Gc_l.append(gc); Gh_l.append(gh)
    print(f'[gate] {e}/{N_STATES}', flush=True)
Gc, Gh = torch.cat(Gc_l), torch.cat(Gh_l)

pick_c, pick_h = Gc.argmax(1), Gh.argmax(1)
idx = torch.arange(N_STATES)
disagree = (pick_c != pick_h).float().mean().item()
gain = (Gh[idx, pick_h] - Gh[idx, pick_c])
rel = gain / Gh[idx, pick_h].clamp(min=1e-6)
np.savez_compressed(OUT, G_cls=Gc.numpy(), G_hyb=Gh.numpy(),
                    pick_cls=pick_c.numpy(), pick_hyb=pick_h.numpy())
print('\n==== HYBRID-CONTINUATION GATE ====')
print(f'argmax disagreement: {100*disagree:.1f}%')
print(f'value left on table by cls-continuation labels '
      f'(deployed units): mean {gain.mean():.3f}, '
      f'mean rel {100*rel.mean():.1f}%, P90 {gain.quantile(0.9):.3f}')
print(f'G_hyb scale: mean(best) {Gh[idx, pick_h].mean():.2f}')
print(f'(r12 anchor: search beat classical by 8.6% under cls-continuation)')
print('[gate] done', flush=True)
