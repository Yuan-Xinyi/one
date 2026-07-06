"""r13: MPC teacher with FAITHFUL HYBRID CONTINUATION (user-motivated fix).

Gate result (hybcont_gate): 63.7% argmax disagreement vs classical
continuation; +3.0% deployed value left on the table by the old labels.

Protocol mirrors r12 for a controlled comparison, changing ONLY the scoring
continuation: hold candidate H steps, then classical while in the shell,
hand back to the FROZEN student pi_D once rho < tau_exit (0.985/0.96).
Two DAgger rounds; belt rows blended with pi0 in the band (same as r12);
final merges with r6 clean labels at boundaries {0.965, 0.975}, warm-start
soup2; eval pure + hybrid @0.985/0.96.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch, yaml

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.RL_controller.self_improve.distill import fit_actor, TARGET_CLAMP
from Yuan.RL_controller.self_improve.mpc_teacher import (
    make_candidates, _classical_action, GAMMA)
from Yuan.RL_controller.self_improve.loop import eval_ckpt_on_10k

RUNS = Path('Yuan/RL_controller/runs')
OUT = RUNS / 'distill_r13_hybcont'
OUT.mkdir(exist_ok=True)
PI_D = RUNS / 'distill_r12m_b0.965_soup2'   # frozen continuation student
PI0 = RUNS / 'p0_progress_only_30M_0520'
WARM = RUNS / 'distill_soup2'
TAU_E, TAU_X = 0.985, 0.96
K, HOLD_H, CONT_CAP, SIGMA = 32, 16, 240, 0.2
N_TASKS = 12288
BELT_LO, TAU_HI, BAND = 0.955, 0.975, 0.02
MPC_CHUNK = 640

dev = torch.device('cuda')
pi_d, cfg_yaml = load_agent(PI_D, dev)     # frozen continuation
pi0, _ = load_agent(PI0, dev)
env_kw = load_env_kw(cfg_yaml)
line_cfg = cfg_yaml['line_distribution']
threshold_m = (float(line_cfg['feasibility_threshold_m'])
               if line_cfg.get('feasibility_filter', False) else None)
proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': 1}),
                       line_dist=None, device=dev)
pool = LineDistribution.load_or_build(
    kin=proxy.kin, collision=proxy.collision, n_pool=line_cfg['n_pool'],
    n_target_noise_deg=line_cfg['n_target_noise_deg'],
    seed=line_cfg['train_seed'], env_cfg=EnvConfig(**{**env_kw, 'n_envs': 1}),
    feasibility_threshold_m=threshold_m)
del proxy


@torch.no_grad()
def collect_belt(behavior, n_tasks, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    tasks = pool.sample(n_tasks, generator=gen)
    out = {'obs': [], 'q': [], 'ld': [], 'nt': []}
    for s in range(0, n_tasks, 4096):
        e = min(s + 4096, n_tasks)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': e - s}),
                             line_dist=None, device=dev)
        env.line_dist = ScriptedLineDistribution(
            {k: v[s:e].clone() for k, v in tasks.items()})
        env.reset()
        for _ in range(env.max_steps + 1):
            active = ~env.done_persistent
            qn = ((env.q - env.q_mid).abs() / env.q_half).max(-1).values
            sel = active & (qn >= BELT_LO)
            if bool(sel.any()):
                out['obs'].append(env.current_obs()[sel].float().cpu())
                out['q'].append(env.q[sel].float().cpu())
                out['ld'].append(env.line_dir[sel].float().cpu())
                out['nt'].append(env.n_target[sel].float().cpu())
            a = behavior.actor_mean(env.current_obs()).clamp(-1, 1)
            env.step(a, auto_reset=False)
            if bool(env.done_persistent.all().item()):
                break
        del env
        torch.cuda.empty_cache()
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def hybcont_label(q0, ld, nt, seed):
    """a_best (B,4) under hold->hybrid continuation scoring."""
    B = q0.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    probe = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': B}),
                           line_dist=None, device=dev)
    spec = {'q0': q0.to(dev, probe.kin.dtype),
            'line_dir': ld.to(dev, probe.kin.dtype),
            'n_target': nt.to(dev, probe.kin.dtype)}
    probe.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
    probe.reset()
    a_cls = _classical_action(probe, ClassicalNullspaceController(probe.kin)).float()
    a_pol = pi_d.actor_mean(probe.current_obs()).clamp(-1, 1).float()
    del probe
    torch.cuda.empty_cache()
    cands = make_candidates(a_cls, a_pol, K, gen, SIGMA)
    env = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': B * K}),
                         line_dist=None, device=dev)
    env.line_dist = ScriptedLineDistribution(
        {k: v.repeat_interleave(K, dim=0) for k, v in spec.items()})
    env.reset()
    ctrl = ClassicalNullspaceController(env.kin)
    flat = cands.reshape(B * K, -1).to(env.kin.dtype)
    using_rl = torch.zeros(B * K, dtype=torch.bool, device=dev)
    G = torch.zeros(B * K, dtype=torch.float64, device=dev)
    disc = 1.0
    for t in range(HOLD_H + CONT_CAP):
        if bool(env.done_persistent.all().item()):
            break
        if t < HOLD_H:
            a = flat
        else:
            qn = ((env.q - env.q_mid).abs() / env.q_half).max(-1).values
            using_rl = torch.where(using_rl, qn < TAU_E, qn < TAU_X)
            a_c = _classical_action(env, ctrl)
            a_s = pi_d.actor_mean(env.current_obs()).clamp(-1, 1).to(env.kin.dtype)
            a = torch.where(using_rl.unsqueeze(-1), a_s, a_c)
        _, r, _, _, _ = env.step(a, auto_reset=False)
        G += disc * r.double()
        disc *= GAMMA
    del env
    torch.cuda.empty_cache()
    pick = G.float().view(B, K).argmax(1)
    return cands[torch.arange(B), pick.cpu()]


def label_round(rnd, behavior, seed):
    ds = OUT / f'dataset_round{rnd}.npz'
    if ds.exists():
        d = np.load(ds)
        return torch.from_numpy(d['obs']), torch.from_numpy(d['act'])
    st = collect_belt(behavior, N_TASKS, seed)
    n = st['obs'].shape[0]
    print(f'[r13] round {rnd}: {n} belt states', flush=True)
    a_best = torch.zeros((n, 4))
    for s in range(0, n, MPC_CHUNK):
        e = min(s + MPC_CHUNK, n)
        a_best[s:e] = hybcont_label(st['q'][s:e], st['ld'][s:e],
                                    st['nt'][s:e], seed + s)
        print(f'[r13]   labeled {e}/{n}', flush=True)
    # band blend with pi0 (same as r12 recipe)
    with torch.no_grad():
        a_p0 = pi0.actor_mean(st['obs'].to(dev)).clamp(-1, 1).float().cpu()
    qn = st['obs'][:, :7].abs().max(1).values
    w = ((qn - BELT_LO) / BAND).clamp(0, 1).unsqueeze(-1)
    act = (1 - w) * a_p0 + w * a_best
    np.savez_compressed(ds, obs=st['obs'].numpy(), act=act.numpy())
    return st['obs'], act


# ---- round 0: behavior = frozen pi_D ----
obs0_r, act0_r = label_round(0, pi_d, 11000)

# ---- intermediate fit for DAgger behavior ----
d6 = np.load(RUNS / 'distill_r6_soft/distill_dataset.npz')
obs6 = torch.from_numpy(d6['obs']).float()
act6 = torch.from_numpy(d6['act']).float()
qn6 = obs6[:, :7].abs().max(1).values
inter_pt = OUT / 'intermediate.pt'
inter, _ = load_agent(WARM, dev)
if inter_pt.exists():
    inter.load_state_dict(torch.load(inter_pt, map_location=dev, weights_only=False))
else:
    keep6 = qn6 < 0.965
    qn0 = obs0_r[:, :7].abs().max(1).values
    keep0 = qn0 >= 0.965
    obs_m = torch.cat([obs6[keep6], obs0_r[keep0]])
    act_m = torch.cat([act6[keep6], act0_r[keep0]]).clamp(-TARGET_CLAMP, TARGET_CLAMP)
    torch.manual_seed(8900)
    fit_actor(inter, obs_m, act_m, dev, epochs=60)
    torch.save(inter.state_dict(), inter_pt)
    print('[r13] intermediate fitted', flush=True)

# ---- round 1: behavior = intermediate student ----
obs1_r, act1_r = label_round(1, inter, 12000)
del inter
torch.cuda.empty_cache()

# ---- final merges + eval ----
obs13 = torch.cat([obs0_r, obs1_r])
act13 = torch.cat([act0_r, act1_r])
qn13 = obs13[:, :7].abs().max(1).values
for tb in (0.965, 0.975):
    out_dir = RUNS / f'distill_r13m_b{tb:.3f}'
    if (out_dir / 'eval_10k.npz').exists():
        print(f'[r13] {out_dir.name} done, skip', flush=True)
        continue
    out_dir.mkdir(exist_ok=True)
    obs = torch.cat([obs6[qn6 < tb], obs13[qn13 >= tb]])
    act = torch.cat([act6[qn6 < tb], act13[qn13 >= tb]]).clamp(
        -TARGET_CLAMP, TARGET_CLAMP)
    print(f'[r13m] b{tb}: {int((qn6 < tb).sum())} clean + '
          f'{int((qn13 >= tb).sum())} hybcont-MPC rows', flush=True)
    student, _ = load_agent(WARM, dev)
    torch.manual_seed(8910)
    val = fit_actor(student, obs, act, dev, epochs=80)
    torch.save(student.state_dict(), out_dir / 'agent.pt')
    cfg = dict(cfg_yaml)
    cfg['distill'] = {'note': f'r13: r6 clean (qn<{tb}) + hybrid-continuation '
                              f'MPC K=32/hold16 (qn>={tb}), cont student pi_D '
                              f'frozen, warm-start soup2', 'val_mse': float(val)}
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    del student
    torch.cuda.empty_cache()
    eval_ckpt_on_10k(out_dir, out_dir / 'eval_10k.npz',
                     tau_enter=TAU_E, tau_exit=TAU_X, device=dev)
print('[r13] all done', flush=True)
