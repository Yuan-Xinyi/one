"""Can the SAME critic that ranks successors also rank START configurations?

For each held-out task: generate K alternative IK solutions of the start pose
(random joint perturbation projected back onto the pen-tip position by damped
least squares, constraints checked), score each candidate with the critic at
its fresh-start observation, roll the value-lookahead controller from every
candidate, and compare: original q0 / random candidate / critic-picked /
oracle-picked (best realized).
"""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, damped_pinv)
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
kw = dict(y['env']); kw['dt'] = kw['dt'] / hl.SUB
kw['max_steps'] = int(y['env']['max_steps'] * hl.SUB)

NT, K = 100, 8                      # tasks x candidates per task
env1 = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env1.kin, collision=env1.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env1.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
perm = torch.randperm(valid.numel(),
                      generator=torch.Generator().manual_seed(7))
sel = valid[perm[:12000 + 300 + 24]][12000:12000 + NT]   # held-out slice
Q0 = pool.q_pool[sel].to(dev)
DIR = pool.line_dir_pool[sel].to(dev)
NTG = pool.n_target_pool[sel].to(dev)
kin = env1.kin
P0 = kin.tcp_fk_jac(Q0)[0]
model = hl.StraightModel(env1)

# ---- candidate generation: perturb + project back to the pen-tip point ----
rng = torch.Generator(device='cpu').manual_seed(0)
cands, owner = [], []
t0 = time.time()
for i in range(NT):
    got, tries = [Q0[i]], 0
    while len(got) < K and tries < 200:
        tries += 1
        scale = 0.2 + 0.6 * torch.rand(1, generator=rng).item()
        q = Q0[i] + scale * torch.randn(7, generator=rng).to(dev).to(Q0.dtype)
        q = q.clamp(kin.lmt_lo + 0.02, kin.lmt_up - 0.02)
        ok = True
        for it in range(60):                    # project position error out
            p, _, J, _ = kin.tcp_fk_jac(q[None])
            err = P0[i] - p[0]
            if float(err.norm()) < 1e-4:
                break
            J_plus, _ = damped_pinv(J[:, :3, :], env1.cfg.lambda_0,
                                    env1.cfg.sigma_thr)
            q = (q + (J_plus[0] @ err)).clamp(kin.lmt_lo + 0.02,
                                              kin.lmt_up - 0.02)
        else:
            ok = False
        if not ok:
            continue
        m = model.margins(q[None], P0[i][None], DIR[i][None], NTG[i][None])
        if not bool(m.amin() > 0):
            continue
        if min(float((q - g).abs().max()) for g in got) < 0.08:
            continue                            # duplicate branch
        got.append(q)
    cands.extend(got)
    owner.extend([i] * len(got))
CAND = torch.stack(cands)
OWNER = np.array(owner)
n_multi = sum(1 for i in range(NT) if (OWNER == i).sum() >= 3)
print(f"candidates: {CAND.shape[0]} for {NT} tasks "
      f"({n_multi} tasks with >=3) ({time.time()-t0:.0f}s)", flush=True)

# ---- critic score at the fresh-start observation of each candidate ----
B = CAND.shape[0]
envB = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ii = torch.as_tensor(OWNER, device=dev)
spec = {'q0': CAND.clone(), 'line_dir': DIR[ii].clone(),
        'n_target': NTG[ii].clone()}
envB.line_dist = ScriptedLineDistribution({k: v.clone()
                                           for k, v in spec.items()})
envB.reset()
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', envB.obs_dim, dev,
            act_dim=envB.act_dim)
with torch.no_grad():
    V0 = ag.critic(envB.current_obs()).squeeze(-1).cpu().numpy()

# ---- roll the value-lookahead controller from every candidate ----
mdl = hl.StraightModel(envB)
vlook = hl.make_vlook(mdl, envB, ag)
envB.line_dist = ScriptedLineDistribution({k: v.clone()
                                           for k, v in spec.items()})
prog, term = hl.rollout_env(envB, vlook)
print(f"rollouts done ({time.time()-t0:.0f}s)", flush=True)

# ---- selection comparison on tasks with >=3 candidates ----
rows = []
from scipy.stats import spearmanr
sp_all = []
for i in range(NT):
    m = OWNER == i
    if m.sum() < 3:
        continue
    p, v = prog[m], V0[m]
    orig = p[0]                                  # candidate 0 is pool q0
    rows.append((orig, p.mean(), p[np.argmax(v)], p.max(), p.min()))
    if np.std(p) > 1e-6 and np.std(v) > 1e-6:
        sp_all.append(spearmanr(v, p).statistic)
rows = np.array(rows)
orig, rnd, vpick, oracle, worst = rows.T
print(f"\ntasks with >=3 IK candidates: {len(rows)}")
print(f"mean progress (m):")
print(f"  original pool q0        {orig.mean():.4f}")
print(f"  random candidate        {rnd.mean():.4f}")
print(f"  critic-picked candidate {vpick.mean():.4f}")
print(f"  oracle (best realized)  {oracle.mean():.4f}   worst {worst.mean():.4f}")
cap = (vpick.mean() - rnd.mean()) / max(oracle.mean() - rnd.mean(), 1e-9)
print(f"capture of the oracle-over-random gap: {cap:.1%}")
print(f"critic pick == oracle pick on {np.mean(vpick >= oracle - 1e-6):.1%} "
      f"of tasks; within-task Spearman(V, progress) mean "
      f"{np.mean(sp_all):.3f}")
out = REPO / 'Yuan/IJRR/runs/vlook_ablation/seed_by_value.npz'
np.savez(out, cand=CAND.cpu().numpy(), owner=OWNER, v0=V0, prog=prog,
         rows=rows)
print('wrote', out)