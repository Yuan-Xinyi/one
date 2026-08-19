"""Transfer test: the ISRR seed-selection ranker (ens10, pair+list, trained
on 60k tasks for the classical/hybrid controller) scoring TODAY's start
candidates, judged by realized progress of the value-lookahead controller.
Same candidates and labels as seed_calib.npz (regenerated deterministically).
"""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn as nn, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
kw = dict(y['env']); kw['dt'] = kw['dt'] / hl.SUB
kw['max_steps'] = int(y['env']['max_steps'] * hl.SUB)
OUT = REPO / 'Yuan/IJRR/runs/vlook_ablation'

env1 = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env1.kin, collision=env1.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env1.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
perm = torch.randperm(valid.numel(),
                      generator=torch.Generator().manual_seed(7))
sel = valid[perm[:12000 + 300 + 24]]
kin = env1.kin
model = hl.StraightModel(env1)

# ---- regenerate EXACTLY the eval candidates of seed_calib (seed=1) ----
EV = list(range(12000, 12300))
rng = torch.Generator(device='cpu').manual_seed(1)
Q0 = pool.q_pool[sel[EV]].to(dev)
DIR = pool.line_dir_pool[sel[EV]].to(dev)
NTG = pool.n_target_pool[sel[EV]].to(dev)
P0 = kin.tcp_fk_jac(Q0)[0]
cands, owner = [], []
for i in range(len(EV)):
    got, tries = [Q0[i]], 0
    while len(got) < 8 and tries < 150:
        tries += 1
        scale = 0.2 + 0.6 * torch.rand(1, generator=rng).item()
        q = Q0[i] + scale * torch.randn(7, generator=rng).to(dev).to(Q0.dtype)
        q = q.clamp(kin.lmt_lo + 0.02, kin.lmt_up - 0.02)
        ok = True
        for it in range(60):
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
            continue
        got.append(q)
    cands.extend(got)
    owner.extend([i] * len(got))
CAND = torch.stack(cands)
OWNER = np.array(owner)

# sanity: labels from seed_calib must match this candidate set
sc = np.load(OUT / 'seed_calib.npz')
assert sc['prog'].shape[0] == CAND.shape[0], \
    f"candidate regen mismatch: {sc['prog'].shape[0]} vs {CAND.shape[0]}"
assert np.array_equal(sc['owner'], OWNER)
PROG = sc['prog']

# ---- ranker features: fresh-start obs31 + log manipulability ----
ii = torch.as_tensor(OWNER, device=dev)
envB = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': CAND.shape[0]}), None, dev)
envB.line_dist = ScriptedLineDistribution(
    {'q0': CAND.clone(), 'line_dir': DIR[ii].clone(),
     'n_target': NTG[ii].clone()})
envB.reset()
OBS = envB.current_obs().float()
_, _, J, _ = kin.tcp_fk_jac(envB.q)
Jp = J[:, :3, :]
mu = torch.sqrt(torch.det(Jp @ Jp.transpose(-1, -2)).clamp(min=1e-12))
X = torch.cat([OBS, torch.log(mu.float() + 1e-9)[:, None]], -1)

ck = torch.load('/home/lqin/one/Yuan/seed_selection/runs/rank_train/'
                'ranker_v4.pt', map_location=dev, weights_only=False)
mean = torch.tensor(ck['mean'], device=dev, dtype=torch.float32)
std = torch.tensor(ck['std'], device=dev, dtype=torch.float32)
Xn = (X - mean) / std


class Rank(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


scores = torch.zeros(CAND.shape[0], device=dev)
with torch.no_grad():
    for sd in ck['nets']:
        net = Rank().to(dev)
        net.load_state_dict(sd)
        net.eval()
        scores += net(Xn)
S = scores.cpu().numpy()

# ---- capture on the same per-task protocol ----
from scipy.stats import spearmanr
res = {k: [] for k in ('rnd', 'pick', 'oracle')}
sp = []
for i in range(len(EV)):
    m = OWNER == i
    if m.sum() < 3:
        continue
    p = PROG[m]
    res['rnd'].append(p.mean())
    res['pick'].append(p[np.argmax(S[m])])
    res['oracle'].append(p.max())
    if np.std(p) > 1e-6:
        sp.append(spearmanr(S[m], p).statistic)
res = {k: np.array(v) for k, v in res.items()}
gap = res['oracle'].mean() - res['rnd'].mean()
cap = (res['pick'].mean() - res['rnd'].mean()) / gap
print(f"ISRR ens10 ranker on today's candidates ({len(res['rnd'])} tasks):")
print(f"  picked {res['pick'].mean():.4f} m   random {res['rnd'].mean():.4f}"
      f"   oracle {res['oracle'].mean():.4f}")
print(f"  capture {cap:+.1%}   within-task Spearman {np.mean(sp):.3f}")
np.savez(OUT / 'isrr_ranker_transfer.npz', score=S, prog=PROG, owner=OWNER,
         capture=cap)
print('wrote', OUT / 'isrr_ranker_transfer.npz')
