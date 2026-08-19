"""Transfer test for the IJRR paper's OWN selector (5-member SetSel, 45-D
features, labels = return tables under the RL/classical hybrid controller):
score today's start candidates, judge by realized value-lookahead progress.
Same candidates/labels as seed_calib.npz."""
import sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.stage1_seed.candidate_batch import SeedCandidateBatch
from Yuan.IJRR.stage1_seed.features import initial_observation_features
from Yuan.IJRR.stage1_seed.setsel import SetSel, _picks

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

# regenerate the same eval candidates (seed=1) as seed_calib
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
    cands.append(torch.stack(got + [got[0]] * (8 - len(got))))
    owner.append(len(got))
QK = torch.stack(cands)                       # (B, 8, 7), padded w/ dup
NVAL = np.array(owner)
V = (torch.arange(8, device=dev)[None, :]
     < torch.as_tensor(NVAL, device=dev)[:, None])

sc = np.load(OUT / 'seed_calib.npz')
PROG_flat, OWNER_flat = sc['prog'], sc['owner']

batch = SeedCandidateBatch(q0=QK, p0=P0, line_dir=DIR, n_target=NTG,
                           valid=V)
X = initial_observation_features(kin, batch, include_ray_error=True,
                                 include_log_manip=True,
                                 include_directional_dynamics=True).float()
print('features', tuple(X.shape))

ck = torch.load(MAIN / 'runs/iksel_final/sel_iksel_run0.pt',
                map_location=dev, weights_only=False)
nets = []
for sd_ in ck['members']:
    n = SetSel().to(dev)
    n.load_state_dict(sd_)
    n.eval()
    nets.append(n)
mu = torch.as_tensor(ck['mu'], device=dev, dtype=torch.float32)
sd = torch.as_tensor(ck['sd'], device=dev, dtype=torch.float32)
picks = _picks(nets, mu, sd, X.to(dev), V).cpu().numpy()

from scipy.stats import spearmanr
res = {k: [] for k in ('rnd', 'pick', 'oracle')}
sp = []
Xz = ((X.to(dev) - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
with torch.no_grad():
    S = torch.stack([n(Xz, V)[0] for n in nets]).mean(0).cpu().numpy()
for i in range(len(EV)):
    m = OWNER_flat == i
    k = int(m.sum())
    if k < 3:
        continue
    p = PROG_flat[m]
    res['rnd'].append(p.mean())
    res['pick'].append(p[min(picks[i], k - 1)])
    res['oracle'].append(p.max())
    if np.std(p) > 1e-6:
        sp.append(spearmanr(S[i, :k], p).statistic)
res = {k: np.array(v) for k, v in res.items()}
gap = res['oracle'].mean() - res['rnd'].mean()
cap = (res['pick'].mean() - res['rnd'].mean()) / gap
print(f"IJRR SetSel (hybrid-labeled) on today's candidates "
      f"({len(res['rnd'])} tasks):")
print(f"  picked {res['pick'].mean():.4f} m   random {res['rnd'].mean():.4f}"
      f"   oracle {res['oracle'].mean():.4f}")
print(f"  capture {cap:+.1%}   within-task Spearman {np.mean(sp):.3f}")
np.savez(OUT / 'ijrr_selector_transfer.npz', picks=picks, score=S,
         capture=cap)
print('wrote', OUT / 'ijrr_selector_transfer.npz')
