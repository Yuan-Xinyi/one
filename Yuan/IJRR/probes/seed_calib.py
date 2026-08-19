"""Calibrate the critic on the START distribution and re-test start
selection. Labels are free: the realized progress of the value-lookahead
controller from each candidate start. Train on 3000 pool tasks, evaluate
capture on 300 held-out tasks; baselines = raw critic and the handcrafted
start heuristic (largest softmin margin at t=0)."""
import sys, time, copy
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn as nn, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

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
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env1.obs_dim, dev,
            act_dim=env1.act_dim)


def gen_candidates(ids, K=8, seed=0):
    rng = torch.Generator(device='cpu').manual_seed(seed)
    Q0 = pool.q_pool[sel[ids]].to(dev)
    DIR = pool.line_dir_pool[sel[ids]].to(dev)
    NTG = pool.n_target_pool[sel[ids]].to(dev)
    P0 = kin.tcp_fk_jac(Q0)[0]
    cands, owner, msoft = [], [], []
    for i in range(len(ids)):
        got = [Q0[i]]
        tries = 0
        while len(got) < K and tries < 150:
            tries += 1
            scale = 0.2 + 0.6 * torch.rand(1, generator=rng).item()
            q = Q0[i] + scale * torch.randn(7, generator=rng).to(dev).to(
                Q0.dtype)
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
            m = model.margins(q[None], P0[i][None], DIR[i][None],
                              NTG[i][None])
            if not bool(m.amin() > 0):
                continue
            if min(float((q - g).abs().max()) for g in got) < 0.08:
                continue
            got.append(q)
        for q in got:
            m = model.softmin_margin(q[None], P0[i][None], DIR[i][None],
                                     NTG[i][None])
            msoft.append(float(m))
        cands.extend(got)
        owner.extend([i] * len(got))
    return (torch.stack(cands), np.array(owner), np.array(msoft),
            DIR, NTG)


def obs_and_roll(CAND, OWNER, DIR, NTG, chunk=2048):
    obs_l, prog_l = [], []
    for base in range(0, CAND.shape[0], chunk):
        cc = CAND[base:base + chunk]
        oo = OWNER[base:base + chunk]
        ii = torch.as_tensor(oo, device=dev)
        eB = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': cc.shape[0]}),
                            None, dev)
        spec = {'q0': cc.clone(), 'line_dir': DIR[ii].clone(),
                'n_target': NTG[ii].clone()}
        eB.line_dist = ScriptedLineDistribution({k: v.clone()
                                                 for k, v in spec.items()})
        eB.reset()
        obs_l.append(eB.current_obs().clone())
        mdl = hl.StraightModel(eB)
        vl = hl.make_vlook(mdl, eB, ag)
        eB.line_dist = ScriptedLineDistribution({k: v.clone()
                                                 for k, v in spec.items()})
        pr, _ = hl.rollout_env(eB, vl)
        prog_l.append(pr)
        del eB
    return torch.cat(obs_l), np.concatenate(prog_l)


t0 = time.time()
TR = list(range(0, 3000))
CAND, OWNER, MSOFT, DIR, NTG = gen_candidates(TR, seed=0)
print(f"train candidates {CAND.shape[0]} ({time.time()-t0:.0f}s)",
      flush=True)
OBS, PROG = obs_and_roll(CAND, OWNER, DIR, NTG)
print(f"train rollouts done ({time.time()-t0:.0f}s)  "
      f"label mean {PROG.mean():.3f} m", flush=True)

# ---- calibrate a copy of the critic on the start distribution ----
head = copy.deepcopy(ag.critic).train()
opt = torch.optim.Adam(head.parameters(), lr=1e-4)
LAB = torch.tensor(PROG, device=dev, dtype=torch.float32)
n = OBS.shape[0]
hold = torch.arange(n, device=dev) % 10 == 0
for ep in range(15000):
    idx = torch.randint(0, n, (2048,), device=dev)
    m = ~hold[idx]
    loss = nn.functional.smooth_l1_loss(
        head(OBS[idx][m].float()).squeeze(-1), LAB[idx][m])
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 3000 == 0:
        with torch.no_grad():
            pr = head(OBS[hold].float()).squeeze(-1)
            r2 = 1 - ((pr - LAB[hold]) ** 2).mean() / LAB[hold].var()
        print(f"  calib {ep:>6}  holdout R2 {float(r2):.3f}", flush=True)
head.eval()
torch.save(head.state_dict(), OUT / 'start_value_head.pt')

# ---- evaluate on held-out tasks ----
EV = list(range(12000, 12300))
CANDe, OWNERe, MSOFTe, DIRe, NTGe = gen_candidates(EV, seed=1)
OBSe, PROGe = obs_and_roll(CANDe, OWNERe, DIRe, NTGe)
print(f"eval rollouts done ({time.time()-t0:.0f}s)", flush=True)
with torch.no_grad():
    Vraw = ag.critic(OBSe.float()).squeeze(-1).cpu().numpy()
    Vcal = head(OBSe.float()).squeeze(-1).cpu().numpy()

from scipy.stats import spearmanr
res = {k: [] for k in ('orig', 'rnd', 'raw', 'cal', 'marg', 'oracle')}
sp_raw, sp_cal = [], []
for i in range(len(EV)):
    m = OWNERe == i
    if m.sum() < 3:
        continue
    p = PROGe[m]
    res['orig'].append(p[0])
    res['rnd'].append(p.mean())
    res['raw'].append(p[np.argmax(Vraw[m])])
    res['cal'].append(p[np.argmax(Vcal[m])])
    res['marg'].append(p[np.argmax(MSOFTe[m])])
    res['oracle'].append(p.max())
    if np.std(p) > 1e-6:
        sp_raw.append(spearmanr(Vraw[m], p).statistic)
        sp_cal.append(spearmanr(Vcal[m], p).statistic)
res = {k: np.array(v) for k, v in res.items()}
gap = res['oracle'].mean() - res['rnd'].mean()
print(f"\nheld-out tasks with >=3 candidates: {len(res['orig'])}")
for k, name in (('orig', 'original pool q0'), ('rnd', 'random candidate'),
                ('marg', 'largest softmin margin'),
                ('raw', 'raw critic'), ('cal', 'CALIBRATED critic'),
                ('oracle', 'oracle')):
    cap = (res[k].mean() - res['rnd'].mean()) / gap
    print(f"  {name:<24s} {res[k].mean():.4f} m   capture {cap:+.1%}")
print(f"Spearman raw {np.mean(sp_raw):.3f} -> calibrated "
      f"{np.mean(sp_cal):.3f}")
np.savez(OUT / 'seed_calib.npz', prog=PROGe, owner=OWNERe, vraw=Vraw,
         vcal=Vcal, msoft=MSOFTe, **{k: v for k, v in res.items()})
print('wrote', OUT / 'seed_calib.npz')
