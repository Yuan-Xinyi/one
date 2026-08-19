"""Capture ladder for one-seed-one-rollout start selection, no peeking:
  A. SetSel recipe, 45-D features, labels = value-lookahead rollouts
  B. A + critic value at the start (1 feature)
  C. B + one-step successor statistics from the controller's own first
     glance (4 features; no unrolling beyond the single control step)
Train on 3000 tasks (candidates seed=0), evaluate capture on the 300
held-out tasks of seed_calib.npz (candidates seed=1, labels cached)."""
import sys, time, copy
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn as nn, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage1_seed.candidate_batch import SeedCandidateBatch
from Yuan.IJRR.stage1_seed.features import initial_observation_features
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
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * env1.act_dim,
                     indexing='ij'), -1).reshape(-1, env1.act_dim),
                     dtype=torch.float32, device=dev)


def gen_candidates(ids, seed):
    rng = torch.Generator(device='cpu').manual_seed(seed)
    Q0 = pool.q_pool[sel[ids]].to(dev)
    DIR = pool.line_dir_pool[sel[ids]].to(dev)
    NTG = pool.n_target_pool[sel[ids]].to(dev)
    P0 = kin.tcp_fk_jac(Q0)[0]
    QK, NV = [], []
    for i in range(len(ids)):
        got, tries = [Q0[i]], 0
        while len(got) < 8 and tries < 150:
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
        NV.append(len(got))
        QK.append(torch.stack(got + [got[0]] * (8 - len(got))))
    return (torch.stack(QK), np.array(NV), Q0, DIR, NTG, P0)


def labels_for(QK, NV, DIR, NTG, chunk=2048):
    flat_q, flat_t = [], []
    for i in range(QK.shape[0]):
        for k in range(NV[i]):
            flat_q.append(QK[i, k]); flat_t.append(i)
    FQ = torch.stack(flat_q); FT = np.array(flat_t)
    prog = np.zeros(FQ.shape[0], dtype=np.float32)
    for base in range(0, FQ.shape[0], chunk):
        cc = FQ[base:base + chunk]
        oo = FT[base:base + chunk]
        ii = torch.as_tensor(oo, device=dev)
        eB = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': cc.shape[0]}),
                            None, dev)
        spec = {'q0': cc.clone(), 'line_dir': DIR[ii].clone(),
                'n_target': NTG[ii].clone()}
        eB.line_dist = ScriptedLineDistribution({k_: v.clone()
                                                 for k_, v in spec.items()})
        mdl = hl.StraightModel(eB)
        vl = hl.make_vlook(mdl, eB, ag)
        eB.line_dist = ScriptedLineDistribution({k_: v.clone()
                                                 for k_, v in spec.items()})
        pr, _ = hl.rollout_env(eB, vl)
        prog[base:base + chunk] = pr
        del eB
    Y = np.full((QK.shape[0], 8), -1.0, dtype=np.float32)
    ptr = 0
    for i in range(QK.shape[0]):
        for k in range(NV[i]):
            Y[i, k] = prog[ptr]; ptr += 1
    return Y


@torch.no_grad()
def features_for(QK, NV, DIR, NTG, P0):
    B = QK.shape[0]
    V = (torch.arange(8, device=dev)[None, :]
         < torch.as_tensor(NV, device=dev)[:, None])
    batch = SeedCandidateBatch(q0=QK, p0=P0, line_dir=DIR, n_target=NTG,
                               valid=V)
    X45 = initial_observation_features(
        kin, batch, include_ray_error=True, include_log_manip=True,
        include_directional_dynamics=True).float().to(dev)
    # critic value at the start (fresh obs, a_prev = 0)
    flat_q = QK.reshape(B * 8, 7)
    ii = torch.arange(B, device=dev).repeat_interleave(8)
    eB = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B * 8}), None, dev)
    eB.line_dist = ScriptedLineDistribution(
        {'q0': flat_q.clone(), 'line_dir': DIR[ii].clone(),
         'n_target': NTG[ii].clone()})
    eB.reset()
    obs0 = eB.current_obs()
    v0 = ag.critic(obs0.float()).squeeze(-1)
    # one-step successor statistics (the controller's own first glance)
    qe = flat_q.repeat_interleave(16, 0)
    ae = verts.unsqueeze(0).expand(B * 8, -1, -1).reshape(-1, 4)
    de = DIR[ii].repeat_interleave(16, 0)
    ne = NTG[ii].repeat_interleave(16, 0)
    pe = P0[ii].repeat_interleave(16, 0)
    CH = 32768
    qn = torch.cat([model.step(qe[i:i + CH], de[i:i + CH], ne[i:i + CH],
                               ae[i:i + CH])
                    for i in range(0, qe.shape[0], CH)])
    mg = torch.cat([model.margins(qn[i:i + CH], pe[i:i + CH],
                                  de[i:i + CH], ne[i:i + CH])
                    for i in range(0, qe.shape[0], CH)])
    alive = (mg.amin(-1) > 0).reshape(B * 8, 16)
    vn = torch.cat([ag.critic(hl._obs_of(eB, qn[i:i + CH], de[i:i + CH],
                                         ne[i:i + CH],
                                         ae[i:i + CH])).squeeze(-1)
                    for i in range(0, qe.shape[0], CH)]).reshape(B * 8, 16)
    vn_m = torch.where(alive, vn, torch.full_like(vn, -1e9))
    top2 = vn_m.topk(2, dim=-1).values
    stats = torch.stack([
        top2[:, 0], (top2[:, 0] - top2[:, 1]).clamp(-50, 50),
        alive.float().mean(-1),
        torch.where(alive, vn, torch.zeros_like(vn)).sum(-1)
        / alive.float().sum(-1).clamp_min(1)], -1)
    del eB
    XB = torch.cat([X45, v0.reshape(B, 8, 1)], -1)
    XC = torch.cat([XB, stats.reshape(B, 8, 4)], -1)
    return {'A': X45, 'B': XB, 'C': XC}, V


class SetSelD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                   nn.Linear(256, 1))

    def forward(self, X, V):
        e = self.enc(X)
        vf = V.unsqueeze(-1).float()
        ctx = (e * vf).sum(1) / vf.sum(1).clamp_min(1)
        h = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
        return self.score(h).squeeze(-1)


def train_ens(X, V, Y, n_members=5, epochs=3000, temp=0.05):
    Yt = torch.tensor(Y, device=dev)
    mu = X[V].mean(0); sd = X[V].std(0) + 1e-6
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    tgt = torch.softmax(
        torch.where(V, Yt / temp, torch.full_like(Yt, -1e9)), -1)
    nets = []
    for s in range(n_members):
        torch.manual_seed(s)
        net = SetSelD(X.shape[-1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for ep in range(epochs):
            sc = net(Xz, V).masked_fill(~V, -1e9)
            loss = -(tgt * torch.log_softmax(sc, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        nets.append(net.eval())
    return nets, mu, sd


@torch.no_grad()
def capture(nets, mu, sd, X, V, Y):
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    sc = torch.stack([n(Xz, V).masked_fill(~V, -1e9)
                      for n in nets]).mean(0)
    pick = sc.argmax(1).cpu().numpy()
    caps_n, caps_d = [], []
    picked, rnd, orc = [], [], []
    for i in range(X.shape[0]):
        k = int(V[i].sum())
        if k < 3:
            continue
        yv = Y[i, :k]
        picked.append(yv[min(pick[i], k - 1)])
        rnd.append(yv.mean()); orc.append(yv.max())
    picked, rnd, orc = map(np.array, (picked, rnd, orc))
    return (picked.mean() - rnd.mean()) / (orc.mean() - rnd.mean()), \
        picked.mean(), rnd.mean(), orc.mean()


t0 = time.time()
TR = list(range(0, 3000))
QKt, NVt, _, DIRt, NTGt, P0t = gen_candidates(TR, seed=0)
print(f"train candidates ({time.time()-t0:.0f}s)", flush=True)
Yt = labels_for(QKt, NVt, DIRt, NTGt)
print(f"train labels ({time.time()-t0:.0f}s)", flush=True)
Xt, Vt = features_for(QKt, NVt, DIRt, NTGt, P0t)
print(f"train features ({time.time()-t0:.0f}s)", flush=True)

EV = list(range(12000, 12300))
QKe, NVe, _, DIRe, NTGe, P0e = gen_candidates(EV, seed=1)
sc_ = np.load(OUT / 'seed_calib.npz')
Ye = np.full((len(EV), 8), -1.0, dtype=np.float32)
ptr = 0
for i in range(len(EV)):
    for k in range(NVe[i]):
        Ye[i, k] = sc_['prog'][ptr]; ptr += 1
assert ptr == sc_['prog'].shape[0]
Xe, Ve = features_for(QKe, NVe, DIRe, NTGe, P0e)
print(f"eval features ({time.time()-t0:.0f}s)", flush=True)

results = {}
for tag, d in (('A 45-D, vlook labels', 'A'),
               ('B + critic value', 'B'),
               ('C + one-step stats', 'C')):
    nets, mu, sd = train_ens(Xt[d], Vt, Yt)
    cap, pk, rd, orc = capture(nets, mu, sd, Xe[d], Ve, Ye)
    results[tag] = cap
    print(f"[ladder] {tag:<24s} capture {cap:+.1%}  "
          f"(picked {pk:.4f} / random {rd:.4f} / oracle {orc:.4f})",
          flush=True)
    torch.save({'members': [n.state_dict() for n in nets],
                'mu': mu.cpu(), 'sd': sd.cpu(), 'variant': d},
               OUT / f'setsel_vlook_{d}.pt')
np.savez(OUT / 'seed_ladder.npz', **{k: v for k, v in results.items()})
print('done', results)
