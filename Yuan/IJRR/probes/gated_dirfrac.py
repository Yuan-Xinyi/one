"""Diagnostic gate on the dirfrac v2 actor: one-step-checked candidates that
preserve actor intent (rho backoff -> pure tracking -> reversed direction).
Not a deployment proposal; measures how much of the residual gap to the
value law is pure boundary accidents."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent

dev = torch.device('cuda')
hl.SUB = 2
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N, B = 10000, 2048
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_v2.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
ag = ContAgent(env.obs_dim, env.act_dim_policy).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_dirfrac_v2_30M/agent.pt',
    map_location=dev))
ag.eval()


def dirfrac_step(q, d, u, rho, substeps=2):
    """Exact copy of the env's dir_frac==2 update, straight ray
    (k_lateral = 0), one full decision period = `substeps` integrations
    at env.cfg.dt (already the halved dt)."""
    for _ in range(substeps):
        _, _, J, _ = env.kin.tcp_fk_jac(q)
        J_p = J[:, :3, :]
        J_plus, _ = damped_pinv(J_p, env.cfg.lambda_0, env.cfg.sigma_thr)
        x_dot = (env.cfg.v * d).unsqueeze(-1)
        qdot_task = (J_plus @ x_dot).squeeze(-1)
        _, _, Vh = torch.linalg.svd(J_p.double(), full_matrices=True)
        Nn = Vh.transpose(-1, -2)[..., 3:]
        xi = (Nn @ (Nn.transpose(-1, -2)
                    @ u.double().unsqueeze(-1))).squeeze(-1)
        xi = (xi / xi.norm(dim=-1, keepdim=True).clamp_min(1e-6)) \
            .to(env.kin.dtype)
        head = (env.qd_limit - qdot_task) / xi.clamp_min(1e-9)
        room = (env.qd_limit + qdot_task) / (-xi).clamp_min(1e-9)
        bound = torch.where(xi >= 0, head, room)
        alpha = bound.amin(dim=-1).clamp_min(0.0)
        q = q + (qdot_task + (rho * alpha).unsqueeze(-1) * xi) * env.cfg.dt
    return q


# candidate ladder: (direction sign, rho scale), in intent-preserving order
CAND = [(1, 1.0), (1, 0.5), (1, 0.25), (1, 0.0),
        (-1, 0.25), (-1, 0.5), (-1, 1.0)]
K = len(CAND)


@torch.no_grad()
def gated_actor(e, done):
    a = ag.actor_mean(e.current_obs()).clamp(-1., 1.)
    u = a[:, :7].to(e.kin.dtype)
    rho = 0.5 * (a[:, 7].to(e.kin.dtype) + 1.0)
    Bn = e.n_envs
    qe = e.q.repeat_interleave(K, 0)
    de = e.line_dir.repeat_interleave(K, 0)
    pe = e.p_start.repeat_interleave(K, 0)
    ne = e.n_target.repeat_interleave(K, 0)
    sg = torch.tensor([c[0] for c in CAND], dtype=e.kin.dtype, device=dev)
    sc = torch.tensor([c[1] for c in CAND], dtype=e.kin.dtype, device=dev)
    ue = u.repeat_interleave(K, 0) * sg.repeat(Bn).unsqueeze(-1)
    re = rho.repeat_interleave(K, 0) * sc.repeat(Bn)
    qn = dirfrac_step(qe, de, ue, re)
    alive = (model.margins(qn, pe, de, ne).amin(-1) > 0).reshape(Bn, K)
    pick = torch.where(alive.any(-1),
                       alive.float().argmax(-1),
                       torch.zeros(Bn, dtype=torch.long, device=dev))
    out = a.clone()
    out[:, :7] = u * sg[pick].unsqueeze(-1)
    out[:, 7] = 2.0 * (rho * sc[pick]) - 1.0
    return out


@torch.no_grad()
def run(afn, tag):
    out = np.zeros(N, np.float32)
    dt = env.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        ids = np.arange(lo, hi)
        ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        env.line_dist = ScriptedLineDistribution(
            {'q0': torch.tensor(tz['q0_seed'][ip], dtype=dt, device=dev),
             'line_dir': torch.tensor(tz['cs_line_dir'][ip], dtype=dt,
                                      device=dev),
             'n_target': torch.tensor(tz['cs_n_target'][ip], dtype=dt,
                                      device=dev)})
        env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(env.cfg.max_steps // 2):
            a = afn(env, done)
            for _ in range(2):
                env.step(a, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'{tag} {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
    return out


# ---- replay exactness check: model vs env under the raw actor ----------
with torch.no_grad():
    ids = np.arange(2048)
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(tz['q0_seed'][ids], dtype=env.kin.dtype,
                            device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][ids],
                                  dtype=env.kin.dtype, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][ids],
                                  dtype=env.kin.dtype, device=dev)})
    env.reset()
    worst = 0.0
    for t in range(40):
        a = ag.actor_mean(env.current_obs()).clamp(-1., 1.)
        u = a[:, :7].to(env.kin.dtype)
        rho = 0.5 * (a[:, 7].to(env.kin.dtype) + 1.0)
        qp = dirfrac_step(env.q.clone(), env.line_dir, u, rho)
        alive_before = ~env.done_persistent
        for _ in range(2):
            env.step(a, auto_reset=False)
        still = alive_before & ~env.done_persistent
        if still.any():
            worst = max(worst, float((env.q - qp)[still].abs().max()))
    print(f'replay check: max |dq| over 40 steps = {worst:.2e}', flush=True)

r = run(gated_actor, 'gated dirfrac v2')
np.savez(FU + 'gated_dirfrac_10k.npz', prog=r)

A = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
b = np.load(A + 'bound_pool_fr3.npz')
w = np.load(A + 'witness_pool_fr3.npz')
base = np.load(FU + 'pool_fr3_straight.npz')
v2 = np.load(FU + 'dirfrac_v2_10k.npz')['prog']
ref = np.maximum(b['L_hi'], w['prog'])
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]:
    ref = np.maximum(ref, base[f'{a2}_progress'])
ref = np.maximum(np.maximum(ref, v2), r)
def stat(v, tag):
    rt = v / np.maximum(ref, 1e-9)
    print(f'{tag}: {v.mean():.4f}  {rt.mean()*100:.1f} / '
          f'{np.percentile(rt, 10)*100:.1f}   t27 {v[27]:.3f}', flush=True)
stat(v2, 'dirfrac v2 raw actor  ')
stat(r, 'dirfrac v2 gated actor')
stat(base['vlook_progress'], 'vertex vlook          ')
d = r - v2
print(f'gated-raw: improved {(d>0.02).mean()*100:.1f}%  '
      f'hurt {(d<-0.02).mean()*100:.1f}%')
