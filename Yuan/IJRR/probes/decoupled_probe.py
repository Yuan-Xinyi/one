"""Direction-magnitude decoupled value controller (user formulation):

    p_k   = P_N g_k            (projected physical gradients, magnitudes kept)
    G     = eps*I + sum p_k p_k^T          (null-space capability tensor)
    d     = G P_N grad_q V / ||.||         (continuous direction)
    alpha*= argmax_alpha V(F(q, alpha d))  over a log grid, feasibility-gated
    xi*   = alpha* d   (executed via a = B^T xi / a_max, exact since xi in ker J)

No a_max box, no vertices, no retraining: geometry gives the direction,
value gives the objective, the exact model gives the amplitude.
"""
import sys, dataclasses, time
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent
from Yuan.IJRR.eval.eval_curve import _agent
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
dev = torch.device('cuda')
hl.SUB = 2
y = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
model.terms = [0, 1]
A_MAX = env.cfg.a_max          # only used to re-encode xi for env.step
LAM = env.cfg.manip_damping

agV = _agent(REPO / hl.ROBOTS['fr3'][1], env.obs_dim, dev,
             act_dim=env.act_dim)
agC = ContAgent(env.obs_dim, env.act_dim).to(dev)
agC.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev))
agC.eval()

ALPHAS = torch.tensor(
    [0.0] + [s * a for a in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8,
                             1.6, 3.2) for s in (1.0, -1.0)],
    dtype=torch.float32, device=dev)                 # 19 candidates
NA = ALPHAS.shape[0]
EPS_G = 1e-4


def make_decoupled(agent, chunk=32768, use_G=True):
    @torch.no_grad()
    def _nograd_guard():
        pass

    def fn(e, done):
        q0 = e.q.detach()
        dline, ntgt = e.line_dir, e.n_target
        Bn = e.n_envs
        dt_k = e.kin.dtype
        # ---- gradients of the three physical objectives + V ----
        with torch.enable_grad():
            qg = q0.clone().requires_grad_(True)
            p, R, J, _ = e.kin.tcp_fk_jac(qg)
            J_p = J[:, :3, :]
            eye3 = torch.eye(3, device=dev, dtype=dt_k).expand(Bn, 3, 3)
            JJt = J_p @ J_p.transpose(-1, -2) + (LAM ** 2) * eye3
            uc = dline.unsqueeze(-1)
            iq = (uc.transpose(-1, -2) @ torch.linalg.solve(JJt, uc)
                  ).squeeze(-1).squeeze(-1).clamp_min(1e-12)
            wu = iq.pow(-0.5)
            cosv = (R[:, :, 2] * ntgt).sum(-1)
            g1 = torch.autograd.grad(wu.sum(), qg, retain_graph=True)[0]
            g2 = torch.autograd.grad(cosv.sum(), qg, retain_graph=True)[0]
            a0 = torch.zeros(Bn, e.act_dim, device=dev)
            obs = hl._obs_of(e, qg, dline, ntgt, a0)
            V = agent.critic(obs).sum()
            gV = torch.autograd.grad(V, qg)[0]
        qn_ = (q0 - e.kin.q_mid) / e.q_half
        g3 = -(2.0 / q0.shape[-1]) * qn_ / e.q_half
        with torch.no_grad():
            # ---- null projector (fp64 SVD as in the basis builder) ----
            _, _, Vh = torch.linalg.svd(J_p.detach().double(),
                                        full_matrices=True)
            m = q0.shape[-1] - 3
            N = Vh.transpose(-1, -2)[..., -m:]
            P = (N @ N.transpose(-1, -2))
            def proj(g):
                return (P @ g.double().unsqueeze(-1)).squeeze(-1)
            p1, p2, p3 = proj(g1), proj(g2), proj(g3)
            pV = proj(gV)
            # ---- capability tensor and direction ----
            if use_G:
                G = EPS_G * torch.eye(q0.shape[-1], device=dev,
                                      dtype=torch.float64).expand(
                                          Bn, -1, -1).clone()
                for pk in (p1, p2, p3):
                    G = G + pk.unsqueeze(-1) @ pk.unsqueeze(-2)
                dvec = (G @ pV.unsqueeze(-1)).squeeze(-1)
            else:
                dvec = pV
            dnorm = dvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dunit = (dvec / dnorm).to(dt_k)                     # (B, 7)
            # ---- 1D value search along d with feasibility gate ----
            qe = q0.repeat_interleave(NA, 0)
            de = dline.repeat_interleave(NA, 0)
            ne = ntgt.repeat_interleave(NA, 0)
            pe = e.p_start.repeat_interleave(NA, 0)
            xi = (ALPHAS.view(1, NA, 1).to(dt_k)
                  * dunit.unsqueeze(1)).reshape(Bn * NA, -1)    # (B*NA, 7)
            # exact one-step model with raw null velocity xi (mirrors
            # StraightModel.step but with xi instead of B a)
            qq = qe
            substeps = hl.SUB
            ddt = model.cfg.dt / substeps
            for _ in range(substeps):
                _, _, Jj, _ = model.kin.tcp_fk_jac(qq)
                Jp2 = Jj[:, :3, :]
                Jpl, _ = hl.damped_pinv(Jp2, model.cfg.lambda_0,
                                        model.cfg.sigma_thr)
                xdot = (model.cfg.v * de).unsqueeze(-1)
                qdot = (Jpl @ xdot).squeeze(-1) + xi
                qq = qq + qdot * ddt
            mg = torch.cat([model.margins(qq[i:i + chunk], pe[i:i + chunk],
                                          de[i:i + chunk], ne[i:i + chunk])
                            for i in range(0, Bn * NA, chunk)])
            alive = (mg.amin(-1) > 0).reshape(Bn, NA)
            av = torch.zeros(Bn * NA, e.act_dim, device=dev)
            v = torch.cat([agent.critic(
                hl._obs_of(e, qq[i:i + chunk], de[i:i + chunk],
                           ne[i:i + chunk], av[i:i + chunk])).squeeze(-1)
                for i in range(0, Bn * NA, chunk)]).reshape(Bn, NA)
            v = torch.where(alive, v, torch.full_like(v, -1e9))
            best = v.argmax(-1)
            alpha = ALPHAS[best].to(dt_k)                       # (B,)
            xi_star = alpha.unsqueeze(-1) * dunit               # (B, 7)
            # encode for env.step: a = B^T xi / a_max (exact: xi in span B)
            from Yuan.IJRR.env.env import build_task_aligned_basis
            Bb, _ = build_task_aligned_basis(
                e.kin, q0, dline, ntgt, e.kin.q_mid, e.q_half, LAM)
            a_exec = torch.einsum('bij,bi->bj', Bb, xi_star) / A_MAX
            return a_exec.float()
    return fn


tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
dt2 = env.kin.dtype


def run(afn, tag):
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(tz['q0_seed'][:B], dtype=dt2, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][:B], dtype=dt2,
                                  device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][:B], dtype=dt2,
                                  device=dev)})
    env.reset()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    t0 = time.time()
    for _ in range(env.cfg.max_steps // 2):
        a = afn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    r = env.arc_progress.float().cpu().numpy().copy()
    print(f'{tag:34s} mean {r.mean():.4f}  t27 {r[27]:.3f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return r


def make_twostage(agent, chunk=32768):
    import itertools
    verts = torch.tensor(list(itertools.product([-1., 1.],
                                                repeat=env.act_dim)),
                         dtype=torch.float32, device=dev)
    K = verts.shape[0]
    from Yuan.IJRR.env.env import build_task_aligned_basis

    @torch.no_grad()
    def fn(e, done):
        Bn = e.n_envs
        dt_k = e.kin.dtype
        q0 = e.q
        dline, ntgt = e.line_dir, e.n_target
        # stage 1: standard vertex comparison picks the direction
        qe = q0.repeat_interleave(K, 0)
        ae = verts.unsqueeze(0).expand(Bn, -1, -1).reshape(Bn * K, -1)
        de = dline.repeat_interleave(K, 0)
        ne = ntgt.repeat_interleave(K, 0)
        pe = e.p_start.repeat_interleave(K, 0)
        qn = torch.cat([model.step(qe[i:i+chunk], de[i:i+chunk],
                                   ne[i:i+chunk], ae[i:i+chunk])
                        for i in range(0, Bn*K, chunk)])
        mg = torch.cat([model.margins(qn[i:i+chunk], pe[i:i+chunk],
                                      de[i:i+chunk], ne[i:i+chunk])
                        for i in range(0, Bn*K, chunk)])
        alive = (mg.amin(-1) > 0).reshape(Bn, K)
        av = torch.zeros(Bn*K, e.act_dim, device=dev)
        v = torch.cat([agent.critic(hl._obs_of(e, qn[i:i+chunk],
                       de[i:i+chunk], ne[i:i+chunk],
                       av[i:i+chunk])).squeeze(-1)
                       for i in range(0, Bn*K, chunk)]).reshape(Bn, K)
        v = torch.where(alive, v, torch.full_like(v, -1e9))
        abest = verts[v.argmax(-1)]                      # (B, m)
        Bb, _ = build_task_aligned_basis(
            e.kin, q0, dline, ntgt, e.kin.q_mid, e.q_half, LAM)
        xi_dir = torch.einsum('bij,bj->bi', Bb, abest.to(dt_k))
        xi_dir = xi_dir / xi_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # stage 2: 1D alpha search along that direction
        qe2 = q0.repeat_interleave(NA, 0)
        de2 = dline.repeat_interleave(NA, 0)
        ne2 = ntgt.repeat_interleave(NA, 0)
        pe2 = e.p_start.repeat_interleave(NA, 0)
        xi = (ALPHAS.view(1, NA, 1).to(dt_k)
              * xi_dir.unsqueeze(1)).reshape(Bn*NA, -1)
        qq = qe2
        ddt = model.cfg.dt / hl.SUB
        for _ in range(hl.SUB):
            _, _, Jj, _ = model.kin.tcp_fk_jac(qq)
            Jpl, _ = hl.damped_pinv(Jj[:, :3, :], model.cfg.lambda_0,
                                    model.cfg.sigma_thr)
            xdot = (model.cfg.v * de2).unsqueeze(-1)
            qq = qq + ((Jpl @ xdot).squeeze(-1) + xi) * ddt
        mg2 = torch.cat([model.margins(qq[i:i+chunk], pe2[i:i+chunk],
                                       de2[i:i+chunk], ne2[i:i+chunk])
                         for i in range(0, Bn*NA, chunk)])
        alive2 = (mg2.amin(-1) > 0).reshape(Bn, NA)
        av2 = torch.zeros(Bn*NA, e.act_dim, device=dev)
        v2 = torch.cat([agent.critic(hl._obs_of(e, qq[i:i+chunk],
                        de2[i:i+chunk], ne2[i:i+chunk],
                        av2[i:i+chunk])).squeeze(-1)
                        for i in range(0, Bn*NA, chunk)]).reshape(Bn, NA)
        v2 = torch.where(alive2, v2, torch.full_like(v2, -1e9))
        alpha = ALPHAS[v2.argmax(-1)].to(dt_k)
        xi_star = alpha.unsqueeze(-1) * xi_dir
        a_exec = torch.einsum('bij,bi->bj', Bb, xi_star) / A_MAX
        return a_exec.float()
    return fn


def run10k(afn, tag):
    N=10000
    out=np.zeros(N,np.float32)
    for lo in range(0,N,B):
        hi=min(lo+B,N); pad=B-(hi-lo)
        ids=np.arange(lo,hi)
        ip=np.concatenate([ids,np.full(pad,ids[0])]) if pad else ids
        env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][ip],dtype=dt2,device=dev),
          'line_dir':torch.tensor(tz['cs_line_dir'][ip],dtype=dt2,device=dev),
          'n_target':torch.tensor(tz['cs_n_target'][ip],dtype=dt2,device=dev)})
        env.reset()
        done=torch.zeros(B,dtype=torch.bool,device=dev)
        for _ in range(env.cfg.max_steps//2):
            a=afn(env,done)
            for _ in range(2): env.step(a,auto_reset=False)
            done=env.done_persistent.clone()
            if bool(done.all()): break
        out[lo:hi]=env.arc_progress.float().cpu().numpy()[:hi-lo]
    print(f'{tag:38s} 10k mean {out.mean():.4f}  t27 {out[27]:.3f}', flush=True)
    return out

res = {}
with torch.no_grad():
    res['twostage_V_10k'] = run10k(make_twostage(agV),
                                   'two-stage vertex-dir + alpha (10k)')
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
         'decoupled_probe.npz', **res)
print('DECOUPLED PROBE DONE')
