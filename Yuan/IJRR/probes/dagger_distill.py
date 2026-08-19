"""Item 4: Value -> DirFrac distillation.

Teacher = the exact-model value law (16-vertex enumeration + feasibility
gate + vertex critic), whose chosen command's executed null-space velocity
is converted into dirfrac coordinates:
    xi* = qdot_null* / |qdot_null*|,   rho* = min(1, |qdot_null*| /
    alpha_feas(q, xi*)).
Student = dirfrac v2 actor, warm-started from the PPO checkpoint and
BC-trained on states visited by BOTH the student and the teacher (DAgger
round 0 mixture). Deployment stays one forward + one projection.

Dynamics identity teacher/student verified: configs differ only in the
action interface. Labels are generated at TRAIN granularity (50 ms, one
integration per decision) to match the rollout env used here; the final
eval runs the standard 25 ms substep protocol via dirfrac_eval_any.py.
"""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml, itertools
import torch.nn.functional as F
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, damped_pinv,
                               build_task_aligned_basis)
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
from Yuan.IJRR.eval.eval_curve import _agent

dev = torch.device('cuda')
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/rl_dirfrac_bcdistill')
OUT.mkdir(exist_ok=True)

# ---- envs: dirfrac rollout env at TRAIN granularity ---------------------
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_v2.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
B = 1024
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
lc = y['line_distribution']
env.line_dist = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=lc['n_pool'],
    n_target_noise_deg=lc['n_target_noise_deg'], seed=lc['train_seed'],
    env_cfg=env.cfg,
    feasibility_threshold_m=(float(lc['feasibility_threshold_m'])
                             if lc.get('feasibility_filter') else None),
    swing_max_deg=lc.get('swing_max_deg', 0.0))

# teacher assets: same dynamics, box interface (a_max = 0.5)
model = hl.StraightModel(env)
A_MAX = 0.5
vy = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
assert vy['env']['v'] == y['env']['v'] and vy['env']['dt'] == y['env']['dt']
vobs = 2 * env.n_joints + 13 + env.act_dim
teacher = _agent(REPO / hl.ROBOTS['fr3'][1], vobs, dev,
                 act_dim=env.act_dim)
verts = torch.tensor(list(itertools.product([-1., 1.], repeat=env.act_dim)),
                     dtype=torch.float32, device=dev)
K = verts.shape[0]

student = Agent(env.obs_dim, env.act_dim_policy).to(dev)
student.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_dirfrac_v2_30M/agent.pt',
    map_location=dev))


@torch.no_grad()
def teacher_label(e):
    """(xi*, rho*, has_alive) at the env's current state."""
    q, d, n = e.q, e.line_dir, e.n_target
    Bn = q.shape[0]
    qe = q.repeat_interleave(K, 0)
    ae = verts.unsqueeze(0).expand(Bn, -1, -1).reshape(Bn * K, -1)
    de = d.repeat_interleave(K, 0)
    ne = n.repeat_interleave(K, 0)
    pe = e.p_start.repeat_interleave(K, 0)
    qn = model.step(qe, de, ne, ae, substeps=1)      # train granularity
    mg = model.margins(qn, pe, de, ne)
    alive = (mg.amin(-1) > 0).reshape(Bn, K)
    vv = teacher.critic(hl._obs_of(e, qn, de, ne, ae)).squeeze(-1) \
        .reshape(Bn, K)
    vv = torch.where(alive, vv, torch.full_like(vv, -1e9))
    pick = vv.argmax(-1)
    a_star = verts[pick].to(e.kin.dtype)             # (B, 4)
    Bb, _ = build_task_aligned_basis(
        e.kin, q, d, n, e.kin.q_mid, e.q_half, e.cfg.manip_damping,
        raw_scale=e.cfg.basis_raw_scale)
    qdn = (Bb @ (A_MAX * a_star).unsqueeze(-1)).squeeze(-1)   # (B, 7)
    nrm = qdn.norm(dim=-1)
    xi = qdn / nrm.clamp_min(1e-9).unsqueeze(-1)
    # alpha_feas along xi, velocity form (matches the v2 env exactly)
    _, _, J, _ = e.kin.tcp_fk_jac(q)
    J_plus, _ = damped_pinv(J[:, :3, :], e.cfg.lambda_0, e.cfg.sigma_thr)
    qdot_task = (J_plus @ (e.cfg.v * d).unsqueeze(-1)).squeeze(-1)
    head = (e.qd_limit - qdot_task) / xi.clamp_min(1e-9)
    room = (e.qd_limit + qdot_task) / (-xi).clamp_min(1e-9)
    alpha = torch.where(xi >= 0, head, room).amin(-1).clamp_min(1e-9)
    rho = (nrm / alpha).clamp(0.0, 1.0)
    return xi, rho, alive.any(-1)


@torch.no_grad()
def collect(driver, n_steps, tag):
    """Roll `driver` (student|teacher) with auto_reset, label every state."""
    OBS, XI, RHO = [], [], []
    env.reset()
    for t in range(n_steps):
        obs = env.current_obs()
        xi, rho, ok = teacher_label(env)
        OBS.append(obs[ok].float().cpu())
        XI.append(xi[ok].float().cpu())
        RHO.append(rho[ok].float().cpu())
        if driver == 'student':
            a = student.actor_mean(obs)
        else:
            a = torch.cat([xi.float(), (2 * rho - 1).float().unsqueeze(-1)],
                          -1)
        env.step(a.to(env.kin.dtype), auto_reset=True)
        if (t + 1) % 25 == 0:
            print(f'[collect:{tag}] {t+1}/{n_steps}  '
                  f'{sum(o.shape[0] for o in OBS)} states', flush=True)
    return torch.cat(OBS), torch.cat(XI), torch.cat(RHO)


def bc_train(X, XI, RHO, epochs=25, lr=1e-4, tag='bc'):
    n = X.shape[0]
    nva = min(50000, n // 10)
    perm = torch.randperm(n)
    X, XI, RHO = X[perm], XI[perm], RHO[perm]
    Xv, XIv, RHv = X[:nva].to(dev), XI[:nva].to(dev), RHO[:nva].to(dev)
    Xt, XIt, RHt = X[nva:], XI[nva:], RHO[nva:]
    params = (list(student._actor_trunk.parameters())
              + list(student._mean_head.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    ntr = Xt.shape[0]
    for ep in range(epochs):
        p2 = torch.randperm(ntr)
        tot = 0.0
        for i in range(0, ntr, 8192):
            b = p2[i:i + 8192]
            xb, xib, rb = Xt[b].to(dev), XIt[b].to(dev), RHt[b].to(dev)
            a = student.actor_mean(xb)
            cosl = 1 - F.cosine_similarity(a[:, :7], xib, dim=-1).mean()
            rhol = F.mse_loss(0.5 * (a[:, 7] + 1), rb)
            loss = cosl + 4.0 * rhol
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        with torch.no_grad():
            av = student.actor_mean(Xv)
            cv = F.cosine_similarity(av[:, :7], XIv, dim=-1).mean()
            rv = (0.5 * (av[:, 7] + 1) - RHv).abs().mean()
        print(f'[{tag}] ep {ep+1}/{epochs} loss {tot/ntr:.4f} '
              f'val_cos {cv:.4f} val_|drho| {rv:.4f}', flush=True)


# ---- round 0: student states + teacher states (coverage) ----------------
Xs, XIs, RHs = collect('student', 120, 'student')
Xt_, XIt_, RHt_ = collect('teacher', 120, 'teacher')
print(f'dataset: student {Xs.shape[0]}  teacher {Xt_.shape[0]}', flush=True)
X = torch.cat([Xs, Xt_]); XI = torch.cat([XIs, XIt_])
RHO = torch.cat([RHs, RHt_])
bc_train(X, XI, RHO, tag='bc0')
torch.save(student.state_dict(), OUT / 'agent.pt')

# ---- round 1: relabel under the BC student, fine-tune -------------------
Xs2, XIs2, RHs2 = collect('student', 120, 'student-r1')
X = torch.cat([X, Xs2]); XI = torch.cat([XI, XIs2])
RHO = torch.cat([RHO, RHs2])
bc_train(X, XI, RHO, epochs=15, lr=5e-5, tag='bc1')
torch.save(student.state_dict(), OUT / 'agent.pt')
print('[dagger] done, ckpt ->', OUT / 'agent.pt', flush=True)
