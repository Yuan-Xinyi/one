"""#11 assets: (A) per-task-set path sample tables (points + local cone
axes on a 1 cm arc grid) for the generalized bound march, and (B) random-FK
warm-start tables for xArm7 / Cobotta."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.env.path_geometry import arc_point, serpentine_point
from Yuan.IJRR.eval.horizon_ladder import ROBOTS

STEP, MAXL = 0.01, 1.8
NG = int(round(MAXL / STEP)) + 1
OUT = MAIN / 'runs/paper_fill/ratio_assets'
OUT.mkdir(exist_ok=True)
dev = torch.device('cuda')


def envfor(robot):
    y = yaml.safe_load(open(REPO / ROBOTS[robot][0]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 8}), None, dev)


def straight_samples(p0, d, n):
    s = torch.arange(NG, dtype=torch.float64) * STEP
    pts = p0[:, None, :] + s[None, :, None] * d[:, None, :]
    axes = n[:, None, :].expand(-1, NG, -1).clone()
    return pts, axes


# ---- (A1) pool task sets, straight, per robot ------------------------------
for robot in ('fr3', 'xarm7', 'cobotta'):
    f = OUT / f'tasks_pool_{robot}.npz'
    if f.exists():
        continue
    env = envfor(robot)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    ids = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:10000]
    q0 = pool.q_pool[ids].to(dev, env.kin.dtype)
    p0 = env.kin.tcp_fk_jac(q0)[0].double().cpu()
    d = pool.line_dir_pool[ids].double().cpu()
    n = pool.n_target_pool[ids].double().cpu()
    pts, axes = straight_samples(p0, d, n)
    np.savez_compressed(f, cs_p0=p0.numpy().astype(np.float32),
                        cs_line_dir=d.numpy().astype(np.float32),
                        cs_n_target=n.numpy().astype(np.float32),
                        q0_seed=q0.float().cpu().numpy(),
                        path_pts=pts.numpy().astype(np.float32),
                        path_axes=axes.numpy().astype(np.float32))
    print('wrote', f.name)

# ---- (A2) selector test sets (straight + 3 curved), FR3 --------------------
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
labels = torch.load(MAIN / 'runs/selector_ood/v2_k32/labels.pt',
                    weights_only=False)
for fam in ('straight', 'arc', 'serpentine', 'nonplanar'):
    key = f'test_{fam}'
    f = OUT / f'tasks_sel_{fam}.npz'
    if f.exists():
        continue
    sp = {k: v.double() for k, v in tasks[key].items()}
    N = sp['p0'].shape[0]
    p0, d, n = sp['p0'], sp['line_dir'], sp['n_target']
    d = d / d.norm(dim=-1, keepdim=True)
    n = n / n.norm(dim=-1, keepdim=True)
    s = torch.arange(NG, dtype=torch.float64) * STEP
    if fam == 'straight':
        pts, axes = straight_samples(p0, d, n)
    elif fam == 'arc':
        pts = torch.stack([arc_point(p0, d, n, sp['kappa'],
                                     torch.full((N,), float(si), dtype=torch.float64))
                           for si in s], dim=1)
        axes = n[:, None, :].expand(-1, NG, -1).clone()
    elif fam == 'serpentine':
        # invert axis-coordinate -> arc length per task on a dense grid
        xs = torch.arange(0, 2.6, 0.001, dtype=torch.float64)
        k = 2.0 * torch.pi / sp['wavelen'].clamp_min(1e-3)
        dy = sp['amp'][:, None] * k[:, None] * torch.cos(k[:, None] * xs[None, :])
        ds = torch.sqrt(1.0 + dy ** 2) * 0.001
        cum = torch.cumsum(ds, dim=1) - ds
        xi = torch.empty(N, NG, dtype=torch.float64)
        for i in range(N):
            xi[i] = torch.from_numpy(
                np.interp(s.numpy(), cum[i].numpy(), xs.numpy()))
        pts = serpentine_point(p0[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                               d[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                               n[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                               sp['amp'][:, None].expand(-1, NG).reshape(-1),
                               sp['wavelen'][:, None].expand(-1, NG).reshape(-1),
                               xi.reshape(-1)).reshape(N, NG, 3)
        axes = n[:, None, :].expand(-1, NG, -1).clone()
    else:                                      # rotating axis, straight line
        pts, _ = straight_samples(p0, d, n)
        ax = sp['n_rot_axis'] / sp['n_rot_axis'].norm(dim=-1, keepdim=True)
        th = sp['n_rot_rate'][:, None] * s[None, :]          # (N, NG)
        kxn = torch.linalg.cross(ax[:, None, :].expand(-1, NG, -1),
                                 n[:, None, :].expand(-1, NG, -1), dim=-1)
        kdn = (ax * n).sum(-1)[:, None, None]
        axes = (n[:, None, :] * torch.cos(th)[..., None]
                + kxn * torch.sin(th)[..., None]
                + ax[:, None, :] * kdn * (1 - torch.cos(th))[..., None])
        axes = axes / axes.norm(dim=-1, keepdim=True)
    # oracle candidate start (for witnesses)
    L = labels[key]; nf = cands[key]['n_found']
    V = torch.arange(L.shape[1])[None, :] < nf[:, None]
    best = torch.where(V, L, torch.full_like(L, -1e9)).argmax(1)
    qbest = cands[key]['cands'][torch.arange(N), best]
    np.savez_compressed(f, cs_p0=p0.float().numpy(),
                        cs_line_dir=d.float().numpy(),
                        cs_n_target=n.float().numpy(),
                        q0_seed=qbest.numpy(),
                        path_pts=pts.float().numpy(),
                        path_axes=axes.float().numpy(),
                        **{kk: sp[kk].float().numpy() for kk in
                           ('kappa', 'amp', 'wavelen', 'n_rot_axis',
                            'n_rot_rate') if kk in sp})
    print('wrote', f.name)

# ---- (B) random-FK warm tables for xarm7 / cobotta -------------------------
for robot in ('xarm7', 'cobotta'):
    f = OUT / f'fk_table_{robot}.npz'
    if f.exists():
        continue
    env = envfor(robot)
    kin = env.kin
    nj = kin.lmt_lo.shape[0]
    g = torch.Generator(device='cpu').manual_seed(3)
    Q = (torch.rand(300000, nj, generator=g).to(dev, kin.dtype)
         * (kin.lmt_up - kin.lmt_lo) + kin.lmt_lo)
    POS, ZAX, JI = [], [], []
    for i in range(0, Q.shape[0], 8192):
        q = Q[i:i + 8192]
        p, R, J, _ = kin.tcp_fk_jac(q)
        POS.append(p.float().cpu()); ZAX.append(R[:, :, 2].float().cpu())
        JI.append(torch.linalg.pinv(J).float().cpu())
    np.savez_compressed(f, q=Q.float().cpu().numpy(),
                        pos=torch.cat(POS).numpy(),
                        zax=torch.cat(ZAX).numpy(),
                        jinv6=torch.cat(JI).numpy())
    print('wrote', f.name)
print('ASSETS DONE')
