"""xArm7 selection-stage data under the unified (critic-based) scheme.

Per task set (straight = the 10k pool; serpentine / nonplanar = the 2500
selx sets): collect all admissible IK starts at the seam start (per-task
cone directions x warm starts from fk_table_xarm7), farthest-point-sample
K=32 candidates, roll each with the xArm7 DirFrac mainline (labels), and
score each candidate with the same run's critic. Saves L, V, n_found per
set for the selection tables (rows: first / random / heuristics / shared
critic / oracle)."""
import sys, math, dataclasses, time, os
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
K_SEEDS = 32
SMOKE = bool(int(os.environ.get('SMOKE', '0')))

env = lb.build_env(dev, 'stock', 512, robot='xarm7')
T = np.load(A / 'fk_table_xarm7.npz')
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
dt = env.kin.dtype
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_xarm7.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2500
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_xarm7_XXL/agent.pt',
    map_location=dev))
ag.eval()
rdt = renv.kin.dtype

SETS = {
    'straight': dict(npz=A / 'tasks_pool_xarm7.npz', curved=False),
    'serpentine': dict(npz=A / 'tasks_selx_serpentine_xarm7.npz',
                       curved=True),
    'nonplanar': dict(npz=A / 'tasks_selx_nonplanar_xarm7.npz',
                      curved=True),
}

for name, meta in SETS.items():
    out_f = FU / f'xarm7_sel_{name}.npz'
    if out_f.exists():
        print(f'[xsel] {name}: exists, skip', flush=True)
        continue
    tz = np.load(meta['npz'])
    N = len(tz['cs_line_dir']) if not SMOKE else 128
    p0 = (tz['cs_p0'][:N] if 'cs_p0' in tz.files else None)
    if p0 is None:
        # pool npz: start position from FK of the seed q0
        q0s = torch.tensor(tz['q0_seed'][:N], dtype=dt, device=dev)
        p0 = env.kin.tcp_fk_jac(q0s)[0].cpu().numpy().astype(np.float32)
    nt = tz['cs_n_target'][:N].astype(np.float32)
    nt /= np.linalg.norm(nt, axis=1, keepdims=True)
    ld = tz['cs_line_dir'][:N].astype(np.float32)
    ld /= np.linalg.norm(ld, axis=1, keepdims=True)

    # per-task cone directions (axis + 7 sampled)
    M_DIRS = 8
    dirs = np.empty((N, M_DIRS, 3), np.float32)
    for i in range(N):
        pool = _sample_in_cone(torch.as_tensor(nt[i]), lb.CONE_DEG, 16,
                               np.random.default_rng(31 + i)).numpy()
        dirs[i, 0] = nt[i]
        dirs[i, 1:] = pool[:M_DIRS - 1]

    # admissible starts
    K_NN = 200
    sol_pid, sol_q = [], []
    t0 = time.time()
    for m in range(M_DIRS):
        zs = dirs[:, m]
        feat = np.concatenate([p0 * POS_SCALE, zs], 1).astype(np.float32)
        _, ids = tree.query(feat, k=K_NN, workers=-1)
        cand = T['q'][ids]
        fq = torch.as_tensor(cand.reshape(-1, 7), device=dev, dtype=dt)
        fp = torch.as_tensor(np.repeat(p0, K_NN, 0), device=dev, dtype=dt)
        fz = torch.as_tensor(np.repeat(zs, K_NN, 0), device=dev, dtype=dt)
        fn = torch.as_tensor(np.repeat(nt, K_NN, 0), device=dev, dtype=dt)
        CH = 8192
        for lo in range(0, fq.shape[0], CH):
            q_o, _, _ = _batched_ik_project(
                env.kin, fq[lo:lo + CH], fp[lo:lo + CH],
                _build_R_with_z(fz[lo:lo + CH], hint), branch_action=None)
            coll = env.collision.is_collided(env.kin.link_transforms(q_o))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
            in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                      & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
            fine = ((~coll) & in_lmt
                    & ((p_fk - fp[lo:lo + CH]).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * fn[lo:lo + CH]).sum(-1) >= cos_lim))
            f = fine.cpu().numpy()
            rows = (np.arange(lo, lo + q_o.shape[0]) // K_NN)[f]
            sol_pid.append(rows)
            sol_q.append(q_o[fine].cpu().numpy())
        print(f'[xsel] {name} dir {m + 1}/{M_DIRS} '
              f'({sum(len(x) for x in sol_pid)} sols, '
              f'{time.time() - t0:.0f}s)', flush=True)
    PID = np.concatenate(sol_pid)
    QS = np.concatenate(sol_q)
    order = np.argsort(PID, kind='stable')
    PID, QS = PID[order], QS[order]

    # FPS K=32 per task
    C = np.full((N, K_SEEDS, 7), np.nan, np.float32)
    nf = np.zeros(N, np.int32)
    lo = 0
    while lo < len(PID):
        hi = lo
        while hi < len(PID) and PID[hi] == PID[lo]:
            hi += 1
        qs = np.unique(np.round(QS[lo:hi], 2), axis=0)
        if len(qs) > K_SEEDS:
            pick = [0]
            dmin = np.linalg.norm(qs - qs[0], axis=1)
            for _ in range(K_SEEDS - 1):
                j = int(dmin.argmax())
                pick.append(j)
                dmin = np.minimum(dmin, np.linalg.norm(qs - qs[j], axis=1))
            qs = qs[pick]
        C[PID[lo], :len(qs)] = qs
        nf[PID[lo]] = len(qs)
        lo = hi
    print(f'[xsel] {name}: median candidates '
          f'{int(np.median(nf[nf > 0]))}', flush=True)

    # spec fields for rollouts / critic obs
    spec = {'line_dir': torch.tensor(ld), 'n_target': torch.tensor(nt)}
    for k2 in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
        if k2 in tz.files:
            spec[k2] = torch.tensor(tz[k2][:N])

    L = np.zeros((N, K_SEEDS), np.float32)
    V = np.full((N, K_SEEDS), np.nan, np.float32)
    t0 = time.time()
    for k in range(K_SEEDS):
        valid = nf > k
        for lo in range(0, N, B):
            hi = min(lo + B, N)
            pad = B - (hi - lo)
            q0b = np.nan_to_num(C[lo:hi, k], nan=0.0)
            sub = {'q0': torch.tensor(q0b, dtype=rdt)}
            for kk, v in spec.items():
                sub[kk] = v[lo:hi]
            if pad:
                sub = {kk: torch.cat([v, v[-1:].expand(pad,
                                                       *v.shape[1:])])
                       for kk, v in sub.items()}
            for kk in ('q0', 'line_dir', 'n_target'):
                sub[kk] = sub[kk].to(device=dev, dtype=rdt)
            renv.line_dist = ScriptedLineDistribution(sub)
            renv.reset()
            with torch.no_grad():
                V[lo:hi, k] = ag.get_value(
                    renv.current_obs()).float().cpu().numpy()[:hi - lo]
                for _ in range(renv.cfg.max_steps // 2):
                    a = ag.actor_mean(renv.current_obs())
                    for _ in range(2):
                        renv.step(a, auto_reset=False)
                    if bool(renv.done_persistent.all()):
                        break
            L[lo:hi, k] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
        L[~valid, k] = 0.0
        V[~valid, k] = np.nan
        print(f'[xsel] {name}: cand {k + 1}/{K_SEEDS} '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    if not SMOKE:
        np.savez_compressed(out_f, L=L, V=V, n_found=nf, cands=C)
        print(f'[xsel] wrote {out_f.name}', flush=True)
print('all done', flush=True)
