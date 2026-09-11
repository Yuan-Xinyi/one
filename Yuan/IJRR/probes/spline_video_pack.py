"""Re-roll one spline scenario and save a render pack (+ the spline)."""
import sys
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg')
import numpy as np
src = open('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/spline_probe.py').read()
exec(src.split("SPL = [")[0].replace("N_SP = int(sys.argv[1])", "N_SP = 100").replace("TAG = sys.argv[2] if len(sys.argv) > 2 else 'smoke'", "TAG='x'"))
SPL = [sample_spline(rng) for _ in range(100)]
d = np.load(str(FU/'spline_probe_v1.npz'))
prog, L = d['prog'], d['length']
fin = prog >= L - 0.03
cand = np.nonzero(fin)[0]
IDX = int(cand[np.argmax(L[cand])]) if len(cand) else int(np.argmax(prog))
print(f'chosen spline {IDX}: len {L[IDX]:.2f} prog {prog[IDX]:.2f}')
pts, tan, Li = SPL[IDX]
env0 = lb.build_env(dev, 'stock', 512)
dt0 = env0.kin.dtype
T = np.load(REPO/lb.TABLE)
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
import math as _m
cos_lim = _m.cos(_m.radians(lb.CONE_DEG)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)
# start config
feat = np.concatenate([pts[0]*POS_SCALE, [0, 0, 1]], 0)[None].astype(np.float32)
_, ids = tree.query(feat, k=96, workers=-1)
fq = torch.as_tensor(T['q'][ids[0]], device=dev, dtype=dt0)
fp = torch.as_tensor(pts[0], device=dev, dtype=dt0).expand(96, 3)
fz = torch.tensor([0.0, 0.0, 1.0], dtype=dt0, device=dev).expand(96, 3)
q_o, _, _ = _batched_ik_project(env0.kin, fq, fp, _build_R_with_z(fz, hint), branch_action=None)
coll = env0.collision.is_collided(env0.kin.link_transforms(q_o))
p_fk, R_fk, _, _ = env0.kin.tcp_fk_jac(q_o)
in_lmt = ((q_o >= env0.kin.lmt_lo-1e-5) & (q_o <= env0.kin.lmt_up+1e-5)).all(-1)
fine = ((~coll) & in_lmt & ((p_fk-fp).norm(dim=-1) <= tube)
        & ((R_fk[:, :, 2]*fz).sum(-1) >= cos_lim))
q0 = q_o[int(fine.float().argmax())].cpu().numpy()
del env0; torch.cuda.empty_cache()

import dataclasses, yaml, torch as th
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
class SplineEnv(NSRLBatchedEnv):
    def set_tables(self, p_, t_): self._sp, self._st = p_, t_
    def _path_frame(self, p):
        d2 = ((p.unsqueeze(1) - self._sp)**2).sum(-1)
        j = d2.argmin(1); ar = th.arange(len(j), device=p.device)
        lat = self._sp[ar, j] - p
        return self._st[ar, j], lat, lat.norm(dim=-1)
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 3000; kw['k_lateral'] = 5.0
env = SplineEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
dt_t = env.kin.dtype
env.set_tables(th.tensor(pts[None], dtype=dt_t, device=dev),
               th.tensor(tan[None], dtype=dt_t, device=dev))
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(th.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
env.line_dist = ScriptedLineDistribution(
    {'q0': th.tensor(q0[None], dtype=dt_t, device=dev),
     'line_dir': th.tensor(tan[0][None], dtype=dt_t, device=dev),
     'n_target': th.tensor([[0.0, 0.0, 1.0]], dtype=dt_t, device=dev)})
env.reset()
QS, SS = [env.q[0].cpu().numpy().copy()], [0.0]
with th.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
            QS.append(env.q[0].cpu().numpy().copy())
            SS.append(float(env.arc_progress[0]))
        if bool(env.done_persistent.all()):
            break
QS, SS = np.array(QS), np.array(SS)
keep = np.concatenate([[True], np.diff(SS) > 1e-9])
QS, SS = QS[keep], SS[keep]
g = np.arange(0.0, SS[-1], 0.03)
QG = np.stack([np.interp(g, SS, QS[:, j]) for j in range(7)], 1).astype(np.float32)
qt = th.tensor(QG, dtype=dt_t, device=dev)
p, R, _, _ = env.kin.tcp_fk_jac(qt)
np.savez_compressed('/home/lqin/one/Yuan/IJRR/runs/paper_fill/search_compare/spline_showcase_pack.npz',
                    q=QG, s=g.astype(np.float32),
                    tip=p.cpu().numpy().astype(np.float32),
                    zax=R[:, :, 2].cpu().numpy().astype(np.float32),
                    ring=pts[::10].astype(np.float32), circ=np.float32(1e9))
print(f'pack: {len(QG)} frames, {SS[-1]:.2f} m')
