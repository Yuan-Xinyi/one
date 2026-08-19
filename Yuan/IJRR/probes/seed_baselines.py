"""Reviewer comment 5: heuristic candidate-ranking baselines on the SAME
candidate pool as the learned selector (selector_ood v1, test_straight,
FR3), scored against the cached per-candidate rollout labels. Also the
no-privileged-slot variant (slot 0 = task-generating configuration).
Pure slicing of cached data; no new rollouts."""
import sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'selector_ood', MAIN / 'stage1_seed/selector_ood.py')
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)
cand_features, road_table, K_CAND = so.cand_features, so.road_table, so.K_CAND

dev = torch.device('cuda')
hl.SUB = 2
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env)

d = MAIN / 'runs/selector_ood/v1'
tasks = torch.load(d / 'tasks.pt', weights_only=False)
cands = torch.load(d / 'cands.pt', weights_only=False)
labels = torch.load(d / 'labels.pt', weights_only=False)
rk = torch.load(d / 'rankers.pt', weights_only=False)

KEY = 'test_straight'
spec = tasks[KEY]
C = cands[KEY]['cands'].to(dev)                  # (N,8,7)
NF = cands[KEY]['n_found']
Y = labels[KEY].float()                          # (N,8) stroke lengths [m]
N = C.shape[0]
M = (torch.arange(K_CAND)[None, :] < NF[:, None])          # valid mask

P0 = spec['p0'].to(dev)
D = spec['line_dir'].to(dev)
NT = spec['n_target'].to(dev)

# ---- heuristic scores at the start configuration --------------------------
CH = 4096
flat_q = C.reshape(-1, 7)
flat_p0 = P0.repeat_interleave(K_CAND, 0)
flat_d = D.repeat_interleave(K_CAND, 0)
flat_n = NT.repeat_interleave(K_CAND, 0)
mjl, mcone, phim, wdir = [], [], [], []
with torch.no_grad():
    for i in range(0, flat_q.shape[0], CH):
        q = flat_q[i:i + CH]
        m = model.margins(q, flat_p0[i:i + CH], flat_d[i:i + CH],
                          flat_n[i:i + CH])
        mjl.append(m[:, 0]); mcone.append(m[:, 1])
        phim.append(-0.1 * torch.logsumexp(-m / 0.1, dim=-1))
        _, _, J, _ = env.kin.tcp_fk_jac(q)
        J_plus, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                                env.cfg.sigma_thr)
        wdir.append(1.0 / ((J_plus @ flat_d[i:i + CH].unsqueeze(-1))
                           .squeeze(-1).norm(dim=-1) + 1e-9))
S_heur = {
    'max joint-limit margin': torch.cat(mjl).reshape(N, K_CAND).cpu(),
    'max orientation margin': torch.cat(mcone).reshape(N, K_CAND).cpu(),
    'margin field  (softmin)': torch.cat(phim).reshape(N, K_CAND).cpu(),
    'max directional manip.': torch.cat(wdir).reshape(N, K_CAND).cpu(),
}

# ---- learned selector scores ----------------------------------------------
Xc = cand_features(env, C.cpu().numpy(), spec)
rt = road_table(spec)
Xr = torch.tensor(rt.reshape(rt.shape[0], -1))
netc = so.Ranker(Xc.shape[-1], Xr.shape[-1], conditioned=True).to(dev)
netc.load_state_dict(rk['cond'])
netc.eval()
with torch.no_grad():
    S_sel = netc(Xc.to(dev).float(), Xr.to(dev).float()).cpu()

def rows(mask, tag):
    """mean length + capture (first-feasible normalization) per scorer"""
    Ym = torch.where(mask, Y, torch.full_like(Y, -1e9))
    oracle = Ym.max(1).values
    # first feasible = first valid slot
    first_idx = mask.float().argmax(1)
    first = Y.gather(1, first_idx[:, None]).squeeze(1)
    # random = mean over valid
    rnd = (Y * mask).sum(1) / mask.sum(1).clamp(min=1)
    keep = mask.any(1)
    res = {'first feasible': first, 'random valid candidate': rnd}
    for name, S in list(S_heur.items()) + [('learned selector', S_sel)]:
        pick = torch.where(mask, S, torch.full_like(S, -1e9)).argmax(1)
        res[name] = Y.gather(1, pick[:, None]).squeeze(1)
    res['within-pool oracle'] = oracle
    print(f'\n== {tag}  (n={int(keep.sum())}) ==')
    f0 = first[keep].mean(); orc = oracle[keep].mean()
    for name, v in res.items():
        vm = v[keep].mean()
        cap = (vm - f0) / (orc - f0) * 100
        print(f'  {name:28s} {vm:.3f} m   capture {cap:6.1f}%')

rows(M, 'full pool (slot 0 = task-generating config)')
M2 = M.clone(); M2[:, 0] = False                 # drop the privileged slot
rows(M2 & (M2.any(1, keepdim=True)), 'without the privileged slot')
