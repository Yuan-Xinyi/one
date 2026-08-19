"""General witness generator for the ratio-unification bound runs.
Sets: pool_fr3 / pool_xarm7 / pool_cobotta (ladder env, task q0) and
sel_straight / sel_arc / sel_serpentine / sel_nonplanar (label env:
k_lateral=5, task-p0 anchor, oracle-candidate start; FR3).
Crossing detection and stroke use env.arc_progress (label convention)."""
import sys, os, time, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent

SET = sys.argv[1]
STARTS = int(os.environ.get('STARTS', '1'))
hl.SUB = 2
STEP, MAXL = 0.01, 1.8
NG = int(round(MAXL / STEP)) + 1
dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'

robot = 'fr3'
if SET == 'pool_xarm7':
    robot = 'xarm7'
elif SET == 'pool_cobotta':
    robot = 'cobotta'
elif SET.startswith('selx_'):          # selx_{fam}_{robot}
    robot = SET.rsplit('_', 1)[-1]
CFG, CKPT = hl.ROBOTS[robot]
y = yaml.safe_load(open(REPO / CFG))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] = kw['dt'] / 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
if SET.startswith('sel_') or SET.startswith('selx_'):
    kw['k_lateral'] = 5.0                     # label-env convention
tz = np.load(A / f'tasks_{SET}.npz')
N = tz['cs_p0'].shape[0]
Q0S = tz['q0_seed'][:, None, :]                      # (N, 1, NJ)
if STARTS > 1 and SET.startswith('sel_'):
    fam = SET.split('_', 1)[1]
    L = torch.load(MAIN / 'runs/selector_ood/v3_k32_fixedgate/labels.pt',
                   weights_only=False)[f'test_{fam}'].float()
    Cn = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                    weights_only=False)[f'test_{fam}']
    M = torch.arange(L.shape[1])[None, :] < Cn['n_found'][:, None]
    order = torch.where(M, L, torch.full_like(L, -1e9)).argsort(
        dim=1, descending=True)
    picks = order[:, :STARTS]
    qs = Cn['cands'][torch.arange(N)[:, None], picks]
    Q0S = qs.numpy()
    print(f'[wit {SET}] multi-start k={STARTS} from label top-k', flush=True)
B = 2500 if N >= 2500 else N
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
ag = _agent(REPO / CKPT, env.obs_dim, dev, act_dim=env.act_dim)
vfn = hl.make_vlook(model, env, ag)
NJ = int(env.kin.lmt_lo.shape[0])

W = np.full((N, NG, NJ), np.nan, np.float32)
prog_all = np.zeros(N, np.float32)
t0 = time.time()
for j_start in range(Q0S.shape[1]):
  for base in range(0, N, B):
      ids = np.arange(base, min(base + B, N))
      pad = B - len(ids)
      ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
      spec = {'q0': torch.as_tensor(Q0S[ip, j_start], device=dev),
              'line_dir': torch.as_tensor(tz['cs_line_dir'][ip],
                                          dtype=torch.float32, device=dev),
              'n_target': torch.as_tensor(tz['cs_n_target'][ip],
                                          dtype=torch.float32, device=dev)}
      if SET.startswith('sel_') or SET.startswith('selx_'):
          spec['p0'] = torch.as_tensor(tz['cs_p0'][ip], dtype=torch.float32,
                                       device=dev)
          for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
              if k in tz.files:
                  spec[k] = torch.as_tensor(tz[k][ip], dtype=torch.float32,
                                            device=dev)
      env.line_dist = ScriptedLineDistribution(spec)
      env.reset()
      done = torch.zeros(B, dtype=torch.bool, device=dev)
      nxt = torch.ones(B, dtype=torch.long, device=dev)
      Wb = torch.full((B, NG, NJ), float('nan'), device=dev)
      Wb[:, 0] = env.q
      q_last = env.q.clone()
      cur_last = env.arc_progress.clone().float()
      blocks = env.cfg.max_steps // 2
      for _ in range(blocks):
          a = vfn(env, done)
          for _ in range(2):
              env.step(a, auto_reset=False)
              live = ~env.done_persistent
              cur = env.arc_progress.float()
              for _ in range(4):
                  s_g = nxt.float() * STEP
                  fill = live & (cur >= s_g) & (nxt < NG) \
                      & (cur > cur_last + 1e-9)
                  if not bool(fill.any()):
                      break
                  alpha = ((s_g - cur_last) / (cur - cur_last)
                           ).clamp(0, 1)[fill].unsqueeze(-1)
                  Wb[fill, nxt[fill]] = (q_last[fill]
                                         + alpha * (env.q[fill] - q_last[fill]))
                  nxt = torch.where(fill, nxt + 1, nxt)
              q_last = env.q.clone()
              cur_last = torch.where(live, cur, cur_last)
          done = env.done_persistent.clone()
          if bool(done.all()):
              break
      fin = env.arc_progress.float()
      prog = torch.where(env.done_persistent, fin, fin)
      Wn = Wb[:len(ids)].cpu().numpy()
      keep = np.isnan(W[ids]) & ~np.isnan(Wn)
      W[ids] = np.where(keep, Wn, W[ids])
      prog_all[ids] = np.maximum(prog_all[ids],
                                 cur_last[:len(ids)].cpu().numpy())
      print(f'[wit {SET}] start {j_start} {ids[-1]+1}/{N} '
            f'mean {prog_all[:ids[-1]+1].mean():.4f} '
            f'({time.time()-t0:.0f}s)', flush=True)

np.savez_compressed(A / f'witness_{SET}.npz', W=W, prog=prog_all, step=STEP)
print(f'[wit {SET}] wrote witness_{SET}.npz  mean stroke {prog_all.mean():.4f}')
