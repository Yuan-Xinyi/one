"""Generate witness configurations for the pointwise bound: re-roll the
deployed value law from the within-pool best candidate of every 10k eval
task, and record the configuration at each 2 cm arc-length crossing.
Witnesses certify pointwise feasibility up to each task's best stroke."""
import sys, time, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent

hl.SUB = 2
STEP = 0.01
MAXL = 1.8
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))

t = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
fs = np.load(MAIN / 'runs/selector_ood/v2_k32/benchmark_arrays.npz')
L, cands, nf = fs['L'], fs['cands'], fs['n_found']
N = L.shape[0]
valid = np.arange(L.shape[1])[None, :] < nf[:, None]
best = np.where(valid, L, -1e9).argmax(1)
q0 = cands[np.arange(N), best]

B = 2500
kw = dict(y['env']); kw['dt'] = kw['dt'] / 2; kw['k_lateral'] = 5.0
kw['max_steps'] = int(kw['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
vfn = hl.make_vlook(model, env, ag)

n_grid = int(round(MAXL / STEP)) + 1
W = np.full((N, n_grid, 7), np.nan, np.float32)
P0_exec = np.zeros((N, 3), np.float32)      # executed ray origin = FK(q0)
prog_all = np.zeros(N, np.float32)
t0 = time.time()
for base in range(0, N, B):
    ids = np.arange(base, min(base + B, N))
    pad = B - len(ids)
    ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.as_tensor(q0[ip], device=dev),
         'p0': torch.as_tensor(
             t['cs_p0'][ip], dtype=torch.float32, device=dev),
         'line_dir': torch.as_tensor(
             t['cs_line_dir'][ip], dtype=torch.float32, device=dev),
         'n_target': torch.as_tensor(
             t['cs_n_target'][ip], dtype=torch.float32, device=dev)})
    env.reset()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    nxt = torch.zeros(B, dtype=torch.long, device=dev)   # next grid index
    Wb = torch.full((B, n_grid, 7), float('nan'), device=dev)
    prog = torch.zeros(B, device=dev)
    q_last = env.q.clone()
    cur_last = torch.zeros(B, device=dev)
    Wb[:, 0] = env.q                    # the start config witnesses s=0
    nxt += 1
    blocks = env.cfg.max_steps // 2
    for _ in range(blocks):
        a = vfn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
            live = ~env.done_persistent
            p, _, _, _ = env.kin.tcp_fk_jac(env.q)
            cur = ((p - env.p_start) * env.line_dir).sum(-1)
            prog = torch.maximum(prog, torch.where(live | done, cur, prog))
            # interpolate a config whose tip sits ON each crossed grid arc,
            # so the raw constraint check certifies it without projection
            for _ in range(4):
                s_g = nxt.float() * STEP
                fill = live & (cur >= s_g) & (nxt < n_grid) \
                    & (cur > cur_last + 1e-9)
                if not bool(fill.any()):
                    break
                alpha = ((s_g - cur_last) / (cur - cur_last)
                         ).clamp(0, 1)[fill].unsqueeze(-1)
                qw = q_last[fill] + alpha * (env.q[fill] - q_last[fill])
                Wb[fill, nxt[fill]] = qw
                nxt = torch.where(fill, nxt + 1, nxt)
            q_last = env.q.clone()
            cur_last = torch.where(live, cur, cur_last)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    W[ids] = Wb[:len(ids)].cpu().numpy()
    P0_exec[ids] = env.p_start[:len(ids)].float().cpu().numpy()
    prog_all[ids] = prog[:len(ids)].cpu().numpy()
    print(f'[witness] {ids[-1]+1}/{N}  mean stroke so far '
          f'{prog_all[:ids[-1]+1].mean():.4f} m  ({time.time()-t0:.0f}s)',
          flush=True)

ref = np.where(valid, L, -1e9).max(1)
print(f'[witness] re-rolled mean {prog_all.mean():.4f} vs cached oracle '
      f'{ref.mean():.4f}; per-task |diff| mean '
      f'{np.abs(prog_all - ref).mean():.4f}')
np.savez_compressed(MAIN / 'runs/paper_fill/witness_10k_v4.npz',
                    W=W, prog=prog_all, step=STEP)
print('[witness] wrote witness_10k_v4.npz')
