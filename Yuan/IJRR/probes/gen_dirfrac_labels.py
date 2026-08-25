"""Roll every IKSel candidate of the selection sets with the DirFrac
mainline (rh2048XXL) and record the executed stroke: L[N, K=32].
These labels regenerate the selection-stage tables under the new
motion-generation controller.

argv: set names among train_straight train_arc test_straight test_arc
      test_serpentine test_nonplanar benchmark
env:  SMOKE=1 -> 128 tasks, K=4."""
import sys, os, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
SMOKE = bool(int(os.environ.get('SMOKE', '0')))
SETS = sys.argv[1:]

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_rh2048XXL.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)

cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                   weights_only=False)


def make_env(curved, batch):
    kw2 = dict(kw)
    if curved:
        kw2['k_lateral'] = 5.0
    env = NSRLBatchedEnv(EnvConfig(**{**kw2, 'n_envs': batch}), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(
        REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL/agent.pt',
        map_location=dev))
    ag.eval()
    return env, ag


def get_spec(name):
    if name == 'benchmark':
        tz = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
        ld = torch.tensor(tz['cs_line_dir'], dtype=torch.float32)
        nt = torch.tensor(tz['cs_n_target'], dtype=torch.float32)
        ld = ld / ld.norm(dim=-1, keepdim=True)
        nt = nt / nt.norm(dim=-1, keepdim=True)
        return {'line_dir': ld, 'n_target': nt}, False
    sp = tasks[name]
    spec = {'line_dir': sp['line_dir'].clone(),
            'n_target': sp['n_target'].clone(),
            'p0': sp['p0'].clone()}
    curved = False
    for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
        if k in sp:
            spec[k] = sp[k].clone()
            curved = True
    return spec, curved


@torch.no_grad()
def roll(env, ag, sub):
    env.line_dist = ScriptedLineDistribution(sub)
    env.reset()
    for _ in range(env.cfg.max_steps // 2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    return env.arc_progress.float().cpu().numpy().copy()


for name in SETS:
    out_f = FU / f'dirfrac_labels_{name}.npz'
    if out_f.exists():
        print(f'[labels] {name}: exists, skip', flush=True)
        continue
    spec, curved = get_spec(name)
    C = cands[name]['cands']
    nf = cands[name]['n_found'].numpy()
    N, K = C.shape[:2]
    if SMOKE:
        N, K = 128, 4
    B = min(2500, N)
    env, ag = make_env(curved, B)
    dt = env.kin.dtype
    L = np.zeros((N, K), np.float32)
    t0 = time.time()
    for k in range(K):
        for lo in range(0, N, B):
            hi = min(lo + B, N)
            pad = B - (hi - lo)
            sub = {'q0': C[lo:hi, k]}
            for kk, v in spec.items():
                sub[kk] = v[lo:hi]
            if pad:
                sub = {kk: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                       for kk, v in sub.items()}
            for kk in ('q0', 'line_dir', 'n_target'):
                sub[kk] = sub[kk].to(device=dev, dtype=dt)
            L[lo:hi, k] = roll(env, ag, sub)[:hi - lo]
        print(f'[labels] {name}: cand {k + 1}/{K}  '
              f'({(time.time() - t0) / 60:.1f} min)', flush=True)
    if not SMOKE:
        np.savez_compressed(out_f, L=L, n_found=nf[:N])
        print(f'[labels] wrote {out_f.name}', flush=True)
    else:
        V = np.arange(K)[None, :] < nf[:N, None]
        best = np.where(V[:, :K], L, -1e9).max(1)
        print(f'[labels-smoke] {name}: oracle-of-{K} mean {best.mean():.4f}',
              flush=True)
    del env, ag
    torch.cuda.empty_cache()
print('all done', flush=True)
