"""Can the PPO critic replace the rollout-supervised selector?
Score every IKSel candidate by V_hat(o(q0)) from the rh2048XXL run and
compare the picked candidates' TRUE strokes (dirfrac labels) against the
deployed selector and the oracle."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from scipy.stats import spearmanr
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
K = 32
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                   weights_only=False)
tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
bench = {'line_dir': torch.tensor(tb['cs_line_dir'], dtype=torch.float32),
         'n_target': torch.tensor(tb['cs_n_target'], dtype=torch.float32)}
for k2 in bench:
    bench[k2] = bench[k2] / bench[k2].norm(dim=-1, keepdim=True)

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_rh2048XXL.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2500
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL/agent.pt',
    map_location=dev))
ag.eval()
dt = env.kin.dtype

sel_rep = np.load(FU / 'sel_report_dirfrac.npz')

for key in ('test_straight', 'test_arc', 'test_serpentine',
            'test_nonplanar', 'benchmark'):
    L = np.load(FU / f'dirfrac_labels_{key}.npz')['L']
    nf = cands[key]['n_found'].numpy()[:L.shape[0]]
    C = cands[key]['cands'][:L.shape[0]]
    if key == 'benchmark':
        spec = bench
    else:
        sp = tasks[key]
        spec = {'line_dir': sp['line_dir'], 'n_target': sp['n_target']}
        for k2 in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
            if k2 in sp:
                spec[k2] = sp[k2]
    N = L.shape[0]
    V = np.full((N, K), -1e9, np.float32)
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
            env.line_dist = ScriptedLineDistribution(sub)
            env.reset()
            with torch.no_grad():
                V[lo:hi, k] = ag.get_value(
                    env.current_obs()).float().cpu().numpy()[:hi - lo]
    M = np.arange(K)[None, :] < nf[:, None]
    Vm = np.where(M, V, -1e9)
    pick = Vm.argmax(1)
    got = L[np.arange(N), pick]
    Lm = np.where(M, L, -1e9)
    orc = Lm.max(1)
    ok = orc > 1e-6
    # per-task rank corr between V and labels (valid cands only)
    rhos = []
    for i in np.nonzero(ok)[0][:2000]:
        if M[i].sum() >= 5:
            r = spearmanr(V[i, M[i]], L[i, M[i]]).statistic
            if np.isfinite(r):
                rhos.append(r)
    near = (got >= orc - 0.01)
    np.savez_compressed(FU / f'critic_scores_{key}.npz', V=V)
    sel_stroke = sel_rep[f'{key}__selector']
    print(f'{key:15s} critic-pick stroke {got[ok].mean():.3f} | '
          f'selector {sel_stroke[ok].mean():.3f} | oracle {orc[ok].mean():.3f} | '
          f'near-oracle {near[ok].mean()*100:.1f}% | '
          f'rank-rho {np.mean(rhos):.3f}', flush=True)
