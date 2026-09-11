"""Pause-usage probe for the gate and gate+beta arms: fraction of steps
with the gate closed, split by wall proximity (min margin < 0.05)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'
tz = np.load(A/'tasks_pool_fr3.npz')
N = 2048
for suf in ('gate', 'gb'):
    y = yaml.safe_load(open(REPO/f'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_{suf}.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO/f'Yuan/IJRR/runs/rl_dirfrac_e8k_{suf}/agent.pt', map_location=dev))
    ag.eval()
    dt_t = env.kin.dtype
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(tz['q0_seed'][:N], dtype=dt_t, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][:N], dtype=dt_t, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][:N], dtype=dt_t, device=dev)})
    env.reset()
    n_pause = n_step = n_pw = n_wall = 0
    with torch.no_grad():
        for _ in range(env.cfg.max_steps//2):
            a = ag.actor_mean(env.current_obs())
            g0 = (a[:, -1] <= 0)
            act = ~env.done_persistent
            m_jl = ((env.q_half - (env.q - env.q_mid).abs())/env.q_half).amin(-1)
            wall = m_jl < 0.05
            n_pause += int((g0 & act).sum()); n_step += int(act.sum())
            n_pw += int((g0 & act & wall).sum()); n_wall += int((act & wall).sum())
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
    deep = (n_pause-n_pw)/max(n_step-n_wall,1)
    print(f'{suf}: 总暂停率 {n_pause/max(n_step,1)*100:.1f}%  贴墙 {n_pw/max(n_wall,1)*100:.1f}%  '
          f'深水 {deep*100:.1f}%', flush=True)
    del env, ag; torch.cuda.empty_cache()
