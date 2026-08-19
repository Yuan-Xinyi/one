import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.rollout import rollout_first_episode
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
task = np.load(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2/task.npz')
spec = {k: torch.tensor(task[k], device=dev, dtype=env.kin.dtype).unsqueeze(0)
        for k in ('q0', 'line_dir', 'n_target')}
for name, terms in (('softmin jl+cone (deployed)', [0, 1]),
                    ('cone-greedy only', [1]),
                    ('jl-greedy only', [0]),
                    ('all four margins', None)):
    model = hl.StraightModel(env); model.terms = terms
    myo = hl.make_myopic(model)
    env.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
    st = rollout_first_episode(env, lambda e: myo(e, e.done_persistent))
    print(f"{name:<28s} progress {float(st['episode_progress'][0]):.4f} m  "
          f"len {int(st['episode_len'][0]):>3}  "
          f"term {TERM_NAMES[int(st['term_reason'][0])]}")
