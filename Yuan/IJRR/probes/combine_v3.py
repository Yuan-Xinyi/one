"""Compose dirfrac v3 from the single-ingredient results: any flag whose
10k mean is not worse than v2 - 0.002 goes in (neutral ingredients are
kept for their semantic value). Writes the v3 yaml only when >= 2 flags
qualify (a single winner is already its own trained config)."""
import numpy as np

FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
ST = ('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/'
      'Yuan/IJRR/stage2_traj/')
v2 = np.load(FU + 'dirfrac_v2_10k.npz')['prog'].mean()
flags = {'aprev': 'a_prev_executed: true',
         'headroom': 'observe_headroom: true',
         'ajoint': 'alpha_joint: true'}
keep = []
for v, line in flags.items():
    m = np.load(FU + f'dirfrac_{v}_10k.npz')['prog'].mean()
    tag = 'KEEP' if m >= v2 - 0.002 else 'DROP'
    print(f'{v:9s} mean {m:.4f}  (v2 {v2:.4f})  -> {tag}')
    if tag == 'KEEP':
        keep.append(line)
if len(keep) >= 2:
    s = open(ST + 'config_line_cont_dirfrac_v2.yaml').read()
    s = s.replace('env:\n  dir_frac_action: 2',
                  'env:\n  dir_frac_action: 2\n  ' + '\n  '.join(keep))
    open(ST + 'config_line_cont_dirfrac_v3.yaml', 'w').write(s)
    print('v3 config written with:', keep)
else:
    print('fewer than 2 qualifying flags; no v3 combo training')
