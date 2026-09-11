"""Re-project pool q0 to satisfy the 1-deg cone exactly; invalidate rows
that cannot be tightened. Operates on the two n_pool=100002 caches."""
import sys, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
dev = torch.device('cuda')
env = lb.build_env(dev, 'stock', 512)
dt = env.kin.dtype
cos_ok = math.cos(math.radians(0.9))
for key in ('7a4bdde9e1', 'cefa081b44'):
    p = REPO/f'Yuan/IJRR/runs/_pool_cache/pool_{key}.pt'
    d = torch.load(p, map_location='cpu', weights_only=False)
    q = d['q_pool'].to(dev, dt)
    nt = d['n_target_pool'].to(dev, dt)
    nt = nt/nt.norm(dim=-1, keepdim=True)
    ok_all = torch.zeros(len(q), dtype=torch.bool, device=dev)
    q_new = q.clone()
    B = 8192
    for lo in range(0, len(q), B):
        hi = min(lo+B, len(q))
        qs = q[lo:hi]
        p_fk0, _, _, _ = env.kin.tcp_fk_jac(qs)
        R_t = _build_R_with_z(nt[lo:hi], torch.tensor([1.0, 0, 0], dtype=dt, device=dev))
        q_o, _, _ = _batched_ik_project(env.kin, qs, p_fk0, R_t, branch_action=None)
        # second pass tightens further
        q_o, _, _ = _batched_ik_project(env.kin, q_o, p_fk0, R_t, branch_action=None)
        coll = env.collision.is_collided(env.kin.link_transforms(q_o))
        p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
        in_lmt = ((q_o >= env.kin.lmt_lo-1e-5) & (q_o <= env.kin.lmt_up+1e-5)).all(-1)
        fine = ((~coll) & in_lmt
                & ((p_fk-p_fk0).norm(dim=-1) <= 0.005)
                & ((R_fk[:, :, 2]*nt[lo:hi]).sum(-1) >= cos_ok))
        ok_all[lo:hi] = fine
        q_new[lo:hi] = torch.where(fine.unsqueeze(-1), q_o, qs)
    d['q_pool'] = q_new.cpu().to(d['q_pool'].dtype)
    vm = d['valid_mask']
    d['valid_mask'] = vm & ok_all.cpu()
    torch.save(d, p)
    print(f'{key}: repaired {ok_all.float().mean()*100:.1f}% rows valid '
          f'(mask now {d["valid_mask"].float().mean()*100:.1f}%)', flush=True)
