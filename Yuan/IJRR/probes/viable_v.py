import numpy as np, torch, yaml, dataclasses, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz'); tr = np.load(RUN / 'traj_compare.npz')
env2 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
qp = torch.tensor(tr['PPO_q'], device=dev, dtype=env2.kin.dtype)
verts = torch.tensor(np.stack(np.meshgrid(*[[-1.,1.]]*4, indexing='ij'), -1).reshape(-1,4),
                     dtype=torch.float32, device=dev)
d = torch.tensor(task['line_dir'], device=dev, dtype=env2.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env2.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env2.kin.dtype)
p0 = env2.kin.tcp_fk_jac(q0_t[None])[0][0]

def viable(q_state, v, H=40, W=1024, grid=0.02):
    model = hl.StraightModel(env2)
    model.cfg = dataclasses.replace(env2.cfg, v=v)
    out = []
    for ai in range(16):
        qn = model.step(q_state[None], d[None], n_t[None], verts[ai][None])
        m = model.margins(qn, p0[None], d[None], n_t[None])
        if not bool(m.amin() > 0): out.append(0); continue
        q = qn; depth = 1
        for _ in range(H - 1):
            P = q.shape[0]
            qe = q.unsqueeze(1).expand(-1,16,-1).reshape(P*16,-1)
            ae = verts.unsqueeze(0).expand(P,-1,-1).reshape(P*16,-1)
            CH = 32768
            qq = torch.cat([model.step(qe[i:i+CH], d.expand(min(CH,P*16-i),3),
                                       n_t.expand(min(CH,P*16-i),3), ae[i:i+CH])
                            for i in range(0,P*16,CH)])
            mm = torch.cat([model.margins(qq[i:i+CH], p0.expand(min(CH,P*16-i),3),
                                          d.expand(min(CH,P*16-i),3),
                                          n_t.expand(min(CH,P*16-i),3))
                            for i in range(0,P*16,CH)])
            alive = (mm.amin(-1) > 0)
            if not bool(alive.any()): break
            qq = qq[alive]
            key = torch.round(qq/grid).to(torch.int32)
            _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
            keep = torch.as_tensor(np.sort(first), device=dev)
            if keep.numel() > W:
                keep = keep[torch.randperm(keep.numel(), device=dev)[:W]]
            q = qq[keep]; depth += 1
        out.append(depth)
    return np.array(out)

for name, qs in (('PPO@s=0.40', qp[40]), ('PPO@s=0.50', qp[50])):
    for v in (0.2, 0.1):
        Dv = viable(qs, v)
        print(f"{name} v={v}: viable(D>=40) {(Dv>=40).sum():>2}/16  Dmax {Dv.max():>2}  {Dv.tolist()}", flush=True)
