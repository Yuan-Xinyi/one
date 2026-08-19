"""Death forensics for vlook: how does the alive-candidate count evolve
in the last steps before death? Cliff (many->0) vs gradual (->1->0)."""
import sys, dataclasses, itertools
sys.path.insert(0,'/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent
from pathlib import Path
REPO=Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
dev=torch.device('cuda'); hl.SUB=2
y=yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
kw={k:v for k,v in y['env'].items() if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt']/=2; kw['max_steps']=int(kw['max_steps']*2)
B=1024
env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':B}),None,dev)
model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
ag=_agent(REPO/hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
verts=torch.tensor(list(itertools.product([-1.,1.],repeat=env.act_dim)),dtype=torch.float32,device=dev)
K=16; chunk=32768
tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
dt=env.kin.dtype
env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][:B],dtype=dt,device=dev),
  'line_dir':torch.tensor(tz['cs_line_dir'][:B],dtype=dt,device=dev),
  'n_target':torch.tensor(tz['cs_n_target'][:B],dtype=dt,device=dev)})
env.reset()
done=torch.zeros(B,dtype=torch.bool,device=dev)
HIST=8
alive_hist=torch.zeros(B,HIST,device=dev)   # rolling: alive counts of last HIST steps
death_snapshot=np.full((B,HIST),-1,np.int16)
prev_done=done.clone()
@torch.no_grad()
def vstep(e):
    Bn=e.n_envs
    qe=e.q.repeat_interleave(K,0)
    ae=verts.unsqueeze(0).expand(Bn,-1,-1).reshape(Bn*K,-1)
    de=e.line_dir.repeat_interleave(K,0); ne=e.n_target.repeat_interleave(K,0); pe=e.p_start.repeat_interleave(K,0)
    qn=torch.cat([model.step(qe[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],ae[i:i+chunk]) for i in range(0,Bn*K,chunk)])
    mg=torch.cat([model.margins(qn[i:i+chunk],pe[i:i+chunk],de[i:i+chunk],ne[i:i+chunk]) for i in range(0,Bn*K,chunk)])
    alive=(mg.amin(-1)>0).reshape(Bn,K)
    v=torch.cat([ag.critic(hl._obs_of(e,qn[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],
        torch.zeros(min(chunk,Bn*K-i),e.act_dim,device=dev))).squeeze(-1) for i in range(0,Bn*K,chunk)]).reshape(Bn,K)
    v=torch.where(alive,v,torch.full_like(v,-1e9))
    return verts[v.argmax(-1)], alive.sum(-1).float()
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a,cnt=vstep(env)
        alive_hist=torch.cat([alive_hist[:,1:],cnt.unsqueeze(1)],1)
        for _ in range(2): env.step(a,auto_reset=False)
        done=env.done_persistent.clone()
        newly=(done & ~prev_done).cpu().numpy()
        if newly.any():
            death_snapshot[newly]=alive_hist[torch.tensor(newly,device=dev)].cpu().numpy().astype(np.int16)
        prev_done=done.clone()
        if bool(done.all()): break
d=death_snapshot[death_snapshot[:,0]>=0]
print('deaths captured:', len(d))
print('alive-count trajectory over the last 8 decisions before death (median / mean):')
for i in range(HIST):
    col=d[:,i]
    print(f'  t-{HIST-1-i}: median {np.median(col):.0f}  mean {col.mean():.2f}')
last=d[:,-1]
print('at the final decision: alive==0:', (last==0).mean().round(3), ' alive==1:', (last==1).mean().round(3),
      ' alive>=8:', (last>=8).mean().round(3))
cliff=(d[:,-2]>=6)&(d[:,-1]<=1)
print('cliff deaths (>=6 alive at t-1 -> <=1 at t):', cliff.mean().round(3))
np.save('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/death_alive_hist.npy', d)
