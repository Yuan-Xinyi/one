"""Dense per-dim grids vs vertices for cont-critic vlook (few trajectories)."""
import sys, dataclasses, itertools, time
sys.path.insert(0,'/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib.pyplot
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent
from pathlib import Path
REPO=Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
dev=torch.device('cuda'); hl.SUB=2
y=yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
kw={k:v for k,v in y['env'].items() if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt']/=2; kw['max_steps']=int(kw['max_steps']*2)
tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')

def run(task_ids, cands, tag):
    B=len(task_ids)
    env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':B}),None,dev)
    model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
    ag=ContAgent(env.obs_dim, env.act_dim).to(dev)
    ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev))
    ag.eval()
    verts=cands.to(dev); K=verts.shape[0]; chunk=32768
    @torch.no_grad()
    def fn(e,done):
        qe=e.q.repeat_interleave(K,0)
        ae=verts.unsqueeze(0).expand(B,-1,-1).reshape(B*K,-1)
        de=e.line_dir.repeat_interleave(K,0)
        ne=e.n_target.repeat_interleave(K,0)
        pe=e.p_start.repeat_interleave(K,0)
        best=torch.zeros(B,dtype=torch.long,device=dev)
        bestv=torch.full((B,),-1e9,device=dev)
        # chunk over candidates to bound memory: iterate candidate blocks
        for c0 in range(0,K,max(1,chunk//B)):
            c1=min(c0+max(1,chunk//B),K)
            kk=c1-c0
            qe2=e.q.repeat_interleave(kk,0)
            ae2=verts[c0:c1].unsqueeze(0).expand(B,-1,-1).reshape(B*kk,-1)
            de2=e.line_dir.repeat_interleave(kk,0)
            ne2=e.n_target.repeat_interleave(kk,0)
            pe2=e.p_start.repeat_interleave(kk,0)
            qn=model.step(qe2,de2,ne2,ae2)
            mg=model.margins(qn,pe2,de2,ne2)
            alive=(mg.amin(-1)>0).reshape(B,kk)
            v=ag.critic(hl._obs_of(e,qn,de2,ne2,ae2)).squeeze(-1).reshape(B,kk)
            v=torch.where(alive,v,torch.full_like(v,-1e9))
            m,idx=v.max(-1)
            upd=m>bestv
            best=torch.where(upd, idx+c0, best)
            bestv=torch.where(upd,m,bestv)
        return verts[best]
    dt2=env.kin.dtype
    env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][task_ids],dtype=dt2,device=dev),
      'line_dir':torch.tensor(tz['cs_line_dir'][task_ids],dtype=dt2,device=dev),
      'n_target':torch.tensor(tz['cs_n_target'][task_ids],dtype=dt2,device=dev)})
    env.reset()
    done=torch.zeros(B,dtype=torch.bool,device=dev)
    t0=time.time()
    for _ in range(env.cfg.max_steps//2):
        a=fn(env,done)
        for _ in range(2): env.step(a,auto_reset=False)
        done=env.done_persistent.clone()
        if bool(done.all()): break
    r=env.arc_progress.float().cpu().numpy()
    print(f'{tag:24s} K={K:6d}  n={B:4d}  mean {r.mean():.4f}  ({time.time()-t0:.0f}s)  '
          f'per-task: {np.round(r[:8],3)}', flush=True)
    return r

m=4
def grid(step):
    vals=np.arange(-1,1+1e-9,step)
    return torch.tensor(list(itertools.product(vals,repeat=m)),dtype=torch.float32)
V16=torch.tensor(list(itertools.product([-1.,1.],repeat=m)),dtype=torch.float32)

ids256=np.arange(256)
r={}
r['V16@256']=run(ids256, V16, 'V16 vertices')
r['G0.5@256']=run(ids256, grid(0.5), 'grid 0.5 (5^4=625)')
ids64=np.arange(64)
r['V16@64']=run(ids64, V16, 'V16 vertices (n=64)')
r['G0.25@64']=run(ids64, grid(0.25), 'grid 0.25 (9^4=6561)')
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/grid_probe.npz', **r)
print('GRID PROBE DONE')
