"""Two-step (viability) gate: a candidate counts as viable only if it is
alive AND at least one of its 16 grandchildren is alive. Rank viable
candidates by the critic; fall back to alive-only, then to raw argmax."""
import sys, dataclasses, itertools, time
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
B=2048
env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':B}),None,dev)
model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
ag=_agent(REPO/hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
verts=torch.tensor(list(itertools.product([-1.,1.],repeat=env.act_dim)),dtype=torch.float32,device=dev)
K=16; chunk=32768
tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
dt=env.kin.dtype

def batched_step(q,d,n,a):
    return torch.cat([model.step(q[i:i+chunk],d[i:i+chunk],n[i:i+chunk],a[i:i+chunk]) for i in range(0,q.shape[0],chunk)])
def batched_marg(q,p0,d,n):
    return torch.cat([model.margins(q[i:i+chunk],p0[i:i+chunk],d[i:i+chunk],n[i:i+chunk]) for i in range(0,q.shape[0],chunk)])

@torch.no_grad()
def vlook2(e,done):
    Bn=e.n_envs
    qe=e.q.repeat_interleave(K,0)
    ae=verts.unsqueeze(0).expand(Bn,-1,-1).reshape(Bn*K,-1)
    de=e.line_dir.repeat_interleave(K,0); ne=e.n_target.repeat_interleave(K,0); pe=e.p_start.repeat_interleave(K,0)
    qn=batched_step(qe,de,ne,ae)
    mg=batched_marg(qn,pe,de,ne)
    alive=(mg.amin(-1)>0).reshape(Bn,K)
    # grandchildren of every candidate (also for dead ones; masked later)
    qg=qn.repeat_interleave(K,0)                         # (B*K*K, 7)
    ag2=verts.unsqueeze(0).expand(Bn*K,-1,-1).reshape(Bn*K*K,-1)
    dg=de.repeat_interleave(K,0); ng=ne.repeat_interleave(K,0); pg=pe.repeat_interleave(K,0)
    qgg=batched_step(qg,dg,ng,ag2)
    mgg=batched_marg(qgg,pg,dg,ng)
    galive=(mgg.amin(-1)>0).reshape(Bn,K,K).any(-1)      # candidate has a living grandchild
    viable=alive & galive
    v=torch.cat([ag.critic(hl._obs_of(e,qn[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],
        torch.zeros(min(chunk,Bn*K-i),e.act_dim,device=dev))).squeeze(-1) for i in range(0,Bn*K,chunk)]).reshape(Bn,K)
    NEG=torch.full_like(v,-1e9)
    v2=torch.where(viable, v, NEG)
    has_v=viable.any(-1)
    v1=torch.where(alive, v, NEG)
    has_a=alive.any(-1)
    pick=torch.where(has_v, v2.argmax(-1), torch.where(has_a, v1.argmax(-1), v.argmax(-1)))
    return verts[pick]

def run(afn,tag):
    env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][:B],dtype=dt,device=dev),
      'line_dir':torch.tensor(tz['cs_line_dir'][:B],dtype=dt,device=dev),
      'n_target':torch.tensor(tz['cs_n_target'][:B],dtype=dt,device=dev)})
    env.reset()
    done=torch.zeros(B,dtype=torch.bool,device=dev)
    t0=time.time()
    for _ in range(env.cfg.max_steps//2):
        a=afn(env,done)
        for _ in range(2): env.step(a,auto_reset=False)
        done=env.done_persistent.clone()
        if bool(done.all()): break
    r=env.arc_progress.float().cpu().numpy().copy()
    print(f'{tag:24s} mean {r.mean():.4f}  t27 {r[27]:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    return r
def run10k(afn,tag):
    N=10000
    out=np.zeros(N,np.float32)
    for lo in range(0,N,B):
        hi=min(lo+B,N); pad=B-(hi-lo)
        ids=np.arange(lo,hi)
        ip=np.concatenate([ids,np.full(pad,ids[0])]) if pad else ids
        env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][ip],dtype=dt,device=dev),
          'line_dir':torch.tensor(tz['cs_line_dir'][ip],dtype=dt,device=dev),
          'n_target':torch.tensor(tz['cs_n_target'][ip],dtype=dt,device=dev)})
        env.reset()
        done=torch.zeros(B,dtype=torch.bool,device=dev)
        for _ in range(env.cfg.max_steps//2):
            a=afn(env,done)
            for _ in range(2): env.step(a,auto_reset=False)
            done=env.done_persistent.clone()
            if bool(done.all()): break
        out[lo:hi]=env.arc_progress.float().cpu().numpy()[:hi-lo]
        print(f'{tag} {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
    return out
r2=run10k(vlook2,'vlook2')
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/gate2_10k.npz', g2=r2)
base=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/pool_fr3_straight.npz')
d=r2-base['vlook_progress']
print(f'10k: vlook2 {r2.mean():.4f} vs vlook {base["vlook_progress"].mean():.4f}  delta {d.mean():+.4f}  improved {(d>0.01).mean()*100:.1f}%  hurt {(d<-0.01).mean()*100:.1f}%  t27 {r2[27]:.3f}')
