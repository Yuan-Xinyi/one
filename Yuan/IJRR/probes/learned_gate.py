"""Learned feasibility gate: classifier (obs, a) -> alive?, trained on
exact-model labels from on-trajectory states; deployed replacing the gate."""
import sys, dataclasses, itertools, time
sys.path.insert(0,'/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot
import numpy as np, torch, yaml
import torch.nn as nn
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
env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':2048}),None,dev)
model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
ag=_agent(REPO/hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
verts=torch.tensor(list(itertools.product([-1.,1.],repeat=env.act_dim)),dtype=torch.float32,device=dev)
K=16; chunk=32768
W=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/witness_pool_fr3.npz')['W']
tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
rng=np.random.default_rng(5)
# ---- label generation: 60k on-trajectory states x 16 actions ----
NSAMP=60000
ts=rng.integers(0,10000,NSAMP)
ks=np.zeros(NSAMP,np.int64)
for i,t in enumerate(ts):
    ok=np.nonzero(~np.isnan(W[t,:,0]))[0]
    ks[i]=rng.choice(ok)
qs=torch.tensor(W[ts,ks],dtype=env.kin.dtype,device=dev)
ds=torch.tensor(tz['cs_line_dir'][ts],dtype=env.kin.dtype,device=dev)
ns_=torch.tensor(tz['cs_n_target'][ts],dtype=env.kin.dtype,device=dev)
OBS=[];LAB=[]
BB=2048
with torch.no_grad():
    for lo in range(0,NSAMP,BB):
        sl=slice(lo,min(lo+BB,NSAMP))
        q=qs[sl]; d=ds[sl]; n=ns_[sl]
        Bn=q.shape[0]
        qe=q.repeat_interleave(K,0); de=d.repeat_interleave(K,0); ne=n.repeat_interleave(K,0)
        ae=verts.unsqueeze(0).expand(Bn,-1,-1).reshape(Bn*K,-1)
        qn=model.step(qe,de,ne,ae)
        # p0 anchor: current TCP (labels for the lateral margin need an anchor;
        # use current TCP so lat measures the step's own drift)
        p_now,_ ,_,_=env.kin.tcp_fk_jac(q)
        pe=p_now.repeat_interleave(K,0)
        mg=model.margins(qn,pe,de,ne)
        alive=(mg.amin(-1)>0).float()
        a0=torch.zeros(Bn,env.act_dim,device=dev)
        obs=hl._obs_of(env,q,d,n,a0)             # state obs (31)
        OBS.append(obs.float().cpu()); LAB.append(alive.reshape(Bn,K).cpu())
X=torch.cat(OBS); Y=torch.cat(LAB)
print('labels:', X.shape, Y.shape, 'alive rate', float(Y.mean()), flush=True)
# ---- train classifier: obs -> 16 logits (one per vertex) ----
net=nn.Sequential(nn.Linear(31,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,16)).to(dev)
opt=torch.optim.Adam(net.parameters(),lr=1e-3)
ntr=50000
Xtr,Ytr,Xva,Yva=X[:ntr].to(dev),Y[:ntr].to(dev),X[ntr:].to(dev),Y[ntr:].to(dev)
for ep in range(30):
    perm=torch.randperm(ntr,device=dev)
    for i in range(0,ntr,4096):
        b=perm[i:i+4096]
        loss=nn.functional.binary_cross_entropy_with_logits(net(Xtr[b]),Ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pv=(net(Xva)>0).float()
    acc=(pv==Yva).float().mean()
    # false-permit rate: predicted alive but actually dead (the fatal error)
    fp=((pv==1)&(Yva==0)).float().sum()/ (Yva==0).float().sum()
    fn=((pv==0)&(Yva==1)).float().sum()/ (Yva==1).float().sum()
print(f'gate classifier: acc {acc:.4f}  false-permit {fp:.4f}  false-block {fn:.4f}', flush=True)
# ---- deploy: vlook with LEARNED gate ----
@torch.no_grad()
def vlook_learned_gate(e,done):
    Bn=e.n_envs
    a0=torch.zeros(Bn,e.act_dim,device=dev)
    obs=hl._obs_of(e,e.q,e.line_dir,e.n_target,a0).float()
    alive=(net(obs)>0)
    qe=e.q.repeat_interleave(K,0)
    ae=verts.unsqueeze(0).expand(Bn,-1,-1).reshape(Bn*K,-1)
    de=e.line_dir.repeat_interleave(K,0); ne=e.n_target.repeat_interleave(K,0)
    qn=torch.cat([model.step(qe[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],ae[i:i+chunk]) for i in range(0,Bn*K,chunk)])
    v=torch.cat([ag.critic(hl._obs_of(e,qn[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],
        torch.zeros(min(chunk,Bn*K-i),e.act_dim,device=dev))).squeeze(-1) for i in range(0,Bn*K,chunk)]).reshape(Bn,K)
    NEG=torch.full_like(v,-1e9)
    v2=torch.where(alive,v,NEG)
    pick=torch.where(alive.any(-1), v2.argmax(-1), v.argmax(-1))
    return verts[pick]
def run(afn,tag,N=10000,B=2048):
    out=np.zeros(N,np.float32)
    dt2=env.kin.dtype
    for lo in range(0,N,B):
        hi=min(lo+B,N); pad=B-(hi-lo)
        ids=np.arange(lo,hi)
        ip=np.concatenate([ids,np.full(pad,ids[0])]) if pad else ids
        env.line_dist=ScriptedLineDistribution({'q0':torch.tensor(tz['q0_seed'][ip],dtype=dt2,device=dev),
          'line_dir':torch.tensor(tz['cs_line_dir'][ip],dtype=dt2,device=dev),
          'n_target':torch.tensor(tz['cs_n_target'][ip],dtype=dt2,device=dev)})
        env.reset()
        done=torch.zeros(B,dtype=torch.bool,device=dev)
        for _ in range(env.cfg.max_steps//2):
            a=afn(env,done)
            for _ in range(2): env.step(a,auto_reset=False)
            done=env.done_persistent.clone()
            if bool(done.all()): break
        out[lo:hi]=env.arc_progress.float().cpu().numpy()[:hi-lo]
    print(f'{tag}: mean {out.mean():.4f}  t27 {out[27]:.3f}', flush=True)
    return out
r=run(vlook_learned_gate,'vlook w/ LEARNED gate')
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/learned_gate_10k.npz', prog=r)
base=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/pool_fr3_straight.npz')
print(f'exact-gate vlook   : {base["vlook_progress"].mean():.4f}  t27 {base["vlook_progress"][27]:.3f}')
torch.save(net.state_dict(),'/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/gate_classifier.pt')
