"""Probing-scale SNR test: are critic value differences between nearby
candidates too small to rank reliably? Ground truth = full rollout
continuation from every candidate successor."""
import sys, dataclasses, itertools, time
sys.path.insert(0,'/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot
import numpy as np, torch, yaml
from scipy.stats import spearmanr
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent
from Yuan.IJRR.eval.eval_curve import _agent
from pathlib import Path
REPO=Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
dev=torch.device('cuda'); hl.SUB=2
y=yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
kw={k:v for k,v in y['env'].items() if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt']/=2; kw['max_steps']=int(kw['max_steps']*2)
NS=128           # probe states
K=16
env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':2048}),None,dev)
model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
agV=_agent(REPO/hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
agC=ContAgent(env.obs_dim, env.act_dim).to(dev)
agC.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev)); agC.eval()

# --- probe states from real trajectories (witness grid) ---
W=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/witness_pool_fr3.npz')['W']
tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
rng=np.random.default_rng(3)
states=[]
while len(states)<NS:
    t=int(rng.integers(0,10000))
    ok=np.nonzero(~np.isnan(W[t,:,0]))[0]
    if len(ok)<10: continue
    k=int(rng.choice(ok[2:]))          # skip the very start
    states.append((t,k))
qs=torch.tensor(np.stack([W[t,k] for t,k in states]),dtype=env.kin.dtype,device=dev)
ds=torch.tensor(np.stack([tz['cs_line_dir'][t] for t,_ in states]),dtype=env.kin.dtype,device=dev)
ns=torch.tensor(np.stack([tz['cs_n_target'][t] for t,_ in states]),dtype=env.kin.dtype,device=dev)
verts=torch.tensor(list(itertools.product([-1.,1.],repeat=env.act_dim)),dtype=torch.float32,device=dev)

def continuation(q0batch, dbatch, nbatch):
    B=env.n_envs; n=q0batch.shape[0]
    out=np.zeros(n,np.float32)
    for lo in range(0,n,B):
        hi=min(lo+B,n); pad=B-(hi-lo)
        sl=slice(lo,hi)
        def padd(t):
            x=t[sl]
            return torch.cat([x,x[-1:].expand(pad,*x.shape[1:])]) if pad else x
        env.line_dist=ScriptedLineDistribution({'q0':padd(q0batch),'line_dir':padd(dbatch),'n_target':padd(nbatch)})
        env.reset()
        vfn=hl.make_vlook(model,env,agV)
        done=torch.zeros(B,dtype=torch.bool,device=dev)
        for _ in range(env.cfg.max_steps//2):
            a=vfn(env,done)
            for _ in range(2): env.step(a,auto_reset=False)
            done=env.done_persistent.clone()
            if bool(done.all()): break
        out[lo:hi]=env.arc_progress.float().cpu().numpy()[:hi-lo]
    return out

print('scale | dV(top1-top2) medn | Spearman(V,true) | regret[m] med | topA(V-critic)=true% | Vcrit~Ccrit top1 agree%', flush=True)
RES={}
for s in (0.05, 0.1, 0.25, 0.5, 1.0):
    cand=(s*verts)                          # (K, m)
    qe=qs.repeat_interleave(K,0)
    de=ds.repeat_interleave(K,0); ne=ns.repeat_interleave(K,0)
    ae=cand.unsqueeze(0).expand(NS,-1,-1).reshape(NS*K,-1)
    with torch.no_grad():
        qn=model.step(qe,de,ne,ae)
        a0=torch.zeros(NS*K,env.act_dim,device=dev)
        obs=hl._obs_of(env,qn,de,ne,a0)
        vV=agV.critic(obs).squeeze(-1).reshape(NS,K).float().cpu().numpy()
        vC=agC.critic(obs).squeeze(-1).reshape(NS,K).float().cpu().numpy()
    true=continuation(qn,de,ne).reshape(NS,K)
    top2=np.sort(vV,axis=1)[:,-2:]
    dv=top2[:,1]-top2[:,0]
    rho=np.array([spearmanr(vV[i],true[i]).statistic for i in range(NS)])
    pick=vV.argmax(1); best=true.argmax(1)
    regret=true.max(1)-true[np.arange(NS),pick]
    agree=(vV.argmax(1)==vC.argmax(1)).mean()
    hit=(true[np.arange(NS),pick]>=true.max(1)-0.02).mean()
    RES[s]=(dv,rho,regret)
    print(f'{s:5.2f} | {np.median(dv):.4f} | {np.nanmedian(rho):.3f} | {np.median(regret):.3f} | {hit*100:5.1f}% | {agree*100:5.1f}%', flush=True)
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/snr_probe.npz',
         **{f's{s}_{n}':v for s,(dv,rho,rg) in RES.items() for n,v in (('dv',dv),('rho',rho),('regret',rg))})
print('SNR PROBE DONE')
