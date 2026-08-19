"""Decompose the value-law advantage: gate vs critic ranking.
  gated vertex actor : feasibility gate + actor-logit ranking (no critic)
  filtered cont actor: backtracking safety filter on the continuous action
"""
import sys, dataclasses, itertools
sys.path.insert(0,'/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot
import numpy as np, torch, yaml
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
N,B=10000,2048
env=NSRLBatchedEnv(EnvConfig(**{**kw,'n_envs':B}),None,dev)
model=hl.StraightModel(env); model.cfg=dataclasses.replace(env.cfg,dt=y['env']['dt']); model.terms=[0,1]
agV=_agent(REPO/hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
agC=ContAgent(env.obs_dim, env.act_dim).to(dev)
agC.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev)); agC.eval()
verts=torch.tensor(list(itertools.product([-1.,1.],repeat=env.act_dim)),dtype=torch.float32,device=dev)
K=verts.shape[0]; chunk=32768

@torch.no_grad()
def gated_vertex_actor(e,done):
    Bn=e.n_envs
    obs=e.current_obs()
    logits=agV._logits_head(agV._actor_trunk(obs))          # (B, 16)
    qe=e.q.repeat_interleave(K,0)
    ae=verts.unsqueeze(0).expand(Bn,-1,-1).reshape(Bn*K,-1)
    de=e.line_dir.repeat_interleave(K,0); ne=e.n_target.repeat_interleave(K,0); pe=e.p_start.repeat_interleave(K,0)
    qn=torch.cat([model.step(qe[i:i+chunk],de[i:i+chunk],ne[i:i+chunk],ae[i:i+chunk]) for i in range(0,Bn*K,chunk)])
    mg=torch.cat([model.margins(qn[i:i+chunk],pe[i:i+chunk],de[i:i+chunk],ne[i:i+chunk]) for i in range(0,Bn*K,chunk)])
    alive=(mg.amin(-1)>0).reshape(Bn,K)
    z=torch.where(alive, logits, torch.full_like(logits,-1e9))
    return verts[z.argmax(-1)]

BETAS=torch.tensor([1.0,0.5,0.25,0.125,0.0],device=dev)
NB=BETAS.shape[0]
@torch.no_grad()
def filtered_cont_actor(e,done):
    Bn=e.n_envs
    a=agC.actor_mean(e.current_obs())                        # (B, m)
    cand=(BETAS.view(1,NB,1)*a.unsqueeze(1)).reshape(Bn*NB,-1)
    qe=e.q.repeat_interleave(NB,0)
    de=e.line_dir.repeat_interleave(NB,0); ne=e.n_target.repeat_interleave(NB,0); pe=e.p_start.repeat_interleave(NB,0)
    qn=model.step(qe,de,ne,cand)
    mg=model.margins(qn,pe,de,ne)
    alive=(mg.amin(-1)>0).reshape(Bn,NB)
    # first feasible beta in descending order
    idx=torch.argmax(alive.float() * torch.arange(NB,0,-1,device=dev).view(1,NB), dim=-1)
    beta=BETAS[idx]
    return beta.unsqueeze(-1)*a

tz=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/tasks_pool_fr3.npz')
dt=env.kin.dtype
def roll(afn,tag):
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
    print(f'{tag} done {out.mean():.4f}', flush=True)
    return out
gv=roll(gated_vertex_actor,'gated vertex actor')
fc=roll(filtered_cont_actor,'filtered cont actor')
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/gated_actor_10k.npz', gated_vertex=gv, filtered_cont=fc)
A='/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
b=np.load(A+'bound_pool_fr3.npz'); w=np.load(A+'witness_pool_fr3.npz')
base=np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/pool_fr3_straight.npz')
ref=np.maximum(b['L_hi'],w['prog'])
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]: ref=np.maximum(ref,base[f'{a2}_progress'])
for v in (gv,fc): ref=np.maximum(ref,v)
def stat(v):
    rt=v/np.maximum(ref,1e-9)
    return f'{v.mean():.4f}  {rt.mean()*100:.1f} / {np.percentile(rt,10)*100:.1f}  t27 {v[27]:.3f}'
print('raw vertex actor   :', stat(base['vertex_progress']))
print('GATED vertex actor :', stat(gv))
print('vlook (gate+critic):', stat(base['vlook_progress']))
print('raw cont actor     :', stat(base['cont_progress']))
print('FILTERED cont actor:', stat(fc))
