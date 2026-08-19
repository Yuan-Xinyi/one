"""Policy iteration on the critic, no policy gradients anywhere:
round r: collect eps-noised rollouts of the CURRENT value-lookahead
controller on fresh pool tasks (training protocol, 50 ms steps), refit a
copy of the critic on the realized discounted returns (affine-matched to
the critic's output scale), rebuild the controller, score it on the
horizon ladder (1024 tasks, paper protocol SUB=2). Keep the best round.
"""
import sys, time, copy, subprocess, re
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn as nn, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

hl.SUB = 1                                   # collection at the training protocol
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
GAMMA = float(y['ppo']['gamma'])
OUT = REPO / 'Yuan/IJRR/runs/polit'
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / 'polit.log', 'a')


def say(m):
    print(m, flush=True)
    log.write(m + '\n'); log.flush()


B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': B}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
# the ladder consumes valid[:1024]; collect strictly beyond it
COLLECT_BASE = 2048
model = hl.StraightModel(env)
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * env.act_dim,
                     indexing='ij'), -1).reshape(-1, env.act_dim),
                     dtype=torch.float32, device=dev)


@torch.no_grad()
def collect(critic, task_ids, eps_levels=(0.0, 0.1, 0.25),
            eps_frac=(0.5, 0.3, 0.2)):
    """Roll eps-noised vlook on the given pool tasks; return (obs, ret)."""
    OBS_l, RET_l = [], []
    for base in range(0, len(task_ids), B):
        ids = task_ids[base:base + B]
        pad = B - len(ids)
        ids_p = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        ii = torch.as_tensor(ids_p, device=dev)
        env.line_dist = ScriptedLineDistribution(
            {'q0': pool.q_pool[valid[ii]].to(dev),
             'line_dir': pool.line_dir_pool[valid[ii]].to(dev),
             'n_target': pool.n_target_pool[valid[ii]].to(dev)})
        env.reset()
        eps = torch.as_tensor(
            np.random.default_rng(base).choice(eps_levels, B, p=eps_frac),
            device=dev, dtype=torch.float32)
        obs_t, rew_t, alive_t = [], [], []
        for t in range(env.max_steps):
            live = ~env.done_persistent
            if not bool(live.any()):
                break
            obs_t.append(env.current_obs().clone())
            alive_t.append(live.clone())
            qe = env.q.repeat_interleave(16, 0)
            ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(-1, 4)
            de = env.line_dir.repeat_interleave(16, 0)
            ne = env.n_target.repeat_interleave(16, 0)
            pe = env.p_start.repeat_interleave(16, 0)
            CH = 32768
            qn = torch.cat([model.step(qe[i:i + CH], de[i:i + CH],
                                       ne[i:i + CH], ae[i:i + CH])
                            for i in range(0, B * 16, CH)])
            mg = torch.cat([model.margins(qn[i:i + CH], pe[i:i + CH],
                                          de[i:i + CH], ne[i:i + CH])
                            for i in range(0, B * 16, CH)])
            aliveA = (mg.amin(-1) > 0).reshape(B, 16)
            v = torch.cat([critic(hl._obs_of(env, qn[i:i + CH],
                                             de[i:i + CH], ne[i:i + CH],
                                             ae[i:i + CH])).squeeze(-1)
                           for i in range(0, B * 16, CH)]).reshape(B, 16)
            vm = torch.where(aliveA, v, torch.full_like(v, -1e9))
            act = vm.argmax(-1)
            # eps-noise: random ALIVE successor
            r = torch.rand(B, device=dev)
            noise = (r < eps) & aliveA.any(-1)
            if bool(noise.any()):
                w = aliveA.float() + 1e-9
                rnd_act = torch.multinomial(w, 1).squeeze(-1)
                act = torch.where(noise, rnd_act, act)
            _, rew, _, _, _ = env.step(verts[act], auto_reset=False)
            rew_t.append(rew.clone())
        T = len(rew_t)
        RW = torch.stack(rew_t)                 # (T, B)
        AL = torch.stack(alive_t)
        OBSt = torch.stack(obs_t)
        ret = torch.zeros_like(RW)
        acc = torch.zeros(B, device=dev)
        for t in range(T - 1, -1, -1):
            acc = RW[t] + GAMMA * acc
            ret[t] = acc
        keep = AL.clone()
        if pad:
            keep[:, len(ids):] = False
        OBS_l.append(OBSt[keep])
        RET_l.append(ret[keep])
    return torch.cat(OBS_l).float(), torch.cat(RET_l).float()


def refit(critic0, OBS, RET, steps=20000, lr=1e-4):
    crit = copy.deepcopy(critic0)
    with torch.no_grad():
        pred0 = torch.cat([crit(OBS[i:i + 65536]).squeeze(-1)
                           for i in range(0, OBS.shape[0], 65536)])
    tgt = ((RET - RET.mean()) / (RET.std() + 1e-8)
           * pred0.std() + pred0.mean())       # affine match, rank-preserving
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    n = OBS.shape[0]
    hold = torch.arange(n, device=dev) % 10 == 0
    best_r2, best_state, bad = -1e9, None, 0
    for ep in range(steps):
        idx = torch.randint(0, n, (4096,), device=dev)
        m = ~hold[idx]
        loss = nn.functional.smooth_l1_loss(
            crit(OBS[idx][m]).squeeze(-1), tgt[idx][m])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 1000 == 999:
            with torch.no_grad():
                hp = torch.cat([crit(OBS[hold][i:i + 65536]).squeeze(-1)
                                for i in range(0, int(hold.sum()), 65536)])
                r2 = float(1 - ((hp - tgt[hold]) ** 2).mean()
                           / tgt[hold].var())
            if r2 > best_r2 + 1e-4:
                best_r2, bad = r2, 0
                best_state = copy.deepcopy(crit.state_dict())
            else:
                bad += 1
                if bad >= 3:
                    break
    crit.load_state_dict(best_state)
    return crit.eval(), best_r2


def ladder_eval(critic, tag):
    p = OUT / f'critic_{tag}.pt'
    torch.save(critic.state_dict(), p)
    r = subprocess.run(
        [sys.executable, '-m', 'Yuan.IJRR.eval.horizon_ladder',
         '--arms', 'vlook', '--n-tasks', '1024', '--sub', '2',
         '--vlook-value', str(p.relative_to(REPO))],
        capture_output=True, text=True, cwd=REPO)
    m = re.search(r'vlook\s+ratio to classical\s+([\d.]+)', r.stdout)
    mm = re.search(r'vlook\s+mean progress ([\d.]+)', r.stdout)
    if not m:
        say(f'[polit] ladder FAILED for {tag}: {r.stdout[-400:]} '
            f'{r.stderr[-400:]}')
        return None, None
    return float(m.group(1)), float(mm.group(1))


say(f"[polit] start; gamma {GAMMA}; baseline vlook (PPO critic) = "
    f"1.9179 x classical / 0.5644 m on this ladder")
critic = copy.deepcopy(ag.critic).eval()
best = (0, 1.9179)
N_TASKS, ROUNDS = 6144, 4
for rnd in range(1, ROUNDS + 1):
    t0 = time.time()
    ids = np.arange(COLLECT_BASE + (rnd - 1) * N_TASKS,
                    COLLECT_BASE + rnd * N_TASKS)
    OBS, RET = collect(critic, ids)
    say(f"[polit] r{rnd}: collected {OBS.shape[0]} states from "
        f"{N_TASKS} tasks ({time.time()-t0:.0f}s)")
    critic_new, r2 = refit(critic, OBS, RET)
    say(f"[polit] r{rnd}: refit holdout R2 {r2:.3f} "
        f"({time.time()-t0:.0f}s)")
    ratio, prog = ladder_eval(critic_new, f'r{rnd}')
    say(f"[polit] r{rnd}: ladder vlook = {ratio} x classical "
        f"({prog} m)  [best so far {best[1]}]  "
        f"({time.time()-t0:.0f}s)")
    if ratio is None:
        break
    if ratio > best[1]:
        best = (rnd, ratio)
        critic = critic_new                   # iterate from the improvement
    else:
        say(f"[polit] r{rnd}: no improvement; iterating from previous best")
say(f"[polit] done: best round {best[0]} ratio {best[1]}")
log.close()
