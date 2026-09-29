"""Real-time closed-loop execution of the xArm7 flagship controller.

The policy is a state-feedback law q_dot = pi(q): every cycle the measured
joint angles are written into a one-env copy of the training environment,
the policy is queried, and the joint velocity the environment would have
integrated is streamed to the arm as 100 Hz servo set-points through the
one control interface (one.control.manipulators.xarm7.XArm7X, xArm mode 1).  The
environment's own step() computes the command, so projection, amplitude
bound, lateral feedback and the predictive termination test are exactly the
paper's; nothing is re-implemented here.  The policy tolerates any decision
period (5 to 50 ms give the same stroke in simulation), but one env.step on
a single configuration costs ~17 ms on the CPU (the task-aligned basis and
its autograd gradients dominate), so the default keeps the training period
of 50 ms task time; a uniform time scale then slows the whole motion without
changing the joint path (0.5 -> 100 ms wall per cycle).

    # closed loop from the current configuration, path direction +y, cone
    # centred on the current tool axis, stop after 0.6 m or at a violation
    python -m Yuan.IJRR.deploy.xarm7_realtime run --ip 192.168.1.203 \
        --dir 0 1 0 --stroke 0.6 --time-scale 0.5 --log run1.npz

    # same task, but first pick the start with the critic (paper protocol:
    # cone-IK candidates from the xArm7 table, critic at the reset
    # observation), move there in position mode, then run
    python -m Yuan.IJRR.deploy.xarm7_realtime select --ip 192.168.1.203 \
        --p0 0.45 0.0 0.30 --dir 0 1 0 --normal 0 0 -1 --stroke 0.6

    # no hardware: a simulated arm with a first-order velocity lag stands in
    # for the SDK (start from task 17 of the 10k pool)
    python -m Yuan.IJRR.deploy.xarm7_realtime run --sim --sim-task 17 \
        --dir 0 1 0 --stroke 0.6 --time-scale 1.0

Safety: every command is bounded per joint by the training velocity limits
through alpha_feas, then scaled by --time-scale and capped by --qd-cap; the
velocity command carries a timeout of three cycles so the arm stops if the
loop dies; the loop stops on the environment's predictive termination, on a
measured-state violation, on the stroke target, on Ctrl-C, and on any
exception, always by sending zero velocity and returning to position mode.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from pathlib import Path

import matplotlib  # noqa: F401  (import before torch: conda libstdc++)
matplotlib.use('Agg')
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, TERM_NAMES,  # noqa: E402
                               LATERAL_SAFETY_NET)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution  # noqa: E402
from Yuan.IJRR.stage2_traj.ppo import Agent  # noqa: E402

ASSETS = Path(os.environ.get('IJRR_ASSETS',
                             '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets'))
CFG = REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml'
# The xArm7 flagship (2026-09-02, 240M steps, 8192 envs; the paper's xArm7
# rows). Trained runs live in the research worktree, so look there too.
_CKPT_REL = 'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt'
_CKPT_CANDIDATES = [
    Path(os.environ['IJRR_CKPT']) if 'IJRR_CKPT' in os.environ else None,
    REPO / _CKPT_REL,
    Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05') / _CKPT_REL,
]
CKPT = next(p for p in _CKPT_CANDIDATES if p is not None and p.exists())
N_J = 7
DEFAULT_IP = os.environ.get('ONE_ARM_IP', '192.168.1.205')
# Tool presets: point of the tool tip in the flange frame (tool axis = flange z).
# xhand_index: XHand mounted with Rz(270 deg), hand open (all finger joints 0),
# index fingertip = far end of index_rota_link2, from the hand model.
TOOLS = {
    'pen': (0.0, 0.0, 0.10),
    'xhand_index': (-0.0065, -0.0265, 0.206),
}


def tool_xyz_from_args(args):
    if getattr(args, 'tool_xyz', None):
        return tuple(float(v) for v in args.tool_xyz)
    if getattr(args, 'tool', None):
        return TOOLS[args.tool]
    return (0.0, 0.0, float(args.tcp))


def unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


# ------------------------------------------------------------------ arms
class XArm:
    """Hardware through the one control interface (one.control.manipulators
    .xarm7.XArm7X): joints via get_jnt_values, point-to-point via move_j,
    and the closed loop as servo streaming (xArm mode 1): each control cycle
    the commanded joint velocity is integrated into set-points streamed at
    100 Hz with servo_j, starting from the last commanded set-point so the
    stream stays continuous.  The wrapper blocks for the cycle (streams=True)."""
    streams = True
    STREAM_HZ = 100.0
    TRACK_TOL = 0.08          # rad: commanded set-point vs measured joints

    def __init__(self, ip: str):
        from one.control.manipulators.xarm7.xarm7 import XArm7X
        self.x = XArm7X(ip=ip)
        self._q_cmd = None

    def q(self) -> np.ndarray:
        q = np.asarray(self.x.get_jnt_values(), np.float64)[:N_J]
        if np.abs(q).max() > 2.0 * math.pi + 0.1:
            raise RuntimeError(f'joint readout looks like degrees, not radians: {q}')
        return q

    def velocity_mode(self):
        self._q_cmd = self.q()
        self.x.enter_servo_mode()

    def send_qdot(self, qdot: np.ndarray, timeout: float):
        """Stream set-points for one cycle (= timeout / 3) at STREAM_HZ."""
        period = float(timeout) / 3.0
        q_meas = self.q()
        if self._q_cmd is None:
            self._q_cmd = q_meas
        if np.abs(self._q_cmd - q_meas).max() > self.TRACK_TOL:
            raise RuntimeError('servo tracking error above tolerance: commanded '
                               f'{self._q_cmd.round(3)} measured {q_meas.round(3)}')
        n = max(1, int(round(period * self.STREAM_HZ)))
        dt = period / n
        next_t = time.perf_counter()
        for _ in range(n):
            self._q_cmd = self._q_cmd + np.asarray(qdot, np.float64) * dt
            code = self.x.servo_j(self._q_cmd)
            if code != 0:
                raise RuntimeError(f'servo_j failed, code {code}')
            next_t += dt
            s = next_t - time.perf_counter()
            if s > 0:
                time.sleep(s)

    def stop(self):
        # leaving servo mode holds the last set-point; position mode = idle
        self.x.enter_position_mode()
        self._q_cmd = None

    def move_to(self, q: np.ndarray, speed: float = 0.3):
        self.x.move_j(np.asarray(q, np.float64), speed=speed, wait=True)

    def close(self):
        try:
            self.x.enter_position_mode()
        except Exception:               # noqa: BLE001
            pass


class SimArm:
    """Stand-in for the SDK: integrates the commanded velocity through a
    first-order lag (time constant --sim-lag) at 1 kHz."""

    def __init__(self, q0: np.ndarray, lag: float = 0.03):
        self._q = np.asarray(q0, np.float64).copy()
        self._v = np.zeros(N_J)
        self._cmd = np.zeros(N_J)
        self.lag = float(lag)
        self._t = time.perf_counter()

    def _advance(self):
        now = time.perf_counter()
        n = max(int((now - self._t) * 1000), 0)
        h = 0.001
        a = 1.0 - math.exp(-h / self.lag) if self.lag > 0 else 1.0
        for _ in range(n):
            self._v += (self._cmd - self._v) * a
            self._q += self._v * h
        self._t += n * h

    def q(self):
        self._advance()
        return self._q.copy()

    def velocity_mode(self):
        pass

    def send_qdot(self, qdot, timeout):
        self._advance()
        self._cmd = np.asarray(qdot, np.float64).copy()

    def stop(self):
        self._advance()
        self._cmd[:] = 0.0
        self._v[:] = 0.0

    def move_to(self, q, speed=0.3):
        self._advance()
        self._q = np.asarray(q, np.float64).copy()
        self._v[:] = 0.0
        self._cmd[:] = 0.0

    def close(self):
        pass


# ------------------------------------------------------------ controller
class Controller:
    """One-env copy of the training environment driven by measured joints."""

    def __init__(self, period: float, tcp, cone_deg: float,
                 k_lateral: float, device: str = 'cuda', n_envs: int = 1):
        """tcp: on-axis tool length, or an (x, y, z) tool point in the
        flange frame."""
        y = yaml.safe_load(open(CFG))
        keys = {f.name for f in dataclasses.fields(EnvConfig)}
        kw = {k: v for k, v in y['env'].items() if k in keys}
        tool = (tuple(float(v) for v in tcp) if hasattr(tcp, '__len__')
                else (0.0, 0.0, float(tcp)))
        kw.update(dt=float(period), max_steps=10 ** 7, tcp_offset=tool[2],
                  tool_xyz=tool, cone_deg=float(cone_deg),
                  k_lateral=float(k_lateral), n_envs=n_envs)
        self.tool_xyz = tool
        self.dev = torch.device(device)
        self.env = NSRLBatchedEnv(EnvConfig(**kw), None, self.dev)
        self.rdt = self.env.kin.dtype
        self.agent = Agent(self.env.obs_dim, self.env.act_dim_policy,
                           hidden_dim=y['ppo']['hidden_dim']).to(self.dev)
        self.agent.load_state_dict(torch.load(CKPT, map_location=self.dev))
        self.agent.eval()
        self.cos_cone = math.cos(math.radians(cone_deg))
        self.qd_limit = np.asarray(y['env']['qd_limit'], np.float64)

    def _t(self, x):
        return torch.as_tensor(np.asarray(x, np.float64), dtype=self.rdt,
                               device=self.dev)

    def fk(self, q: np.ndarray):
        p, R, _, _ = self.env.kin.tcp_fk_jac(self._t(q)[None])
        return p[0].cpu().numpy(), R[0].cpu().numpy()

    def reset(self, q_meas: np.ndarray, d: np.ndarray, n: np.ndarray,
              p0: np.ndarray | None = None):
        spec = {'q0': self._t(q_meas)[None], 'line_dir': self._t(d)[None],
                'n_target': self._t(n)[None]}
        if p0 is not None:
            spec['p0'] = self._t(p0)[None]
        self.env.line_dist = ScriptedLineDistribution(spec)
        self.env.reset()
        self.d, self.n = unit(d), unit(n)
        self.p_start = self.env.p_start[0].cpu().numpy()

    @torch.no_grad()
    def command(self, q_meas: np.ndarray):
        """Joint velocity the environment would integrate from q_meas, plus
        the environment's own (predictive) termination verdict."""
        self.env.q[0] = self._t(q_meas)
        obs = self.env.current_obs()
        a = self.agent.actor_mean(obs)
        q_before = self.env.q.clone()
        _, _, _, _, info = self.env.step(a, auto_reset=False)
        qdot = ((self.env.q - q_before) / self.env.dt)[0].cpu().numpy()
        done = bool(self.env.done_persistent[0].item())
        reason = TERM_NAMES[int(info['term_reason'][0].item())]
        return qdot.astype(np.float64), done, reason

    def measured_state(self, q_meas: np.ndarray):
        """Progress, lateral error, cone angle and the measured-state
        violations (the same constraint set, evaluated on the real state)."""
        qt = self._t(q_meas)[None]
        p, R, _, _ = self.env.kin.tcp_fk_jac(qt)
        p, z = p[0].cpu().numpy(), R[0, :, 2].cpu().numpy()
        delta = p - self.p_start
        prog = float(delta @ self.d)
        lat = float(np.linalg.norm(delta - prog * self.d))
        cosang = float(z @ self.n)
        coll = bool(self.env.collision.is_collided(
            self.env.kin.link_transforms(qt))[0].item())
        lo = self.env.kin.lmt_lo.cpu().numpy()
        up = self.env.kin.lmt_up.cpu().numpy()
        jl = bool(((q_meas < lo) | (q_meas > up)).any())
        viol = []
        if coll:
            viol.append('collision')
        if jl:
            viol.append('joint limit')
        if cosang < self.cos_cone:
            viol.append('cone')
        if lat > LATERAL_SAFETY_NET:
            viol.append('lateral')
        return dict(p=p, progress=prog, lateral=lat,
                    tilt_deg=math.degrees(math.acos(max(-1.0, min(1.0, cosang)))),
                    violations=viol)

    @torch.no_grad()
    def critic_pick(self, cands: np.ndarray, d, n, p0):
        """Paper protocol: value of every candidate at its reset observation."""
        B = len(cands)
        env = NSRLBatchedEnv(dataclasses.replace(self.env.cfg, n_envs=B), None,
                             self.dev)
        env.line_dist = ScriptedLineDistribution({
            'q0': self._t(cands), 'line_dir': self._t(np.tile(unit(d), (B, 1))),
            'n_target': self._t(np.tile(unit(n), (B, 1))),
            'p0': self._t(np.tile(p0, (B, 1)))})
        V = self.agent.get_value(env.reset()).float().cpu().numpy().reshape(-1)
        return int(V.argmax()), V


# --------------------------------------------------------- start selection
def cone_ik_candidates(ctrl: Controller, p0, d, n, cone_deg, n_dirs=8,
                       k_nn=200, seed=0):
    """Admissible start configurations at p0: cone directions x warm starts
    from the xArm7 FK table, projected by the same IK as the paper."""
    from scipy.spatial import cKDTree
    from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
    from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
    from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
    env, dev, rdt = ctrl.env, ctrl.dev, ctrl.rdt
    T = np.load(ASSETS / 'fk_table_xarm7.npz')
    T = {k: T[k] for k in T.files}
    if np.linalg.norm(np.asarray(ctrl.tool_xyz) - np.asarray(TOOLS['pen'])) > 1e-6:
        # the table stores tip positions for the on-axis pen; re-evaluate the
        # stored configurations with the actual tool (seconds on the GPU)
        pos = np.empty_like(T['pos'])
        for lo in range(0, len(T['q']), 65536):
            qq = torch.as_tensor(T['q'][lo:lo + 65536], device=dev, dtype=rdt)
            pos[lo:lo + 65536] = env.kin.tcp_fk_jac(qq)[0].cpu().numpy()
        T['pos'] = pos
    tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
                   .astype(np.float32))
    n = unit(n).astype(np.float32)
    dirs = [n] + list(_sample_in_cone(torch.as_tensor(n), cone_deg, n_dirs - 1,
                                      np.random.default_rng(seed)).numpy())
    hint = torch.tensor([1.0, 0.0, 0.0], dtype=rdt, device=dev)
    cos_lim = math.cos(math.radians(cone_deg))
    p0 = np.asarray(p0, np.float32)
    sols = []
    for z in dirs:
        feat = np.concatenate([p0 * POS_SCALE, z]).astype(np.float32)
        _, ids = tree.query(feat, k=k_nn)
        fq = torch.as_tensor(T['q'][ids], device=dev, dtype=rdt)
        fp = torch.as_tensor(np.tile(p0, (k_nn, 1)), device=dev, dtype=rdt)
        fz = torch.as_tensor(np.tile(z, (k_nn, 1)), device=dev, dtype=rdt)
        q_o, _, _ = _batched_ik_project(env.kin, fq, fp,
                                        _build_R_with_z(fz, hint),
                                        branch_action=None)
        coll = env.collision.is_collided(env.kin.link_transforms(q_o))
        p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
        ok = ((~coll)
              & ((q_o >= env.kin.lmt_lo - 1e-5) & (q_o <= env.kin.lmt_up + 1e-5)).all(-1)
              & ((p_fk - fp).norm(dim=-1) <= LATERAL_SAFETY_NET)
              & ((R_fk[:, :, 2] * torch.as_tensor(n, device=dev, dtype=rdt)).sum(-1) >= cos_lim))
        sols.append(q_o[ok].cpu().numpy())
    Q = np.concatenate(sols) if sols else np.zeros((0, N_J))
    Q = np.unique(np.round(Q, 3), axis=0)
    return Q.astype(np.float64)


# --------------------------------------------------------------- the loop
def run_loop(arm, ctrl: Controller, d, n, stroke, time_scale, qd_cap,
             log_path, p0=None, max_wall=120.0, on_cycle=None):
    """Closed loop until the stroke target, a violation, Ctrl-C or max_wall.
    on_cycle(q, state, cmd) is called once per cycle (e.g. for a live view)."""
    q = arm.q()
    ctrl.reset(q, d, n, p0)
    st = ctrl.measured_state(q)
    if st['violations']:
        raise RuntimeError(f'start state violates: {st["violations"]}')
    wall_period = ctrl.env.dt / time_scale
    print(f'[rt] start tip {st["p"].round(3)}, tilt {st["tilt_deg"]:.1f} deg, '
          f'cycle {ctrl.env.dt * 1000:.0f} ms task time = '
          f'{wall_period * 1000:.0f} ms wall, time scale {time_scale}',
          flush=True)
    log = {k: [] for k in ('t', 'q', 'qdot_cmd', 'p', 'progress', 'lateral',
                           'tilt_deg')}
    reason = 'stroke target'
    n_over = 0
    arm.velocity_mode()
    t0 = time.perf_counter()
    next_t = t0
    try:
        while True:
            q = arm.q()
            st = ctrl.measured_state(q)
            if st['violations']:
                reason = 'measured ' + ','.join(st['violations'])
                break
            if st['progress'] >= stroke:
                break
            if time.perf_counter() - t0 > max_wall:
                reason = 'wall-clock limit'
                break
            qdot, done, why = ctrl.command(q)
            if done:
                reason = 'predicted ' + why
                break
            cmd = qdot * time_scale
            peak = np.abs(cmd).max()
            if peak > qd_cap:
                cmd = cmd * (qd_cap / peak)
            arm.send_qdot(cmd, timeout=3.0 * wall_period)
            if on_cycle is not None:
                on_cycle(q, st, cmd)
            now = time.perf_counter()
            for k, v in (('t', now - t0), ('q', q), ('qdot_cmd', cmd),
                         ('p', st['p']), ('progress', st['progress']),
                         ('lateral', st['lateral']), ('tilt_deg', st['tilt_deg'])):
                log[k].append(v)
            if len(log['t']) % max(int(0.5 / wall_period), 1) == 0:
                print(f'[rt] t {now - t0:6.2f}s  progress {st["progress"]:.3f} m  '
                      f'lateral {st["lateral"] * 1000:5.1f} mm  tilt '
                      f'{st["tilt_deg"]:4.1f} deg  |qdot| {peak:.2f} rad/s',
                      flush=True)
            next_t += wall_period
            if getattr(arm, 'streams', False):
                # the wrapper streamed set-points for the whole cycle itself
                next_t = time.perf_counter()
                dt_sleep = 0.0
            else:
                dt_sleep = next_t - time.perf_counter()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            elif dt_sleep < 0:
                n_over += 1
                if n_over == 1:
                    print(f'[rt] WARNING: cycle overran the {wall_period * 1000:.0f} ms '
                          f'period by {-dt_sleep * 1000:.1f} ms; the arm keeps the '
                          f'last velocity meanwhile (lengthen --period or lower '
                          f'--time-scale)', flush=True)
                next_t = time.perf_counter()   # fell behind: do not burst
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        arm.stop()
    q = arm.q()
    st = ctrl.measured_state(q)
    print(f'[rt] stopped: {reason}; progress {st["progress"]:.3f} m, '
          f'lateral {st["lateral"] * 1000:.1f} mm, tilt {st["tilt_deg"]:.1f} deg, '
          f'{len(log["t"])} cycles, {n_over} overruns', flush=True)
    if log_path:
        np.savez(log_path, reason=reason, d=unit(d), n=unit(n),
                 p_start=ctrl.p_start, time_scale=time_scale,
                 period=ctrl.env.dt, **{k: np.asarray(v) for k, v in log.items()})
        print(f'[rt] log -> {log_path}', flush=True)
    return st['progress'], reason


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'select'])
    ap.add_argument('--ip', default=DEFAULT_IP)
    ap.add_argument('--sim', action='store_true', help='simulated arm, no hardware')
    ap.add_argument('--sim-task', type=int, default=0,
                    help='--sim: start from this task of the 10k xArm7 pool')
    ap.add_argument('--sim-lag', type=float, default=0.03,
                    help='--sim: velocity lag time constant [s]')
    ap.add_argument('--dir', type=float, nargs=3, required=True,
                    help='path direction in the base frame')
    ap.add_argument('--normal', type=float, nargs=3, default=None,
                    help='plane normal / cone axis (default: current tool axis)')
    ap.add_argument('--p0', type=float, nargs=3, default=None,
                    help='select: task start point in the base frame')
    ap.add_argument('--stroke', type=float, default=0.5, help='stop after this length [m]')
    ap.add_argument('--period', type=float, default=0.05,
                    help='decision period in task time [s] (training: 0.05)')
    ap.add_argument('--time-scale', type=float, default=0.5,
                    help='execute at this fraction of the training speed')
    ap.add_argument('--qd-cap', type=float, default=1.0,
                    help='uniform cap on the commanded joint speed [rad/s]')
    ap.add_argument('--tcp', type=float, default=0.10,
                    help='on-axis tool length along the flange z [m] (model: 0.10)')
    ap.add_argument('--tool', choices=sorted(TOOLS), default=None,
                    help='tool preset (overrides --tcp): ' + ', '.join(sorted(TOOLS)))
    ap.add_argument('--tool-xyz', type=float, nargs=3, default=None,
                    help='tool tip (x y z) in the flange frame (overrides --tool)')
    ap.add_argument('--cone', type=float, default=30.0)
    ap.add_argument('--k-lateral', type=float, default=5.0,
                    help='task-space feedback gain on the path error [1/s]')
    ap.add_argument('--n-dirs', type=int, default=8)
    ap.add_argument('--k-nn', type=int, default=200)
    ap.add_argument('--move-speed', type=float, default=0.3,
                    help='select: joint speed of the point-to-point move [rad/s]')
    ap.add_argument('--log', default=None)
    ap.add_argument('--device', default='cpu',
                    help='cpu is faster than cuda for one configuration')
    args = ap.parse_args()

    if args.device == 'cpu':
        torch.set_num_threads(1)     # 7x7 problems: threading only adds overhead
    ctrl = Controller(args.period, tool_xyz_from_args(args), args.cone,
                      args.k_lateral, args.device)

    if args.sim:
        tz = np.load(ASSETS / 'tasks_pool_xarm7.npz')
        q_init = tz['q0_seed'][args.sim_task].astype(np.float64)
        arm = SimArm(q_init, lag=args.sim_lag)
        print(f'[rt] simulated arm from pool task {args.sim_task}', flush=True)
    else:
        arm = XArm(args.ip)

    try:
        q = arm.q()
        p_now, R_now = ctrl.fk(q)
        n = unit(args.normal) if args.normal is not None else R_now[:, 2]
        d = np.asarray(args.dir, np.float64)
        d = unit(d - (d @ n) * n)          # path direction lies in the plane
        print(f'[rt] tip now {p_now.round(3)}, d {d.round(3)}, n {n.round(3)}',
              flush=True)
        p0 = None
        if args.mode == 'select':
            p0 = np.asarray(args.p0, np.float64) if args.p0 is not None else p_now
            cands = cone_ik_candidates(ctrl, p0, d, n, args.cone, args.n_dirs,
                                       args.k_nn)
            if len(cands) == 0:
                raise RuntimeError('no admissible start at p0 for this task')
            best, V = ctrl.critic_pick(cands, d, n, p0)
            print(f'[rt] {len(cands)} admissible starts; critic pick #{best} '
                  f'value {V[best]:.3f} (min {V.min():.3f}, median '
                  f'{np.median(V):.3f})', flush=True)
            print(f'[rt] moving to {cands[best].round(3)}', flush=True)
            arm.move_to(cands[best], speed=args.move_speed)
            time.sleep(0.5)
        run_loop(arm, ctrl, d, n, args.stroke, args.time_scale, args.qd_cap,
                 args.log, p0=p0)
    finally:
        try:
            arm.stop()
        finally:
            arm.close()


if __name__ == '__main__':
    main()
