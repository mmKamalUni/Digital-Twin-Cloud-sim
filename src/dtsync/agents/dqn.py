"""Deep Q-Network synchronisation agent.

A single Q-network is shared by every stream (parameter sharing, decentralised execution):
each stream feeds its own observation and acts on its own Q-values, while all streams
contribute transitions to one replay buffer.

Two value heads are supported:

* ``heads=1`` - scalar reward (reference method: r = w1*I - w2*C - w3*D).
* ``heads=2`` - SLA-constrained, decomposed value function.  Head 0 estimates the
  discounted (congestion-priced) message cost, head 1 the discounted SLA-violation cost,
  and actions maximise ``Q_cost + lambda * Q_viol``.  The network is *conditioned on
  lambda* (log-lambda is an input feature) and lambda is randomised per stream during
  training, so one network represents the whole cost/fidelity trade-off family.  At run
  time the cloud orchestrator adjusts lambda online (integral control in log space) so the
  fleet tracks a target SLA-violation rate without retraining.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..env import EnvCfg, SyncEnv
from ..features import make_obs_fn


class QNet(nn.Module):
    def __init__(self, d_in: int, n_act: int = 2, heads: int = 1, hidden: int = 128):
        super().__init__()
        self.n_act, self.heads = n_act, heads
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_act * heads))

    def forward(self, x):
        return self.body(x).view(-1, self.heads, self.n_act)


@dataclass
class DQNCfg:
    total_slots: int = 24000
    episode_len: int = 2000
    warmup_slots: int = 1500
    buffer: int = 400_000
    batch: int = 1024
    lr: float = 5e-4
    gamma: float = 0.95
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_frac: float = 0.5
    hidden: int = 128
    explore_send_p: float = 0.5   # P(send) of an exploratory action
    lam_lo: float = 0.3           # lambda range for the constrained variant
    lam_hi: float = 30.0


LOG_LAM_SCALE = np.log(30.0)


def lam_feature(lam):
    return (np.log(lam) / LOG_LAM_SCALE).astype(np.float32)


class Replay:
    def __init__(self, cap, d, heads):
        self.o = np.zeros((cap, d), np.float32)
        self.o2 = np.zeros((cap, d), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros((cap, heads), np.float32)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, o, a, r, o2):
        k = len(o)
        idx = (self.i + np.arange(k)) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx] = o, a, r, o2
        self.i = (self.i + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, b, rng):
        j = rng.integers(0, self.n, b)
        return self.o[j], self.a[j], self.r[j], self.o2[j]


class SLAController:
    """Cloud-side integral controller that sets the violation price lambda.

    The fleet-level violation estimate is aggregated from edge heartbeats: with delivery
    feedback the edge knows the twin's true state (acknowledged updates); without it the
    edge can only use its own optimistic mirror.
    """

    def __init__(self, target: float, lam0: float = 2.0, gain: float = 0.003, ema: float = 0.99,
                 lo: float = 0.3, hi: float = 30.0, window: str = "ema", prior_slots: int = 200):
        self.target, self.lam, self.gain, self.ema, self.lo, self.hi = target, lam0, gain, ema, lo, hi
        self.lam0, self.window, self.prior_slots = lam0, window, prior_slots
        self.reset()

    def reset(self):
        """Start a new SLA accounting window (e.g. a billing period)."""
        self.lam = self.lam0
        self.v = self.target
        self.v_sum, self.n = self.target * self.prior_slots, float(self.prior_slots)

    def update(self, viol_est: float):
        # "cumulative": violation rate over the SLA window so far (bursts are not forgotten
        # during the quiet periods that follow them); "ema": short-horizon moving average
        if self.window == "cumulative":
            self.v_sum += viol_est
            self.n += 1.0
            self.v = self.v_sum / self.n
        else:
            self.v = self.ema * self.v + (1 - self.ema) * viol_est
        self.lam = float(np.clip(self.lam * np.exp(self.gain * (self.v - self.target)
                                                   / max(self.target, 1e-3)), self.lo, self.hi))
        return self.lam


class DQNPolicy:
    always_sense = True

    def __init__(self, net: QNet, obs_kind: str, lam: float = 2.0, controller: SLAController | None = None):
        self.net, self.obs_kind, self.lam = net, obs_kind, lam
        self.obs_fn = make_obs_fn(obs_kind)
        self.controller = controller
        self.lam_trace = []

    @property
    def constrained(self):
        return self.net.heads == 2

    def build_obs(self, env, lam_vec=None):
        o = self.obs_fn(env)
        if self.constrained:
            lv = np.full(env.S, self.lam) if lam_vec is None else lam_vec
            o = np.concatenate([o, lam_feature(lv)[:, None]], 1)
        return o

    def q(self, o, lam_vec=None):
        with torch.no_grad():
            q = self.net(torch.as_tensor(o)).numpy()
        if not self.constrained:
            return q[:, 0]
        lv = np.full(len(o), self.lam) if lam_vec is None else lam_vec
        return q[:, 0] + lv[:, None] * q[:, 1]

    def reset(self, env):
        if self.controller is not None:
            self.controller.reset()
            self.lam = self.controller.lam
        self.lam_trace = []

    def __call__(self, env: SyncEnv):
        if self.controller is not None and env.t > env.t0_ep:
            t1 = env.t - 1
            x = env.X[t1]
            if self.obs_kind in ("cong", "full"):
                ref = env._predict(env.c_a, env.c_b, env.c_t, t1)   # acknowledged twin state
            else:
                ref = env._predict(env.m_a, env.m_b, env.m_t, t1)   # optimistic mirror only
            v_est = float((np.abs(x - ref) > env.eps).mean())
            self.lam = self.controller.update(v_est)
            self.lam_trace.append(self.lam)
        return self.q(self.build_obs(env)).argmax(1).astype(bool)

    def state(self):
        return dict(net=self.net.state_dict(), obs_kind=self.obs_kind, lam=self.lam,
                    d_in=self.net.body[0].in_features, heads=self.net.heads,
                    hidden=self.net.body[0].out_features)

    @staticmethod
    def load(path):
        s = torch.load(path, weights_only=False)
        net = QNet(s["d_in"], heads=s["heads"], hidden=s["hidden"])
        net.load_state_dict(s["net"])
        return DQNPolicy(net, s["obs_kind"], s["lam"])


def train_dqn(ds, env_cfg: EnvCfg, obs_kind: str, cfg: DQNCfg = DQNCfg(), seed: int = 0,
              log_every: int = 2000, verbose: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.always_sense = True
    constrained = env_cfg.reward.mode == "lagrangian"
    heads = 2 if constrained else 1
    env = SyncEnv(ds, env_cfg, seed=seed)
    a0, b0 = ds.splits["train"]
    env.reset(a0 + 8, cfg.episode_len)
    d = make_obs_fn(obs_kind)(env).shape[1] + (1 if constrained else 0)
    net = QNet(d, heads=heads, hidden=cfg.hidden)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer, d, heads)
    policy = DQNPolicy(net, obs_kind)
    hist = []
    t_start = time.perf_counter()
    lam_vec = np.ones(env.S)

    def new_episode():
        nonlocal lam_vec
        s = int(rng.integers(a0 + 8, max(a0 + 9, b0 - cfg.episode_len)))
        env.reset(s, cfg.episode_len)
        lam_vec = np.exp(rng.uniform(np.log(cfg.lam_lo), np.log(cfg.lam_hi), env.S))

    new_episode()
    o = policy.build_obs(env, lam_vec)
    for step in range(cfg.total_slots):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step /
                  (cfg.eps_frac * cfg.total_slots))
        greedy = policy.q(o, lam_vec).argmax(1)
        explore = rng.random(env.S) < eps
        act = np.where(explore, (rng.random(env.S) < cfg.explore_send_p).astype(np.int64), greedy)
        r, comp = env.step(act)
        # time-limit truncation: bootstrap from the true next state of the same episode
        o2 = policy.build_obs(env, lam_vec)
        rr = np.stack([-comp["cost"], -comp["viol_social"]], 1) if constrained else r[:, None]
        buf.add(o, act, rr.astype(np.float32), o2)
        if env.done:
            new_episode()
            o = policy.build_obs(env, lam_vec)
        else:
            o = o2

        if step >= cfg.warmup_slots:
            ob, ab, rb, o2b = buf.sample(cfg.batch, rng)
            ob, o2b = torch.as_tensor(ob), torch.as_tensor(o2b)
            ab, rb = torch.as_tensor(ab), torch.as_tensor(rb)
            q = net(ob)                                    # (B, H, A)
            q_sa = q.gather(2, ab.view(-1, 1, 1).expand(-1, heads, 1)).squeeze(2)
            with torch.no_grad():
                qn = tgt(o2b)
                if constrained:
                    lam_b = torch.exp(o2b[:, -1] * LOG_LAM_SCALE).unsqueeze(1)
                    a_star = (qn[:, 0] + lam_b * qn[:, 1]).argmax(1)
                else:
                    a_star = qn[:, 0].argmax(1)
                q_next = qn.gather(2, a_star.view(-1, 1, 1).expand(-1, heads, 1)).squeeze(2)
                y = rb + cfg.gamma * q_next
            loss = nn.functional.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        if step % log_every == 0:
            rec = dict(step=step, eps=eps, send=float(act.mean()), viol=float(comp["viol"].mean()),
                       reward=float(r.mean()), backlog=float(env.W))
            hist.append(rec)
            if verbose:
                print(f"  [dqn {obs_kind}/{env_cfg.reward.mode}] slot {step:6d} eps={eps:.2f} "
                      f"send={rec['send']:.3f} viol={rec['viol']:.3f} backlog={rec['backlog']:.1f}",
                      flush=True)
    info = dict(train_seconds=time.perf_counter() - t_start, history=hist,
                n_params=sum(p.numel() for p in net.parameters()))
    return policy, info
