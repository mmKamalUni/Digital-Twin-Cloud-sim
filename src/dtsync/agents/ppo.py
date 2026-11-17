"""PPO synchronisation agent (comparison method listed in the reference paper)."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..env import EnvCfg, SyncEnv
from ..features import make_obs_fn


class ActorCritic(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, 2))
        self.v = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))


@dataclass
class PPOCfg:
    total_slots: int = 24000
    episode_len: int = 2000
    rollout: int = 128
    epochs: int = 4
    minibatch: int = 4096
    lr: float = 3e-4
    gamma: float = 0.95
    gae: float = 0.95
    clip: float = 0.2
    ent: float = 0.01
    hidden: int = 128


class PPOPolicy:
    always_sense = True

    def __init__(self, ac, obs_kind):
        self.ac, self.obs_kind = ac, obs_kind
        self.obs_fn = make_obs_fn(obs_kind)

    def __call__(self, env: SyncEnv):
        with torch.no_grad():
            logits = self.ac.pi(torch.as_tensor(self.obs_fn(env)))
        return logits.argmax(1).numpy().astype(bool)


def train_ppo(ds, env_cfg: EnvCfg, obs_kind: str = "huang", cfg: PPOCfg = PPOCfg(), seed: int = 0,
              verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.always_sense = True
    env = SyncEnv(ds, env_cfg, seed=seed)
    obs_fn = make_obs_fn(obs_kind)
    a0, b0 = ds.splits["train"]

    def new_episode():
        s = int(rng.integers(a0 + 8, max(a0 + 9, b0 - cfg.episode_len)))
        env.reset(s, cfg.episode_len)

    new_episode()
    o = obs_fn(env)
    ac = ActorCritic(o.shape[1], cfg.hidden)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    t_start = time.perf_counter()
    n_updates = cfg.total_slots // cfg.rollout
    for u in range(n_updates):
        O, A, LP, R, V, D = [], [], [], [], [], []
        for _ in range(cfg.rollout):
            ot = torch.as_tensor(o)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=ac.pi(ot))
                a = dist.sample()
                v = ac.v(ot).squeeze(1)
            r, _ = env.step(a.numpy())
            done = env.done
            O.append(o); A.append(a.numpy()); LP.append(dist.log_prob(a).numpy())
            R.append(r.astype(np.float32)); V.append(v.numpy()); D.append(done)
            if done:
                new_episode()
            o = obs_fn(env)
        with torch.no_grad():
            v_last = ac.v(torch.as_tensor(o)).squeeze(1).numpy()
        T = len(R)
        adv = np.zeros((T, env.S), np.float32)
        last = np.zeros(env.S, np.float32)
        for t in reversed(range(T)):
            nv = v_last if t == T - 1 else V[t + 1]
            nonterm = 0.0 if D[t] else 1.0
            delta = R[t] + cfg.gamma * nv * nonterm - V[t]
            last = delta + cfg.gamma * cfg.gae * nonterm * last
            adv[t] = last
        ret = adv + np.array(V)
        Ob = torch.as_tensor(np.concatenate(O)); Ab = torch.as_tensor(np.concatenate(A))
        LPb = torch.as_tensor(np.concatenate(LP)); ADV = torch.as_tensor(adv.reshape(-1))
        RET = torch.as_tensor(ret.reshape(-1))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = len(Ob)
        for _ in range(cfg.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, cfg.minibatch):
                j = perm[i:i + cfg.minibatch]
                dist = torch.distributions.Categorical(logits=ac.pi(Ob[j]))
                ratio = torch.exp(dist.log_prob(Ab[j]) - LPb[j])
                pl = -torch.min(ratio * ADV[j], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV[j]).mean()
                vl = ((ac.v(Ob[j]).squeeze(1) - RET[j]) ** 2).mean()
                loss = pl + 0.5 * vl - cfg.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        if verbose and u % 30 == 0:
            print(f"  [ppo {obs_kind}] update {u}/{n_updates} send={np.mean(A):.3f} "
                  f"reward={np.mean(R):.4f}", flush=True)
    info = dict(train_seconds=time.perf_counter() - t_start,
                n_params=sum(p.numel() for p in ac.pi.parameters()))
    return PPOPolicy(ac, obs_kind), info
