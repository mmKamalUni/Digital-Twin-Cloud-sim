"""Observation builders.

``huang``      - state used by the reference DQN adaptive-sampling method: the current
                 (normalised) observation, recent changes, deviation from the last sampled
                 value, time since the last sample, recent sampling decisions and the
                 remaining energy budget.
``proposed``   - adds the components of the proposed method on top of the reference state:
                 (A) deviation from the *mirrored predictive twin* plus a one-step-ahead
                     violation forecast, and
                 (B) congestion / delivery feedback from the ingestion gateway: backlog in
                     slots of service, backlog trend, in-flight updates and their age, the
                     acknowledged (true) twin error and the twin's age of information.
"""
from __future__ import annotations

import numpy as np

from .env import SyncEnv

K_HIST = 6


def _clip(a, lo=-5.0, hi=5.0):
    return np.clip(a, lo, hi)


def huang_obs(env: SyncEnv) -> np.ndarray:
    t = env.t
    x = env.X[t]
    z = (x - env.mu) / env.sd
    d = np.stack([(x - env.X[t - k]) / env.eps for k in range(1, K_HIST + 1)], 1)
    dev = (x - env.mirror_pred()) / env.eps
    budget = env.cfg.cost.e_tx_mj * 0.1 * (env.t_end - env.t0_ep) + 1e-9
    remaining = 1.0 - np.clip(env.energy / budget, 0, 1)
    return np.concatenate([
        _clip(z)[:, None],
        _clip(d),
        _clip(dev)[:, None],
        np.abs(_clip(dev))[:, None],
        np.minimum(env.since / 50.0, 2.0)[:, None],
        env.act_hist,
        remaining[:, None],
    ], 1).astype(np.float32)


def proposed_obs(env: SyncEnv, use_pred: bool = True, use_cong: bool = True) -> np.ndarray:
    parts = [huang_obs(env)]
    t = env.t
    x = env.X[t]
    if use_pred:
        dev = (x - env.mirror_pred()) / env.eps
        # one-step-ahead forecast of the deviation using the local slope estimate
        t1 = t + 1
        x_next = x + env.slope[t]
        pred_next = env._predict(env.m_a, env.m_b, env.m_t, t1)
        dev_next = (x_next - pred_next) / env.eps
        vol = np.abs(env.diffn[max(t - 7, 0):t + 1]).mean(0)
        parts.append(np.stack([_clip(dev), _clip(dev_next), np.abs(_clip(dev_next)),
                               np.minimum(vol, 5.0)], 1))
    if use_cong:
        ch = env.cfg.channel
        cap = ch.capacity if ch.kind != "ideal" else 1e9
        backlog = np.full(env.S, min(env.W / cap, 10.0) / 5.0)
        trend = np.full(env.S, np.clip((env.W - env.W_prev) / max(cap, 1e-9), -5, 5) / 5.0)
        age_if = np.where(env.inflight_since >= 0, t - env.inflight_since, 0) / 10.0
        cerr = (x - env.cloud_pred()) / env.eps
        aoi = np.minimum((t - env.c_gen) / 50.0, 2.0)
        parts.append(np.stack([backlog, trend, np.minimum(env.inflight, 3) / 3.0,
                               np.minimum(age_if, 2.0), _clip(cerr), np.abs(_clip(cerr)), aoi,
                               np.minimum(env.last_lat / env.ds.dt, 5.0) / 5.0], 1))
    return np.concatenate(parts, 1).astype(np.float32)


def make_obs_fn(kind: str):
    if kind == "huang":
        return huang_obs
    if kind == "pred":
        return lambda e: proposed_obs(e, True, False)
    if kind == "cong":
        return lambda e: proposed_obs(e, False, True)
    if kind == "full":
        return lambda e: proposed_obs(e, True, True)
    raise ValueError(kind)
