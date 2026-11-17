"""Event-level SimPy validation of the edge-cloud synchronisation pipeline.

The slotted environment (``env.py``) uses a deterministic fluid FIFO gateway so that
training is fast.  This module re-runs trained/heuristic policies in a discrete-event
model with:

* edge updates emitted at a random offset inside each slot (unsynchronised device clocks),
* a cloud ingestion tier of ``workers`` parallel consumers (e.g. serverless/stream workers)
  with exponentially distributed service times and the same aggregate capacity,
* a bounded ingestion buffer (throttling) and optional random loss,
* propagation delay on the uplink.

The cloud twin is updated at the exact completion time of each message; fidelity is
measured at slot boundaries exactly as in the slotted environment, so the two engines are
directly comparable.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import simpy

from .env import EnvCfg, SyncEnv


class SimPyEnv(SyncEnv):
    def __init__(self, ds, cfg: EnvCfg, seed: int = 0, workers: int = 4, jitter: float = 1.0):
        super().__init__(ds, cfg, seed)
        self.workers, self.jitter = workers, jitter

    def reset(self, start, length):
        super().reset(start, length)
        self.sim = simpy.Environment(initial_time=start * self.ds.dt)
        self.gw = simpy.Resource(self.sim, capacity=self.workers)
        self.q_len = 0
        return self

    # message life-cycle ------------------------------------------------------------
    def _message(self, s, v, b, gen, t_emit):
        ch, dt = self.cfg.channel, self.ds.dt
        yield self.sim.timeout(t_emit - self.sim.now)
        yield self.sim.timeout(ch.prop_delay)
        if self.q_len >= ch.buffer or (ch.loss > 0 and self.rng.random() < ch.loss):
            self.n_dropped += 1
            self.inflight[s] -= 1
            return
        self.q_len += 1
        with self.gw.request() as req:
            yield req
            service = self.rng.exponential(self.workers * dt / ch.capacity)
            yield self.sim.timeout(service)
        self.q_len -= 1
        # FIFO per stream is not guaranteed with parallel workers: keep the freshest update
        if gen >= self.c_gen[s]:
            self.c_a[s], self.c_b[s], self.c_t[s], self.c_gen[s] = v, b, gen, gen
        self.inflight[s] -= 1
        if self.inflight[s] == 0:
            self.inflight_since[s] = -1
        lat = self.sim.now - gen * dt
        self.last_lat[s] = lat
        self._lat.append(lat)

    def step(self, action):
        cfg, t, dt = self.cfg, self.t, self.ds.dt
        if not hasattr(self, "_lat"):
            self._lat = []
        x = self.X[t]
        send = action.astype(bool)
        idx = np.flatnonzero(send)
        redundant = send & (np.abs(x - self.mirror_pred()) < cfg.redundant_frac * self.eps)
        b = self.slope[t] if cfg.predictor == "holt" else np.zeros(self.S)
        self.m_a[idx], self.m_b[idx], self.m_t[idx] = x[idx], b[idx], t
        for s in idx:
            off = self.rng.uniform(0, self.jitter * dt * 0.5)
            self.inflight[s] += 1
            if self.inflight_since[s] < 0:
                self.inflight_since[s] = t
            self.sim.process(self._message(int(s), x[s], b[s], t, t * dt + off))
        self.sim.run(until=(t + 1) * dt - 1e-6)

        e = self.Xtrue[t] - self.cloud_pred()
        ae = np.abs(e) / self.eps
        viol = (ae > 1.0).astype(np.float64)
        self.err_log[t - self.t0_ep] = e / self.eps
        self.aoi_sum += t - self.c_gen
        self.W_prev = self.W
        self.W = float(self.q_len)           # backlog advertised to the edge for the next slot
        self.backlog_log[t - self.t0_ep] = self.W
        self.energy += cfg.cost.e_tx_mj * send + (cfg.cost.e_sense_mj if cfg.always_sense
                                                  else cfg.cost.e_sense_mj * send)
        self.n_sent += send
        self.n_redundant += redundant
        self.act_hist = np.roll(self.act_hist, 1, axis=1)
        self.act_hist[:, 0] = send
        self.since = np.where(send, 0, self.since + 1)
        self.t += 1
        return -viol, dict(viol=viol)

    def metrics(self):
        self.latencies = [np.asarray(self._lat)] if getattr(self, "_lat", None) else []
        return super().metrics()


def run_policy_simpy(ds, cfg: EnvCfg, policy, split="test", seed=0, workers=4):
    cfg = copy.deepcopy(cfg)
    cfg.always_sense = getattr(policy, "always_sense", True)
    env = SimPyEnv(ds, cfg, seed=seed, workers=workers)
    a, b = ds.splits[split]
    env.reset(a, b - a)
    t0 = time.perf_counter()
    while not env.done:
        env.step(policy(env))
    m = env.metrics()
    m["wall_s"] = time.perf_counter() - t0
    return m
