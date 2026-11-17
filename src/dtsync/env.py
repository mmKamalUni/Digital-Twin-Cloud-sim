"""Vectorised edge-to-cloud Digital Twin synchronisation environment.

One environment instance simulates *all* S telemetry streams of a fleet in lock-step
slots of ``dt`` seconds.  In every slot each edge node decides whether to push an update
of its stream to the cloud twin.  Updates travel through a cloud ingestion channel:

* ``ideal``  - every update reaches the twin within the slot (the assumption made by the
               reference DQN adaptive-sampling work).
* ``shared`` - a single FIFO ingestion gateway with a service capacity of ``capacity``
               messages per slot, a finite buffer (throttling) and optional random loss.
               Correlated bursts therefore create queueing delay, which leaves the twin
               stale even after the edge has decided to synchronise.

The cloud twin either holds the last value (``zoh``) or runs a damped-trend predictor
(``holt``) whose parameters (level, slope) are carried by each update; the edge mirrors
the same predictor so it always knows what the twin is showing (dual prediction).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .data import TelemetryDataset


@dataclass
class ChannelCfg:
    kind: str = "shared"          # "ideal" | "shared"
    capacity: float = 12.0        # messages served per slot by the ingestion gateway
    buffer: float = 240.0         # max messages queued before the gateway throttles
    prop_delay: float = 0.05      # one-way propagation + processing delay (s)
    loss: float = 0.0             # random message loss probability


@dataclass
class CostCfg:
    e_tx_mj: float = 2.0          # radio energy per transmitted update (mJ)
    e_sense_mj: float = 0.15      # energy per sensor reading (mJ)
    bytes_zoh: int = 48           # MQTT/TLS framed payload with one value
    bytes_holt: int = 52          # + one float32 slope
    usd_per_million_msgs: float = 1.0


@dataclass
class RewardCfg:
    mode: str = "huang"           # "huang" | "lagrangian"
    w: tuple = (0.4, 0.4, 0.2)    # (information, energy, redundancy) weights
    lam: float = 4.0              # violation price for the lagrangian reward
    congestion_price: float = 0.0 # extra cost per update per slot of gateway backlog


@dataclass
class EnvCfg:
    predictor: str = "zoh"        # "zoh" | "holt"
    phi: float = 0.97             # trend damping of the holt twin
    slope_window: int = 6
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    event_k: float = 3.0          # critical event = |x_t - x_{t-2}| >= event_k * eps
    event_window: int = 2         # twin must be back within eps after <= this many slots
    redundant_frac: float = 0.5   # update is redundant if twin error < frac*eps when sent
    always_sense: bool = True
    obs_noise: float = 0.0        # sensor interference: N(0, obs_noise*eps) added to readings


def precompute_slopes(X: np.ndarray, eps: np.ndarray, w: int) -> np.ndarray:
    """Least-squares slope over the last ``w`` samples, zeroed around step changes."""
    T, S = X.shape
    k = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    denom = (k ** 2).sum()
    slope = np.zeros((T, S), dtype=np.float32)
    for t in range(w - 1, T):
        win = X[t - w + 1:t + 1]
        slope[t] = (k[:, None] * (win - win.mean(0))).sum(0) / denom
    d = np.abs(np.diff(X, axis=0, prepend=X[:1]))
    # a jump inside the window means the series is piecewise constant, not trending
    jump = np.zeros((T, S), dtype=bool)
    for j in range(w):
        jump[j:] |= d[: T - j] > 2.0 * eps
    slope[jump] = 0.0
    return slope


class SyncEnv:
    def __init__(self, ds: TelemetryDataset, cfg: EnvCfg, seed: int = 0):
        self.ds, self.cfg = ds, cfg
        self.eps = ds.eps.astype(np.float64)
        self.Xtrue = ds.X.astype(np.float64)
        if cfg.obs_noise > 0:
            nrng = np.random.default_rng(10_000 + seed)
            self.X = self.Xtrue + nrng.normal(0, 1, self.Xtrue.shape) * cfg.obs_noise * self.eps
        else:
            self.X = self.Xtrue
        self.S = ds.S
        self.mu = self.X.mean(0)
        self.sd = self.X.std(0) + 1e-9
        self.slope = precompute_slopes(self.X, self.eps, cfg.slope_window).astype(np.float64)
        d = np.diff(self.X, axis=0, prepend=self.X[:1]) / self.eps
        self.diffn = d
        self.rng = np.random.default_rng(seed)
        self.lam = cfg.reward.lam

    # ------------------------------------------------------------------ predictors
    def _predict(self, a, b, t0, t):
        if self.cfg.predictor == "zoh":
            return a
        k = np.maximum(t - t0, 0).astype(np.float64)
        phi = self.cfg.phi
        return a + b * phi * (1.0 - phi ** k) / (1.0 - phi)

    # ------------------------------------------------------------------ episode
    def reset(self, start: int, length: int):
        T = self.X.shape[0]
        start = max(start, 8)
        self.t0_ep, self.t_end = start, min(start + length, T)
        self.t = start
        S = self.S
        x = self.X[start]
        # edge mirror (what the edge believes the twin shows) and true cloud twin
        self.m_a, self.m_b, self.m_t = x.copy(), np.zeros(S), np.full(S, start)
        self.c_a, self.c_b, self.c_t = x.copy(), np.zeros(S), np.full(S, start)
        self.c_gen = np.full(S, start)          # generation slot of the twin's latest update
        self.W = 0.0                            # gateway backlog (messages)
        self.W_prev = 0.0
        self.W_prev_arrival = 0.0
        self.pending = defaultdict(list)
        self.inflight = np.zeros(S, dtype=np.int32)
        self.inflight_since = np.full(S, -1)
        self.last_lat = np.zeros(S)             # last observed end-to-end latency (s)
        self.since = np.zeros(S)                # slots since last transmission
        self.act_hist = np.zeros((S, 6))
        self.energy = np.zeros(S)
        # bookkeeping for metrics
        L = self.t_end - start
        self.err_log = np.zeros((L, S), dtype=np.float32)
        self.n_sent = np.zeros(S)
        self.n_redundant = np.zeros(S)
        self.n_dropped = 0
        self.latencies = []
        self.aoi_sum = np.zeros(S)
        self.backlog_log = np.zeros(L, dtype=np.float32)
        return self

    @property
    def done(self):
        return self.t >= self.t_end

    def mirror_pred(self):
        return self._predict(self.m_a, self.m_b, self.m_t, self.t)

    def cloud_pred(self):
        return self._predict(self.c_a, self.c_b, self.c_t, self.t)

    # ------------------------------------------------------------------ dynamics
    def step(self, action: np.ndarray):
        cfg, S, t = self.cfg, self.S, self.t
        x = self.X[t]
        send = action.astype(bool)
        idx = np.flatnonzero(send)

        err_before = x - self.cloud_pred()
        redundant = send & (np.abs(x - self.mirror_pred()) < cfg.redundant_frac * self.eps)

        b = self.slope[t] if cfg.predictor == "holt" else np.zeros(S)
        # edge mirror assumes its update will be applied
        self.m_a[idx], self.m_b[idx], self.m_t[idx] = x[idx], b[idx], t

        ch = cfg.channel
        self.W_prev_arrival = self.W
        dropped = np.zeros(S, dtype=bool)
        if len(idx):
            if ch.kind == "ideal":
                lat = np.full(len(idx), ch.prop_delay)
                self._deliver(idx, x[idx], b[idx], t, t)
                self.latencies.append(lat)
                self.last_lat[idx] = lat
            else:
                order = self.rng.permutation(idx)
                pos = self.W + np.arange(len(order))
                ok = pos < ch.buffer
                if ch.loss > 0:
                    ok &= self.rng.random(len(order)) >= ch.loss
                dropped[order[~ok]] = True
                self.n_dropped += int((~ok).sum())
                keep = order[ok]
                wait_slots = (pos[ok] + 1.0) / ch.capacity
                lat = ch.prop_delay + wait_slots * self.ds.dt
                dslot = t + np.floor(lat / self.ds.dt).astype(int)
                for s, v, bb, d_, l_ in zip(keep, x[keep], b[keep], dslot, lat):
                    self.pending[int(d_)].append((s, v, bb, t, l_))
                self.inflight[keep] += 1
                self.inflight_since[keep] = np.where(self.inflight_since[keep] < 0, t,
                                                     self.inflight_since[keep])
                self.W += len(keep)
                self.latencies.append(lat)
        # deliveries scheduled for this slot (FIFO order preserved by append order)
        if ch.kind != "ideal" and t in self.pending:
            for s, v, bb, gen, l_ in self.pending.pop(t):
                self.c_a[s], self.c_b[s], self.c_t[s], self.c_gen[s] = v, bb, gen, gen
                self.inflight[s] -= 1
                if self.inflight[s] == 0:
                    self.inflight_since[s] = -1
                self.last_lat[s] = l_

        e = self.Xtrue[t] - self.cloud_pred()
        ae = np.abs(e) / self.eps
        viol = (ae > 1.0).astype(np.float64)
        self.err_log[t - self.t0_ep] = e / self.eps
        self.aoi_sum += t - self.c_gen
        self.backlog_log[t - self.t0_ep] = self.W

        e_tx = cfg.cost.e_tx_mj * send
        e_sense = cfg.cost.e_sense_mj if cfg.always_sense else cfg.cost.e_sense_mj * send
        self.energy += e_tx + e_sense
        self.n_sent += send
        self.n_redundant += redundant

        # reward
        rc = cfg.reward
        cost = send.astype(np.float64)
        # shadow price of the queueing delay an update imposes on the rest of the fleet:
        # proportional to the backlog (in slots of service) it has to join
        ext = np.zeros(S)
        if rc.congestion_price > 0 and ch.kind != "ideal":
            ext = rc.congestion_price * send * (self.W_prev_arrival / ch.capacity)
        if rc.mode == "huang":
            info = 1.0 - np.minimum(ae, 1.0)          # data quality of the twin this slot
            r = rc.w[0] * info - rc.w[1] * (cost + ext) - rc.w[2] * redundant
            viol_social = viol
        else:
            # externality is expressed in violation units, so it scales with lambda
            viol_social = viol + ext
            r = -(cost + self.lam * viol_social)
        comp = dict(cost=cost, viol=viol, viol_social=viol_social, redundant=redundant.astype(np.float64),
                    dropped=dropped, err_before=err_before)

        # history / counters
        self.act_hist = np.roll(self.act_hist, 1, axis=1)
        self.act_hist[:, 0] = send
        self.since = np.where(send, 0, self.since + 1)
        self.W_prev = self.W
        if ch.kind != "ideal":
            self.W = max(0.0, self.W - ch.capacity)
        self.t += 1
        return r, comp

    def _deliver(self, idx, v, b, gen, t):
        self.c_a[idx], self.c_b[idx], self.c_t[idx], self.c_gen[idx] = v, b, gen, gen

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        cfg, ds = self.cfg, self.ds
        L = self.t_end - self.t0_ep
        ae = np.abs(self.err_log)
        X = self.Xtrue[self.t0_ep:self.t_end]
        # critical events
        lag = np.abs(X[2:] - X[:-2]) / self.eps
        ev = np.zeros_like(ae, dtype=bool)
        ev[2:] = lag >= cfg.event_k
        last = np.full(self.S, -10 ** 9)
        n_ev = n_det = 0
        ok = ae < 1.0
        W = cfg.event_window
        for t, s in zip(*np.nonzero(ev)):
            if t - last[s] < 10:
                continue
            last[s] = t
            n_ev += 1
            n_det += bool(ok[t:t + W + 1, s].any())
        lat = np.concatenate(self.latencies) if self.latencies else np.zeros(1)
        msgs = self.n_sent.sum()
        size = cfg.cost.bytes_holt if cfg.predictor == "holt" else cfg.cost.bytes_zoh
        hours = L * ds.dt / 3600.0
        group_mae = {}
        for g, gname in enumerate(ds.group_names):
            m = ds.stream_group == g
            if m.any():
                group_mae[gname] = float((ae[:, m] * self.eps[m]).mean())
        return dict(
            data_quality=float((1.0 - np.minimum(ae, 1.0)).mean()),
            mae_eps=float(ae.mean()),
            sla_violation_pct=100.0 * float((ae > 1.0).mean()),
            p99_err_eps=float(np.percentile(ae, 99)),
            msgs_per_stream_hour=float(msgs / self.S / hours),
            send_rate_pct=100.0 * float(msgs / (self.S * L)),
            bandwidth_Bps=float(msgs * size / (L * ds.dt)),
            redundancy_pct=100.0 * float(self.n_redundant.sum() / max(msgs, 1)),
            energy_mJ_per_stream_day=float(self.energy.mean() / (L * ds.dt / 86400.0)),
            event_detection_pct=100.0 * n_det / max(n_ev, 1),
            n_events=int(n_ev),
            latency_mean_s=float(lat.mean()),
            latency_p95_s=float(np.percentile(lat, 95)),
            aoi_mean_s=float(self.aoi_sum.mean() / L * ds.dt),
            backlog_mean=float(self.backlog_log.mean()),
            backlog_max=float(self.backlog_log.max()),
            dropped_pct=100.0 * self.n_dropped / max(msgs, 1),
            usd_per_1k_streams_month=float(msgs / self.S * 1000 / hours * 24 * 30
                                           * cfg.cost.usd_per_million_msgs / 1e6),
            group_mae=group_mae,
        )
