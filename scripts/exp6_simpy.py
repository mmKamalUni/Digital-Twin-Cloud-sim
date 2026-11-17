"""Experiment 6 - event-level SimPy validation of the main comparison (Intel Lab test split).

Stochastic (exponential) service at a multi-worker ingestion tier, jittered edge clocks.
Output: results/exp6_simpy.csv
"""
from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg
from dtsync.simpy_engine import run_policy_simpy


def main():
    log = logger("exp6")
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    rows = []
    zoh, holt = EnvCfg(predictor="zoh", channel=SHARED), EnvCfg(predictor="holt", phi=PHI, channel=SHARED)
    ref = EnvCfg(predictor="zoh", channel=SHARED, reward=RewardCfg("huang"))
    full, _ = variant(True, True, True)
    for workers in [1, 4]:
        for name, cfg, pol in [("Send-on-delta (1.0 eps)", zoh, DeltaThreshold(1.0)),
                               ("Dual-prediction DPS (1.0 eps)", holt, DeltaThreshold(1.0)),
                               ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold())]:
            for s in SEEDS:
                rows.append(dict(method=name, workers=workers, seed=s,
                                 **flat(run_policy_simpy(ds, cfg, pol, "test", s, workers))))
        for s in SEEDS:
            pol, _ = load_agent("ref_dqn_ideal", s)
            rows.append(dict(method="DQN-ref (published setting)", workers=workers, seed=s,
                             **flat(run_policy_simpy(ds, ref, pol, "test", s, workers))))
            pol, _ = proposed(s)
            rows.append(dict(method="Proposed (A+B+C)", workers=workers, seed=s,
                             **flat(run_policy_simpy(ds, full, pol, "test", s, workers))))
        log(f"workers {workers} done")
    save(agg(rows, ["method", "workers"]), "exp6_simpy.csv")


if __name__ == "__main__":
    main()
