"""Experiment 4 - cloud ingestion provisioning (capacity sweep, Intel Lab).

Policies trained at capacity 12 are deployed against gateways of different size.  The
minimum capacity that meets the SLA target is the cloud resource each policy requires.
Output: results/exp4_capacity.csv
"""
from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg

CAPS = [4, 6, 8, 10, 12, 16, 20, 24, 32]


def main():
    log = logger("exp4")
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    rows = []
    for cap in CAPS:
        ch = shared(cap)
        zoh, holt = EnvCfg(predictor="zoh", channel=ch), EnvCfg(predictor="holt", phi=PHI, channel=ch)
        for name, cfg, pol in [("Send-on-delta (1.0 eps)", zoh, DeltaThreshold(1.0)),
                               ("Dual-prediction DPS (1.0 eps)", holt, DeltaThreshold(1.0)),
                               ("Congestion-adaptive DPS", holt, CongestionAdaptiveThreshold())]:
            rows.append(dict(method=name, capacity=cap, seed=0, **flat(run_policy(ds, cfg, pol, "test", 0))))
        ref = EnvCfg(predictor="zoh", channel=ch, reward=RewardCfg("huang"))
        full, _ = variant(True, True, True, ch)
        for s in SEEDS:
            pol, _ = load_agent("ref_dqn_ideal", s)
            rows.append(dict(method="DQN-ref (published setting)", capacity=cap, seed=s,
                             **flat(run_policy(ds, ref, pol, "test", s))))
            pol, _ = proposed(s)
            rows.append(dict(method="Proposed (A+B+C)", capacity=cap, seed=s,
                             **flat(run_policy(ds, full, pol, "test", s))))
        log(f"capacity {cap} done")
    save(agg(rows, ["method", "capacity"]), "exp4_capacity.csv")


if __name__ == "__main__":
    main()
