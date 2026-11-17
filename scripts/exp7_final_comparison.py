"""Experiment 7 - final comparison: original paper vs reproduced baseline vs proposed method.

Intel Lab test split, both in the published setting (ideal delivery) and behind the shared
cloud gateway.  The proposed method keeps its validated price lambda* in both settings.
Output: results/exp7_final.csv
"""
from common import *  # noqa: F401,F403

from dtsync.env import EnvCfg

# Table 1 of the original paper: data quality, energy (mJ), redundancy (%), event detection (%)
PAPER = {"Fixed-frequency": (0.72, 145.3, 38.6, 82.1), "Random": (0.65, 96.7, 27.4, 76.4),
         "Threshold": (0.78, 121.5, 24.9, 85.6), "PPO": (0.87, 89.2, 12.3, 91.8),
         "DQN": (0.83, 93.6, 15.1, 89.3)}


def main():
    log = logger("exp7")
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    fixed_energy = {}
    rows = []
    for ch_name, ch in [("ideal", IDEAL), ("shared", SHARED)]:
        ref = EnvCfg(predictor="zoh", channel=ch, reward=RewardCfg("huang"))
        full, _ = variant(True, True, True, ch)
        m = run_policy(ds, EnvCfg(predictor="zoh", channel=ch), FixedRate(4), "test", 0)
        fixed_energy[ch_name] = m["energy_mJ_per_stream_day"]
        for s in SEEDS:
            pol, _ = load_agent("ref_dqn_ideal", s)
            rows.append(dict(method="DQN (reproduced baseline)", channel=ch_name, seed=s,
                             **flat(run_policy(ds, ref, pol, "test", s))))
            pol, _ = proposed(s)
            rows.append(dict(method="Proposed (A+B+C)", channel=ch_name, seed=s,
                             **flat(run_policy(ds, full, pol, "test", s))))
        log(f"{ch_name} done")
    df = agg(rows, ["method", "channel"])
    df["energy_vs_fixed"] = [r.energy_mJ_per_stream_day / fixed_energy[r.channel] for r in df.itertuples()]
    p = PAPER["DQN"]
    df = pd.concat([pd.DataFrame([dict(method="DQN (original paper)", channel="ideal", data_quality=p[0],
                                       energy_vs_fixed=p[1] / PAPER["Fixed-frequency"][1],
                                       redundancy_pct=p[2], event_detection_pct=p[3])]), df],
                   ignore_index=True)
    save(df, "exp7_final.csv")


if __name__ == "__main__":
    main()
