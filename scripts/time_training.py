"""Clean wall-clock timing of one training run per agent type (single process, 2 threads).

Output: results/training_time.csv
"""
from common import *  # noqa: F401,F403

import psutil

from dtsync.env import EnvCfg


def main():
    torch.set_num_threads(2)
    ds = load_dataset("intel", root=os.path.join(ROOT, "data"))
    ref = EnvCfg(predictor="zoh", channel=IDEAL, reward=RewardCfg("huang"))
    full, obs = variant(True, True, True)
    rows = []
    proc = psutil.Process()
    for name, fn in [("DQN-ref", lambda: train_dqn(ds, ref, "huang", dqn_cfg(False), seed=0, verbose=False)),
                     ("PPO", lambda: train_ppo(ds, ref, "huang", PPOCfg(total_slots=TRAIN_SLOTS), seed=0,
                                               verbose=False)),
                     ("Proposed (A+B+C)", lambda: train_dqn(ds, full, obs, dqn_cfg(True), seed=0, verbose=False))]:
        t = time.perf_counter()
        _, info = fn()
        rows.append(dict(method=name, train_seconds=time.perf_counter() - t, n_params=info["n_params"],
                         train_slots=TRAIN_SLOTS, streams=ds.S,
                         rss_MB=proc.memory_info().rss / 2 ** 20))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "training_time.csv")


if __name__ == "__main__":
    main()
