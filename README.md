# Dynamic Digital Twin Synchronization & Adaptive Cloud Orchestration under Bursty Edge-IoT Workloads

Simulation framework and experiments for congestion-aware, SLA-constrained adaptive synchronization of
edge-to-cloud digital twins. Developed for FAST-NUCES Cloud Computing Research.

**Author**: Malik Murtaza Kamal (Roll No: i222622)
**Department**: Software Engineering, FAST-NUCES, Islamabad
**Course**: Cloud Computing (100-Day Research Assignment)
**Supervisor**: Dr. Basharat Hussain (basharat.hussain.v@isb.nu.edu.pk)
**Approved Research Track**: Track 5 — Digital Twins and Cloud Computing

---

## Research question

> How does a deep-reinforcement-learning (DRL) adaptive synchronization policy compare with fixed-rate and
> threshold-based baselines in minimizing cloud bandwidth and twin state drift while meeting a fidelity SLA
> under bursty IoT workloads?

## Baseline study

The baseline is the DQN context-aware adaptive sampling method of Huang et al., *Context-Aware Adaptive
Sampling for Intelligent Data Acquisition Systems Using DQN*, arXiv:2504.09344 (2025). No official code is
available, so the method was re-implemented from the paper (state, action, reward
`r = w1·I − w2·C − w3·D`, soft-target DQN, PPO comparison) and evaluated on the Intel Berkeley Lab data used
in the paper. See [docs/reproduction_report.md](docs/reproduction_report.md).

## Proposed method

The baseline assumes every transmitted reading reaches the cloud instantly. Behind a shared cloud ingestion
gateway, correlated bursts congest the gateway and the twin falls out of tolerance exactly when it matters.
The proposed method adds three components, each of which can be switched off for ablation:

| | Component | What it does |
|---|---|---|
| **A** | Predictive mirrored twin | Cloud twin extrapolates a damped trend between updates; the edge mirrors the same predictor and decides on the predicted error. |
| **B** | Congestion awareness | Gateway backlog, in-flight age, acknowledged cloud error and age of information are observed; each transmission is charged a congestion externality. |
| **C** | SLA-constrained control | Two-head (cost / violation) Q-network conditioned on a price λ; a cloud-side controller adapts λ online to hold the violation rate at the SLA target. |

## Repository layout

```
Digital-Twin-Cloud-sim
├── src/dtsync/              # framework
│   ├── data.py              # Intel Lab, NAB and synthetic bursty loaders (60/10/30 time split)
│   ├── env.py               # vectorised slotted environment: twin predictors, shared FIFO gateway, metrics
│   ├── simpy_engine.py      # event-level SimPy engine (multi-worker, stochastic service) for validation
│   ├── features.py          # observation builders (reference state, +A, +B, +A+B)
│   ├── evaluate.py          # policy roll-out and metric collection
│   ├── policies/heuristics.py   # fixed-rate, random, send-on-delta / dual prediction, congestion-adaptive
│   └── agents/              # dqn.py (reference + constrained decomposed DQN, SLA controller), ppo.py
├── scripts/
│   ├── download_data.py     # fetch Intel Lab and NAB into data/raw
│   ├── exp1_reproduce.py    # reproduction of the baseline (Table 1, weights, interference) + gap
│   ├── exp3_ablation.py     # proposed method, ablation of A/B/C, Pareto fronts
│   ├── exp4_capacity.py     # gateway provisioning sweep
│   ├── exp5_generalization.py   # NAB and synthetic burst-correlation sweep
│   ├── exp6_simpy.py        # event-level validation
│   └── make_figures.py      # figures/ and results/tables/*.tex
├── results/                 # CSV results, trained models (results/models), λ calibration
├── figures/                 # generated figures
├── docs/                    # reproduction report
├── tests/                   # unit tests
├── src/sim_engine.py, src/telemetry.py, src/baselines/   # initial prototype (kept for reference)
└── configs/sim_config.yaml  # prototype configuration
```

## Quick start

```bash
python -m venv venv && venv\Scripts\activate        # Windows (source venv/bin/activate on Linux)
pip install -r requirements.txt
python scripts/download_data.py
python -m pytest tests -q
```

## Reproducing the experiments

Run in this order (exp4–exp6 use the λ calibration written by exp3). Trained models are cached in
`results/models/` and are re-used, so scripts can be interrupted and re-started.

```bash
python scripts/exp1_reproduce.py       # ~15 min on 16 cores
python scripts/exp3_ablation.py        # ~20 min
python scripts/exp4_capacity.py
python scripts/exp5_generalization.py
python scripts/exp6_simpy.py
python scripts/make_figures.py
```

Seeds 0, 1, 2 are used throughout; results are reported as mean ± standard deviation over seeds.
Everything runs on CPU (PyTorch CPU build is sufficient; the networks have about 20k parameters).

## Datasets

* **Intel Berkeley Research Lab** sensor data (54 Mica2Dot motes, 2004): http://db.csail.mit.edu/labdata/labdata.html
  — 40 motes with ≥ 50 % coverage × temperature / humidity / light = 120 streams, 31 s epochs.
* **Numenta Anomaly Benchmark (NAB)**, realKnownCause and realAWSCloudwatch series (MIT licence):
  https://github.com/numenta/NAB — 21 series, 5-min slots.
* **Synthetic** bursty fleet with controllable cross-stream burst correlation ρ (generated in code).

## Reproducibility Statement

Reproducibility: The code and reproducibility materials for this study are publicly available at: https://github.com/mmKamalUni/Digital-Twin-Cloud-sim
