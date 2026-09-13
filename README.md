# Dynamic Digital Twin Synchronization & Adaptive Cloud Orchestration under Bursty Edge-IoT Workloads

Python simulation framework benchmarking DRL-driven adaptive telemetry synchronization against static baselines for Edge-Cloud Digital Twins under bursty IoT workloads. Developed for FAST-NUCES Cloud Computing Research.

**Author**: Malik Murtaza Kamal (Roll No: i222622)  
**Department**: Software Engineering, FAST-NUCES, Islamabad  
**Course**: Cloud Computing (100-Day Research Assignment)  
**Supervisor**: Dr. Basharat Hussain (basharat.hussain.v@isb.nu.edu.pk)  
**Approved Research Track**: Track 5 — Digital Twins and Cloud Computing  

---

## Research Motivation
Real-time IoT Digital Twins rely on continuous telemetry streaming to stay synchronized with physical assets in cloud environments. However, fixed-rate sampling and static threshold rules cause unnecessary bandwidth consumption during steady states or severe state-drift errors during transient operational bursts. 

This repository provides a discrete-event simulation framework implemented in Python using SimPy and NumPy to evaluate adaptive dynamic synchronization mechanisms against standard heuristic baselines.

---

## Primary Research Question
> *How does a Deep Reinforcement Learning (DRL)-driven adaptive synchronization policy compare to standard fixed-rate and threshold-based baselines in minimizing cloud bandwidth overhead and state-drift latency while maintaining twin fidelity under dynamic IoT workloads?*

---

## Repository Architecture

Digital-Twin-Cloud-sim
├── configs/
│   └── sim_config.yaml       # Simulation hyperparameters and baseline thresholds
├── data/                     # Generated telemetry traces and experimental logs
├── docs/                     # Research proposal and PDF poster documents
├── src/
│   ├── agents/               # DRL policy implementations (PyTorch)
│   ├── baselines/            # Fixed-rate and Delta-threshold policies
│   │   ├── delta_threshold.py
│   │   └── fixed_rate.py
│   ├── __init__.py
│   ├── sim_engine.py         # Main SimPy simulation engine runner
│   └── telemetry.py          # Synthetic Edge telemetry stream generator
├── tests/                    # Unit testing suite
├── .gitignore                # Git environment ignore rules
├── requirements.txt          # Python dependency specifications
└── README.md                 # Public documentation


---

## Quick Start

### 1. Environment Activation

.\venv\Scripts\Activate


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run Benchmark Simulation Engine

python src/sim_engine.py


---

## Reproducibility Statement

Reproducibility: The code and reproducibility materials for this study are publicly available at: https://github.com/mmKamalUni/Digital-Twin-Cloud-sim
