import os
import sys
import simpy
import yaml
import numpy as np

# Resolve local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telemetry import TelemetryGenerator
from baselines.fixed_rate import FixedRatePolicy
from baselines.delta_threshold import DeltaThresholdPolicy

class CloudDigitalTwinSim:
    """SimPy discrete-event simulation engine for Edge-Cloud Digital Twin synchronization."""
    def __init__(self, config_path="configs/sim_config.yaml", policy_type="fixed"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.env = simpy.Environment()
        self.generator = TelemetryGenerator(
            base_val=self.config["workload"]["base_val"],
            noise_std=self.config["workload"]["noise_std"],
            burst_lambda=self.config["workload"]["burst_lambda"],
            burst_magnitude=self.config["workload"]["burst_magnitude"]
        )

        self.policy_type = policy_type
        if policy_type == "fixed":
            self.policy = FixedRatePolicy(interval=self.config["baselines"]["fixed_interval"])
        elif policy_type == "delta":
            self.policy = DeltaThresholdPolicy(threshold=self.config["baselines"]["delta_threshold"])
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        self.physical_state = self.config["workload"]["base_val"]
        self.cloud_twin_state = self.config["workload"]["base_val"]
        
        self.sync_count = 0
        self.drift_errors = []

    def run_device(self):
        while True:
            self.physical_state = self.generator.step()
            current_time = self.env.now

            if self.policy.should_sync(current_time, self.physical_state, self.cloud_twin_state):
                self.cloud_twin_state = self.physical_state
                self.sync_count += 1
                print(f"[{current_time:.1f}s] SYNC ({self.policy_type.upper()}) | Twin State Updated: {self.cloud_twin_state:.2f}")

            drift = abs(self.physical_state - self.cloud_twin_state)
            self.drift_errors.append(drift)

            yield self.env.timeout(self.config["simulation"]["time_step"])

    def execute(self):
        self.env.process(self.run_device())
        self.env.run(until=self.config["simulation"]["duration"])
        
        mae = np.mean(self.drift_errors)
        print(f"--- Results ({self.policy_type.upper()}) ---")
        print(f"Simulation Runtime: {self.config['simulation']['duration']}s")
        print(f"Total Cloud Sync Events: {self.sync_count}")
        print(f"State-Drift MAE: {mae:.4f}\n")

if __name__ == "__main__":
    print("=== Executing Baseline Simulation Benchmarks ===\n")
    sim_fixed = CloudDigitalTwinSim(policy_type="fixed")
    sim_fixed.execute()

    sim_delta = CloudDigitalTwinSim(policy_type="delta")
    sim_delta.execute()