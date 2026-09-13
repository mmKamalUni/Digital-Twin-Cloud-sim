import numpy as np
import random

class TelemetryGenerator:
    """Generates continuous physical asset telemetry with Gaussian noise and Poisson-driven operational bursts."""
    def __init__(self, base_val=100.0, noise_std=0.5, burst_lambda=0.1, burst_magnitude=5.0):
        self.current_val = base_val
        self.noise_std = noise_std
        self.burst_lambda = burst_lambda
        self.burst_magnitude = burst_magnitude

    def step(self):
        """Calculates the physical asset metric transition for a single time step."""
        noise = np.random.normal(0, self.noise_std)
        burst = np.random.poisson(lam=self.burst_lambda) * random.choice([-1.0, 1.0]) * self.burst_magnitude
        self.current_val += noise + burst
        return self.current_val
