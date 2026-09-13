class DeltaThresholdPolicy:
    """Delta-Threshold Baseline: Syncs telemetry when state drift exceeds a set delta limit."""
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def should_sync(self, current_time, current_val, last_synced_val):
        if last_synced_val is None:
            return True
        return abs(current_val - last_synced_val) >= self.threshold