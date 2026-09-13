class FixedRatePolicy:
    """Fixed-Rate Baseline: Syncs telemetry at static periodic time intervals."""
    def __init__(self, interval=2.0):
        self.interval = interval
        self.last_sync_time = -interval

    def should_sync(self, current_time, current_val, last_synced_val):
        if (current_time - self.last_sync_time) >= self.interval:
            self.last_sync_time = current_time
            return True
        return False