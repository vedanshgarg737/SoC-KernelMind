class Process:
    def __init__(self, pid: int, arrival_time: int, burst_time: int, priority: int = 1):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.priority = priority
        
        # Extended fields for Direct Control Scheduler
        self.start_time = -1
        self.finish_time = -1
        self.wait_time = 0
        self.virtual_runtime = 0.0  # Cumulative service time executed on CPU

    @property
    def turn_around_time(self) -> int:
        return self.finish_time - self.arrival_time if self.finish_time != -1 else 0

    def copy(self):
        p = Process(self.pid, self.arrival_time, self.burst_time, self.priority)
        p.start_time = self.start_time
        p.finish_time = self.finish_time
        p.wait_time = self.wait_time
        p.virtual_runtime = self.virtual_runtime
        p.remaining_time = self.remaining_time
        return p