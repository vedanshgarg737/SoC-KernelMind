import numpy as np

class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        
        self.start_time = None
        self.finish_time = None
        self.wait_time = 0

    def reset(self):
        self.remaining_time = self.burst_time
        self.start_time = None
        self.finish_time = None
        self.wait_time = 0

def generate_workload(num_processes=10, mode="uniform"):
    processes = []
    if mode == "uniform":
        for i in range(num_processes):
            arrival = int(np.random.randint(0, 15))
            burst = int(np.random.randint(2, 25))
            processes.append(Process(i, arrival, burst))
    elif mode == "io_storm":
        # 8 brief jobs arriving instantly, 2 massive background hogs
        for i in range(8):
            processes.append(Process(i, 0, int(np.random.randint(1, 3))))
        for i in range(8, 10):
            processes.append(Process(i, 0, int(np.random.randint(50, 65))))
    
    # Sort by arrival time to maintain physical consistency
    processes.sort(key=lambda p: p.arrival_time)
    return processes