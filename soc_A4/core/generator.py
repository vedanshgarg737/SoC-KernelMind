import random
from typing import List
from .process import Process

class WorkloadGenerator:
    @staticmethod
    def generate_uniform(num_processes: int = 20, max_arrival: int = 30, max_burst: int = 20, seed: int = None) -> List[Process]:
        if seed is not None:
            random.seed(seed)
        processes = []
        for pid in range(num_processes):
            arr = random.randint(0, max_arrival)
            burst = random.randint(1, max_burst)
            prio = random.randint(1, 5)
            processes.append(Process(pid, arr, burst, prio))
        return sorted(processes, key=lambda x: x.arrival_time)

    @staticmethod
    def generate_io_storm(seed: int = None) -> List[Process]:
        if seed is not None:
            random.seed(seed)
        processes = []
        # 8 short IO-bound jobs arriving at tick 0
        for pid in range(8):
            processes.append(Process(pid=pid, arrival_time=0, burst_time=random.randint(1, 2), priority=1))
        # 2 long CPU-bound jobs arriving at tick 0
        for pid in range(8, 10):
            processes.append(Process(pid=pid, arrival_time=0, burst_time=random.randint(50, 60), priority=1))
        return sorted(processes, key=lambda x: x.arrival_time)