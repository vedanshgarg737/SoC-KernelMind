import random
from typing import List

class Process:
    def __init__(self, pid: int, arrival_time: int, burst_time: int):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        
        # Simulation tracking states
        self.start_time = -1
        self.finish_time = -1
        self.wait_time = 0
        self.turnaround_time = 0

    def reset(self):
        """Resets the process state for fresh simulator runs."""
        self.remaining_time = self.burst_time
        self.start_time = -1
        self.finish_time = -1
        self.wait_time = 0
        self.turnaround_time = 0

    def __repr__(self):
        return (f"Process(PID={self.pid}, Arrival={self.arrival_time}, "
                f"Burst={self.burst_time}, Remaining={self.remaining_time}, "
                f"Wait={self.wait_time}, Finish={self.finish_time})")


def generate_workload(num_processes: int = 10, seed: int = None) -> List[Process]:
    """
    Generates an array of random processes with randomized burst times 
    and staggered arrivals.
    """
    if seed is not None:
        random.seed(seed)
        
    processes = []
    current_arrival = 0
    
    for pid in range(num_processes):
        # Staggered arrivals: increment arrival times randomly
        current_arrival += random.randint(0, 4) 
        burst_time = random.randint(2, 25)
        processes.append(Process(pid=pid, arrival_time=current_arrival, burst_time=burst_time))
        
    return processes