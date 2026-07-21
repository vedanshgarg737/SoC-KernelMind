from typing import List, Tuple
from .process import Process
from utils.metrics import calculate_metrics

class BaselineSimulators:
    @staticmethod
    def simulate_fcfs(processes: List[Process]) -> Tuple[List[Process], dict]:
        procs = [p.copy() for p in processes]
        current_time = 0
        ready_queue = []
        completed = []
        unarrived = sorted(procs, key=lambda p: p.arrival_time)
        
        running_proc = None
        while unarrived or ready_queue or running_proc:
            while unarrived and unarrived[0].arrival_time <= current_time:
                ready_queue.append(unarrived.pop(0))
            
            if running_proc is None and ready_queue:
                running_proc = ready_queue.pop(0)
                if running_proc.start_time == -1:
                    running_proc.start_time = current_time
                    
            if running_proc:
                running_proc.remaining_time -= 1
                running_proc.virtual_runtime += 1.0
                for p in ready_queue:
                    p.wait_time += 1
                current_time += 1
                if running_proc.remaining_time == 0:
                    running_proc.finish_time = current_time
                    completed.append(running_proc)
                    running_proc = None
            else:
                current_time += 1
                
        return completed, calculate_metrics(completed)

    @staticmethod
    def simulate_sjf(processes: List[Process]) -> Tuple[List[Process], dict]:
        procs = [p.copy() for p in processes]
        current_time = 0
        ready_queue = []
        completed = []
        unarrived = sorted(procs, key=lambda p: p.arrival_time)
        running_proc = None
        
        while unarrived or ready_queue or running_proc:
            while unarrived and unarrived[0].arrival_time <= current_time:
                ready_queue.append(unarrived.pop(0))
            
            if running_proc is None and ready_queue:
                ready_queue.sort(key=lambda p: p.remaining_time)
                running_proc = ready_queue.pop(0)
                if running_proc.start_time == -1:
                    running_proc.start_time = current_time
                    
            if running_proc:
                running_proc.remaining_time -= 1
                running_proc.virtual_runtime += 1.0
                for p in ready_queue:
                    p.wait_time += 1
                current_time += 1
                if running_proc.remaining_time == 0:
                    running_proc.finish_time = current_time
                    completed.append(running_proc)
                    running_proc = None
            else:
                current_time += 1
                
        return completed, calculate_metrics(completed)

    @staticmethod
    def simulate_rr(processes: List[Process], quantum: int = 2) -> Tuple[List[Process], dict]:
        procs = [p.copy() for p in processes]
        current_time = 0
        ready_queue = []
        completed = []
        unarrived = sorted(procs, key=lambda p: p.arrival_time)
        running_proc = None
        quantum_clock = 0
        
        while unarrived or ready_queue or running_proc:
            while unarrived and unarrived[0].arrival_time <= current_time:
                ready_queue.append(unarrived.pop(0))
            
            if running_proc is None and ready_queue:
                running_proc = ready_queue.pop(0)
                quantum_clock = 0
                if running_proc.start_time == -1:
                    running_proc.start_time = current_time

            if running_proc:
                running_proc.remaining_time -= 1
                running_proc.virtual_runtime += 1.0
                quantum_clock += 1
                for p in ready_queue:
                    p.wait_time += 1
                current_time += 1
                
                if running_proc.remaining_time == 0:
                    running_proc.finish_time = current_time
                    completed.append(running_proc)
                    running_proc = None
                elif quantum_clock == quantum:
                    # Time quantum expired, re-queue
                    while unarrived and unarrived[0].arrival_time <= current_time:
                        ready_queue.append(unarrived.pop(0))
                    ready_queue.append(running_proc)
                    running_proc = None
            else:
                current_time += 1
                
        return completed, calculate_metrics(completed)