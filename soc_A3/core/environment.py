import numpy as np
import copy
from core.simulator import run_fcfs, run_sjf, run_rr

class MetaSchedulerEnv:
    def __init__(self, context_switch_tax=0):
        self.context_switch_tax = context_switch_tax
        self.processes = []
        self.current_time = 0
        self.ready_queue = []
        self.unarrived = []
        self.completed = []
        self.last_pid = None
        
        # Action space: 0: FCFS, 1: SJF, 2: RR-Short (q=2), 3: RR-Long (q=8)
        self.action_space_size = 4
        
        # State space dimensions: Queue Length Buckets x Avg Burst Buckets
        self.state_dims = (5, 5)

    def _get_state_index(self):
        q_len = len(self.ready_queue)
        q_bucket = min(q_len // 2, self.state_dims[0] - 1)
        
        if q_len == 0:
            avg_burst = 0
        else:
            avg_burst = np.mean([p.remaining_time for p in self.ready_queue])
            
        burst_bucket = min(int(avg_burst // 4), self.state_dims[1] - 1)
        return (q_bucket, burst_bucket)

    def reset(self, processes):
        self.processes = copy.deepcopy(processes)
        for p in self.processes:
            p.reset()
        self.current_time = 0
        self.completed = []
        self.last_pid = None
        self.unarrived = sorted(self.processes, key=lambda x: x.arrival_time)
        
        # Advance clock to first arrival if queue empty
        if self.unarrived:
            self.current_time = self.unarrived[0].arrival_time
            
        self.ready_queue = [p for p in self.unarrived if p.arrival_time <= self.current_time]
        self.unarrived = [p for p in self.unarrived if p.arrival_time > self.current_time]
        
        return self._get_state_index()

    def step(self, action):
        if not self.ready_queue:
            done = (len(self.unarrived) == 0)
            return self._get_state_index(), 0, done

        # Sort queue based on selected macro heuristic framework
        if action == 0:  # FCFS structure
            self.ready_queue.sort(key=lambda x: x.arrival_time)
            quantum = 999999
        elif action == 1:  # SJF structure
            self.ready_queue.sort(key=lambda x: x.remaining_time)
            quantum = 999999
        elif action == 2:  # RR-Short
            quantum = 2
        elif action == 3:  # RR-Long
            quantum = 8

        p = self.ready_queue.pop(0)

        # Apply context switch overhead cost
        if self.last_pid is not None and self.last_pid != p.pid:
            self.current_time += self.context_switch_tax
            while self.unarrived and self.unarrived[0].arrival_time <= self.current_time:
                self.ready_queue.append(self.unarrived.pop(0))

        if p.start_time is None:
            p.start_time = self.current_time

        exec_time = min(p.remaining_time, quantum)
        p.remaining_time -= exec_time
        self.current_time += exec_time

        # Absorb incoming process packets
        while self.unarrived and self.unarrived[0].arrival_time <= self.current_time:
            self.ready_queue.append(self.unarrived.pop(0))

        if p.remaining_time > 0:
            self.ready_queue.append(p)
            self.last_pid = p.pid
        else:
            p.finish_time = self.current_time
            p.wait_time = p.finish_time - p.arrival_time - p.burst_time
            self.completed.append(p)
            self.last_pid = None

        # Age remaining items & calculate step performance metrics
        step_wait_penalty = 0
        starvation_penalty = 0
        
        for item in self.ready_queue:
            wait_in_queue = self.current_time - max(item.arrival_time, self.current_time - exec_time)
            step_wait_penalty += wait_in_queue
            
            # Non-linear quadratic penalty to combat severe localized long-term starvation
            current_total_wait = self.current_time - item.arrival_time
            if current_total_wait > 30:
                starvation_penalty += 0.05 * (current_total_wait ** 2)

        reward = -(step_wait_penalty + starvation_penalty)

        # Advance timeline if system idle but jobs remain
        if not self.ready_queue and self.unarrived:
            self.current_time = self.unarrived[0].arrival_time
            while self.unarrived and self.unarrived[0].arrival_time <= self.current_time:
                self.ready_queue.append(self.unarrived.pop(0))
            self.last_pid = None

        done = (len(self.unarrived) == 0 and len(self.ready_queue) == 0)
        return self._get_state_index(), reward, done