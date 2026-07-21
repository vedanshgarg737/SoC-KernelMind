import numpy as np
import torch
from typing import Tuple, List, Dict
from core.process import Process

class DirectSchedulerEnvironment:
    def __init__(self, queue_window_size: int = 10, max_burst_ref: float = 20.0, max_wait_ref: float = 100.0):
        self.N = queue_window_size
        self.max_burst_ref = max_burst_ref
        self.max_wait_ref = max_wait_ref
        self.reset_state()

    def reset_state(self):
        self.current_time = 0
        self.ready_queue: List[Process] = []
        self.unarrived_processes: List[Process] = []
        self.completed_processes: List[Process] = []

    def reset(self, processes: List[Process]) -> Tuple[np.ndarray, np.ndarray]:
        self.reset_state()
        self.unarrived_processes = sorted([p.copy() for p in processes], key=lambda x: x.arrival_time)
        self._admit_processes()
        return self._get_observation()

    def _admit_processes(self):
        # Admit all processes that have arrived up to current_time
        while self.unarrived_processes and self.unarrived_processes[0].arrival_time <= self.current_time:
            self.ready_queue.append(self.unarrived_processes.pop(0))

        # CPU Idle handling: If ready_queue is empty but processes are still waiting to arrive,
        # advance current_time directly to the next process's arrival time.
        if not self.ready_queue and self.unarrived_processes:
            self.current_time = self.unarrived_processes[0].arrival_time
            while self.unarrived_processes and self.unarrived_processes[0].arrival_time <= self.current_time:
                self.ready_queue.append(self.unarrived_processes.pop(0))

    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        # Feature Matrix: (N, 3) -> [norm_remaining_burst, norm_wait_time, norm_priority]
        state = np.zeros((self.N, 3), dtype=np.float32)
        mask = np.zeros(self.N, dtype=bool)

        visible_procs = self.ready_queue[:self.N]
        for idx, proc in enumerate(visible_procs):
            norm_burst = min(proc.remaining_time / self.max_burst_ref, 1.0)
            norm_wait = min(proc.wait_time / self.max_wait_ref, 1.0)
            norm_prio = proc.priority / 5.0
            
            state[idx] = [norm_burst, norm_wait, norm_prio]
            mask[idx] = True  # Valid process slot

        return state, mask

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        state_tensor, mask = self._get_observation()
        
        # Action space validation against mask
        if not mask[action]:
            raise ValueError(f"Invalid Action Selected: Action slot {action} is padded/invalid!")

        # Execute chosen process for 1 tick
        selected_proc = self.ready_queue.pop(action)
        if selected_proc.start_time == -1:
            selected_proc.start_time = self.current_time

        selected_proc.remaining_time -= 1
        selected_proc.virtual_runtime += 1.0

        # Increment wait time for all other processes in the queue
        for proc in self.ready_queue:
            proc.wait_time += 1

        self.current_time += 1
        self._admit_processes()

        job_completed = False
        if selected_proc.remaining_time == 0:
            selected_proc.finish_time = self.current_time
            self.completed_processes.append(selected_proc)
            job_completed = True
        else:
            # Re-insert back into ready queue
            self.ready_queue.append(selected_proc)

        # Calculate Bounded Reward Structure
        reward = self._compute_reward(job_completed, selected_proc)
        
        done = (len(self.ready_queue) == 0) and (len(self.unarrived_processes) == 0)
        next_obs = self._get_observation()

        return next_obs, reward, done, {"completed_proc": selected_proc if job_completed else None}

    def _compute_reward(self, job_completed: bool, executed_proc: Process) -> float:
        # 1. Base Step Cost (Encourages system-wide urgency)
        r_step = -0.1

        # 2. Completion Bonus (Bounded)
        r_comp = +2.0 if job_completed else 0.0

        # 3. CFS Virtual Runtime Variance Penalty (Bounded using Tanh)
        if self.ready_queue:
            v_runtimes = [p.virtual_runtime for p in self.ready_queue] + [executed_proc.virtual_runtime]
            v_var = float(np.var(v_runtimes))
            r_cfs = -1.0 * np.tanh(v_var / 10.0)
        else:
            r_cfs = 0.0

        # 4. Starvation Cap Penalty (Escalating, capped via Tanh)
        threshold = 20.0
        max_wait = max([p.wait_time for p in self.ready_queue], default=0.0)
        if max_wait > threshold:
            excess = max_wait - threshold
            r_starve = -1.5 * np.tanh(excess / 25.0)
        else:
            r_starve = 0.0

        total_reward = r_step + r_comp + r_cfs + r_starve
        return float(total_reward)