import copy

def run_fcfs(processes_input):
    processes = copy.deepcopy(processes_input)
    current_time = 0
    ready_queue = []
    unarrived = sorted(processes, key=lambda p: p.arrival_time)
    completed = []

    while unarrived or ready_queue:
        # Transfer arrived processes
        while unarrived and unarrived[0].arrival_time <= current_time:
            ready_queue.append(unarrived.pop(0))
            
        if not ready_queue:
            current_time = unarrived[0].arrival_time
            continue
            
        p = ready_queue.pop(0)
        p.start_time = current_time
        p.wait_time = p.start_time - p.arrival_time
        current_time += p.burst_time
        p.finish_time = current_time
        p.remaining_time = 0
        completed.append(p)
        
    return completed

def run_sjf(processes_input):
    processes = copy.deepcopy(processes_input)
    current_time = 0
    ready_queue = []
    unarrived = sorted(processes, key=lambda p: p.arrival_time)
    completed = []

    while unarrived or ready_queue:
        while unarrived and unarrived[0].arrival_time <= current_time:
            ready_queue.append(unarrived.pop(0))
            
        if not ready_queue:
            current_time = unarrived[0].arrival_time
            continue
            
        # Non-preemptive shortest burst choice
        ready_queue.sort(key=lambda x: x.burst_time)
        p = ready_queue.pop(0)
        p.start_time = current_time
        p.wait_time = p.start_time - p.arrival_time
        current_time += p.burst_time
        p.finish_time = current_time
        p.remaining_time = 0
        completed.append(p)
        
    return completed

def run_rr(processes_input, quantum=4, context_switch_tax=0):
    processes = copy.deepcopy(processes_input)
    current_time = 0
    unarrived = sorted(processes, key=lambda p: p.arrival_time)
    ready_queue = []
    completed = []
    last_pid = None

    while unarrived or ready_queue:
        while unarrived and unarrived[0].arrival_time <= current_time:
            ready_queue.append(unarrived.pop(0))
            
        if not ready_queue:
            current_time = unarrived[0].arrival_time
            continue
            
        p = ready_queue.pop(0)
        
        # Enforce context switch tax if transitioning between different processes
        if last_pid is not None and last_pid != p.pid:
            current_time += context_switch_tax
            # Re-verify arrivals during context switch slice
            while unarrived and unarrived[0].arrival_time <= current_time:
                ready_queue.append(unarrived.pop(0))
                
        if p.start_time is None:
            p.start_time = current_time
            
        execution_slice = min(p.remaining_time, quantum)
        p.remaining_time -= execution_slice
        current_time += execution_slice
        
        # Pull new arrivals during execution interval
        while unarrived and unarrived[0].arrival_time <= current_time:
            ready_queue.append(unarrived.pop(0))
            
        if p.remaining_time > 0:
            ready_queue.append(p)
            last_pid = p.pid
        else:
            p.finish_time = current_time
            p.wait_time = p.finish_time - p.arrival_time - p.burst_time
            completed.append(p)
            last_pid = None
            
    return completed