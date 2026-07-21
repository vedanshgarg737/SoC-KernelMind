import numpy as np
from typing import List, Dict
from core.process import Process

def calculate_metrics(completed_processes: List[Process]) -> Dict[str, float]:
    if not completed_processes:
        return {"mean_wait_time": 0.0, "p90_wait_time": 0.0, "jain_fairness": 0.0}
    
    wait_times = [p.wait_time for p in completed_processes]
    mean_wait = float(np.mean(wait_times))
    p90_wait = float(np.percentile(wait_times, 90))
    
    # Jain's Fairness Index on Wait Times: (sum(w)^2) / (n * sum(w^2))
    # Note: Using wait time directly where lower wait time is better.
    # Standard formula mapped on service/fairness metrics:
    # Scale wait times using inverse/effective service factor or standard allocation
    n = len(wait_times)
    sum_w = sum(wait_times)
    sum_sq_w = sum(w ** 2 for w in wait_times)
    
    if sum_sq_w == 0:
        jain = 1.0
    else:
        # Measure fairness over completion turnaround / wait distribution
        # Invert wait times to model quality-of-service (QoS)
        qos = [1.0 / (w + 1.0) for w in wait_times]
        jain = float((sum(qos) ** 2) / (n * sum(q ** 2 for q in qos)))
        
    return {
        "mean_wait_time": mean_wait,
        "p90_wait_time": p90_wait,
        "jain_fairness": jain
    }