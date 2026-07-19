import numpy as np

def calculate_metrics(processes):
    n = len(processes)
    if n == 0:
        return 0.0, 0.0, 1.0
        
    wait_times = np.array([p.wait_time for p in processes], dtype=np.float64)
    
    mean_wait = float(np.mean(wait_times))
    p90_wait = float(np.percentile(wait_times, 90))
    
    # Jain's Fairness Index Formula
    sum_x = np.sum(wait_times)
    sum_sq_x = np.sum(wait_times ** 2)
    
    if sum_sq_x == 0:
        jains_index = 1.0  # Perfectly fair distribution if nobody waited
    else:
        jains_index = float((sum_x ** 2) / (n * sum_sq_x))
        
    return mean_wait, p90_wait, jains_index