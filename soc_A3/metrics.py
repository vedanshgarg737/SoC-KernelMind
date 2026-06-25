import numpy as np
from typing import List
from core.process import Process

class MetricsUtility:
    @staticmethod
    def calculate_metrics(completed_processes: List[Process]) -> dict:
        """
        Calculates Mean Wait Time, P90 Wait Time, and Jain's Fairness Index 
        from completed processes.
        """
        if not completed_processes:
            return {"mean_wait": 0.0, "p90_wait": 0.0, "jains_fairness": 1.0}

        wait_times = np.array([p.wait_time for p in completed_processes], dtype=float)
        n = len(wait_times)

        # 1. Mean Wait Time
        mean_wait = float(np.mean(wait_times))

        # 2. P90 Wait Time (90th percentile)
        p90_wait = float(np.percentile(wait_times, 90))

        # 3. Jain's Fairness Index
        sum_x = np.sum(wait_times)
        sum_x_sq = np.sum(wait_times ** 2)
        
        # Handle edge case where all wait times are exactly 0 (perfectly fair initialization)
        if sum_x_sq == 0:
            jains_fairness = 1.0
        else:
            jains_fairness = float((sum_x ** 2) / (n * sum_x_sq))

        return {
            "mean_wait": round(mean_wait, 2),
            "p90_wait": round(p90_wait, 2),
            "jains_fairness": round(jains_fairness, 4)
        }