from typing import List
from core.process import Process
import copy

class BaseSimulator:
    def __init__(self, name: str):
        self.name = name

    def run(self, workload: List[Process]) -> List[Process]:
        """Deep copies workload and executes the scheduling simulation loop."""
        raise NotImplementedError("Simulators must implement the run method.")