# cell-eval-from-scratch/src/celleval_fs/metrics/__init__.py
from typing import Callable, Dict
import numpy as np

Registry: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}

def register(name: str):
	def _wrap(fn: Callable[[np.ndarray, np.ndarray], float]):
		Registry[name] = fn
		return fn
	return _wrap



