# cell-eval-from-scratch/src/celleval_fs/types.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class EvalData:
	X: np.ndarray  # shape: [n_cells, n_features] from obsm["X_selected_genes"]
	pert: np.ndarray  # shape: [n_cells], from obs["gene"]
	cell_type: np.ndarray  # shape: [n_cells], from obs["cell_type"]
	batch: np.ndarray  # shape: [n_cells], from obs["gem_group"]
	feature_names: Optional[List[str]]  # len = n_features (genes or features)
	control_label: str  # "non-targeting"

