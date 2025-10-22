# cell-eval-from-scratch/src/celleval_fs/de.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from .types import EvalData

@dataclass
class DERecord:
	delta: np.ndarray  # shape [n_features]
	n_pert: int
	n_ctrl: int
	feature_names: List[str]

@dataclass
class DEResult:
	# keys:
	# 	mode == "pooled": ("pooled", pert)
	# 	mode == "by_celltype": (cell_type, pert)
	records: Dict[Tuple[str, str], DERecord]
	mode: str  # "pooled" or "by_celltype"

def _mean_diff(X_pert: np.ndarray, X_ctrl: np.ndarray) -> np.ndarray:
	# simple, interpretable effect size
	return X_pert.mean(axis=0) - X_ctrl.mean(axis=0)

def run_de(
	ed: EvalData,
	mode: str = "pooled",  # "pooled" or "by_celltype"
	control_label: Optional[str] = None,  # override if needed
	min_cells: int = 3,  # guardrail for toy data
) -> DEResult:
	assert mode in ("pooled", "by_celltype")
	ctrl = control_label or ed.control_label

	X = ed.X
	pert = ed.pert
	ctype = ed.cell_type
	feats = ed.feature_names

	mask_ctrl_global = (pert == ctrl)
	if mask_ctrl_global.sum() < min_cells:
		raise RuntimeError(f"Not enough global control cells for label '{ctrl}'.")

	records: Dict[Tuple[str, str], DERecord] = {}

	# all perts to evaluate (exclude contrl)
	perts = np.unique(pert)
	perts = [p for p in perts if p != ctrl]

	if mode == "pooled":
		X_ctrl = X[mask_ctrl_global]
		n_ctrl = int(mask_ctrl_global.sum())
		for p in perts:
			mask_p = (pert == p)
			n_pert = int(mask_p.sum())
			if n_pert < min_cells:
				continue
			delta = _mean_diff(X[mask_p], X_ctrl)
			records[("pooled", str(p))] = DERecord(
			delta=delta, n_pert=n_pert, n_ctrl=n_ctrl, feature_names=feats
			)
	else:  # by celltype
		ctypes = np.unique(ctype)
		for ct in ctypes:
			mask_ct = (ctype == ct)
			# controls within this ct (fallback to global if none)
			mask_ctrl_ct = mask_ct & (pert == ctrl)
			if mask_ctrl_ct.sum() >= min_cells:
				X_ctrl = X[mask_ctrl_ct]
				n_ctrl = int(mask_ctrl_ct.sum())
			else:
				X_ctrl = X[mask_ctrl_global]
				n_ctrl = int(mask_ctrl_global.sum())

			for p in perts:
				mask_p = mask_ct & (pert == p)
				n_pert = int(mask_p.sum())
				if n_pert < min_cells:
					continue
				delta = _mean_diff(X[mask_p], X_ctrl)
				records[(str(ct), str(p))] = DERecord(
				delta=delta, n_pert=n_pert, n_ctrl=n_ctrl, feature_names=feats
				)
	return DEResult(records=records, mode=mode)
