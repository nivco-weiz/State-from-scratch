# cell-eval-from-scratch/src/celleval_fs/evaluate.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from .types import EvalData
from .de import run_de, DEResult, DERecord
from .metrics import Registry
from .metrics.basic import topk_overlap, sign_prec_at_topk

@dataclass
class EvalConfig:
	mode: str = "by_celltype"  # "pooled" or "by_celltype"
	topk: int = 50

def _keys_in_both(a: DEResult, b: DEResult) -> List[Tuple[str, str]]:
	return sorted(set(a.records.keys()).intersection(set(b.records.keys())))

def evaluate_predictions(
	real: EvalData,
	pred: EvalData,
	cfg: Optional[EvalConfig] = None,
) -> Dict[str, pd.DataFrame]:
	cfg = cfg or EvalConfig()
	assert cfg.mode in ("pooled", "by_celltype")

	# run DE both sides
	de_real = run_de(real, mode=cfg.mode)
	de_pred = run_de(pred, mode=cfg.mode)

	rows = []
	keys = _keys_in_both(de_real, de_pred)
	for key in keys:
		rec_r: DERecord = de_real.records[key]
		rec_p: DERecord = de_pred.records[key]
		# feature order is identical by construction; if not, you'd align here.

		a = rec_p.delta
		b = rec_r.delta

		metrics = {
			"pearson": Registry["pearson"](a,b),
                        "spearman": Registry["spearman"](a,b),
                        "cosine": Registry["cosine"](a,b),
                       	"mse": Registry["mse"](a,b),
                        f"overlap_top{cfg.topk}": topk_overlap(a, b, cfg.topk),
                        f"sign_prec_at_top{cfg.topk}": sign_prec_at_topk(a, b, cfg.topk),
		}
		row = {
			"group": key[0],
			"pert": key[1],
			"n_pert": rec_r.n_pert,
			"n_ctrl": rec_r.n_ctrl,
			**metrics,
		}
		rows.append(row)

	df = pd.DataFrame(rows).sort_values(["group", "pert"]).reset_index(drop=True)
	agg = df.drop(columns=["group", "pert"]).mean(numeric_only=True).to_frame().T
	return {"results": df, "agg_results": agg}

