# cell-eval-from-scratch/scripts/smoke_test_metrics.py
from pathlib import Path
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "cell-load-from-scratch" / "src"))

from cellload_fs import Config
from celleval_fs.data_io import load_eval_data
from celleval_fs.evaluate import evaluate_predictions, EvalConfig

root = Path(__file__).parents[2]
cfg = Config.from_toml(str(root / "cell-load-from-scratch" / "configs" / "toy.toml"))
folders = list(cfg.datasets.values())

# Real data
ed_real = load_eval_data(folders, feature_names_source="uns:selected_genes")

# Fake a model prediction by adding small Gaussian noise to X
ed_pred = load_eval_data(folders, feature_names_source="uns:selected_genes")
rng = np.random.default_rng(0)
ed_pred.X = ed_pred.X + rng.normal(0, 0.1, size=ed_pred.X.shape).astype(ed_pred.X.dtype)

# Evaluate
out = evaluate_predictions(ed_real, ed_pred, EvalConfig(mode="by_celltype", topk=50))
print("=== Per-group results (head) ===")
print(out["results"].head().to_string(index=False))
print("\n=== Aggregate ===")
print(out["agg_results"].head().to_string(index=False))

# Optional: save CSVs
out_dir = root / "cell-eval-from-scratch" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out["results"].to_csv(out_dir / "results.csv", index=False)
out["agg_results"].to_csv(out_dir / "agg_results.csv", index=False)
print(f"\nWrote: {out_dir/'results.csv'} and {out_dir/'agg_results.csv'}")

