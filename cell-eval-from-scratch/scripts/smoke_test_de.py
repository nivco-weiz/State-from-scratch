# cell-eval-from-scratch/scripts/smoke_test_de.py
from pathlib import Path
import sys
# add eval src
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
# add cell-load src so read_adata_folder/Config are importable
sys.path.insert(0, str(Path(__file__).parents[2] / "cell-load-from-scratch" / "src"))

from celleval_fs.data_io import load_eval_data
from celleval_fs.de import run_de
from cellload_fs import Config

root = Path(__file__).parents[2]
cfg = Config.from_toml(str(root / "cell-load-from-scratch" / "configs" / "toy.toml"))
folders = list(cfg.datasets.values())

ed = load_eval_data(
	folders,
	embed_key="X_selected_genes",
	pert_col="gene",
	cell_type_key="cell_type",
	batch_col="gem_group",
	control_label="non-targeting",
	feature_names_source="uns:selected_genes",
)

# 1) pooled
pooled = run_de(ed, mode="pooled")
print("POOLED keys:", list(pooled.records.keys())[:5])
for key, rec in pooled.records.items():
	print(f"{key} n_pert={rec.n_pert} n_ctrl={rec.n_ctrl} delta[:5]={rec.delta[:5]}")
	break

# 2) by cell type
byct = run_de(ed, mode="by_celltype")
print("BY-CT keys:", list(byct.records.keys())[:8])
for key, rec in list(byct.records.items())[:2]:
	print(f"{key} n_pert={rec.n_pert} n_ctrl={rec.n_ctrl} delta[:5]={rec.delta[:5]}")


