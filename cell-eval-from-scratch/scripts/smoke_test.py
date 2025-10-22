# cell-eval-from-scratch/scripts/smoke_test.py
from pathlib import Path
import sys
# add eval src
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
# add cell-load src so read_adata_folder is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "cell-load-from-scratch" / "src"))

from celleval_fs.data_io import load_eval_data
from cellload_fs import Config  # reuse your toy config and read_adata_folder

root = Path(__file__).parents[2]  # repo root
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

print("X shape:", ed.X.shape)
print("perts (unique):", sorted(set(ed.pert)))
print("cell types:", sorted(set(ed.cell_type)))
print("batches:", sorted(set(ed.batch)))
print("first 5 feature names:", ed.feature_names[:5])
print("control label:", ed.control_label)
