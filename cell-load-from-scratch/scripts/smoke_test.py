# scripts/smoke_test.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


from pathlib import Path
from torch.utils.data import DataLoader
from cellload_fs import Config, PerturbationDataset, ColumnMap, simple_collate

root = Path(__file__).parents[1]
cfg = Config.from_toml(str(root / "configs/toy.toml"))
folders = list(cfg.datasets.values())

ds = PerturbationDataset(
	folders,
	ColumnMap(
		pert_col="gene",
		cell_type_key="cell_type",
		batch_col="gem_group",
		control_pert="non-targeting",
		embed_key="X_selected_genes",
	),
)


dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=simple_collate)
batch = next(iter(dl))
print("keys:", list(batch.keys()))
for k, v in batch.items():
	print(k, getattr(v, "shape", f"list[{len(v)}] sample={v[0]}"))
