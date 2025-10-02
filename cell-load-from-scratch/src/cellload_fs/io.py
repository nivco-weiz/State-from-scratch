# src/cellload_fs/io.py
from pathlib import Path
import anndata as ad

def read_adata_folder(folder: str):
	folder = Path(folder)
	for f in sorted(folder.glob("*.h5ad")):
		yield ad.read_h5ad(f)
