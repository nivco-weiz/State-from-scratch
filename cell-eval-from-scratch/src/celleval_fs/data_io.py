# cell-eval-from-scrath/src/celleval_fs/data_io.py
from typing import List, Optional
import numpy as np
from cellload_fs.io import read_adata_folder
from .types import EvalData

def load_eval_data(
	folders: List[str],
	embed_key: str= "X_selected_genes",
	pert_col: str = "gene",
	cell_type_key: str = "cell_type",
	batch_col: str = "gem_group",
	control_label: str = "non-targeting",
	feature_names_source: Optional[str] = None,  # "var", "uns:selected_genes", or None
) -> EvalData:
	Xs, perts, cts, bas = [], [], [], []

	gene_names: Optional[List[str]] = None

	for folder in folders:
		for adata in read_adata_folder(folder):
			X = adata.obsm[embed_key].astype(np.float32)
			Xs.append(X)

			obs = adata.obs[[pert_col, cell_type_key, batch_col]]
			perts.append(obs[pert_col].values)
			cts.append(obs[cell_type_key].values)
			bas.append(obs[batch_col].values)

			# try to capture feature names once
			if gene_names is None:
				if feature_names_source == "var" and getattr(adata, "var_names", None) is not None:
					# ensure lengths match
					if len(adata.var_names) == X.shape[1]:
						gene_names = list(map(str, adata.var_names))
				elif feature_names_source and feature_names_source.startswith("uns:"):
					key = feature_names_source.split("uns:", 1)[1]
					if key in adata.uns and len(adata.uns[key]) == X.shape[1]:
						gene_names = list(map(str, adata.uns[key]))

	if gene_names is None:
		# fallback
		gene_names = [f"g{i}" for i in range(Xs[0].shape[1])]

	X = np.concatenate(Xs, axis=0)
	pert = np.concatenate(perts)
	cell_type = np.concatenate(cts)
	batch = np.concatenate(bas)

	return EvalData(
		X=X, pert=pert, cell_type=cell_type, batch=batch,
		feature_names=gene_names, control_label=control_label
	)


