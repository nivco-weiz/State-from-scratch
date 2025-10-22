# cell-load-from-scratch/scripts/make_toy_h5ad.py
import numpy as np, pandas as pd, anndata as ad, os
rng = np.random.default_rng(0)

OUT = os.environ.get("OUT", "data/toyset")
os.makedirs(OUT, exist_ok=True)

def make_celltype(name: str, n_cells=600, n_genes=1000):
	X = rng.negative_binomial(n=10, p=0.8, size=(n_cells,n_genes)).astype(np.float32)

	# three perturbations: "AARS", "NUP107", and a non-targeting control
	perts = rng.choice(["AARS", "NUP107","non-targeting"],
		 	size=n_cells, p=[0.3,0.3,0.4])

	# introduce a perturbation signal
	for i, p in enumerate(perts):
		if p == "AARS":
			X[i,:10] -= 1.0
		elif p == "NUP107":
			X[i, 10:20] -= 1.2

	cell_names = [f"cell_{i}" for i in range(n_cells)]
	gene_names = [f"gene_{i}" for i in range(n_genes)]

	obs = pd.DataFrame({
		"gene": perts,
		"cell_type": name,
		"gem_group": "plate1",
	}, index=pd.Index(cell_names, name="cell_name"))
	var = pd.DataFrame({}, index=pd.Index(gene_names, name="gene_name"))
	adata = ad.AnnData(X=X, obs=obs, var=var)

	selected_genes = gene_names[:128]
	adata.obsm["X_selected_genes"] = adata[:,selected_genes].X
	adata.uns["selected_genes"] = selected_genes
	return adata

for ct in ["jurkat", "rpe1"]:
	make_celltype(ct).write_h5ad(f"{OUT}/{ct}.h5ad")

print("Wrote files:", [f for f in os.listdir(OUT) if f.endswith(".h5ad")])

