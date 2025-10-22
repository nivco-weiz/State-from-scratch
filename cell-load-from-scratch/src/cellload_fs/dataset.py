# cell-load-from-scratch/src/cellload_fs/dataset.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import torch
from cellload_fs.io import read_adata_folder

@dataclass
class ColumnMap:
	pert_col: str = "gene"
	cell_type_key: str = "cell_type"
	batch_col: str = "gem_group"
	control_pert: str = "non-targeting"
	embed_key: str = "X_selected_genes"


class PerturbationDataset(torch.utils.data.Dataset):
	"""
	Returns a dictionary with:
	 - pert_cell_emb: tensor[d]
	 - ctrl_cell_emb: tensor[d]
	 - pert_name: str
	 - cell_type: str
	 - batch: str
	"""
	def __init__(self, folders: List[str], cols: ColumnMap):
		self.cols = cols
		embs, perts, cts, bas = [], [], [], []

		for folder in folders:
			for adata in read_adata_folder(folder):
				E = adata.obsm[self.cols.embed_key].astype(np.float32)
				embs.append(E)
				obs = adata.obs[[self.cols.pert_col, self.cols.cell_type_key, self.cols.batch_col]]
				perts.append(obs[self.cols.pert_col].values)
				cts.append(obs[self.cols.cell_type_key].values)
				bas.append(obs[self.cols.batch_col].values)

		self.emb = np.concatenate(embs, axis=0)
		self.pert = np.concatenate(perts)
		self.ct = np.concatenate(cts)
		self.ba = np.concatenate(bas)

		# index control per (cell_type, batch)
		self.ctrl_map: Dict[tuple[str, str], np.ndarray] = {}
		for key in set(zip(self.ct, self.ba)):
			mask = (self.ct==key[0]) & (self.ba==key[1]) & (self.pert==self.cols.control_pert)
			self.ctrl_map[key] = np.flatnonzero(mask)

		self.global_ctrl = np.flatnonzero(self.pert==self.cols.control_pert)
		if self.global_ctrl.size==0:
			raise RuntimeError(f"No control cells ('{self.cols.control_pert}') found.")

		self.indices = np.arange(len(self.emb))

	def __len__(self): return self.indices.size

	def _sample_control(self, ct: str, ba: str, rng: np.random.Generator) -> int:
		idxs = self.ctrl_map.get((ct, ba))
		pool = idxs if (idxs is not None and idxs.size>0) else self.global_ctrl
		return int(rng.choice(pool))

	def __getitem__(self, i: int):
		rng = np.random.default_rng()
		p, ct, ba = self.pert[i], self.ct[i], self.ba[i]
		j = self._sample_control(ct, ba, rng)
		ei, ej = self.emb[i], self.emb[j]
		return {
			"pert_cell_emb": torch.from_numpy(ei),
			"ctrl_cell_emb": torch.from_numpy(ej),
			"pert_name": p,
			"cell_type": ct,
			"batch": ba,
		}

