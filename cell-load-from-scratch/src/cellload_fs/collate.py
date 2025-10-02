# src/cellload_fs/simple_collate.py
import torch

def simple_collate(batch):
	out = {}
	for k in batch[0]:
		v0 = batch[0][k]
		if isinstance(v0, torch.Tensor):
			out[k] = torch.stack([b[k] for b in batch], axis=0)
		else:
			out[k] = [b[k] for b in batch]
	return out
