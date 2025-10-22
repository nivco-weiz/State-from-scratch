# cell-eval-from-scratch/src/celleval_fs/metrics/basic.py
import numpy as np
from . import register

def _safe(x: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64)
	if not np.all(np.isfinite(x)):
		x = np.nan_to_sum(x, nan=0.0, posinf=0.0, neginf=0.0)
	return x

@register("pearson")
def pearson(a: np.ndarray, b: np.ndarray) -> float:
	a, b = _safe(a), _safe(b)
	if a.size < 2: return float("nan")
	va = a - a.mean()
	vb = b - b.mean()
	den = (np.linalg.norm(va) * np.linalg.norm(vb))
	return float(va @ vb / den) if den > 0 else 0.0

@register("spearman")
def spearman(a: np.ndarray, b: np.ndarray) -> float:
	a, b = _safe(a), _safe(b)
	if a.size < 2: return float("nan")
	ra = np.argsort(np.argsort(a))
	rb = np.argsort(np.argsort(b))
	return pearson(ra, rb)

@register("cosine")
def cosine(a: np.ndarray, b: np.ndarray) -> float:
	a, b = _safe(a), _safe(b)
	den = (np.linalg.norm(a) * np.linalg.norm(b))
	return float((a @ b) / den) if den > 0 else 0.0

@register("mse")
def mse(a: np.ndarray, b: np.ndarray) -> float:
	a, b = _safe(a), _safe(b)
	d = a - b
	return float((d @ d) / d.size)

def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
	a, b = _safe(a), _safe(b)
	k = int(min(k, a.size, b.size))
	if k <= 0: return 0.0
	# rank by absolute effect size
	idx_a = np.argpartition(-np.abs(a), k-1)[:k]
	idx_b = np.argpartition(-np.abs(b), k-1)[:k]
	return float(len(set(idx_a).intersection(set(idx_b))) / k)

def sign_prec_at_topk(a: np.ndarray, b: np.ndarray, k: int) -> float:
	a, b = _safe(a), _safe(b)
	k = int(min(k, a.size, b.size))
	if k <= 0: return 0.0
	# test sign precision on top_k genes w.r.t. |b|
	idx_b = np.argpartition(-np.abs(b), k-1)[:k]
	real = np.sign(b[idx_b])
	pred = np.sign(a[idx_b])
	return float((real == pred).mean())
