# src/cellload_fs/config.py
from dataclasses import dataclass
import tomli

@dataclass
class Config:
	datasets: dict
	training: dict

	@staticmethod
	def from_toml(path: str) -> "Config":
		with open(path, "rb") as f:
			raw = tomli.load(f)
		return Config(datasets=raw.get("datasets", {}), training=raw.get("training", {}))

