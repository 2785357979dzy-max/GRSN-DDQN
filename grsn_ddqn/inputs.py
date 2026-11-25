from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RoadNetwork:
    endpoints: np.ndarray
    length_m: np.ndarray
    speed_mps: np.ndarray
    population: np.ndarray
    importance: np.ndarray

    def __post_init__(self):
        for key in ("endpoints", "length_m", "speed_mps", "population", "importance"):
            object.__setattr__(self, key, np.asarray(getattr(self, key)).copy())
        n = self.population.size
        m = self.length_m.size
        if self.population.ndim != 1 or n < 2 or m < 1:
            raise ValueError("Expected at least two nodes and one road")
        if self.endpoints.shape != (2, m) or self.endpoints.dtype.kind not in "iu":
            raise ValueError("endpoints must be an integer array with shape (2, M)")
        if (self.endpoints < 0).any() or (self.endpoints >= n).any():
            raise ValueError("Road endpoint index is outside the node array")
        if (self.endpoints[0] == self.endpoints[1]).any():
            raise ValueError("Road self-loops are not supported")
        pairs = np.sort(self.endpoints, axis=0).T
        if np.unique(pairs, axis=0).shape[0] != m:
            raise ValueError("Provide one record for each undirected road")
        for key in ("length_m", "speed_mps", "importance"):
            value = getattr(self, key)
            if value.shape != (m,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite array with shape (M,)")
        if (self.length_m <= 0).any() or (self.speed_mps <= 0).any():
            raise ValueError("Road lengths and intact speeds must be positive")
        if not np.isfinite(self.population).all() or (self.population < 0).any():
            raise ValueError("Population must be finite and nonnegative")
        if not 0 < self.population.sum(dtype=np.float64) < float("inf"):
            raise ValueError("Total population must be finite and positive")
        if ((self.importance < 0) | (self.importance > 1)).any():
            raise ValueError("Road importance must be a fraction in [0, 1]")
        for key in ("endpoints", "length_m", "speed_mps", "population", "importance"):
            getattr(self, key).setflags(write=False)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**{k: arrays[k] for k in cls.__dataclass_fields__})


def validate_damage(damage, road_count):
    damage = np.asarray(damage, dtype=np.float64)
    if damage.shape != (road_count,) or not np.isin(damage, [0.0, 0.5, 1.0]).all():
        raise ValueError("damage must have shape (M,) and values 0, 0.5 or 1")
    return damage.copy()


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    network = RoadNetwork.load(path.parent / manifest["network"])
    scenarios = []
    for item in manifest["scenarios"]:
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError("Each scenario needs a nonempty string id")
        if item["split"] not in ("train", "test"):
            raise ValueError("Scenario split must be train or test")
        with np.load(path.parent / item["file"], allow_pickle=False) as arrays:
            damage = validate_damage(arrays["damage"], network.length_m.size)
        scenarios.append({"id": item["id"], "split": item["split"], "damage": damage})
    ids = [item["id"] for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario ids must be unique")
    return network, scenarios
