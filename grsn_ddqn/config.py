from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    gamma: float = 0.9
    learning_rate: float = 0.001
    batch_size: int = 32
    episodes_per_scenario: int = 40
    target_update_steps: int = 500
    epsilon_start: float = 0.8
    epsilon_end: float = 0.01
    replay_capacity: int = 14100
    gcn_widths: tuple = (16, 16, 16)
    mlp_widths: tuple = (64, 64, 32)
    repair_length_unit_m: float = 500.0
    seed: int = 43

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Expected 0 <= epsilon_end <= epsilon_start <= 1")
        for name in ("batch_size", "episodes_per_scenario", "target_update_steps", "replay_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if len(self.gcn_widths) != 3 or len(self.mlp_widths) != 3:
            raise ValueError("Expected three GCN layers and three hidden MLP layers")
        if any(type(v) is not int or v <= 0 for v in (*self.gcn_widths, *self.mlp_widths)):
            raise ValueError("Layer widths must be positive integers")
        if not 0 < self.repair_length_unit_m < float("inf"):
            raise ValueError("repair_length_unit_m must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("gcn_widths", "mlp_widths"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def as_dict(self):
        return asdict(self)
