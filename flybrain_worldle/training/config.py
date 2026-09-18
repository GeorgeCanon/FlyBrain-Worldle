from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    vision: str = "flyvis"  # "flyvis" | "pixels"
    graph: str = "cx"  # "cx" | "shuffled" | "stub"
    vision_proj: int = 64
    rnn_steps: int = 5
    iterations: int = 3000
    batch_size: int = 128
    lr: float = 3e-3
    entropy_coef: float = 1e-3
    gamma: float = 1.0
    seed: int = 0
    out_dir: Path = field(default_factory=lambda: Path("runs") / "default")
    log_every: int = 25

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d
