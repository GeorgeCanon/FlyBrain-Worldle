import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.training.config import TrainConfig
from flybrain_worldle.training.reinforce import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train the fly-brain Worldle agent with REINFORCE.")
    p.add_argument("--vision", choices=["flyvis", "pixels"], default="flyvis")
    p.add_argument("--graph", choices=["cx", "shuffled", "stub"], default="cx")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = TrainConfig(
        vision=a.vision,
        graph=a.graph,
        iterations=a.iterations,
        batch_size=a.batch_size,
        lr=a.lr,
        entropy_coef=a.entropy_coef,
        seed=a.seed,
        log_every=a.log_every,
        out_dir=a.out or Path("runs") / f"{a.vision}_{a.graph}_s{a.seed}",
    )
    out = train(cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
