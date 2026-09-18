import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the interactive demo.")
    p.add_argument("--checkpoint", type=Path, default=Path("runs/flyvis_cx_s0/best.pt"))
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    os.environ["FLYBRAIN_CHECKPOINT"] = str(a.checkpoint)

    import uvicorn

    uvicorn.run("flybrain_worldle.demo.backend.app:app", host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
