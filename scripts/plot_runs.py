"""Plot learning curves from runs/*/log.jsonl into a single PNG for the README."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run: Path) -> tuple[str, list[dict]]:
    cfg = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    return f"{cfg['vision']} + {cfg['graph']}", rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("docs/learning_curves.png"))
    a = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for run in a.runs:
        label, rows = load(run)
        it = [r["iteration"] for r in rows]
        axes[0].plot(it, [r["solve_rate"] for r in rows], label=label)
        axes[1].plot(it, [r["first_guess_acc"] for r in rows], label=label)
        axes[2].plot(it, [r["mean_guesses"] for r in rows], label=label)
    for ax, title in zip(axes, ["solve rate (within 6 guesses)", "first-guess accuracy", "mean guesses when solved"]):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training iteration")
        ax.grid(alpha=0.3)
    axes[1].axhline(1 / 168, color="gray", ls="--", lw=1, label="chance")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
