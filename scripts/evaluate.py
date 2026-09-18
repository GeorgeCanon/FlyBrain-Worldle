"""Greedy evaluation of one or more checkpoints over every country, printed as a table."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.training.reinforce import evaluate, load_agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--sample", action="store_true", help="use the stochastic policy instead of the greedy one")
    a = p.parse_args()

    print(f"{'checkpoint':48s} {'vision':8s} {'graph':9s} {'solve':>6s} {'guesses':>8s} {'1st-acc':>8s} {'return':>8s}")
    for ckpt in a.checkpoints:
        agent, cfg, game = load_agent(ckpt)
        s = evaluate(agent, game, sample=a.sample)
        print(f"{str(ckpt):48s} {cfg.vision:8s} {cfg.graph:9s} {s['solve_rate']:6.2f} {s['mean_guesses']:8.2f} {s['first_guess_acc']:8.2f} {s['return']:8.2f}")


if __name__ == "__main__":
    main()
