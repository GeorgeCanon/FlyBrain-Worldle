"""Render one greedy episode (silhouette, guesses, world map) to a PNG for the README."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain_worldle.game.geo import compass_point, render_silhouette
from flybrain_worldle.training.reinforce import load_agent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--country", default="Italy")
    p.add_argument("--out", type=Path, default=Path("docs/demo.png"))
    a = p.parse_args()

    agent, cfg, game = load_agent(a.checkpoint)
    idx = next(i for i, c in enumerate(game.countries) if c.name == a.country)
    target = game.countries[idx]
    steps = []
    with torch.no_grad():
        for s in game.steps(agent, torch.tensor([idx]), sample=False):
            if s["alive"].item() == 0:
                break
            steps.append({k: v[0] for k, v in s.items()})

    fig = plt.figure(figsize=(12, 4.6), facecolor="#0f1115")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 2.6], wspace=0.12)

    ax = fig.add_subplot(gs[0])
    ax.imshow(render_silhouette(target.geometry, 128), cmap="gray", vmin=0, vmax=1)
    ax.set_title("puzzle", color="#8b93a7", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    for i, s in enumerate(steps):
        g = game.countries[int(s["guess"])]
        ok = bool(s["correct"])
        txt = f"{i + 1}. {g.name}" + ("  ✓" if ok else f"  {int(s['distance_km']):,} km {compass_point(float(s['bearing']))}")
        ax.text(0, 0.92 - 0.13 * i, txt, color="#7ee787" if ok else "#e6e8ee", fontsize=11, transform=ax.transAxes, va="top")
    ax.set_title(f"{cfg.vision} + {cfg.graph}", color="#8b93a7", fontsize=10)

    ax = fig.add_subplot(gs[2])
    ax.set_facecolor("#0b0d11")
    for c in game.countries:
        polys = [c.geometry] if c.geometry.geom_type == "Polygon" else list(c.geometry.geoms)
        for poly in polys:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="#161a22", ec="#2f3542", lw=0.5)
    prev = (0.0, 0.0)
    for i, s in enumerate(steps):
        g = game.countries[int(s["guess"])]
        px, py = float(s["proposed_lon"]), float(s["proposed_lat"])
        ax.plot([prev[0], px], [prev[1], py], color="#f2cc60", lw=1, ls="--", alpha=0.7)
        ax.plot(px, py, "s", color="#f2cc60", ms=4)
        ax.plot([px, g.lon], [py, g.lat], color="#58a6ff", lw=1, alpha=0.6)
        ax.plot(g.lon, g.lat, "o", color="#7ee787" if s["correct"] else "#58a6ff", ms=10)
        ax.text(g.lon, g.lat, str(i + 1), color="#0b0d11", fontsize=7, ha="center", va="center", fontweight="bold")
        prev = (g.lon, g.lat)
    ax.plot(target.lon, target.lat, "o", mfc="none", mec="#7ee787", ms=18, mew=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("yellow: raw (heading, distance) proposal · blue: snapped guess · ring: answer", color="#8b93a7", fontsize=9)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {a.out}: {target.name} in {len(steps)} guesses, solved={bool(steps[-1]['correct'])}")


if __name__ == "__main__":
    main()
