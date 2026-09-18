"""REINFORCE with a moving-average baseline over batches of Worldle episodes run in lockstep."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..connectome.agent import FlyAgent, encode_feedback
from ..connectome.central_complex import CXGraph
from ..connectome.vision import FlyVision, PixelVision, country_features
from ..game.countries import Country, load_countries
from ..game.env import MAX_GUESSES
from ..game.geo import EARTH_RADIUS_KM, MAX_DISTANCE_KM
from .config import TrainConfig


def build_graph(kind: str, seed: int = 0) -> CXGraph:
    if kind == "cx":
        return CXGraph.load()
    if kind == "shuffled":
        return CXGraph.load().shuffled(seed)
    if kind == "stub":
        return CXGraph.stub(seed=seed)
    raise ValueError(kind)


def build_vision(kind: str):
    return FlyVision() if kind == "flyvis" else PixelVision()


class BatchedWorldle:
    """Vectorized Worldle over a fixed country table, mirroring WorldleEnv's rules with torch ops."""

    def __init__(self, countries: tuple[Country, ...], features: np.ndarray, device: torch.device):
        self.countries = countries
        self.device = device
        lat = torch.tensor([c.lat for c in countries], dtype=torch.float32, device=device)
        lon = torch.tensor([c.lon for c in countries], dtype=torch.float32, device=device)
        self.lat, self.lon = torch.deg2rad(lat), torch.deg2rad(lon)
        self.xyz = torch.stack([self.lat.cos() * self.lon.cos(), self.lat.cos() * self.lon.sin(), self.lat.sin()], -1)
        self.features = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.n = len(countries)

    def haversine(self, lat1, lon1, lat2, lon2):
        a = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
        return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))

    def bearing(self, lat1, lon1, lat2, lon2):
        dl = lon2 - lon1
        x = torch.sin(dl) * torch.cos(lat2)
        y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dl)
        return torch.rad2deg(torch.atan2(x, y)) % 360.0

    def destination(self, lat, lon, bearing_deg, distance_km):
        d = distance_km / EARTH_RADIUS_KM
        b = torch.deg2rad(bearing_deg)
        lat2 = torch.asin(torch.sin(lat) * torch.cos(d) + torch.cos(lat) * torch.sin(d) * torch.cos(b))
        lon2 = lon + torch.atan2(torch.sin(b) * torch.sin(d) * torch.cos(lat), torch.cos(d) - torch.sin(lat) * torch.sin(lat2))
        return lat2, lon2

    def nearest(self, lat, lon) -> torch.Tensor:
        p = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1)
        return (p @ self.xyz.T).argmax(-1)

    def steps(self, agent: FlyAgent, targets: torch.Tensor, sample: bool = True, max_guesses: int = MAX_GUESSES):
        """Play a batch of episodes in lockstep, yielding one dict of per-episode tensors per guess."""
        B = targets.shape[0]
        state = agent.init_state(B)
        vision = self.features[targets]
        t_lat, t_lon = self.lat[targets], self.lon[targets]

        ref_lat = torch.zeros(B, device=self.device)
        ref_lon = torch.zeros(B, device=self.device)
        valid = torch.zeros(B, device=self.device)
        fb_bearing = torch.zeros(B, device=self.device)
        fb_dist = torch.zeros(B, device=self.device)
        alive = torch.ones(B, device=self.device)

        for t in range(max_guesses):
            step = torch.full((B,), t / max_guesses, device=self.device)
            feedback = encode_feedback(valid, fb_bearing, fb_dist, torch.rad2deg(ref_lat), torch.rad2deg(ref_lon), step)
            out = agent(vision, feedback, state, sample=sample)
            state = out.state

            p_lat, p_lon = self.destination(ref_lat, ref_lon, out.bearing_deg, out.distance_km)
            guess = self.nearest(p_lat, p_lon)
            g_lat, g_lon = self.lat[guess], self.lon[guess]

            correct = (guess == targets).float()
            dist = self.haversine(g_lat, g_lon, t_lat, t_lon) * (1 - correct)
            proximity = (1 - dist / MAX_DISTANCE_KM).clamp(min=0)
            reward = proximity - 1 + correct * (1 + (max_guesses - t - 1) / max_guesses)
            fb_bearing = self.bearing(g_lat, g_lon, t_lat, t_lon) * (1 - correct)

            yield {
                "log_prob": out.log_prob,
                "entropy": out.entropy,
                "reward": reward * alive,
                "guess": guess,
                "alive": alive.clone(),
                "correct": correct,
                "distance_km": dist,
                "bearing": fb_bearing,
                "proximity": proximity,
                "action_bearing": out.bearing_deg,
                "action_distance_km": out.distance_km,
                "proposed_lat": torch.rad2deg(p_lat),
                "proposed_lon": (torch.rad2deg(p_lon) + 540) % 360 - 180,
                "state": state,
            }

            ref_lat, ref_lon = g_lat, g_lon
            fb_dist = dist
            valid = torch.ones(B, device=self.device)
            alive = alive * (1 - correct)

    def rollout(self, agent: FlyAgent, targets: torch.Tensor, sample: bool = True, max_guesses: int = MAX_GUESSES):
        """Returns per-step log_probs, entropies, rewards, guess indices and alive masks, each [B, T]."""
        cols = {k: [] for k in ("log_prob", "entropy", "reward", "guess", "alive")}
        for s in self.steps(agent, targets, sample, max_guesses):
            for k in cols:
                cols[k].append(s[k])
        return tuple(torch.stack(cols[k], 1) for k in cols)


def episode_stats(rewards: torch.Tensor, guesses: torch.Tensor, targets: torch.Tensor, alive: torch.Tensor) -> dict:
    solved_at = ((guesses == targets[:, None]) & (alive > 0)).float()
    solved = solved_at.sum(1) > 0
    first = torch.where(solved, solved_at.argmax(1).float() + 1, torch.full_like(solved_at[:, 0], float("nan")))
    return {
        "return": rewards.sum(1).mean().item(),
        "solve_rate": solved.float().mean().item(),
        "mean_guesses": torch.nanmean(first).item() if solved.any() else float("nan"),
        "first_guess_acc": solved_at[:, 0].mean().item(),
    }


def train(cfg: TrainConfig, device: torch.device | None = None) -> Path:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    countries = load_countries()
    vision = build_vision(cfg.vision)
    features = country_features(vision, countries)
    graph = build_graph(cfg.graph, cfg.seed)
    agent = FlyAgent(graph, vision_dim=features.shape[1], vision_proj=cfg.vision_proj, rnn_steps=cfg.rnn_steps).to(device)
    game = BatchedWorldle(countries, features, device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log_path = cfg.out_dir / "log.jsonl"
    baseline = None
    best_solve = -1.0
    t0 = time.time()

    with open(log_path, "w") as log:
        for it in range(1, cfg.iterations + 1):
            targets = torch.randint(0, game.n, (cfg.batch_size,), device=device)
            log_probs, entropies, rewards, guesses, alive = game.rollout(agent, targets)

            returns = torch.flip(torch.cumsum(torch.flip(rewards, [1]) * cfg.gamma, 1), [1])
            batch_mean = returns.mean(0, keepdim=True).detach()
            baseline = batch_mean if baseline is None else 0.9 * baseline + 0.1 * batch_mean
            advantage = (returns - baseline) * alive
            loss = -(log_probs * advantage).sum(1).mean() - cfg.entropy_coef * (entropies * alive).sum(1).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()

            if it % cfg.log_every == 0 or it == 1:
                stats = episode_stats(rewards, guesses, targets, alive)
                stats.update(iteration=it, loss=loss.item(), elapsed=time.time() - t0)
                log.write(json.dumps(stats) + "\n")
                log.flush()
                print(
                    f"it {it:5d}  return {stats['return']:+.3f}  solve {stats['solve_rate']:.2f}  "
                    f"guesses {stats['mean_guesses']:.2f}  first-acc {stats['first_guess_acc']:.2f}"
                )
                if stats["solve_rate"] >= best_solve:
                    best_solve = stats["solve_rate"]
                    torch.save({"config": cfg.to_dict(), "state_dict": agent.state_dict(), "vision_dim": features.shape[1]}, cfg.out_dir / "best.pt")
    torch.save({"config": cfg.to_dict(), "state_dict": agent.state_dict(), "vision_dim": features.shape[1]}, cfg.out_dir / "last.pt")
    return cfg.out_dir


@torch.no_grad()
def evaluate(agent: FlyAgent, game: BatchedWorldle, sample: bool = False) -> dict:
    """Greedy evaluation over every country exactly once."""
    agent.eval()
    targets = torch.arange(game.n, device=game.device)
    _, _, rewards, guesses, alive = game.rollout(agent, targets, sample=sample)
    agent.train()
    return episode_stats(rewards, guesses, targets, alive)


def load_agent(checkpoint: Path, device: torch.device | None = None) -> tuple[FlyAgent, TrainConfig, BatchedWorldle]:
    device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig(**{**ckpt["config"], "out_dir": Path(ckpt["config"]["out_dir"])})
    countries = load_countries()
    features = country_features(build_vision(cfg.vision), countries)
    graph = build_graph(cfg.graph, cfg.seed)
    agent = FlyAgent(graph, vision_dim=features.shape[1], vision_proj=cfg.vision_proj, rnn_steps=cfg.rnn_steps).to(device)
    agent.load_state_dict(ckpt["state_dict"])
    agent.eval()
    return agent, cfg, BatchedWorldle(countries, features, device)
