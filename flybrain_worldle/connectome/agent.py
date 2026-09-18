"""The agent: frozen fly vision features -> connectome-constrained central-complex RNN -> (heading, distance) policy."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from ..game.geo import MAX_DISTANCE_KM
from .central_complex import CentralComplexRNN, CXGraph

FEEDBACK_DIM = 8
MIN_STD = 0.1


def encode_feedback(
    valid: torch.Tensor, bearing_deg: torch.Tensor, distance_km: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor, step: torch.Tensor
) -> torch.Tensor:
    """What the game tells the agent after a guess, as the CX sees it.

    The bearing is quantized to 8 compass points because that is all Worldle reveals; distance is exact.
    The guessed location is included as a unit vector so constraints from different guesses can be combined.
    """
    quantized = torch.round(bearing_deg / 45.0) * 45.0
    theta = torch.deg2rad(quantized)
    phi, lam = torch.deg2rad(lat), torch.deg2rad(lon)
    return torch.stack(
        [
            valid,
            valid * torch.sin(theta),
            valid * torch.cos(theta),
            valid * distance_km / MAX_DISTANCE_KM,
            torch.cos(phi) * torch.cos(lam),
            torch.cos(phi) * torch.sin(lam),
            torch.sin(phi),
            step,
        ],
        dim=-1,
    )


@dataclass
class PolicyOutput:
    bearing_deg: torch.Tensor
    distance_km: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    state: torch.Tensor


class FlyAgent(nn.Module):
    def __init__(self, graph: CXGraph, vision_dim: int, vision_proj: int = 64, rnn_steps: int = 5):
        super().__init__()
        self.vision_proj = nn.Sequential(nn.Linear(vision_dim, vision_proj), nn.Tanh())
        self.cx = CentralComplexRNN(graph, input_dim=vision_proj + FEEDBACK_DIM)
        self.readout = nn.Linear(graph.n_types, 3)
        self.log_std = nn.Parameter(torch.tensor([math.log(0.5), math.log(0.5)]))
        self.rnn_steps = rnn_steps

    def init_state(self, batch: int) -> torch.Tensor:
        return self.cx.init_state(batch)

    def forward(self, vision: torch.Tensor, feedback: torch.Tensor, state: torch.Tensor, sample: bool = True) -> PolicyOutput:
        x = torch.cat([self.vision_proj(vision), feedback], dim=-1)
        state = self.cx(x, state, steps=self.rnn_steps)
        out = self.readout(state)
        mu_angle = torch.atan2(out[:, 0], out[:, 1])
        mu_dist = out[:, 2]
        std = self.log_std.clamp(min=math.log(MIN_STD)).exp()

        angle_dist = Normal(mu_angle, std[0])
        dist_dist = Normal(mu_dist, std[1])
        angle = angle_dist.sample() if sample else mu_angle
        u = dist_dist.sample() if sample else mu_dist

        log_prob = angle_dist.log_prob(angle) + dist_dist.log_prob(u)
        entropy = angle_dist.entropy() + dist_dist.entropy()
        return PolicyOutput(
            bearing_deg=torch.rad2deg(angle) % 360.0,
            distance_km=MAX_DISTANCE_KM * torch.sigmoid(u),
            log_prob=log_prob,
            entropy=entropy,
            state=state,
        )
