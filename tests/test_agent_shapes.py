import numpy as np
import torch

from flybrain_worldle.connectome.agent import FEEDBACK_DIM, FlyAgent, encode_feedback
from flybrain_worldle.connectome.central_complex import CentralComplexRNN, CXGraph
from flybrain_worldle.connectome.vision import PixelVision, country_features, silhouette_movie
from flybrain_worldle.game import load_countries
from flybrain_worldle.training.reinforce import BatchedWorldle, episode_stats


def test_stub_graph_roundtrips_through_csv(tmp_path):
    g = CXGraph.stub(n_types=12, seed=3)
    g.save(tmp_path / "t.csv", tmp_path / "e.csv")
    g2 = CXGraph.load(tmp_path / "t.csv", tmp_path / "e.csv")
    assert g2.types == g.types
    np.testing.assert_allclose(g2.weights, g.weights, rtol=1e-6)
    np.testing.assert_array_equal(g2.signs, g.signs)


def test_shuffled_graph_preserves_edge_count_and_weights():
    g = CXGraph.stub(n_types=20, seed=1)
    s = g.shuffled(seed=2)
    assert (s.weights != 0).sum() == (g.weights != 0).sum()
    assert np.allclose(np.sort(np.abs(s.weights[s.weights != 0])), np.sort(np.abs(g.weights[g.weights != 0])))
    assert not np.array_equal(s.weights, g.weights)


def test_cx_rnn_respects_connectome_topology():
    g = CXGraph.stub(n_types=16, seed=0)
    rnn = CentralComplexRNN(g, input_dim=4)
    w = rnn.recurrent_weight.detach().numpy()
    assert np.array_equal(w != 0, g.weights != 0)
    assert np.array_equal(np.sign(w), np.sign(g.weights))
    h = rnn(torch.randn(3, 4), rnn.init_state(3))
    assert h.shape == (3, 16)
    assert torch.isfinite(h).all()


def test_encode_feedback_quantizes_bearing():
    valid = torch.ones(2)
    fb = encode_feedback(valid, torch.tensor([10.0, 100.0]), torch.tensor([1000.0, 1000.0]), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert fb.shape == (2, FEEDBACK_DIM)
    assert fb[0, 1].item() == 0.0 and fb[0, 2].item() == 1.0  # 10 deg -> N
    assert fb[1, 1].item() == 1.0 and abs(fb[1, 2].item()) < 1e-6  # 100 deg -> E


def test_agent_forward_and_rollout_shapes(tmp_path):
    countries = load_countries()
    features = country_features(PixelVision(size=8), countries, cache_dir=tmp_path)
    assert features.shape == (len(countries), 64)

    agent = FlyAgent(CXGraph.stub(n_types=24), vision_dim=64, vision_proj=16)
    game = BatchedWorldle(countries, features, torch.device("cpu"))
    targets = torch.arange(5)
    log_probs, entropies, rewards, guesses, alive = game.rollout(agent, targets)
    assert log_probs.shape == entropies.shape == rewards.shape == guesses.shape == alive.shape == (5, 6)
    assert torch.isfinite(log_probs).all() and torch.isfinite(rewards).all()
    assert guesses.min() >= 0 and guesses.max() < len(countries)
    assert alive[:, 0].all()

    stats = episode_stats(rewards, guesses, targets, alive)
    assert 0 <= stats["solve_rate"] <= 1
    assert stats["return"] <= 6


def test_batched_geo_matches_scalar_geo():
    from flybrain_worldle.game.geo import bearing_deg, destination_point, haversine_km

    countries = load_countries()
    game = BatchedWorldle(countries, np.zeros((len(countries), 2), np.float32), torch.device("cpu"))
    lat1, lon1, lat2, lon2 = 51.5, -0.13, -33.9, 151.2
    d = game.haversine(*map(torch.deg2rad, map(torch.tensor, (lat1, lon1, lat2, lon2))))
    assert abs(d.item() - haversine_km(lat1, lon1, lat2, lon2)) < 1e-2
    b = game.bearing(*map(torch.deg2rad, map(torch.tensor, (lat1, lon1, lat2, lon2))))
    assert abs(b.item() - bearing_deg(lat1, lon1, lat2, lon2)) < 1e-3
    dlat, dlon = game.destination(torch.deg2rad(torch.tensor(lat1)), torch.deg2rad(torch.tensor(lon1)), torch.tensor(60.0), torch.tensor(5000.0))
    elat, elon = destination_point(lat1, lon1, 60.0, 5000.0)
    assert abs(torch.rad2deg(dlat).item() - elat) < 1e-3
    assert abs(((torch.rad2deg(dlon).item() - elon + 180) % 360) - 180) < 1e-3
    assert countries[game.nearest(torch.deg2rad(torch.tensor([48.8])), torch.deg2rad(torch.tensor([2.3]))).item()].name == "France"


def test_silhouette_movie_shape_and_range():
    m = silhouette_movie(np.zeros((32, 32), np.float32), hold=3, drift=4)
    assert m.shape == (11, 32, 32)
    assert 0 <= m.min() <= m.max() <= 1
