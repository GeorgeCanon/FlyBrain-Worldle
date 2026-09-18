"""FastAPI app: serves the frontend and lets it watch a trained agent play a full Worldle episode."""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from ...connectome.anatomy import BRAIN_PATH
from ...game.countries import COUNTRIES_PATH
from ...game.geo import compass_point, render_silhouette
from ...training.reinforce import load_agent

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
CHECKPOINT = Path(os.environ.get("FLYBRAIN_CHECKPOINT", "runs/flyvis_cx_s0/best.pt"))

app = FastAPI(title="Fly-brain Worldle")
agent, cfg, game = load_agent(CHECKPOINT)
countries = game.countries
by_code = {c.code: c for c in countries}


class NewGame(BaseModel):
    target: str | None = None
    sample: bool = False


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/brain")
def brain():
    if not BRAIN_PATH.exists():
        raise HTTPException(404, "run scripts/fetch_brain_geometry.py to export the brain geometry")
    return FileResponse(BRAIN_PATH, media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/meta")
def meta():
    return {
        "checkpoint": str(CHECKPOINT),
        "config": cfg.to_dict(),
        "n_countries": len(countries),
        "n_cx_types": agent.cx.n_types,
        "countries": [{"code": c.code, "name": c.name, "lat": c.lat, "lon": c.lon} for c in countries],
    }


@app.get("/api/world")
def world():
    with open(COUNTRIES_PATH, encoding="utf-8") as f:
        fc = json.load(f)
    return JSONResponse(
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": ft["geometry"], "properties": {"code": ft["properties"]["ISO_A3"]}} for ft in fc["features"]],
        }
    )


def silhouette_png(country, size: int = 512) -> str:
    img = render_silhouette(country.geometry, size, supersample=4)
    buf = io.BytesIO()
    Image.fromarray((img * 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/play")
def play(req: NewGame):
    if req.target and req.target not in by_code:
        raise HTTPException(404, f"unknown country code {req.target}")
    idx = countries.index(by_code[req.target]) if req.target else int(torch.randint(0, len(countries), (1,)))
    target = countries[idx]
    targets = torch.tensor([idx], device=game.device)

    guesses = []
    with torch.no_grad():
        for s in game.steps(agent, targets, sample=req.sample):
            if s["alive"].item() == 0:
                break
            g = countries[int(s["guess"].item())]
            correct = bool(s["correct"].item())
            guesses.append(
                {
                    "code": g.code,
                    "name": g.name,
                    "lat": g.lat,
                    "lon": g.lon,
                    "correct": correct,
                    "distance_km": round(s["distance_km"].item()),
                    "direction": "✓" if correct else compass_point(s["bearing"].item()),
                    "bearing": s["bearing"].item(),
                    "proximity": s["proximity"].item(),
                    "action": {
                        "bearing": s["action_bearing"].item(),
                        "distance_km": s["action_distance_km"].item(),
                        "lat": s["proposed_lat"].item(),
                        "lon": s["proposed_lon"].item(),
                    },
                    "cx_activity": [round(v, 3) for v in s["state"][0].tolist()],
                }
            )
    return {
        "target": {"code": target.code, "name": target.name, "lat": target.lat, "lon": target.lon},
        "silhouette": silhouette_png(target),
        "guesses": guesses,
        "solved": any(g["correct"] for g in guesses),
    }
