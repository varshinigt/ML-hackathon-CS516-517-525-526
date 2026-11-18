"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health          liveness + loaded model info
  GET  /trips           the evaluation trips available on this server (ids and fix counts only)
  POST /predict         {"trip_id": "..."} -> predicted route + server-side accuracy

The reference routes stay on this machine: they are read only by the evaluation step after
prediction, and only summary metrics are returned. Every request is logged to the console
(and to logs/api.log) so the evaluator can watch inference happen.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_TRIPS_DIR   folder holding the evaluation trips (default data/raw/eval_test)
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from pathsnap.io import discover_trips, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnapPredictor, evaluate_trip  # noqa: E402

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "raw" / "eval_test"))
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / "api.log")])
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = PathSnapPredictor()
log.info("Loaded model %s | HMM params %s | trips dir %s", predictor.cfg["model_path"], predictor.hmm_params, TRIPS_DIR)


def _trips():
    return discover_trips(TRIPS_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model_name"], "trips_dir": str(TRIPS_DIR),
            "n_trips": len(_trips())}


@app.get("/trips")
def trips():
    out = []
    for tid, files in sorted(_trips().items()):
        with open(files["track"]) as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_gps_points": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    files = _trips().get(req.trip_id)
    if files is None:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("POST /predict trip=%s : running full pipeline", req.trip_id)
    t0 = time.perf_counter()
    trip = load_trip(req.trip_id, files, with_route=False)        # model input: no reference route
    res = predictor.predict(trip)
    total = time.perf_counter() - t0

    metrics = None
    if "route" in files:                                           # server-side evaluation only
        route = load_trip(req.trip_id, files, with_route=True).route
        metrics = evaluate_trip(res, route)

    pts, net = res["points"], res["net"]
    lon, lat = trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy()
    nodes = trip.nodes[["lon", "lat"]].to_numpy()
    hmm = res["hmm_arcs"]
    seq = [int(a) for i, a in enumerate(hmm) if i == 0 or a != hmm[i - 1]]
    segments, cur, end = [], [], None
    for a in seq:  # polyline pieces; break where consecutive arcs are not joined
        if cur and net.u[a] != end:
            segments.append(cur); cur = []
        if not cur:
            cur = [nodes[net.u[a]].tolist()]
        cur.append(nodes[net.v[a]].tolist()); end = net.v[a]
    if cur:
        segments.append(cur)

    # every raw observation gets a segment: fixes removed by cleaning inherit the nearest kept fix
    kept = pts["obs"].to_numpy()
    j = np.clip(np.searchsorted(kept, np.arange(len(lon))), 0, len(kept) - 1)
    jm = np.clip(j - 1, 0, len(kept) - 1)
    near = np.where(np.abs(kept[jm] - np.arange(len(lon))) < np.abs(kept[j] - np.arange(len(lon))), jm, j)
    per_obs = [{"obs": int(o), "arc_id": int(hmm[near[o]]), "cleaned_out": bool(kept[near[o]] != o)}
               for o in range(len(lon))]

    log.info("trip=%s fixes=%d candidates=%d time=%.2fs %s | HMM top-1=%s classifier top-1=%s",
             req.trip_id, len(pts), len(res["cands"]), total,
             {k: round(v, 2) for k, v in res["timings"].items()},
             None if not metrics else f"{metrics['hmm_top1_accuracy']:.4f}",
             None if not metrics else f"{metrics['classifier_top1_accuracy']:.4f}")
    return {
        "trip_id": req.trip_id,
        "gps": [[float(a), float(b)] for a, b in zip(lon, lat)],
        "predicted_route": segments,
        "observations": per_obs,
        "dropped_observations": int(len(lon) - len(pts)),
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in res["timings"].items()}, "total": round(total, 3)},
        "model": predictor.cfg["model_name"],
    }
