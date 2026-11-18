"""End-to-end inference: raw trip files -> per-observation road segment predictions.

Used by the FastAPI backend. Prediction never sees the reference route: the trip is loaded
without it, and evaluation against the reference happens afterwards in `evaluate_trip`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .features import DEFAULT_CFG, FEATURES, prepare_trip
from .hmm import precompute, viterbi
from .io import ROOT, Trip
from .metrics import RELIABLE_REF_M, n_breaks, route_continuity, sequence_metrics
from .preprocess import align_route

ARTIFACTS = ROOT / "artifacts"


class PathSnapPredictor:
    def __init__(self, config_path: Path = ARTIFACTS / "pipeline_config.json"):
        cfg = json.loads(Path(config_path).read_text())
        self.cfg = cfg
        self.model = joblib.load(ROOT / cfg["model_path"])
        self.features = cfg.get("features", FEATURES)
        self.cand_cfg = {**DEFAULT_CFG, **cfg.get("candidates", {})}
        self.hmm_params = json.loads((ROOT / cfg["hmm_config"]).read_text())["params"]

    def predict(self, trip: Trip) -> dict:
        timings = {}
        t0 = time.perf_counter()
        unlabeled = Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, None, trip.files)
        cands, pts, net, log = prepare_trip(unlabeled, self.cand_cfg, with_labels=False)
        timings["preprocess_candidates_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = self.model.predict_proba(cands[self.features].to_numpy(np.float32))[:, 1]
        timings["classifier_s"] = time.perf_counter() - t0
        cands["score"] = proba
        clf = cands.loc[cands.groupby("pt")["score"].idxmax()].sort_values("pt")["arc"].to_numpy()

        t0 = time.perf_counter()
        steps = precompute(cands, pts, net)
        hmm = viterbi(steps, proba, self.hmm_params)
        timings["hmm_viterbi_s"] = time.perf_counter() - t0
        return {"cands": cands, "points": pts, "net": net, "clean_log": log,
                "classifier_arcs": clf, "hmm_arcs": hmm, "timings": timings}


def evaluate_trip(result: dict, route: np.ndarray) -> dict:
    """Server-side scoring against the reference route (never sent to the client)."""
    pts, net = result["points"], result["net"]
    ref, ref_d = align_route(pts, net, route)
    rel = ref_d <= RELIABLE_REF_M
    out = {"n_fixes": int(len(pts)), "n_scored_fixes": int(rel.sum())}
    for tag, arcs in (("hmm", result["hmm_arcs"]), ("classifier", result["classifier_arcs"])):
        out[f"{tag}_top1_accuracy"] = float((arcs == ref)[rel].mean()) if rel.any() else None
        out[f"{tag}_top1_accuracy_all_fixes"] = float((arcs == ref).mean())
        out[f"{tag}_route_continuity"] = route_continuity(arcs, net)
        out[f"{tag}_route_breaks"] = n_breaks(arcs, net)
        out.update({f"{tag}_{k}": float(v) for k, v in sequence_metrics(arcs, route, net).items()})
    return out
