"""Shared experiment helpers: loading cached features, fitting, observation-level scoring."""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from .features import FEATURES
from .io import ROOT, discover_trips, load_trip
from .metrics import RELIABLE_REF_M, candidate_metrics, observation_accuracy, route_continuity, sequence_metrics
from .network import build_network
from .geo import LocalProjection

PROC = ROOT / "data" / "processed"
SEED = 42


def load_partition(part: str):
    c = pd.read_pickle(PROC / f"cands_{part}.pkl")
    p = pd.read_pickle(PROC / f"points_{part}.pkl")
    p = p.reset_index().rename(columns={"index": "pt"})
    rel = p[["trip_id", "pt", "ref_dist", "ref_arc"]]
    c = c.merge(rel, on=["trip_id", "pt"], how="left")
    c["reliable"] = (c["ref_arc"] >= 0) & (c["ref_dist"] <= RELIABLE_REF_M)
    return c, p


def subsample(df: pd.DataFrame, n: int | None, seed=SEED) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def fit_model(factory, train_df: pd.DataFrame, max_rows=None):
    tr = subsample(train_df[train_df["reliable"]], max_rows)
    model = factory()
    t0 = time.perf_counter()
    model.fit(tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy())
    return model, time.perf_counter() - t0, len(tr)


def predict(model, df: pd.DataFrame):
    t0 = time.perf_counter()
    p = model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
    return p, time.perf_counter() - t0


def obs_top1(cands: pd.DataFrame, pts: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    c = cands[["trip_id", "pt", "arc"]].copy()
    c["score"] = score
    best = c.loc[c.groupby(["trip_id", "pt"])["score"].idxmax()]
    out = pts.merge(best[["trip_id", "pt", "arc"]].rename(columns={"arc": "pred_arc"}),
                    on=["trip_id", "pt"], how="left")
    out["pred_arc"] = out["pred_arc"].fillna(-1).astype(int)
    return out


def score_block(cands, pts, proba) -> dict:
    """Candidate metrics on reliable rows + observation-level Top-1 on reliable fixes."""
    rel = cands["reliable"].to_numpy()
    m = candidate_metrics(cands["label"].to_numpy()[rel], proba[rel])
    m.update(observation_accuracy(obs_top1(cands, pts, proba)))
    return m


@lru_cache(maxsize=16)
def trip_and_net(trip_id: str):
    files = discover_trips()[trip_id]
    trip = load_trip(trip_id, files)
    proj = LocalProjection.fit(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    return trip, build_network(trip, proj)


def route_scores(pred_pts: pd.DataFrame, col="pred_arc") -> dict:
    """Mean per-trip route continuity, breaks and sequence metrics."""
    rows = []
    for tid, g in pred_pts.groupby("trip_id"):
        trip, net = trip_and_net(tid)
        arcs = g.sort_values("pt")[col].to_numpy()
        r = {"trip_id": tid, "continuity": route_continuity(arcs, net)}
        r.update(sequence_metrics(arcs, trip.route, net))
        rows.append(r)
    df = pd.DataFrame(rows)
    return {"route_continuity": df["continuity"].mean(), "route_f1": df["route_f1"].mean(),
            "lcs_ratio": df["lcs_ratio"].mean(), "per_trip": df}
