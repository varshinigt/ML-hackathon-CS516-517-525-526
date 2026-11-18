"""Candidate generation and per-candidate feature engineering.

The learning problem is framed as candidate ranking: for each GPS fix we generate the K
closest road arcs within a search radius, describe each (fix, arc) pair with features, and
train a binary classifier to predict "this arc is the reference arc for this fix". The
classifier's per-candidate probabilities are then (1) arg-maxed for classifier-only
prediction and (2) used as HMM emission scores for Viterbi decoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import LocalProjection, angle_diff, point_segment
from .network import build_network
from .preprocess import align_route, clean_track, motion_features

DEFAULT_CFG = {"radius_m": 60.0, "fallback_radius_m": 200.0, "k": 8, "directed": True}

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "heading_diff", "heading_valid", "heading_x_valid",
    "frac", "arc_len", "speed", "dt_prev", "dt_next",
    "dist_prev", "dist_next", "dist_win_mean",
    "near_prev_top1", "conn_prev_top1", "conn_next_top1",
]


def prepare_trip(trip, cfg=None, with_labels=True):
    """Clean a trip, build its network and candidate table. Returns (cands_df, points_df, net, log)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    proj = LocalProjection.fit(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    net = build_network(trip, proj, directed=cfg["directed"])
    pts, log = clean_track(trip.track, proj)
    pts = motion_features(pts)

    ref = np.full(len(pts), -1)
    ref_dist = np.full(len(pts), np.nan)
    if with_labels and trip.route is not None and len(trip.route):
        ref, ref_dist = align_route(pts, net, trip.route)
    pts["ref_arc"] = ref
    pts["ref_dist"] = ref_dist

    rows = []
    x, y = pts["x"].to_numpy(), pts["y"].to_numpy()
    for i in range(len(pts)):
        arcs, d, t = net.nearby_arcs(x[i], y[i], cfg["radius_m"], cfg["k"])
        if len(arcs) == 0:
            arcs, d, t = net.nearby_arcs(x[i], y[i], cfg["fallback_radius_m"], cfg["k"])
        if len(arcs) == 0:  # isolated fix: still give it its single nearest arc
            arcs, d, t = net.nearest_arc(x[i], y[i])
        for r, (a, dd, tt) in enumerate(zip(arcs, d, t)):
            rows.append((i, int(a), dd, r, tt))
    c = pd.DataFrame(rows, columns=["pt", "arc", "dist", "dist_rank", "frac"])
    c = _add_features(c, pts, net)
    c["trip_id"] = trip.trip_id
    pts["trip_id"] = trip.trip_id
    if with_labels:
        c["label"] = (c["arc"].to_numpy() == pts["ref_arc"].to_numpy()[c["pt"].to_numpy()]).astype(int)
    log["n_candidates"] = len(c)
    return c, pts, net, log


def _add_features(c: pd.DataFrame, pts: pd.DataFrame, net) -> pd.DataFrame:
    if c.empty:
        for f in FEATURES:
            c[f] = []
        return c
    pi = c["pt"].to_numpy()
    a = c["arc"].to_numpy()
    g = c.groupby("pt")["dist"]
    c["n_cands"] = g.transform("size")
    dmin = g.transform("min")
    c["dist_minus_min"] = c["dist"] - dmin
    c["dist_ratio"] = (c["dist"] + 1.0) / (dmin + 1.0)
    c["heading_diff"] = angle_diff(pts["heading"].to_numpy()[pi], net.bearing[a])
    c["heading_valid"] = pts["heading_valid"].to_numpy()[pi]
    c["heading_x_valid"] = c["heading_diff"] * c["heading_valid"]
    c["arc_len"] = net.length[a]
    for col in ("speed", "dt_prev", "dt_next"):
        c[col] = pts[col].to_numpy()[pi]

    # Context: how close are the neighbouring fixes to the same arc?
    x, y = pts["x"].to_numpy(), pts["y"].to_numpy()
    n = len(pts)

    def d_at(offset):
        j = np.clip(pi + offset, 0, n - 1)
        d, _ = point_segment(x[j], y[j], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        return d

    c["dist_prev"], c["dist_next"] = d_at(-1), d_at(1)
    c["dist_win_mean"] = np.mean([d_at(o) for o in (-2, -1, 0, 1, 2)], axis=0)

    # Connectivity with neighbours' nearest arc (label-free context).
    top1 = np.full(n, -1)
    first = c[c["dist_rank"] == 0]
    top1[first["pt"].to_numpy()] = first["arc"].to_numpy()
    prev_top = top1[np.clip(pi - 1, 0, n - 1)]
    next_top = top1[np.clip(pi + 1, 0, n - 1)]
    c["near_prev_top1"] = (a == prev_top).astype(int)
    c["conn_prev_top1"] = np.array([int(p >= 0 and net.connected(p, q)) for p, q in zip(prev_top, a)])
    c["conn_next_top1"] = np.array([int(q >= 0 and net.connected(p, q)) for p, q in zip(a, next_top)])
    return c


def candidate_miss_rate(cands: pd.DataFrame, pts: pd.DataFrame) -> float:
    """Fraction of labelled fixes whose reference arc is absent from their candidate set."""
    lab = pts[pts["ref_arc"] >= 0]
    if lab.empty:
        return float("nan")
    hit = cands.groupby(["trip_id", "pt"])["label"].max()
    key = list(zip(lab["trip_id"], lab.index))
    found = np.array([hit.get(k, 0) for k in key])
    return float(1 - found.mean())
