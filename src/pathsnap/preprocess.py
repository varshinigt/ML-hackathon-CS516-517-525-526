"""GPS cleaning, motion features and reference-label alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import LocalProjection, bearing

MAX_SPEED_MS = 70.0  # ~250 km/h: anything faster between consecutive fixes is a glitch


def clean_track(track: pd.DataFrame, proj: LocalProjection) -> tuple[pd.DataFrame, dict]:
    """Drop invalid / duplicate / out-of-order / teleporting fixes.

    Returns the cleaned frame (column `obs` = row index in the raw .track file, so every
    prediction can be mapped back to its original observation) and a log of what was removed.
    """
    df = track.copy()
    df["obs"] = np.arange(len(df))
    log = {"raw": len(df)}
    ok = df[["lon", "lat", "t"]].notna().all(axis=1)
    ok &= df["lon"].between(-180, 180) & df["lat"].between(-90, 90)
    log["invalid"] = int((~ok).sum())
    df = df[ok]
    n = len(df)
    df = df.sort_values("t", kind="stable").drop_duplicates("t", keep="first")
    log["duplicate_t"] = n - len(df)

    x, y = proj.forward(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    # Remove isolated spikes: a fix that is far from both neighbours at impossible speed.
    keep = np.ones(len(df), bool)
    if len(df) > 2:
        t = df["t"].to_numpy()
        dprev = np.r_[np.inf, np.hypot(np.diff(x), np.diff(y))]
        dtp = np.r_[np.inf, np.diff(t)]
        dnext = np.r_[dprev[1:], np.inf]
        dtn = np.r_[dtp[1:], np.inf]
        with np.errstate(invalid="ignore", divide="ignore"):
            spike = (dprev / dtp > MAX_SPEED_MS) & (dnext / dtn > MAX_SPEED_MS)
        keep = ~spike
    log["speed_spikes"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    log["clean"] = len(df)
    return df, log


def motion_features(df: pd.DataFrame, half_window: int = 2) -> pd.DataFrame:
    """Speed, heading and time gaps from neighbouring fixes (centred difference)."""
    x, y, t = df["x"].to_numpy(), df["y"].to_numpy(), df["t"].to_numpy()
    n = len(df)
    i0 = np.clip(np.arange(n) - half_window, 0, n - 1)
    i1 = np.clip(np.arange(n) + half_window, 0, n - 1)
    dx, dy, dt = x[i1] - x[i0], y[i1] - y[i0], t[i1] - t[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, np.hypot(dx, dy) / dt, 0.0)
    out = df.copy()
    out["speed"] = speed
    out["heading"] = bearing(dx, dy)
    out["heading_valid"] = (np.hypot(dx, dy) > 3.0).astype(int)  # heading is noise when stationary
    out["dt_prev"] = np.r_[0.0, np.diff(t)]
    out["dt_next"] = np.r_[np.diff(t), 0.0]
    return out


def align_route(df: pd.DataFrame, net, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each GPS fix a reference arc from the ordered route via monotone DP.

    The .route file lists arcs in travel order but does not say which fix belongs to which arc.
    We find the assignment j(0) <= j(1) <= ... <= j(n-1) (arcs may be skipped, e.g. short arcs
    crossed between two fixes) that minimises the total point-to-arc distance.
    Returns (ref_arc per fix, distance of each fix to its ref arc).
    """
    from .geo import point_segment

    route = np.asarray(route, int)
    route = route[(route >= 0) & (route < net.n_arcs)]
    n, m = len(df), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1), np.full(n, np.nan)
    px, py = df["x"].to_numpy(), df["y"].to_numpy()
    rax, ray, rbx, rby = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    # Row-by-row DP keeps memory at O(n*m) int32 back-pointers only (long trips: 19k x 2k).
    back = np.empty((n, m), dtype=np.int32)
    prev, _ = point_segment(px[0], py[0], rax, ray, rbx, rby)
    back[0] = np.arange(m)
    ar = np.arange(m)
    for i in range(1, n):
        c, _ = point_segment(px[i], py[i], rax, ray, rbx, rby)
        pm = np.minimum.accumulate(prev)
        is_new = np.r_[True, prev[1:] < pm[:-1]]
        back[i] = np.maximum.accumulate(np.where(is_new, ar, 0))
        prev = c + pm
    j = int(np.argmin(prev))
    idx = np.empty(n, int)
    for i in range(n - 1, -1, -1):
        idx[i] = j
        j = back[i, j]
    ref = route[idx]
    d, _ = point_segment(px, py, rax[idx], ray[idx], rbx[idx], rby[idx])
    return ref, d
