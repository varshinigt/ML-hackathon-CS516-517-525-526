"""Candidate-level, observation-level and route-level metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

# A fix whose aligned reference arc is further than this from the fix is treated as an
# unreliable label (reference route does not cover that stretch / GPS outlier) and is
# excluded from training and from the "reliable" accuracy figures. See report §2.
RELIABLE_REF_M = 50.0


def candidate_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def top1_predictions(cands: pd.DataFrame, score_col="score") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, fix)."""
    idx = cands.groupby(["trip_id", "pt"])[score_col].idxmax()
    return cands.loc[idx, ["trip_id", "pt", "arc"]].rename(columns={"arc": "pred_arc"})


def observation_accuracy(pts: pd.DataFrame, pred_col="pred_arc") -> dict:
    lab = pts["ref_arc"] >= 0
    rel = lab & (pts["ref_dist"] <= RELIABLE_REF_M)
    correct = pts[pred_col] == pts["ref_arc"]
    return {
        "top1_acc": float(correct[rel].mean()) if rel.any() else float("nan"),
        "top1_acc_all_labelled": float(correct[lab].mean()) if lab.any() else float("nan"),
        "n_reliable": int(rel.sum()),
        "n_labelled": int(lab.sum()),
    }


CONTINUITY_GAP_M = 100.0


def _continuous(net, p, q, max_gap_m):
    if p == q or net.connected(p, q):
        return True
    if max_gap_m <= 0:
        return False
    # at 1 Hz a vehicle can cross a short arc between two fixes; allow a short connecting path
    return net.node_dist(net.v[p], max_gap_m).get(int(net.u[q]), np.inf) <= max_gap_m


def route_continuity(pred_arcs: np.ndarray, net, max_gap_m: float = CONTINUITY_GAP_M) -> float:
    """Share of consecutive fixes whose predicted arcs are identical, adjacent, or joined by a
    network path of at most max_gap_m metres (max_gap_m=0 gives strict adjacency)."""
    a = np.asarray(pred_arcs)
    if len(a) < 2:
        return float("nan")
    return float(np.mean([_continuous(net, p, q, max_gap_m) for p, q in zip(a[:-1], a[1:])]))


def n_breaks(pred_arcs, net, max_gap_m: float = CONTINUITY_GAP_M) -> int:
    a = np.asarray(pred_arcs)
    return int(sum(not _continuous(net, p, q, max_gap_m) for p, q in zip(a[:-1], a[1:])))


def sequence_metrics(pred_arcs: np.ndarray, route: np.ndarray, net) -> dict:
    """Route-level comparison of the set of traversed arcs, weighted by arc length."""
    if route is None or len(route) == 0:
        return {}
    P, R = set(map(int, pred_arcs)), set(map(int, route))
    L = net.length
    inter = sum(L[a] for a in P & R)
    prec = inter / max(sum(L[a] for a in P), 1e-9)
    rec = inter / max(sum(L[a] for a in R), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # order-aware: longest common subsequence of de-duplicated arc sequences
    seq = [int(a) for i, a in enumerate(pred_arcs) if i == 0 or a != pred_arcs[i - 1]]
    lcs = _lcs(seq, list(map(int, route)))
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1,
            "lcs_ratio": lcs / max(len(route), 1)}


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = np.zeros(len(b) + 1, int)
    bb = np.array(b)
    for x in a:
        cur = np.zeros_like(prev)
        match = (bb == x)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if match[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return int(prev[-1])
