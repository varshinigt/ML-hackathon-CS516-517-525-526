"""HMM map-matching with Viterbi decoding.

Hidden states   : at fix i, the states are its candidate arcs (rows of the candidate table
                  with pt == i). State identity = (fix, arc id, projection fraction), so the
                  decoded state maps straight back to a road segment id.
Emission        : log of the classifier probability renormalised over the fix's candidates,
                  scaled by a weight w_e:  log P_e(s_i) = w_e * log( p_i(s) / sum_s' p_i(s') ).
Transition      : Newson & Krumm (2009) style. With d_gc = straight-line distance between the
                  two fixes and d_rt = shortest network distance between the two candidate
                  positions, log P_t = -|d_rt - d_gc| / beta. Pairs with no path within the
                  search limit (disconnected / against one-way direction) get a large finite
                  penalty instead of -inf, so the decoder always returns a full path.
Gaps            : if the time between fixes exceeds gap_s, the transition is made uniform
                  (the chain is cut and restarts from emissions only), because network
                  distances across long gaps are not informative.
Recurrence      : delta_i(s) = log P_e(s) + max_s' [ delta_{i-1}(s') + log P_t(s' -> s) ],
                  back-pointers psi_i(s) = argmax_s'; the path is recovered by backtracking.

Network distances depend only on geometry, so they are computed once per trip
(`precompute`) and re-used when tuning decoder parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HMM = {"beta": 15.0, "w_emission": 1.0, "gap_s": 30.0,
               "unreachable_logp": -50.0, "p_floor": 1e-4}
SEARCH_LIMIT_M = 3000.0


def precompute(cands: pd.DataFrame, pts: pd.DataFrame, net, search_limit_m=SEARCH_LIMIT_M):
    """Per-fix candidate lists and K_prev x K_cur network-distance matrices."""
    pt = cands["pt"].to_numpy()
    order = np.lexsort((cands["dist"].to_numpy(), pt))       # positional, sorted by fix then distance
    starts = np.searchsorted(pt[order], np.arange(len(pts) + 1))
    arcs_all, fr_all = cands["arc"].to_numpy()[order], cands["frac"].to_numpy()[order]
    x, y, t = pts["x"].to_numpy(), pts["y"].to_numpy(), pts["t"].to_numpy()
    steps = []
    prev = None
    for i in range(len(pts)):
        sl = slice(starts[i], starts[i + 1])
        arcs, fr = arcs_all[sl], fr_all[sl]
        if prev is None:
            D = None
        else:
            pa, pf = prev
            D = np.array([[net.route_distance(a, ta, b, tb, search_limit_m) for b, tb in zip(arcs, fr)]
                          for a, ta in zip(pa, pf)]).reshape(len(pa), len(arcs))
        dgc = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) if i else 0.0
        dt = t[i] - t[i - 1] if i else 0.0
        steps.append({"rows": order[sl], "arcs": arcs, "D": D, "dgc": dgc, "dt": dt})
        prev = (arcs, fr)
    return steps


def viterbi(steps, score: np.ndarray, params=None) -> np.ndarray:
    """Decode; `score[j]` is the classifier probability of candidate-table row j (positional)."""
    prm = {**DEFAULT_HMM, **(params or {})}
    delta, back = None, []
    for st in steps:
        p = np.clip(score[st["rows"]], prm["p_floor"], None)
        emis = prm["w_emission"] * np.log(p / p.sum())
        if delta is None:
            delta = emis
            back.append(np.zeros(len(emis), int))
            continue
        if st["dt"] > prm["gap_s"]:
            trans = np.zeros_like(st["D"])
        else:
            D = st["D"]
            trans = np.where(np.isfinite(D), -np.abs(D - st["dgc"]) / prm["beta"], prm["unreachable_logp"])
        tot = delta[:, None] + trans
        bp = tot.argmax(0)
        delta = tot[bp, np.arange(len(emis))] + emis
        back.append(bp)
    n = len(steps)
    path = np.empty(n, int)
    k = int(np.argmax(delta))
    for i in range(n - 1, -1, -1):
        path[i] = steps[i]["arcs"][k]
        k = back[i][k]
    return path
