"""HMM / Viterbi route reconstruction: decoder tuning and classifier-only vs classifier+HMM.

1. Out-of-fold probabilities for the 80 training trips (3-fold GroupKFold, final model config)
   -> grid search of decoder parameters (beta, w_emission, gap_s) on training trips only.
2. Final model (fit on all training trips) probabilities on the 10 student-testing trips
   -> compare classifier-only arg-max with Viterbi decoding.
Outputs: results/hmm_grid.csv, results/hmm_comparison.csv, results/hmm_per_trip.csv,
         artifacts/hmm_config.json, results/figures/hmm_*.png, results/maps/student_trip_*.html
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap.experiment import fit_model, load_partition, predict, trip_and_net  # noqa: E402
from pathsnap.hmm import DEFAULT_HMM, precompute, viterbi  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.metrics import RELIABLE_REF_M, n_breaks, route_continuity, sequence_metrics  # noqa: E402
from pathsnap.models import MODEL_ZOO, make_tuned  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

RES = ROOT / "results"
GRID = {"beta": [3, 7, 15, 30, 60], "w_emission": [0.5, 1, 2, 4], "gap_s": [10, 30, 120]}


def final_factory(name):
    best = RES / f"bayes_opt_best_{name}.json"
    if best.exists():
        return make_tuned(name, json.loads(best.read_text())["best_params"]), "tuned"
    return MODEL_ZOO[name][1], "untuned"


def per_trip(args):
    tid, cc, pp = args
    _, net = trip_and_net(tid)
    return tid, precompute(cc, pp, net)


def split_trips(c, p):
    out = []
    for tid in sorted(c["trip_id"].unique()):
        cc = c[c["trip_id"] == tid].reset_index(drop=True)
        pp = p[p["trip_id"] == tid].sort_values("pt").reset_index(drop=True)
        out.append((tid, cc, pp))
    return out


def evaluate(trips, steps, params):
    """Return per-trip metrics of classifier-only vs HMM."""
    rows = []
    for tid, cc, pp in trips:
        score = cc["score"].to_numpy()
        top = cc.loc[cc.groupby("pt")["score"].idxmax()].sort_values("pt")["arc"].to_numpy()
        t0 = time.perf_counter()
        hmm = viterbi(steps[tid], score, params)
        t_dec = time.perf_counter() - t0
        rel = ((pp["ref_arc"] >= 0) & (pp["ref_dist"] <= RELIABLE_REF_M)).to_numpy()
        ref = pp["ref_arc"].to_numpy()
        rows.append({"trip_id": tid, "n_fixes": len(pp), "n_reliable": int(rel.sum()),
                     "clf_correct": int((top == ref)[rel].sum()), "hmm_correct": int((hmm == ref)[rel].sum()),
                     "decode_s": t_dec, "_top": top, "_hmm": hmm})
    return rows


def main(name, workers):
    (ROOT / "artifacts").mkdir(exist_ok=True)
    factory, variant = final_factory(name)
    tr_c, tr_p = load_partition("train")
    st_c, st_p = load_partition("student_test")

    # ---- 1. out-of-fold scores on training trips
    oof = np.zeros(len(tr_c))
    for i_tr, i_va in GroupKFold(n_splits=3).split(tr_c, groups=tr_c["trip_id"]):
        m, _, _ = fit_model(factory, tr_c.iloc[i_tr])
        oof[i_va], _ = predict(m, tr_c.iloc[i_va])
    tr_c["score"] = oof
    train_trips = split_trips(tr_c, tr_p)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:  # spawn: forking after LightGBM's OpenMP threads deadlocks
        tr_steps = dict(pool.map(per_trip, train_trips, chunksize=1))
    print(f"precomputed transitions for {len(tr_steps)} training trips in {time.time() - t0:.0f}s", flush=True)

    grid_rows = []
    for beta, we, gap in itertools.product(*GRID.values()):
        prm = {"beta": beta, "w_emission": we, "gap_s": gap}
        r = evaluate(train_trips, tr_steps, prm)
        acc = sum(x["hmm_correct"] for x in r) / sum(x["n_reliable"] for x in r)
        grid_rows.append({**prm, "train_oof_hmm_top1": acc})
        print(prm, f"{acc:.4f}", flush=True)
    grid = pd.DataFrame(grid_rows).sort_values("train_oof_hmm_top1", ascending=False)
    grid.to_csv(RES / "hmm_grid.csv", index=False)
    best = {**DEFAULT_HMM, **grid.iloc[0][list(GRID)].to_dict()}
    clf_oof = sum(x["clf_correct"] for x in r) / sum(x["n_reliable"] for x in r)
    (ROOT / "artifacts" / "hmm_config.json").write_text(json.dumps(
        {"params": best, "selected_on": "out-of-fold scores, 80 training trips", "grid": GRID,
         "train_oof_classifier_top1": clf_oof, "train_oof_hmm_top1": float(grid.iloc[0]["train_oof_hmm_top1"])},
        indent=2, default=float))

    # ---- 2. student test with the final model
    model, _, _ = fit_model(factory, tr_c)
    st_c["score"], _ = predict(model, st_c)
    joblib.dump(model, ROOT / "models" / f"final_{name}_{variant}.joblib", compress=3)
    st_trips = split_trips(st_c, st_p)
    with get_context("spawn").Pool(workers) as pool:  # spawn: forking after LightGBM's OpenMP threads deadlocks
        st_steps = dict(pool.map(per_trip, st_trips, chunksize=1))
    rows = evaluate(st_trips, st_steps, best)
    per = []
    for r, (tid, cc, pp) in zip(rows, st_trips):
        trip, net = trip_and_net(tid)
        d = {k: v for k, v in r.items() if not k.startswith("_")}
        for tag in ("top", "hmm"):
            arcs = r[f"_{tag}"]
            lab = "clf" if tag == "top" else "hmm"
            d[f"{lab}_top1"] = r[f"{lab}_correct"] / max(r["n_reliable"], 1)
            d[f"{lab}_continuity"] = route_continuity(arcs, net)
            d[f"{lab}_breaks"] = n_breaks(arcs, net)
            for k, v in sequence_metrics(arcs, trip.route, net).items():
                d[f"{lab}_{k}"] = v
        per.append(d)
        trip_map(trip, pp, net, reference=trip.route,
                 predictions={"Classifier only": r["_top"], "HMM + Viterbi": r["_hmm"]},
                 title=f"Student-test trip {tid}: HMM top-1 {d['hmm_top1']:.1%} vs classifier {d['clf_top1']:.1%}"
                 ).save(str(RES / "maps" / f"student_trip_{tid}.html"))
    per = pd.DataFrame(per)
    per.to_csv(RES / "hmm_per_trip.csv", index=False)
    tot = per[["n_reliable", "clf_correct", "hmm_correct"]].sum()
    summary = pd.DataFrame([
        {"method": "Classifier only (arg-max)", "top1_acc": tot.clf_correct / tot.n_reliable,
         "mean_trip_top1": per.clf_top1.mean(), "route_continuity": per.clf_continuity.mean(),
         "breaks_per_trip": per.clf_breaks.mean(), "route_f1": per.clf_route_f1.mean(), "lcs_ratio": per.clf_lcs_ratio.mean()},
        {"method": "Classifier + HMM/Viterbi", "top1_acc": tot.hmm_correct / tot.n_reliable,
         "mean_trip_top1": per.hmm_top1.mean(), "route_continuity": per.hmm_continuity.mean(),
         "breaks_per_trip": per.hmm_breaks.mean(), "route_f1": per.hmm_route_f1.mean(), "lcs_ratio": per.hmm_lcs_ratio.mean()},
    ])
    summary.to_csv(RES / "hmm_comparison.csv", index=False)
    print(summary.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(len(per)); w = 0.38
    ax[0].bar(xs - w / 2, per.clf_top1, w, label="classifier only"); ax[0].bar(xs + w / 2, per.hmm_top1, w, label="+ HMM/Viterbi")
    ax[0].set(xticks=xs, title="Student-test Top-1 accuracy per trip", ylim=(max(0, per[["clf_top1", "hmm_top1"]].min().min() - 0.05), 1))
    ax[0].set_xticklabels(per.trip_id, rotation=45); ax[0].legend()
    ax[1].bar(xs - w / 2, per.clf_breaks, w, label="classifier only"); ax[1].bar(xs + w / 2, per.hmm_breaks, w, label="+ HMM/Viterbi")
    ax[1].set(xticks=xs, title="Route breaks (consecutive arcs not connected)", yscale="symlog"); ax[1].set_xticklabels(per.trip_id, rotation=45); ax[1].legend()
    fig.tight_layout(); fig.savefig(RES / "figures" / "hmm_per_trip.png", dpi=130); plt.close(fig)

    piv = grid.pivot_table(index="beta", columns="w_emission", values="train_oof_hmm_top1", aggfunc="max")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set(xticks=range(piv.shape[1]), yticks=range(piv.shape[0]), xlabel="emission weight", ylabel="beta (m)",
           title="Decoder grid: train OOF Top-1 (best gap_s)")
    ax.set_xticklabels(piv.columns); ax.set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(RES / "figures" / "hmm_grid.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.model, a.workers)
