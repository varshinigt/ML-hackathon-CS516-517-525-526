"""Recompute route-level metrics for every model's student-test predictions.

Adds strict continuity (consecutive arcs identical/adjacent) and continuity with the
<=100 m connection tolerance used everywhere else, so all models are scored with the same
definition. Updates results/model_comparison.csv in place.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathsnap.experiment import load_partition, obs_top1, trip_and_net  # noqa: E402
from pathsnap.io import ROOT  # noqa: E402
from pathsnap.metrics import route_continuity, sequence_metrics  # noqa: E402

RES = ROOT / "results"
cmp = pd.read_csv(RES / "model_comparison.csv")
st_c, st_p = load_partition("student_test")
rows = []
for name in cmp["model"]:
    pr = pd.read_pickle(RES / "preds" / f"student_{name}.pkl")
    pts = obs_top1(st_c, st_p, pr["score"].to_numpy())
    for tid, g in pts.groupby("trip_id"):
        trip, net = trip_and_net(tid)
        arcs = g.sort_values("pt")["pred_arc"].to_numpy()
        rows.append({"model": name, "trip_id": tid,
                     "continuity": route_continuity(arcs, net),
                     "continuity_strict": route_continuity(arcs, net, 0),
                     **sequence_metrics(arcs, trip.route, net)})
    print(name, flush=True)
per = pd.DataFrame(rows)
per.to_csv(RES / "route_metrics_per_trip_per_model.csv", index=False)
agg = per.groupby("model").agg(st_route_continuity=("continuity", "mean"),
                               st_route_continuity_strict=("continuity_strict", "mean"),
                               st_route_f1=("route_f1", "mean"), st_lcs_ratio=("lcs_ratio", "mean"))
cmp = cmp.drop(columns=[c for c in agg.columns if c in cmp.columns]).merge(agg, left_on="model", right_index=True)
cmp.to_csv(RES / "model_comparison.csv", index=False)
print(cmp[["model", "st_top1_acc", "st_route_continuity", "st_route_continuity_strict", "st_route_f1"]].to_string())
