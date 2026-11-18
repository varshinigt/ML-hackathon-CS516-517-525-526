"""Exploratory data analysis on the train + student_test partitions.

Outputs (results/eda/ and results/figures/, results/maps/):
  inventory.csv           per-trip file sizes, row counts, time gaps, extent, network stats
  data_quality.md         short record of issues, assumptions and decisions
  eda_*.png               plots used in report section 2
  eda_trip_<id>.html      Folium maps of GPS + network + reference route
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.geo import LocalProjection  # noqa: E402
from pathsnap.io import ROOT, discover_trips, load_splits, load_trip  # noqa: E402
from pathsnap.network import build_network  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

EDA = ROOT / "results" / "eda"
FIG = ROOT / "results" / "figures"
MAPS = ROOT / "results" / "maps"


def trip_stats(trip, part):
    tr = trip.track
    proj = LocalProjection.fit(tr["lon"], tr["lat"])
    x, y = proj.forward(tr["lon"], tr["lat"])
    dt = np.diff(tr["t"].to_numpy())
    step = np.hypot(np.diff(x), np.diff(y))
    net = build_network(trip, proj)
    comp = net.components()
    r = trip.route if trip.route is not None else np.array([], int)
    disc = int((trip.arcs["v"].to_numpy()[r[:-1]] != trip.arcs["u"].to_numpy()[r[1:]]).sum()) if len(r) > 1 else 0
    return {
        "trip_id": trip.trip_id, "partition": part,
        **{f"size_kb_{k}": round(p.stat().st_size / 1024, 1) for k, p in trip.files.items()},
        "n_gps": len(tr), "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs), "route_len_arcs": len(r),
        "missing_values": int(tr.isna().sum().sum()),
        "non_monotonic_t": int((dt <= 0).sum()),
        "frac_dt_1s": float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan,
        "max_gap_s": float(dt.max()) if len(dt) else np.nan,
        "n_gaps_gt_10s": int((dt > 10).sum()),
        "duration_min": float((tr["t"].iloc[-1] - tr["t"].iloc[0]) / 60),
        "track_len_km": float(step.sum() / 1000),
        "max_jump_m": float(step.max()) if len(step) else np.nan,
        "extent_ew_km": float((x.max() - x.min()) / 1000), "extent_ns_km": float((y.max() - y.min()) / 1000),
        "lon_mean": float(tr["lon"].mean()), "lat_mean": float(tr["lat"].mean()),
        "route_len_km": float(net.length[r].sum() / 1000) if len(r) else 0.0,
        "route_discontinuities": disc,
        "mean_arc_len_m": float(net.length.mean()),
        "reverse_arc_frac": float(len(set(zip(trip.arcs.u, trip.arcs.v)) & set(zip(trip.arcs.v, trip.arcs.u))) / max(len(trip.arcs), 1)),
        **comp,
    }


def main():
    for d in (EDA, FIG, MAPS):
        d.mkdir(parents=True, exist_ok=True)
    trips, splits = discover_trips(), load_splits()
    rows, sample_tracks = [], {}
    for part in ("train", "student_test"):
        for tid in splits[part]:
            trip = load_trip(tid, trips[tid])
            rows.append(trip_stats(trip, part))
            sample_tracks[tid] = (part, trip.track)
            print(tid, end=" ", flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(EDA / "inventory.csv", index=False)
    summ = inv.groupby("partition")[["n_gps", "n_arcs", "n_nodes", "route_len_arcs", "duration_min",
                                      "track_len_km", "max_gap_s", "frac_dt_1s"]].describe().T
    summ.to_csv(EDA / "summary_stats.csv")
    part_counts = inv.groupby("partition").agg(trips=("trip_id", "count"), gps_points=("n_gps", "sum"))
    part_counts.to_csv(EDA / "partition_counts.csv")

    # ---------------------------------------------------------------- plots (train + student_test)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in inv.groupby("partition"):
        ax[0].hist(g["n_gps"], bins=30, alpha=0.6, label=part)
        ax[1].hist(np.log10(g["n_arcs"]), bins=30, alpha=0.6, label=part)
        ax[2].hist(g["route_len_km"], bins=30, alpha=0.6, label=part)
    ax[0].set(title="GPS observations per trip", xlabel="# fixes")
    ax[1].set(title="Road-network size per trip", xlabel="log10(# arcs)")
    ax[2].set(title="Reference route length", xlabel="km")
    for a in ax:
        a.legend()
    fig.tight_layout(); fig.savefig(FIG / "eda_distributions.png", dpi=130); plt.close(fig)

    all_dt = np.concatenate([np.diff(tr["t"].to_numpy()) for _, tr in sample_tracks.values()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnt = np.unique(np.round(all_dt), return_counts=True)
    top = np.argsort(-cnt)[:10]
    ax[0].bar([str(int(v)) for v in vals[top]], cnt[top])
    ax[0].set(title="Sampling interval (top-10 values, train+student_test)", xlabel="seconds", ylabel="count", yscale="log")
    ax[1].hist(all_dt[all_dt > 1.5], bins=np.logspace(0.2, 3.5, 40))
    ax[1].set(xscale="log", title="Irregular gaps (> 1.5 s)", xlabel="seconds")
    fig.tight_layout(); fig.savefig(FIG / "eda_sampling_intervals.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for tid, (part, tr) in sample_tracks.items():
        ax.plot(tr["lon"], tr["lat"], lw=0.6, color="tab:blue" if part == "train" else "tab:orange", alpha=0.7)
    ax.set(title="All GPS trajectories (blue=train, orange=student_test)", xlabel="longitude", ylabel="latitude")
    fig.tight_layout(); fig.savefig(FIG / "eda_all_trajectories.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(inv["n_arcs"], inv["n_weak_components"], s=12)
    ax[0].set(xscale="log", title="Network size vs weakly connected components", xlabel="# arcs", ylabel="# components")
    ax[1].hist(inv["largest_weak_component_frac"], bins=20)
    ax[1].set(title="Share of nodes in largest component", xlabel="fraction")
    fig.tight_layout(); fig.savefig(FIG / "eda_connectivity.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- folium maps
    pick = [inv.query("partition=='train'").sort_values("n_gps").iloc[len(inv) // 4]["trip_id"],
            inv.loc[inv["max_gap_s"].idxmax(), "trip_id"]]
    for tid in pick:
        trip = load_trip(tid, trips[tid])
        proj = LocalProjection.fit(trip.track["lon"], trip.track["lat"])
        net = build_network(trip, proj)
        x, y = proj.forward(trip.track["lon"], trip.track["lat"])
        pts = pd.DataFrame({"x": x, "y": y})
        trip_map(trip, pts, net, reference=trip.route,
                 title=f"Trip {tid} ({inv.set_index('trip_id').loc[tid, 'partition']}): raw GPS vs reference").save(
            str(MAPS / f"eda_trip_{tid}.html"))

    anomalies = inv[(inv["max_gap_s"] > 30) | (inv["route_discontinuities"] > 0) | (inv["max_jump_m"] > 200)]
    anomalies.to_csv(EDA / "anomalies.csv", index=False)
    print("\n", part_counts, "\n", anomalies[["trip_id", "max_gap_s", "route_discontinuities", "max_jump_m"]])


if __name__ == "__main__":
    main()
