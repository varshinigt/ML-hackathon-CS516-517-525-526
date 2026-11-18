"""Folium map generation (EDA, report and API responses share the same layers)."""
from __future__ import annotations

import folium
import numpy as np

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "classifier": "#ff7f0e", "hmm": "#d62728"}

LEGEND = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
     padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font: 13px sans-serif;">
<b>{title}</b><br>{rows}</div>"""


def _arc_lines(trip, arc_ids):
    nodes = trip.nodes[["lat", "lon"]].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    return [[nodes[u[a]].tolist(), nodes[v[a]].tolist()] for a in arc_ids]


def _arcs_to_polyline(trip, arcs):
    """Collapse consecutive duplicate arcs and return a list of polylines (break at gaps)."""
    nodes = trip.nodes[["lat", "lon"]].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    seq = [int(a) for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    lines, cur = [], []
    for a in seq:
        if cur and u[a] != cur_end:
            lines.append(cur)
            cur = []
        if not cur:
            cur = [nodes[u[a]].tolist()]
        cur.append(nodes[v[a]].tolist())
        cur_end = v[a]
    if cur:
        lines.append(cur)
    return lines


def trip_map(trip, pts=None, net=None, reference=None, predictions: dict | None = None,
             title="PathSnap", network_buffer_m=150, max_gps=4000):
    lat, lon = trip.track["lat"].to_numpy(), trip.track["lon"].to_numpy()
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=14,
                   tiles="cartodbpositron", control_scale=True)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    rows = []

    if net is not None and pts is not None:
        # only arcs near the trajectory, otherwise maps of 300k-arc networks are unusable
        near = set()
        step = max(1, len(pts) // 1500)
        for x, y in zip(pts["x"].to_numpy()[::step], pts["y"].to_numpy()[::step]):
            a, _, _ = net.nearby_arcs(x, y, network_buffer_m, 40)
            near.update(a.tolist())
        fg = folium.FeatureGroup(name="Road network (near trip)", show=True)
        for line in _arc_lines(trip, sorted(near)):
            folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        rows.append(f'<span style="color:{COLORS["network"]}">&#9473;</span> Road network')

    if reference is not None and len(reference):
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _arcs_to_polyline(trip, reference):
            folium.PolyLine(line, color=COLORS["reference"], weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m)
        rows.append(f'<span style="color:{COLORS["reference"]}">&#9473;</span> Reference route')

    for name, arcs in (predictions or {}).items():
        col = COLORS.get(name.split()[0].lower(), "#9467bd")
        fg = folium.FeatureGroup(name=f"Predicted: {name}", show=True)
        for line in _arcs_to_polyline(trip, arcs):
            folium.PolyLine(line, color=col, weight=3.5, opacity=0.9, dash_array="6 4" if "lassif" in name else None).add_to(fg)
        fg.add_to(m)
        rows.append(f'<span style="color:{col}">&#9473;</span> Predicted ({name})')

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lat) // max_gps)
    for la, lo in zip(lat[::step], lon[::step]):
        folium.CircleMarker([float(la), float(lo)], radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m)
    rows.append(f'<span style="color:{COLORS["gps"]}">&#9679;</span> GPS fixes' + (f" (every {step}th)" if step > 1 else ""))
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=title, rows="<br>".join(rows))))
    return m
