"""Road network: projected arc geometry, spatial lookup and shortest-path distances."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .geo import LocalProjection, bearing, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs_uv: np.ndarray, proj: LocalProjection,
                 directed: bool = True, densify_m: float = 10.0):
        self.proj = proj
        self.directed = directed
        self.n_nodes = len(nodes_lonlat)
        self.nx, self.ny = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.u = arcs_uv[:, 0].astype(int)
        self.v = arcs_uv[:, 1].astype(int)
        self.n_arcs = len(self.u)
        self.ax, self.ay = self.nx[self.u], self.ny[self.u]
        self.bx, self.by = self.nx[self.v], self.ny[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing(self.bx - self.ax, self.by - self.ay)

        # Graph over nodes, edge weight = arc length (metres).
        w = np.maximum(self.length, 1e-3)
        self.graph = csr_matrix((w, (self.u, self.v)), shape=(self.n_nodes, self.n_nodes))
        if not directed:
            self.graph = self.graph.maximum(self.graph.T)
        self._sp_cache: dict[tuple[int, float], dict] = {}

        order = np.argsort(self.u, kind="stable")
        starts = np.searchsorted(self.u[order], np.arange(self.n_nodes + 1))
        self._arcs_from_node = (order, starts)

        # Densified sample points along arcs for radius queries.
        n_pts = np.maximum(1, np.ceil(self.length / densify_m).astype(int)) + 1
        arc_idx = np.repeat(np.arange(self.n_arcs), n_pts)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_pts]) if self.n_arcs else np.array([])
        sx = self.ax[arc_idx] + frac * (self.bx - self.ax)[arc_idx]
        sy = self.ay[arc_idx] + frac * (self.by - self.ay)[arc_idx]
        self._sample_arc = arc_idx
        self._tree = cKDTree(np.c_[sx, sy]) if self.n_arcs else None
        self._densify = densify_m

    # ------------------------------------------------------------------ lookup
    def arcs_from_node(self, node: int) -> np.ndarray:
        order, starts = self._arcs_from_node
        return order[starts[node]:starts[node + 1]]

    def nearby_arcs(self, x: float, y: float, radius: float, k: int):
        """Exact distances to the k nearest arcs within radius. Returns (arc_ids, dist, frac)."""
        if self._tree is None:
            return np.array([], int), np.array([]), np.array([])
        idx = self._tree.query_ball_point([x, y], radius + self._densify)
        if not idx:
            return np.array([], int), np.array([]), np.array([])
        arcs = np.unique(self._sample_arc[idx])
        d, t = point_segment(x, y, self.ax[arcs], self.ay[arcs], self.bx[arcs], self.by[arcs])
        keep = d <= radius
        arcs, d, t = arcs[keep], d[keep], t[keep]
        o = np.argsort(d)[:k]
        return arcs[o], d[o], t[o]

    def nearest_arc(self, x: float, y: float):
        _, j = self._tree.query([x, y], k=1)
        a = np.array([self._sample_arc[j]])
        d, t = point_segment(x, y, self.ax[a], self.ay[a], self.bx[a], self.by[a])
        return a, d, t

    # ------------------------------------------------------------ shortest path
    def node_dist(self, src: int, limit: float = np.inf) -> dict:
        """{node: distance} for nodes reachable from src within `limit` metres (cached, sparse)."""
        key = (int(src), limit)
        hit = self._sp_cache.get(key)
        if hit is None:
            d = dijkstra(self.graph, directed=True, indices=int(src), limit=limit)
            idx = np.flatnonzero(np.isfinite(d))
            hit = dict(zip(idx.tolist(), d[idx].tolist()))
            if len(self._sp_cache) > 20000:
                self._sp_cache.clear()
            self._sp_cache[key] = hit
        return hit

    def route_distance(self, a: int, ta: float, b: int, tb: float, limit: float = np.inf) -> float:
        """Network distance travelling from fraction ta on arc a to fraction tb on arc b."""
        La, Lb = self.length[a], self.length[b]
        # GPS noise can make the projection step slightly backwards along the same arc;
        # treat backward steps up to 15 m as standing still rather than a loop around the block.
        if a == b and (ta - tb) * La <= 15.0:
            return max(tb - ta, 0.0) * La
        if not self.directed and a == b:
            return abs(tb - ta) * La
        d_mid = self.node_dist(self.v[a], limit).get(int(self.u[b]), np.inf)
        best = (1 - ta) * La + d_mid + tb * Lb
        if not self.directed:  # may also leave a through u_a / enter b through v_b
            for s, sa in ((self.u[a], ta * La), (self.v[a], (1 - ta) * La)):
                dd = self.node_dist(s, limit)
                for e, eb in ((self.u[b], tb * Lb), (self.v[b], (1 - tb) * Lb)):
                    best = min(best, sa + dd.get(int(e), np.inf) + eb)
        return best

    def connected(self, a: int, b: int) -> bool:
        """True if arc b can directly follow arc a (same arc or sharing the junction)."""
        if a == b:
            return True
        if self.v[a] == self.u[b]:
            return True
        if not self.directed:
            return bool({self.u[a], self.v[a]} & {self.u[b], self.v[b]})
        return False

    # ----------------------------------------------------------------- summary
    def components(self):
        n_weak, lab_w = connected_components(self.graph, directed=True, connection="weak")
        n_strong, _ = connected_components(self.graph, directed=True, connection="strong")
        deg = np.bincount(self.u, minlength=self.n_nodes) + np.bincount(self.v, minlength=self.n_nodes)
        return {
            "n_weak_components": int(n_weak),
            "n_strong_components": int(n_strong),
            "largest_weak_component_frac": float(np.bincount(lab_w).max() / max(self.n_nodes, 1)),
            "isolated_nodes": int((deg == 0).sum()),
        }


def build_network(trip, proj: LocalProjection | None = None, directed: bool = True) -> RoadNetwork:
    if proj is None:
        proj = LocalProjection.fit(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    return RoadNetwork(trip.nodes[["lon", "lat"]].to_numpy(), trip.arcs[["u", "v"]].to_numpy(),
                       proj, directed=directed)
