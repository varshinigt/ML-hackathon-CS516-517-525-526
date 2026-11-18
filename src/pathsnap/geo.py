"""Geographic helpers.

Longitude/latitude are angles, so Euclidean distance in degrees is not a physical distance.
Each trip covers a few kilometres, so we use a local equirectangular (tangent-plane)
projection centred on the trip: x = R * dlon * cos(lat0), y = R * dlat. At city scale the
error relative to the true geodesic distance is well below GPS noise (< 0.1 % for a 10 km
extent), and it lets every downstream step work in metres with plain vector maths.
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(self.lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def fit(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(np.nanmean(lon), np.nanmean(lat))

    def forward(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inverse(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing(dx, dy):
    """Compass-style bearing in degrees [0, 360) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, degrees in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting) and projection fraction t in [0,1]."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t
