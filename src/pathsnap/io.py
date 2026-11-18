"""Raw dataset discovery and loading.

Each trip ships up to four header-less, whitespace-separated files that share a stem:
    <trip>.track  lon lat t           (GPS observations, ~1 Hz)
    <trip>.nodes  lon lat             (row index = node id, 0-based)
    <trip>.arcs   from_node to_node   (row index = arc id, 0-based)
    <trip>.route  arc_id              (chronological reference arcs)

The loader does not assume a folder layout: it searches recursively for *.track files and
groups sibling files with the same stem. Partition membership is read from data/splits.json
(see scripts/00_make_splits.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_FILE = ROOT / "data" / "splits.json"
PARTITIONS = ("train", "student_test", "eval_test")


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame            # columns lon, lat, t
    nodes: pd.DataFrame            # columns lon, lat ; index = node id
    arcs: pd.DataFrame             # columns u, v ; index = arc id
    route: np.ndarray | None = None  # reference arc ids, or None if absent
    files: dict = field(default_factory=dict)


def _read(path: Path, names: list[str], dtype=float) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=names)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")
    return df.astype(dtype)


def discover_trips(raw_dir: Path = RAW_DIR) -> dict[str, dict[str, Path]]:
    """Return {trip_id: {ext: path}} for every *.track found under raw_dir."""
    trips: dict[str, dict[str, Path]] = {}
    for track in sorted(raw_dir.rglob("*.track")):
        stem = track.stem
        trip_id = stem if stem not in trips else f"{track.parent.name}/{stem}"
        files = {"track": track}
        for ext in ("nodes", "arcs", "route"):
            p = track.with_suffix(f".{ext}")
            if p.exists():
                files[ext] = p
        trips[trip_id] = files
    return trips


def load_trip(trip_id: str, files: dict[str, Path], with_route: bool = True) -> Trip:
    track = _read(files.get("track"), ["lon", "lat", "t"])
    nodes = _read(files.get("nodes"), ["lon", "lat"])
    arcs = _read(files.get("arcs"), ["u", "v"], dtype=int)
    route = None
    if with_route and "route" in files:
        r = _read(files["route"], ["arc"], dtype=int)
        route = r["arc"].to_numpy()
    return Trip(trip_id, track, nodes, arcs, route, files)


FOLDER_ALIASES = {
    "train": "train", "training": "train",
    "student_test": "student_test", "student_testing": "student_test",
    "eval_test": "eval_test", "evaluator_test": "eval_test", "evaluator_testing": "eval_test",
    "eval": "eval_test", "test": "eval_test",
}


def infer_splits(raw_dir: Path = RAW_DIR) -> dict[str, list[str]]:
    """Partition membership comes from the folder each trip was delivered in
    (data/raw/train, data/raw/student_test, data/raw/eval_test). We never re-split:
    the organisers' partitions already keep every trip entirely within one partition."""
    splits = {p: [] for p in PARTITIONS}
    for tid, files in discover_trips(raw_dir).items():
        rel = files["track"].relative_to(raw_dir).parts
        part = next((FOLDER_ALIASES[x.lower()] for x in rel if x.lower() in FOLDER_ALIASES), None)
        if part:
            splits[part].append(tid)
    return {k: sorted(v) for k, v in splits.items()}


def load_splits() -> dict[str, list[str]]:
    """Folder-derived partitions (re-scanned every call so newly added eval trips are seen)."""
    splits = infer_splits()
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(splits, indent=1))
    return splits


def iter_partition(partition: str, with_route: bool = True):
    trips = discover_trips()
    for tid in load_splits()[partition]:
        yield load_trip(tid, trips[tid], with_route=with_route)
