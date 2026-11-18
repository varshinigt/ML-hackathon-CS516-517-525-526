"""Build candidate tables + features for the train and student_test partitions.

Output: data/processed/{cands,points}_{partition}.pkl and results/preprocessing_log.csv
The evaluator partition is NEVER processed here; it is only touched by the API at demo time.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathsnap.features import candidate_miss_rate, prepare_trip  # noqa: E402
from pathsnap.io import ROOT, discover_trips, load_splits, load_trip  # noqa: E402

OUT = ROOT / "data" / "processed"


def work(args):
    tid, files, part = args
    t0 = time.time()
    trip = load_trip(tid, files)
    c, p, net, log = prepare_trip(trip)
    log.update(trip_id=tid, partition=part, n_arcs=net.n_arcs, n_nodes=net.n_nodes,
               seconds=round(time.time() - t0, 2), miss_rate=candidate_miss_rate(c, p))
    return c, p.drop(columns=[]), log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trips, splits = discover_trips(), load_splits()
    logs = []
    for part in ("train", "student_test"):
        jobs = [(t, trips[t], part) for t in splits[part]]
        with Pool(processes=min(4, len(jobs))) as pool:
            res = pool.map(work, jobs, chunksize=1)
        pd.concat([r[0] for r in res], ignore_index=True).to_pickle(OUT / f"cands_{part}.pkl")
        pd.concat([r[1] for r in res], ignore_index=False).to_pickle(OUT / f"points_{part}.pkl")
        logs += [r[2] for r in res]
        print(part, "done", sum(len(r[0]) for r in res), "candidate rows")
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(logs).to_csv(ROOT / "results" / "preprocessing_log.csv", index=False)


if __name__ == "__main__":
    main()
