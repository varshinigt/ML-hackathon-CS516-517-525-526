"""Quick check that the API is up and can predict one trip.

Usage: python scripts/smoke_test_api.py [http://localhost:8000]
"""
import json
import sys
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


print("health:", call("/health"))
trips = call("/trips")["trips"]
print("trips:", len(trips))
res = call("/predict", {"trip_id": trips[0]["trip_id"]})
print("trip", res["trip_id"], "| route pieces:", len(res["predicted_route"]),
      "| HMM top-1:", (res["metrics"] or {}).get("hmm_top1_accuracy"), "| time:", res["timings_s"]["total"], "s")
