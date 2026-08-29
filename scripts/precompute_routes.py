"""Pre-compute OSRM road geometry so live tracking works offline on demo day.

Caches every shelter <-> shelter-relevant pair we might need:
  * each shelter  -> every other shelter (delivery legs)
  * each driver's current position -> each shelter (pickup legs are dynamic,
    so we cache driver -> city-centre and driver -> every shelter)

Writes data/routes_cache.json. Re-run whenever drivers.json changes.

Usage:
    python scripts/precompute_routes.py            # shelters + drivers
    python scripts/precompute_routes.py --light    # shelters only (fast)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing import _cache_key, _load_cache, _save_cache, get_route  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
SLEEP = 1.1  # be polite to the public OSRM demo server


def read(name):
    with open(os.path.join(DATA, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    light = "--light" in sys.argv
    shelters = read("shelters.json")
    drivers = read("drivers.json")
    cache = _load_cache()

    pairs = []
    for a in shelters:
        for b in shelters:
            if a["id"] != b["id"]:
                pairs.append(((a["lat"], a["lng"]), (b["lat"], b["lng"]), "car"))

    if not light:
        for d in drivers:
            if d.get("current_lat") is None:
                continue
            for s in shelters:
                pairs.append(((d["current_lat"], d["current_lng"]),
                              (s["lat"], s["lng"]), d.get("vehicle_type", "car")))

    todo = [p for p in pairs if _cache_key(p[0], p[1]) not in cache]
    print(f"{len(pairs)} pairs, {len(todo)} missing from cache")

    for i, (a, b, veh) in enumerate(todo, 1):
        r = get_route(a, b, veh, allow_network=True)
        print(f"  [{i}/{len(todo)}] {r['km']:.2f} km  {r['source']}")
        if r["source"] != "osrm":
            print("    warning: OSRM miss, stored estimate")
        time.sleep(SLEEP)

    cache = _load_cache()
    _save_cache(cache)
    print(f"cache now holds {len(cache)} routes -> data/routes_cache.json")


if __name__ == "__main__":
    main()
