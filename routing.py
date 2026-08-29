"""Geo + routing helpers shared by tools.py and app.py.

Road geometry comes from OSRM (public demo server) with a local cache
fallback so a network hiccup on demo day cannot break live tracking.
"""

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_PATH = os.path.join(_HERE, "data", "routes_cache.json")

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{fl},{fa};{tl},{ta}?overview=full&geometries=geojson"
OSRM_TIMEOUT = 4.0

# average urban speed by vehicle class, km/h
SPEED_KMH = {
    "van": 26, "car": 28, "motorbike": 30,
    "scooter": 22, "e-bike": 17, "bicycle": 14,
}


def haversine_km(a, b):
    """a, b = (lat, lng)."""
    r = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlng = math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlng / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def path_km(coords):
    return sum(haversine_km(coords[i - 1], coords[i]) for i in range(1, len(coords)))


def point_at(coords, t):
    """Interpolate along a polyline. t in 0..1 measured by distance."""
    if not coords:
        return None
    t = max(0.0, min(1.0, t))
    target = path_km(coords) * t
    acc = 0.0
    for i in range(1, len(coords)):
        seg = haversine_km(coords[i - 1], coords[i])
        if acc + seg >= target:
            f = (target - acc) / seg if seg else 0.0
            return (coords[i - 1][0] + (coords[i][0] - coords[i - 1][0]) * f,
                    coords[i - 1][1] + (coords[i][1] - coords[i - 1][1]) * f)
        acc += seg
    return coords[-1]


def _cache_key(a, b):
    return f"{a[0]:.5f},{a[1]:.5f}->{b[0]:.5f},{b[1]:.5f}"


def _load_cache():
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _synthetic(a, b, n=26):
    """Curved fallback line — only used when OSRM and the cache both miss."""
    mid = ((a[0] + b[0]) / 2 + (b[1] - a[1]) * 0.10,
           (a[1] + b[1]) / 2 - (b[0] - a[0]) * 0.10)
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        out.append((mt * mt * a[0] + 2 * mt * t * mid[0] + t * t * b[0],
                    mt * mt * a[1] + 2 * mt * t * mid[1] + t * t * b[1]))
    return out


def get_route(a, b, vehicle_type="car", allow_network=True):
    """Return {coords, km, minutes, source}. coords = [(lat, lng), ...]."""
    key = _cache_key(a, b)
    cache = _load_cache()

    coords, source = None, "cached"
    if allow_network:
        try:
            import requests
            url = OSRM_URL.format(fa=a[0], fl=a[1], ta=b[0], tl=b[1])
            r = requests.get(url, timeout=OSRM_TIMEOUT)
            data = r.json()
            if data.get("routes"):
                coords = [(c[1], c[0]) for c in data["routes"][0]["geometry"]["coordinates"]]
                source = "osrm"
                cache[key] = coords
                try:
                    _save_cache(cache)
                except Exception:
                    pass
        except Exception:
            coords = None

    if coords is None and key in cache:
        coords = [tuple(c) for c in cache[key]]
        source = "cached"

    if coords is None or len(coords) < 2:
        coords = _synthetic(a, b)
        source = "estimated"

    km = path_km(coords)
    speed = SPEED_KMH.get(vehicle_type, 24)
    return {
        "coords": coords,
        "km": round(km, 2),
        "minutes": max(2, int(round(km / speed * 60))),
        "source": source,
    }
