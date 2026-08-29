"""One-shot fetch of real Edinburgh restaurants from the OpenStreetMap Overpass API.

Run manually to refresh data/restaurants.json:
    python scripts/fetch_osm_restaurants.py

Overpass is free, unauthenticated, and covers Edinburgh well. We save the result to
disk so the runtime app never depends on the network being available.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box around central Edinburgh (south, west, north, east).
# This covers the main city area including Old Town, New Town, Leith, and Southside.
BBOX = (55.90, -3.36, 56.00, -3.05)

QUERY = f"""
[out:json][timeout:90];
(
  node["amenity"~"^(restaurant|cafe|fast_food)$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out center;
"""

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "restaurants.json"


def _cuisine_tags(tags: dict) -> list[str]:
    raw = tags.get("cuisine", "")
    if not raw:
        return []
    return [c.strip() for c in raw.replace(";", ",").split(",") if c.strip()]


def _phone(tags: dict) -> str:
    return tags.get("phone") or tags.get("contact:phone") or ""


def _website(tags: dict) -> str:
    return tags.get("website") or tags.get("contact:website") or ""


def _address(tags: dict) -> str:
    parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
        tags.get("addr:postcode", ""),
    ]
    return ", ".join(p for p in parts if p)


def main() -> int:
    print(f"Querying Overpass for Edinburgh restaurants in bbox {BBOX}...")
    headers = {
        # Overpass returns 406 without a real User-Agent identifying the app.
        "User-Agent": "RescueAgent-Hackathon/1.0 (contact: bharthks29@gmail.com)",
        "Accept": "application/json",
    }
    resp = requests.post(OVERPASS_URL, data={"data": QUERY}, headers=headers, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    elements = payload.get("elements", [])
    print(f"Overpass returned {len(elements)} raw elements.")

    restaurants = []
    seen_names: set[str] = set()
    for i, el in enumerate(elements):
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            continue

        # Dedupe by name (some restaurants have multiple OSM nodes for outdoor seating etc).
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)

        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue

        restaurants.append({
            "id": f"rest_{i:04d}",
            "name": name,
            "amenity_type": tags.get("amenity", "restaurant"),
            "cuisine": _cuisine_tags(tags),
            "address": _address(tags),
            "phone": _phone(tags),
            "website": _website(tags),
            "lat": lat,
            "lng": lng,
        })

    # Sort so the JSON is stable across re-fetches.
    restaurants.sort(key=lambda r: r["name"].lower())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(restaurants)} unique named restaurants -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
