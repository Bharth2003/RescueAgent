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

# Bounding box around Edinburgh (south, west, north, east), widened to take in
# Portobello, Corstorphine, Currie and the airport fringe as well as the centre.
BBOX = (55.86, -3.44, 56.02, -3.02)

# Anywhere that plausibly ends the day with surplus food, not just sit-down
# restaurants: pubs and bars do kitchen service, bakeries and delis bin unsold
# stock, supermarkets and convenience shops have short-date shelves.
AMENITIES = ("restaurant|cafe|fast_food|pub|bar|bakery|ice_cream|food_court|"
             "biergarten|canteen")
SHOPS = "bakery|deli|greengrocer|supermarket|convenience|butcher|pastry|farm|seafood"

# Query nodes AND ways/relations - a lot of venues are mapped as building
# polygons rather than points, and the node-only query silently missed them all.
QUERY = f"""
[out:json][timeout:180];
(
  node["amenity"~"^({AMENITIES})$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["amenity"~"^({AMENITIES})$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  relation["amenity"~"^({AMENITIES})$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["shop"~"^({SHOPS})$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["shop"~"^({SHOPS})$"]["name"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
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
    """Best address we can assemble.

    Includes suburb/neighbourhood, because plenty of chain branches carry no
    housenumber or street and would otherwise all render as bare "Edinburgh",
    leaving the picker unable to tell one branch from another.
    """
    street = " ".join(p for p in (tags.get("addr:housenumber", ""),
                                  tags.get("addr:street", "")) if p)
    area = (tags.get("addr:suburb") or tags.get("addr:neighbourhood")
            or tags.get("addr:place") or tags.get("addr:district") or "")
    parts = [street, area, tags.get("addr:city", ""), tags.get("addr:postcode", "")]
    out, seen = [], set()
    for p in parts:
        p = p.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return ", ".join(out)


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
    # Dedupe on name + rough location, NOT name alone. Deduping by name threw
    # away every branch of every chain (all the Greggs collapsed into one row),
    # while ~3 decimal places still merges the duplicate nodes a single venue
    # gets for its outdoor seating or its building outline.
    seen: set[tuple[str, float, float]] = set()
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            continue

        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue

        key = (name.lower(), round(lat, 3), round(lng, 3))
        if key in seen:
            continue
        seen.add(key)

        restaurants.append({
            "name": name,
            "amenity_type": tags.get("amenity") or tags.get("shop") or "restaurant",
            "cuisine": _cuisine_tags(tags),
            "address": _address(tags),
            "phone": _phone(tags),
            "website": _website(tags),
            "lat": lat,
            "lng": lng,
        })

    # Sort so the JSON is stable across re-fetches, then number them.
    restaurants.sort(key=lambda r: (r["name"].lower(), r["lat"], r["lng"]))
    for i, r in enumerate(restaurants):
        r["id"] = f"rest_{i:04d}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(restaurants)} unique named restaurants -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
