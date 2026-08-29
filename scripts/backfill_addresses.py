"""Fill in addresses for venues that OSM has no addr:* tags for.

Some venues — including a couple of KFC and McDonald's branches — are mapped in
OpenStreetMap with a name and a location but no address tags, so they show up in
the picker as a bare "Edinburgh" and are useless as a rescue pickup. This asks
Nominatim (OSM's own reverse geocoder) what is at those coordinates and writes
the real street address back.

Nothing is invented: every address here comes from OSM data for that exact point.

Nominatim's usage policy allows 1 request/second with an identifying User-Agent,
so this takes a few minutes. Run it after fetch_osm_restaurants.py:

    python scripts/backfill_addresses.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "RescueAgent-Hackathon/1.0 (contact: bharthks29@gmail.com)"}
RATE_LIMIT_SECONDS = 1.1          # policy is 1 req/s; leave headroom

DATA = Path(__file__).resolve().parent.parent / "data" / "restaurants.json"


def _compose(addr: dict) -> str:
    """Street-level address from a Nominatim address object."""
    street = " ".join(p for p in (addr.get("house_number", ""),
                                  addr.get("road", "")) if p)
    area = (addr.get("suburb") or addr.get("neighbourhood")
            or addr.get("city_district") or addr.get("village") or "")
    city = addr.get("city") or addr.get("town") or addr.get("county") or ""
    parts, seen, out = [street, area, city, addr.get("postcode", "")], set(), []
    for p in parts:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return ", ".join(out)


def main() -> int:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    todo = [r for r in rows if not (r.get("address") or "").strip()]
    print(f"{len(todo)} of {len(rows)} venues have no address; reverse-geocoding "
          f"(~{len(todo) * RATE_LIMIT_SECONDS / 60:.1f} min)…")

    filled = 0
    for i, r in enumerate(todo, 1):
        try:
            resp = requests.get(
                NOMINATIM,
                params={"format": "jsonv2", "lat": r["lat"], "lon": r["lng"],
                        "zoom": 18, "addressdetails": 1},
                headers=HEADERS, timeout=30)
            resp.raise_for_status()
            addr = _compose(resp.json().get("address", {}))
            if addr:
                r["address"] = addr
                filled += 1
        except Exception as exc:                       # keep going; partial is fine
            print(f"  ! {r['name']}: {exc}")
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done, {filled} filled")
        time.sleep(RATE_LIMIT_SECONDS)

    DATA.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    still = sum(1 for r in rows if not (r.get("address") or "").strip())
    print(f"Filled {filled}. {still} still without an address -> {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
