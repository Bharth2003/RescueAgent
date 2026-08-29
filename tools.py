"""RescueAgent tools exposed to the Strands agent.

Original three tools are unchanged in behaviour (analyze_food_safety,
find_eligible_shelter, dispatch_driver). Added for live tracking:
  - broadcast_rescue   : offer the job to the 3 nearest capable drivers
  - accept_rescue      : first driver to accept wins, the rest are withdrawn
  - route_lookup       : road geometry + distance + ETA for a leg
  - release_driver     : put a driver back on the available roster
"""

import json
import os

from strands import tool

from routing import get_route, haversine_km

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_HOT_KEYWORDS = ("hot", "warm", "cooked", "baked", "soup", "curry", "biryani",
                 "roast", "fried", "stew", "grill", "pizza", "pasta")
_MEAT_KEYWORDS = ("meat", "chicken", "beef", "pork", "fish", "lamb", "mutton",
                  "turkey", "duck", "bacon", "ham", "sausage", "mince", "steak",
                  "prawn", "salmon")
_DRINKS_KEYWORDS = ("drink", "beverage", "juice", "soda", "water", "tea",
                    "coffee", "smoothie", "lassi")


# ---------------------------------------------------------------- data access
def _read(name):
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(name, payload):
    with open(os.path.join(_DATA_DIR, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _rules():
    r = _read("safety_rules.json")
    return r if isinstance(r, dict) else {}


def estimate_weight_kg(text: str) -> float:
    """Pull a kg figure out of free text, falling back to portion counts."""
    import re
    t = (text or "").lower()
    total = sum(float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:kg|kilo)", t))
    if not total:
        units = re.findall(r"(\d+)\s*(?:portion|meal|tray|box|pizza|naan|loaf|loaves|pastr)", t)
        total = sum(int(u) for u in units) * 0.35
    return round(total or 4.0, 1)


# ------------------------------------------------------------------- tool 1
@tool
def analyze_food_safety(food_description: str) -> str:
    """Classify surplus food and attach the FSA handover window."""
    text = (food_description or "").lower()
    is_hot = any(k in text for k in _HOT_KEYWORDS)
    has_meat = any(k in text for k in _MEAT_KEYWORDS)
    has_drinks = any(k in text for k in _DRINKS_KEYWORDS)
    windows = _rules().get("thermal_windows_minutes", {})
    window = windows.get("hot_cooked", 90) if is_hot else windows.get("cold_chilled", 240)
    weight = estimate_weight_kg(text)

    if is_hot:
        note = (f"Hot cooked food: must reach the shelter within {window} minutes. "
                "Driver requires a thermal bag.")
    else:
        note = f"Chilled/ambient food: {window} minute window. Cool box preferred."

    return json.dumps({
        "needs_hot": is_hot,
        "needs_meat": has_meat,
        "needs_drinks": has_drinks,
        "weight_kg": weight,
        "fsa_window_minutes": window,
        "category": "contains meat" if has_meat else ("vegan" if "vegan" in text else "vegetarian/ambient"),
        "safety_note": note,
    })


# ------------------------------------------------------------------- tool 2
@tool
def find_eligible_shelter(needs_hot: bool, needs_meat: bool, needs_drinks: bool,
                          near_lat: float = None, near_lng: float = None) -> str:
    """Pick the shelter that can take this food, ranked by demand then distance."""
    shelters = _read("shelters.json")
    if not shelters:
        return json.dumps({"error": "No shelters available."})

    eligible = []
    for s in shelters:
        if needs_hot and not s.get("accepts_hot_food"):
            continue
        if not needs_hot and not s.get("accepts_cold_food"):
            continue
        if needs_meat and not s.get("accepts_meat"):
            continue
        if s.get("capacity_meals", 0) - s.get("current_intake_meals", 0) < 6:
            continue
        eligible.append(s)

    if not eligible:
        return json.dumps({"error": "No eligible shelter found."})

    if near_lat is not None and near_lng is not None:
        eligible.sort(key=lambda s: -(s.get("demand_score", 0) * 1.4)
                      + haversine_km((near_lat, near_lng), (s["lat"], s["lng"])))
    else:
        eligible.sort(key=lambda s: -s.get("demand_score", 0))

    return json.dumps(eligible[0])


# ------------------------------------------------------------------- tool 3
@tool
def dispatch_driver(shelter_id: str) -> str:
    """Legacy single-driver dispatch: assigns the first available driver."""
    drivers = _read("drivers.json")
    for d in drivers:
        if d.get("status") == "available":
            d["status"] = "busy"
            _write("drivers.json", drivers)
            return json.dumps({"driver_id": d["id"], "name": d["name"], "status": "dispatched"})
    return json.dumps({"error": "All drivers busy."})


# ------------------------------------------------------------------- tool 4
@tool
def broadcast_rescue(shelter_id: str, needs_hot: bool, needs_meat: bool,
                     weight_kg: float, from_lat: float, from_lng: float,
                     fanout: int = 3) -> str:
    """Offer the job to the nearest capable drivers. First to accept wins.

    A driver is capable when they are available, can carry the load, accept the
    food class, and (for hot food) carry a thermal bag.
    """
    drivers = _read("drivers.json")
    candidates = []
    for d in drivers:
        if d.get("status") != "available":
            continue
        if d.get("max_capacity_kg", 0) < weight_kg:
            continue
        accepts = d.get("accepts", [])
        if needs_hot and not ("hot" in accepts and d.get("has_thermal_bag")):
            continue
        if not needs_hot and "cold" not in accepts:
            continue
        if needs_meat and "meat" not in accepts:
            continue
        if d.get("current_lat") is None:
            continue
        candidates.append({
            **d,
            "distance_km": round(haversine_km((from_lat, from_lng),
                                              (d["current_lat"], d["current_lng"])), 2),
        })

    candidates.sort(key=lambda d: d["distance_km"])
    offered = candidates[:max(1, fanout)]

    if not offered:
        return json.dumps({"error": "No capable driver on shift.",
                           "checked": len(drivers), "shelter_id": shelter_id})

    return json.dumps({
        "shelter_id": shelter_id,
        "offered": [{
            "driver_id": d["id"], "name": d["name"], "vehicle_type": d["vehicle_type"],
            "vehicle_reg": d.get("vehicle_reg", ""), "distance_km": d["distance_km"],
            "max_capacity_kg": d["max_capacity_kg"], "neighbourhood": d.get("neighbourhood", ""),
            "rating": d.get("rating"), "has_thermal_bag": d.get("has_thermal_bag", False),
            "current_lat": d["current_lat"], "current_lng": d["current_lng"],
        } for d in offered],
    })


# ------------------------------------------------------------------- tool 5
@tool
def accept_rescue(driver_id: str) -> str:
    """Mark the accepting driver busy. Other offers on the job are withdrawn."""
    drivers = _read("drivers.json")
    for d in drivers:
        if d["id"] == driver_id:
            if d.get("status") != "available":
                return json.dumps({"error": "Offer already taken.", "driver_id": driver_id})
            d["status"] = "busy"
            _write("drivers.json", drivers)
            return json.dumps({"driver_id": driver_id, "name": d["name"], "status": "assigned"})
    return json.dumps({"error": "Unknown driver.", "driver_id": driver_id})


# ------------------------------------------------------------------- tool 6
@tool
def route_lookup(from_lat: float, from_lng: float, to_lat: float, to_lng: float,
                 vehicle_type: str = "car") -> str:
    """Road geometry, distance and ETA for one leg. OSRM live, cache fallback."""
    r = get_route((from_lat, from_lng), (to_lat, to_lng), vehicle_type)
    return json.dumps({
        "distance_km": r["km"],
        "eta_minutes": r["minutes"],
        "source": r["source"],
        "points": len(r["coords"]),
        "polyline": [[round(a, 5), round(b, 5)] for a, b in r["coords"]],
    })


# ------------------------------------------------------------------- tool 7
@tool
def release_driver(driver_id: str) -> str:
    """Return a driver to the available roster and credit the rescue."""
    drivers = _read("drivers.json")
    for d in drivers:
        if d["id"] == driver_id:
            d["status"] = "available"
            d["rescues_completed"] = d.get("rescues_completed", 0) + 1
            _write("drivers.json", drivers)
            return json.dumps({"driver_id": driver_id, "status": "available",
                               "rescues_completed": d["rescues_completed"]})
    return json.dumps({"error": "Unknown driver.", "driver_id": driver_id})
