"""Generate a seed of past rescues for analytics/history views.

Run:
    python scripts/generate_history.py

Writes data/rescue_history.json and data/seed/rescue_history.seed.json.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FOODS = [
    ("Chicken biryani leftovers",     "hot",  "meat",       4.5),
    ("Vegetable curry and rice",      "hot",  "vegetarian", 3.2),
    ("Ham and cheese sandwiches",     "cold", "meat",       1.8),
    ("Garden salad boxes",            "cold", "vegan",      2.4),
    ("Sourdough loaves (day old)",    "cold", "vegan",      6.0),
    ("Fish and chips (portions)",     "hot",  "fish",       5.5),
    ("Pizza margherita (whole pies)", "hot",  "vegetarian", 3.6),
    ("Lentil soup",                   "hot",  "vegan",      4.0),
    ("Yoghurt pots (unopened)",       "cold", "vegetarian", 2.0),
    ("Falafel wraps",                 "cold", "vegan",      2.8),
    ("Roast beef trays",              "hot",  "meat",       5.0),
    ("Pastries and croissants",       "cold", "vegetarian", 1.5),
    ("Lamb rogan josh",               "hot",  "halal",      4.2),
    ("Pasta bake",                    "hot",  "vegetarian", 3.8),
    ("Fruit boxes (mixed)",           "cold", "vegan",      3.5),
]

RESTAURANTS = [
    "Dishoom", "Mother India", "Mosque Kitchen", "The Kitchin", "Civerinos",
    "Ondine", "Ting Thai", "Mono", "Union of Genius", "Kalpna",
    "The Piemaker", "Wildmanwood Fish Bar", "Nom Vietnamese", "Namaste Kathmandu",
    "Cafe Andaluz", "Mimi's Bakehouse", "Söderberg", "Bross Bagels",
    "Twelve Triangles", "Lovecrumbs",
]

SHELTERS = [
    ("shelter_001", "Grassmarket Community Project"),
    ("shelter_002", "Social Bite - Rose Street"),
    ("shelter_004", "Cyrenians - Norton Park"),
    ("shelter_005", "Bethany Christian Trust Care Van"),
    ("shelter_006", "Edinburgh Food Project - Sighthill"),
    ("shelter_007", "Empty Kitchens Full Hearts"),
    ("shelter_010", "Muirhouse Community Larder"),
    ("shelter_013", "Wester Hailes Foodbank"),
    ("shelter_016", "Sikh Sanjog - Sunday Langar"),
    ("shelter_017", "Edinburgh Central Mosque - Iftar Fund"),
]

DRIVER_NAMES = [
    "Alex Mackenzie", "Priya Sharma", "Callum Fraser", "Aisha Ahmed",
    "Rory MacLeod", "Fatima Hussain", "Hamish Sinclair", "Kwame Nkomo",
    "Eilidh Anderson", "Mateus Da Silva", "Nia Okafor", "Iona Grant",
]


def main() -> int:
    random.seed(20260824)
    now = datetime(2026, 8, 24, 20, 0)

    records = []
    for i in range(60):
        food_desc, temperature, category, weight_kg = random.choice(FOODS)
        weight_kg = round(weight_kg + random.uniform(-1.0, 2.5), 1)
        weight_kg = max(0.8, weight_kg)
        meals = round(weight_kg / 0.4)
        shelter_id, shelter_name = random.choice(SHELTERS)
        # Rescues spread over the last 45 days.
        ts = now - timedelta(days=random.randint(0, 45), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        records.append({
            "id": f"rescue_{i+1:04d}",
            "timestamp": ts.isoformat(timespec="minutes"),
            "restaurant": random.choice(RESTAURANTS),
            "food_summary": food_desc,
            "temperature": temperature,
            "category": category,
            "weight_kg": weight_kg,
            "meals_delivered": meals,
            "co2_kg_saved": round(weight_kg * 2.5, 1),
            "shelter_id": shelter_id,
            "shelter_name": shelter_name,
            "driver_name": random.choice(DRIVER_NAMES),
            "delivery_minutes": random.randint(12, 55),
            "status": "delivered",
        })

    records.sort(key=lambda r: r["timestamp"])

    out = DATA_DIR / "rescue_history.json"
    seed = DATA_DIR / "seed" / "rescue_history.seed.json"
    seed.parent.mkdir(exist_ok=True)
    for p in (out, seed):
        p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    total_meals = sum(r["meals_delivered"] for r in records)
    total_co2 = sum(r["co2_kg_saved"] for r in records)
    print(f"Wrote {len(records)} historical rescues -> {out}")
    print(f"  totals: {total_meals} meals, {total_co2:.1f} kg CO2 avoided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
