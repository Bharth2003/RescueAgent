"""Generate a realistic Edinburgh volunteer driver roster and save to data/drivers.json.

Run manually to regenerate:
    python scripts/generate_drivers.py

Also writes a backup copy to data/seed/drivers.seed.json used by the app's
'Reset Demo Data' button.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A diverse Edinburgh-appropriate name pool (Scottish + international, reflecting
# the actual demographics of the city's volunteer base).
FIRST_NAMES = [
    "Alex", "Priya", "Callum", "Aisha", "Rory", "Fatima", "Hamish", "Zainab",
    "Iona", "Kwame", "Eilidh", "Mateus", "Struan", "Nia", "Fergus", "Amara",
    "Isla", "Sanjay", "Rowan", "Beatrix", "Cormac", "Ling", "Douglas", "Yara",
    "Skye", "Tomasz", "Freya", "Adnan", "Blair", "Chloe", "Duncan", "Ekaterina",
    "Gavin", "Halima", "Ivor", "Josephine", "Kirsty", "Liam", "Morag", "Neve",
]
LAST_NAMES = [
    "Mackenzie", "Sharma", "Fraser", "Ahmed", "MacLeod", "Hussain", "Sinclair",
    "Ramirez", "Buchanan", "Nkomo", "Anderson", "Chen", "Wallace", "Okafor",
    "Grant", "Patel", "Munro", "O'Neill", "Stewart", "Dubois", "Cameron", "Rossi",
    "MacDonald", "Kaur", "Sutherland", "Ali", "Ferguson", "Nowak", "Robertson",
    "Da Silva", "Henderson", "Petrov", "Reid", "Bello", "Watson", "Nakamura",
    "Kerr", "Jansson", "Clark", "Farah",
]

# Vehicle blueprints -- capacity, what they can carry, and what insulation they have.
VEHICLES = [
    # (type, weight, count, accepts, thermal_bag, cool_box)
    ("van",       300, 6,  ["hot", "cold", "frozen", "meat", "veg", "vegan", "halal", "dairy"], True,  True),
    ("car",        60, 12, ["hot", "cold", "meat", "veg", "vegan", "halal", "dairy"],           True,  True),
    ("motorbike",  20, 3,  ["hot", "cold", "meat", "veg", "vegan", "halal"],                    True,  False),
    ("e-bike",     15, 10, ["hot", "cold", "meat", "veg", "vegan", "halal"],                    True,  False),
    ("bicycle",     5, 8,  ["cold", "veg", "vegan"],                                            False, False),
    ("on-foot",     3, 1,  ["cold", "veg", "vegan"],                                            False, False),
]

# Spread starting locations across Edinburgh neighbourhoods.
EDINBURGH_HOTSPOTS = [
    (55.9533, -3.1883, "New Town"),
    (55.9478, -3.1936, "Grassmarket"),
    (55.9748, -3.1687, "Leith"),
    (55.9245, -3.2703, "Sighthill"),
    (55.9709, -3.2377, "Pilton"),
    (55.9188, -3.2779, "Wester Hailes"),
    (55.9612, -3.1730, "Norton Park"),
    (55.9436, -3.1858, "Southside"),
    (55.9295, -3.2495, "Broomhouse"),
    (55.9482, -3.1810, "Pleasance"),
    (55.9569, -3.1969, "West End"),
    (55.9351, -3.1665, "Craigmillar"),
    (55.9773, -3.2280, "Muirhouse"),
    (55.9634, -3.1573, "Restalrig"),
    (55.9581, -3.2110, "Stockbridge"),
]

# Status distribution -- roughly 60% available, 30% busy, 10% off duty.
STATUS_POOL = ["available"] * 24 + ["busy"] * 12 + ["off_duty"] * 4


def main() -> int:
    random.seed(20260824)  # deterministic roster; regenerating gives the same 40 drivers.

    name_pool = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    random.shuffle(name_pool)

    drivers = []
    counter = 1
    for vehicle_type, capacity_kg, count, accepts, has_bag, has_box in VEHICLES:
        for _ in range(count):
            first, last = name_pool.pop()
            lat, lng, neighbourhood = random.choice(EDINBURGH_HOTSPOTS)
            # Jitter position by a few hundred metres so drivers aren't stacked.
            lat += random.uniform(-0.004, 0.004)
            lng += random.uniform(-0.006, 0.006)
            drivers.append({
                "id": f"driver_{counter:03d}",
                "name": f"{first} {last}",
                "vehicle_type": vehicle_type,
                "vehicle_reg": f"SC{random.randint(10, 99)} {random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}",
                "max_capacity_kg": capacity_kg,
                "accepts": accepts,
                "has_thermal_bag": has_bag,
                "has_cool_box": has_box,
                "status": STATUS_POOL[counter - 1] if counter - 1 < len(STATUS_POOL) else "available",
                "current_lat": round(lat, 5),
                "current_lng": round(lng, 5),
                "neighbourhood": neighbourhood,
                "eta_minutes": random.randint(5, 30),
                "phone": f"07{random.randint(100, 999)} {random.randint(100000, 999999)}",
                "rating": round(random.uniform(4.2, 5.0), 1),
                "rescues_completed": random.randint(0, 47),
            })
            counter += 1

    # Shuffle so the roster isn't grouped by vehicle type.
    random.shuffle(drivers)

    out_path = DATA_DIR / "drivers.json"
    seed_dir = DATA_DIR / "seed"
    seed_dir.mkdir(exist_ok=True)
    seed_path = seed_dir / "drivers.seed.json"

    for p in (out_path, seed_path):
        p.write_text(json.dumps(drivers, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(drivers)
    avail = sum(1 for d in drivers if d["status"] == "available")
    busy = sum(1 for d in drivers if d["status"] == "busy")
    off = sum(1 for d in drivers if d["status"] == "off_duty")
    print(f"Wrote {total} drivers -> {out_path}")
    print(f"  status: {avail} available, {busy} busy, {off} off duty")
    print(f"  seed backup -> {seed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
