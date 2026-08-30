"""
RescueAgent — Edinburgh Food Rescue
===================================
Streamlit dashboard with Uber-style live driver tracking.

Built from native Streamlit components; the only custom code is the tracking
map (see live_map.py), which exists because st_folium re-rendered the whole map
on every rerun and made it blink. Type scale, colours and radii live in
.streamlit/config.toml — there is no CSS injected here.

Run:  streamlit run app.py
"""

import collections
import json
import os
import time
from datetime import datetime

import streamlit as st

from strands import Agent
from strands.models import BedrockModel

from live_map import live_map
from routing import get_route, haversine_km
from tools import (accept_rescue, analyze_food_safety, broadcast_rescue,
                   dispatch_driver, estimate_weight_kg, find_eligible_shelter,
                   release_driver, route_lookup)

# ---------------------------------------------------------------- config
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BEDROCK_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"
BEDROCK_REGION = "eu-west-2"

# Journey clock. A leg plays at roughly the real driving time OSRM gives for it,
# so a 0.4 km hop is quick and a 4 km run across town genuinely takes longer —
# a fixed duration for every leg is what made the old demo feel artificial.
# Clamped at both ends so a trip stays watchable without being a fast-forward.
LEG_MIN_SECONDS = 80.0
LEG_MAX_SECONDS = 300.0
OFFER_SECONDS = 20.0    # auto-accept if nobody taps Accept
PICKUP_SECONDS = 35.0   # auto-handover if nobody taps the button


def leg_seconds(leg):
    """How long to play one leg for, in wall-clock seconds."""
    return max(LEG_MIN_SECONDS, min(LEG_MAX_SECONDS, leg["minutes"] * 60.0))

BRAND = "#FF7A2F"
VEH = {"van": "Van", "car": "Car", "motorbike": "Motorbike", "e-bike": "E-bike",
       "bicycle": "Bicycle", "scooter": "Scooter"}
VEH_ICON = {"van": ":material/local_shipping:", "car": ":material/directions_car:",
            "motorbike": ":material/two_wheeler:", "e-bike": ":material/electric_bike:",
            "bicycle": ":material/pedal_bike:", "scooter": ":material/moped:"}
STAGES = ["Assigned", "To pickup", "At kitchen", "Delivering", "Delivered"]
PHASE_STAGE = {"idle": -1, "broadcast": 0, "to_pickup": 1, "at_pickup": 2,
               "to_shelter": 3, "delivered": 4}

# accent -> (badge colour, toast icon, alert renderer)
ACCENTS = {
    "brand": {"badge": "primary", "toast": "🛵", "alert": st.info},
    "green": {"badge": "green", "toast": "✅", "alert": st.success},
    "amber": {"badge": "orange", "toast": "⏳", "alert": st.warning},
    "red": {"badge": "red", "toast": "⚠️", "alert": st.error},
    "violet": {"badge": "violet", "toast": "🏠", "alert": st.info},
}
ROLE_AVATAR = {"manager": "🧑‍🍳", "agent": "🤖", "tool": "🛠️",
               "driver": "🛵", "shelter": "🏠", "system": "⚡"}

SYSTEM_PROMPT = """\
You are RescueAgent, an AI that coordinates food rescue in Edinburgh.

When a manager describes surplus food, work in this order:
1. `analyze_food_safety` to classify the food and get the FSA handover window.
2. `find_eligible_shelter` with the safety flags and the restaurant coordinates.
3. `broadcast_rescue` to offer the job to the nearest capable drivers.

Then reply in at most four short lines: the food classification, the matched
shelter and why it qualifies, the drivers you offered the job to with their
distance, and any safety constraint the driver must respect.
Do not invent names or addresses — only use what the tools return.
"""

st.set_page_config(page_title="RescueAgent — Edinburgh", page_icon="🍽",
                   layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- data
@st.cache_data
def load(name):
    path = os.path.join(_DATA_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if name == "restaurants.json":
        clean = []
        for r in rows:
            nm = (r.get("name") or "").strip().strip('"')
            if not nm or r.get("lat") is None or r.get("lng") is None:
                continue
            cu = r.get("cuisine")
            if isinstance(cu, list):
                cu = cu[0] if cu else ""
            clean.append({**r, "name": nm, "cuisine": cu or "",
                          "address": r.get("address") or "Edinburgh"})
        return clean
    return rows


@st.cache_data
def restaurant_index():
    """id -> record, plus the one-line label the picker searches against.

    The label carries name, address, cuisine and venue type because
    st.selectbox filters on the rendered label — putting them all in there is
    what makes "indian", "bakery" or a street name find the right kitchen.
    """
    by_id, labels = {}, {}
    for r in RESTAURANTS:
        kind = []
        if r.get("cuisine"):
            kind.append(str(r["cuisine"]).replace("_", " "))
        t = (r.get("amenity_type") or "").replace("_", " ")
        if t and t not in kind:
            kind.append(t)
        addr = (r.get("address") or "").strip() or "Edinburgh"
        by_id[r["id"]] = r
        labels[r["id"]] = f'{r["name"]} — {addr}' + (f' · {" · ".join(kind)}' if kind else "")

    # A few dozen venues (mostly chain branches OSM never gave an address) would
    # otherwise render as identical rows; pin those to their coordinates so every
    # option in the picker is distinguishable.
    counts = collections.Counter(labels.values())
    for rid, lab in labels.items():
        if counts[lab] > 1:
            r = by_id[rid]
            labels[rid] = f'{lab} · {r["lat"]:.4f}, {r["lng"]:.4f}'
    return by_id, labels


def drivers_live():
    """Driver roster, re-read each run so status changes are visible."""
    path = os.path.join(_DATA_DIR, "drivers.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _wire_aws_credentials():
    """Streamlit Community Cloud has no ~/.aws — read them from st.secrets instead.

    Add this to the app's Secrets box (Settings -> Secrets):
        AWS_ACCESS_KEY_ID = "AKIA..."
        AWS_SECRET_ACCESS_KEY = "..."
        AWS_DEFAULT_REGION = "eu-west-2"
    Locally, `aws configure` already works and this is a no-op.
    """
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
        try:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass


@st.cache_resource
def get_agent():
    _wire_aws_credentials()
    return Agent(
        system_prompt=SYSTEM_PROMPT,
        tools=[analyze_food_safety, find_eligible_shelter, broadcast_rescue,
               accept_rescue, route_lookup, release_driver, dispatch_driver],
        model=BedrockModel(model_id=BEDROCK_MODEL_ID, region_name=BEDROCK_REGION),
    )


# ---------------------------------------------------------------- session
def init_state():
    defaults = {
        "delivery": None,          # active delivery dict
        "steps": [],               # agent narration
        "notifs": [],              # alert feed
        "toast_cursor": 0,         # notifications already toasted
        "history": [],
        "stats": {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0},
        "restaurant": None,
        "food_text": "",
        "use_bedrock": True,
        "agent_reply": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def say(role, text, meta="", accent="brand"):
    """Add one line of plain-English narration to the agent trace.

    `text` is what a person would say happened; `meta` is the supporting
    machine detail (tool name, coordinates, thresholds) shown underneath.
    """
    st.session_state.steps.append({"role": role, "text": text, "meta": meta,
                                   "accent": accent, "t": now_str()})


def notify(title, body, role, accent="brand"):
    st.session_state.notifs.insert(0, {"title": title, "body": body, "role": role,
                                       "accent": accent, "time": now_str()})


def flush_toasts():
    """Toast anything added since the last run."""
    pending = st.session_state.notifs[:max(0, len(st.session_state.notifs) - st.session_state.toast_cursor)]
    for n in reversed(pending):
        st.toast(f"**{n['title']}**  \n{n['body']}",
                 icon=ACCENTS.get(n["accent"], ACCENTS["brand"])["toast"])
    st.session_state.toast_cursor = len(st.session_state.notifs)


def initials(name):
    return "".join(part[0] for part in name.split()[:2]).upper()


# ---------------------------------------------------------------- flow
def launch_rescue():
    r = st.session_state.restaurant
    text = st.session_state.food_text.strip()
    if not r:
        st.warning("Pick the pickup kitchen first.")
        return
    if not text:
        st.warning("Describe the surplus food first.")
        return

    st.session_state.steps = []
    st.session_state.agent_reply = ""
    say("manager", f'The kitchen at **{r["name"]}** reported surplus: “{text}”',
        f'{r["address"]} · {r["lat"]:.5f}, {r["lng"]:.5f}', "brand")

    safety = json.loads(analyze_food_safety(text))
    kind = "hot cooked food" if safety["needs_hot"] else "chilled or ambient food"
    carry = ("It has to travel in a thermal bag."
             if safety["needs_hot"] else "A cool box is preferred but not required.")
    say("agent",
        f'I read that as **{kind}**, roughly **{safety["weight_kg"]} kg**, '
        f'category *{safety["category"]}*. Food standards give me '
        f'**{safety["fsa_window_minutes"]} minutes** from handover to serving. {carry}',
        f'analyze_food_safety · needs_hot={safety["needs_hot"]} '
        f'needs_meat={safety["needs_meat"]} window={safety["fsa_window_minutes"]}min', "amber")

    shelter = json.loads(find_eligible_shelter(
        safety["needs_hot"], safety["needs_meat"], safety["needs_drinks"],
        r["lat"], r["lng"]))
    if shelter.get("error"):
        say("agent", f'I could not place this load. {shelter["error"]}',
            "find_eligible_shelter returned no match", "red")
        notify("Match failed", shelter["error"], "system", "red")
        return

    free = shelter["capacity_meals"] - shelter["current_intake_meals"]
    km_to_shelter = haversine_km((r["lat"], r["lng"]), (shelter["lat"], shelter["lng"]))
    say("shelter",
        f'**{shelter["name"]}** is the right home for it — they accept this food type, '
        f'they are running at **{shelter["demand_score"]}/5 demand**, and they still have '
        f'**{free} meals** of headroom. They are {km_to_shelter:.1f} km from the kitchen.',
        f'find_eligible_shelter · {shelter["id"]} · {shelter["address"]}', "violet")

    bc = json.loads(broadcast_rescue(shelter["id"], safety["needs_hot"], safety["needs_meat"],
                                     safety["weight_kg"], r["lat"], r["lng"], 3))
    if bc.get("error"):
        say("agent", f'Nobody on shift can take this right now. {bc["error"]}',
            "broadcast_rescue found no capable driver", "red")
        notify("Dispatch failed", bc["error"], "system", "red")
        return

    offers = [{**o, "status": "pending"} for o in bc["offered"]]
    roster = ", ".join(f'{o["name"]} ({o["distance_km"]:.1f} km, '
                       f'{(VEH.get(o["vehicle_type"]) or "vehicle").lower()})' for o in offers)
    say("agent",
        f'I offered the job to the **{len(offers)} nearest capable drivers** — {roster}. '
        f'Whoever accepts first takes it; the rest are released automatically.',
        f'broadcast_rescue · filtered on capacity ≥ {safety["weight_kg"]} kg'
        + (", thermal bag required" if safety["needs_hot"] else ""), "green")

    st.session_state.delivery = {
        "restaurant": r, "shelter": shelter, "safety": safety, "food_text": text,
        "offers": offers, "winner": None, "phase": "broadcast",
        "t0": time.time(), "legs": [], "leg": 0,
    }
    notify("New rescue offer",
           f'{safety["weight_kg"]} kg from {r["name"]} → {shelter["name"]}. '
           f'Offered to {len(offers)} drivers.', "driver", "brand")

    if st.session_state.use_bedrock:
        try:
            with st.spinner("Agent reasoning over the request…"):
                reply = str(get_agent()(
                    f'Surplus food at {r["name"]} ({r["address"]}), '
                    f'coordinates {r["lat"]}, {r["lng"]}. The manager says: "{text}"'))
            st.session_state.agent_reply = reply
            say("agent", reply.strip()[:900], "", "brand")
        except Exception as e:
            say("system",
                "Bedrock is unreachable, so I am running the deterministic tool "
                "pipeline instead — the rescue itself is unaffected.",
                str(e)[:200], "red")


def do_accept(driver_id):
    d = st.session_state.delivery
    if not d or d["phase"] != "broadcast":
        return
    won = next((o for o in d["offers"] if o["driver_id"] == driver_id), None)
    if not won:
        return
    res = json.loads(accept_rescue(driver_id))
    if res.get("error"):
        notify("Offer already taken", res["error"], "driver", "red")
        return

    for o in d["offers"]:
        o["status"] = "won" if o["driver_id"] == driver_id else "lost"
    d["winner"] = won

    r, s = d["restaurant"], d["shelter"]
    leg1 = get_route((won["current_lat"], won["current_lng"]), (r["lat"], r["lng"]), won["vehicle_type"])
    leg2 = get_route((r["lat"], r["lng"]), (s["lat"], s["lng"]), won["vehicle_type"])
    d["legs"] = [leg1, leg2]

    bag = " and is carrying a thermal bag" if won.get("has_thermal_bag") else ""
    ride = (VEH.get(won["vehicle_type"]) or "vehicle").lower()
    say("driver",
        f'**{won["name"]}** accepted the job first — {won["distance_km"]:.1f} km away in '
        f'{won["neighbourhood"]}, on a {ride}{bag}. '
        f'The other {len(d["offers"]) - 1} offers were withdrawn.',
        f'accept_rescue · {won["driver_id"]} · rated {won["rating"]}★ · '
        f'capacity {won["max_capacity_kg"]} kg', "green")
    say("agent",
        f'Route locked: **{leg1["km"]:.1f} km** to {r["name"]} (about {leg1["minutes"]} min), '
        f'then **{leg2["km"]:.1f} km** on to {s["name"]} (about {leg2["minutes"]} min). '
        f'Tracking is live below.',
        f'route_lookup · geometry from {leg1["source"]}', "brand")
    notify("Driver assigned",
           f'{won["name"]} accepted and is heading to {r["name"]}.', "manager", "green")

    d["phase"] = "to_pickup"
    d["leg"] = 0
    d["t0"] = time.time()


def do_arrive():
    d = st.session_state.delivery
    d["phase"] = "at_pickup"
    d["t0"] = time.time()
    w = d["winner"]
    say("driver",
        f'**{w["name"]} has arrived at {d["restaurant"]["name"]}** and is waiting at the '
        f'door for the handover.',
        f'leg 1 complete · {d["legs"][0]["km"]:.1f} km driven', "green")
    notify(f'{w["name"]} has arrived',
           f'Waiting at {d["restaurant"]["name"]} — hand over the food and confirm.',
           "manager", "brand")


def do_handover():
    d = st.session_state.delivery
    if not d or d["phase"] != "at_pickup":
        return
    d["phase"] = "to_shelter"
    d["leg"] = 1
    d["t0"] = time.time()
    w, sf, s = d["winner"], d["safety"], d["shelter"]
    where = ("into a thermal bag" if w.get("has_thermal_bag")
             else ("into a cool box" if w.get("has_cool_box") else "into the carrier"))
    say("driver",
        f'Handover confirmed — **{w["name"]} loaded {sf["weight_kg"]} kg** {where} '
        f'and is now driving to **{s["name"]}**, {d["legs"][1]["km"]:.1f} km away. '
        f'The food has to be served within {sf["fsa_window_minutes"]} minutes.',
        f'FSA clock started at {now_str()} · {sf["category"]}', "amber")
    notify("Food collected",
           f'{w["name"]} is en route to {s["name"]}.', "driver", "green")


def do_deliver():
    d = st.session_state.delivery
    w, s, sf = d["winner"], d["shelter"], d["safety"]
    meals = max(2, int(round(sf["weight_kg"] * 2.4)))
    co2 = round(sf["weight_kg"] * 2.5, 1)
    total_km = d["legs"][0]["km"] + d["legs"][1]["km"]

    release_driver(w["driver_id"])
    st.session_state.stats["rescues"] += 1
    st.session_state.stats["meals"] += meals
    st.session_state.stats["kg"] = round(st.session_state.stats["kg"] + sf["weight_kg"], 1)
    st.session_state.stats["co2"] = round(st.session_state.stats["co2"] + co2, 1)
    st.session_state.history.insert(0, {
        "id": f'rescue_{len(st.session_state.history) + 1:04d}', "time": now_str(),
        "food": d["food_text"], "route": f'{d["restaurant"]["name"]} → {s["name"]}',
        "summary": f'{w["name"]} · {VEH.get(w["vehicle_type"], "")} · '
                   f'{sf["weight_kg"]} kg · {total_km:.1f} km total',
        "meals": meals, "co2": f"{co2} kg", "shelter": s["name"],
    })
    say("shelter",
        f'**Delivered.** {s["name"]} signed for {sf["weight_kg"]} kg — about **{meals} meals** '
        f'— and {w["name"]} is back on the available roster. That is **{co2} kg of CO₂e** '
        f'kept out of the air, over {total_km:.1f} km of driving.',
        f'release_driver · {w["driver_id"]} · rescue logged', "green")
    notify("Delivered",
           f'{s["name"]} signed for {meals} meals. {w["name"]} is back on shift.',
           "manager", "green")
    d["phase"] = "delivered"
    load.clear()


def advance():
    """Drive the state machine off the wall clock. Returns leg progress 0..1."""
    d = st.session_state.delivery
    if not d:
        return 0.0
    el = time.time() - d["t0"]
    ph = d["phase"]
    if ph == "broadcast":
        if el >= OFFER_SECONDS:
            do_accept(d["offers"][0]["driver_id"])
        return 0.0
    if ph == "to_pickup":
        p = min(1.0, el / leg_seconds(d["legs"][0]))
        if p >= 1.0:
            do_arrive()
            return 1.0
        return p
    if ph == "at_pickup":
        if el >= PICKUP_SECONDS:
            do_handover()
        return 0.0
    if ph == "to_shelter":
        p = min(1.0, el / leg_seconds(d["legs"][1]))
        if p >= 1.0:
            do_deliver()
            return 1.0
        return p
    return 1.0


def reset_demo():
    d = st.session_state.delivery
    if d and d.get("winner") and d["phase"] != "delivered":
        release_driver(d["winner"]["driver_id"])
    st.session_state.delivery = None
    st.session_state.steps = []
    load.clear()


# ---------------------------------------------------------------- map plan
def map_plan():
    """Everything the tracking map needs to animate the journey unaided."""
    d = st.session_state.delivery
    fleet = [[dr["current_lat"], dr["current_lng"], dr.get("status") == "available"]
             for dr in DRV if dr.get("current_lat") is not None]

    base = {"brand": BRAND, "fleet": fleet, "route": [], "done_route": [],
            "pickup": None, "shelter": None, "hold": None, "t0": 0, "dur": 0,
            "eta_min": 0, "legend": [["Driver on shift", "#34D399"],
                                     ["Busy or off duty", "#7D8794"]]}

    if not d:
        return {**base, "phase": "idle", "label": "Fleet standby",
                "value": str(len(AVAIL)),
                "sub": f"{len(DRV)} volunteer drivers across Edinburgh"}

    r, s, ph = d["restaurant"], d["shelter"], d["phase"]
    pins = {"pickup": {"lat": r["lat"], "lng": r["lng"], "name": r["name"]},
            "shelter": {"lat": s["lat"], "lng": s["lng"], "name": s["name"]}}
    legend = [["Pickup kitchen", BRAND], ["Shelter", "#A78BFA"]]

    if ph == "broadcast":
        return {**base, **pins, "phase": ph, "legend": legend,
                "label": "Dispatching", "value": str(len(d["offers"])),
                "sub": "offers open — waiting for a driver to accept"}

    leg1, leg2 = d["legs"][0], d["legs"][1]
    l1 = [[a, b] for a, b in leg1["coords"]]
    l2 = [[a, b] for a, b in leg2["coords"]]

    if ph == "to_pickup":
        return {**base, **pins, "phase": ph, "legend": legend, "route": l1,
                "t0": d["t0"], "dur": leg_seconds(leg1), "eta_min": leg1["minutes"],
                "label": "ETA to pickup"}
    if ph == "at_pickup":
        return {**base, **pins, "phase": ph, "legend": legend, "route": l1, "hold": 1.0,
                "label": "At the kitchen", "value": "0<span>min</span>",
                "sub": f'{d["winner"]["name"]} is waiting for the handover'}
    if ph == "to_shelter":
        return {**base, **pins, "phase": ph, "legend": legend, "route": l2,
                "done_route": l1, "t0": d["t0"], "dur": leg_seconds(leg2),
                "eta_min": leg2["minutes"], "label": "ETA to shelter"}
    return {**base, **pins, "phase": ph, "legend": legend, "route": l2,
            "done_route": l1, "hold": 1.0, "label": "Delivered", "value": "✓",
            "sub": f'{s["name"]} signed for the load'}


# ---------------------------------------------------------------- shared UI
def render_stepper(stage_idx):
    cols = st.columns(len(STAGES))
    for i, (col, label) in enumerate(zip(cols, STAGES)):
        if i < stage_idx:
            col.markdown(f":green[:material/check_circle:] **{label}**")
        elif i == stage_idx:
            col.markdown(f":primary[:material/radio_button_checked:] **{label}**")
        else:
            col.markdown(f":gray[:material/radio_button_unchecked: {label}]")


def render_narration(height=380):
    """The agent talking through the rescue in plain English."""
    if not st.session_state.steps:
        st.info("Idle. Send a rescue from **New rescue** and the agent will narrate "
                "every decision here, step by step.")
        return
    with st.container(height=height, border=False):
        for s in st.session_state.steps:
            with st.chat_message(s["role"], avatar=ROLE_AVATAR.get(s["role"], "💬")):
                st.markdown(s["text"])
                if s["meta"]:
                    st.caption(f'{s["meta"]} · {s["t"]}')


# ---------------------------------------------------------------- pages
def render_dashboard():
    c1, c2 = st.columns([4, 1.1])
    with c1:
        st.header("Dashboard")
        st.caption("Session totals. Counters move only as the agent completes work.")
    with c2:
        if st.button("New rescue", type="primary", icon=":material/add:", width="stretch"):
            st.switch_page(page_new_rescue)

    s = st.session_state.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rescues completed", s["rescues"], border=True)
    m2.metric("Meals delivered", s["meals"], border=True)
    m3.metric("Food rescued", f'{s["kg"]:.1f} kg', border=True)
    m4.metric("CO₂ avoided", f'{s["co2"]:.1f} kg', border=True)

    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        active = 1 if st.session_state.delivery and \
            st.session_state.delivery["phase"] != "delivered" else 0
        with st.container(border=True):
            st.subheader("Network")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Restaurants", f"{len(RESTAURANTS):,}")
            n2.metric("Shelters", len(SHELTERS))
            n3.metric("On shift", f"{len(AVAIL)}/{len(DRV)}")
            n4.metric("In flight", active)
            st.divider()
            st.write(
                "A manager describes surplus food in plain English. The agent classifies it "
                "against FSA thermal rules, matches a shelter that accepts it, then broadcasts "
                "the job to the three nearest capable drivers. First to accept gets the route."
            )
    with right:
        with st.container(border=True):
            st.subheader("Recent this session")
            if st.session_state.history:
                for i, h in enumerate(st.session_state.history[:3]):
                    st.markdown(f"**{h['food'][:70]}**")
                    st.caption(f"{h['time']} · {h['shelter']} · {h['meals']} meals")
                    if i < min(2, len(st.session_state.history) - 1):
                        st.divider()
            else:
                st.info("Nothing rescued yet. Open **New rescue** and describe "
                        "what is left over.")


def render_new_rescue():
    st.header("New rescue")
    st.caption("Pick the pickup kitchen, describe the surplus. "
               "The agent handles safety, shelter and driver.")

    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.subheader("1 · Pickup kitchen")
        by_id, labels = restaurant_index()

        kinds = sorted({(r.get("amenity_type") or "").replace("_", " ")
                        for r in RESTAURANTS} - {""})
        picked_kinds = st.pills("Venue type", kinds, selection_mode="multi",
                                label_visibility="collapsed")
        pool = [r for r in RESTAURANTS
                if not picked_kinds
                or (r.get("amenity_type") or "").replace("_", " ") in picked_kinds]

        ids = [r["id"] for r in pool]
        current = st.session_state.restaurant["id"] if st.session_state.restaurant else None
        choice = st.selectbox(
            "Pickup kitchen", ids,
            index=ids.index(current) if current in ids else None,
            format_func=lambda i: labels[i], label_visibility="collapsed",
            placeholder=f"Click to browse all {len(pool):,} kitchens, or type a name, "
                        f"street or cuisine…",
        )
        if choice != current:
            st.session_state.restaurant = by_id.get(choice) if choice else None
            st.rerun()

        if st.session_state.restaurant:
            r = st.session_state.restaurant
            with st.container(border=True):
                st.markdown(f"**{r['name']}**")
                st.caption((r.get("address") or "").strip() or "Edinburgh")
                with st.container(horizontal=True):
                    if r.get("amenity_type"):
                        st.badge(r["amenity_type"].replace("_", " "),
                                 icon=":material/storefront:", color="gray")
                    if r.get("cuisine"):
                        st.badge(str(r["cuisine"]).replace("_", " "), color="gray")
        else:
            st.caption(f"{len(RESTAURANTS):,} kitchens across Edinburgh — restaurants, "
                       f"cafes, pubs, bakeries and food shops.")

        st.subheader("2 · Surplus food")
        st.session_state.food_text = st.text_area(
            "food", value=st.session_state.food_text, height=120,
            placeholder="e.g. 5 kg hot chicken biryani and 2 kg garlic naan, "
                        "needs collecting within the hour",
            label_visibility="collapsed")

        p1, p2, p3 = st.columns(3)
        if p1.button("Hot curry, 6 kg", width="stretch"):
            st.session_state.food_text = ("6 kg hot chicken curry and rice, cooked 40 minutes "
                                          "ago, needs collecting soon")
            st.rerun()
        if p2.button("Chilled sandwiches", width="stretch"):
            st.session_state.food_text = "3 kg chilled sandwiches and salad boxes from the counter"
            st.rerun()
        if p3.button("Bakery, end of day", width="stretch"):
            st.session_state.food_text = "12 loaves and 20 pastries, ambient, end of day"
            st.rerun()

        if st.button("Send to agent", type="primary", icon=":material/send:", width="stretch"):
            launch_rescue()
            dd = st.session_state.delivery
            if dd and dd["phase"] == "broadcast":
                st.switch_page(page_live_tracking)
        st.caption("Broadcasts to the 3 nearest capable drivers. First to accept takes the job.")

    with right:
        sf = json.loads(analyze_food_safety(st.session_state.food_text or ""))
        reqs = [f'Capacity ≥ {sf["weight_kg"]} kg']
        reqs += ["Thermal bag", "Accepts hot"] if sf["needs_hot"] else ["Accepts cold"]
        if sf["needs_meat"]:
            reqs.append("Accepts meat")
        reqs.append("Available now")

        with st.container(border=True):
            st.subheader("Pre-flight read")
            st.dataframe(
                {"Property": ["Temperature class", "Category", "Estimated weight",
                              "FSA handover window"],
                 "Value": ["Hot / cooked" if sf["needs_hot"] else "Cold / ambient",
                           sf["category"].title(), f'{sf["weight_kg"]} kg',
                           f'{sf["fsa_window_minutes"]} min']},
                hide_index=True, width="stretch")
            st.caption(sf["safety_note"])

        with st.container(border=True):
            st.subheader("Driver requirements")
            with st.container(horizontal=True):
                for x in reqs:
                    st.badge(x, color="gray")


def render_live_tracking():
    d = st.session_state.delivery
    head1, head2 = st.columns([4, 1.1])
    with head1:
        st.header("Live tracking")
        if d:
            st.caption(f'{d["restaurant"]["name"]} → {d["shelter"]["name"]}')
        else:
            st.caption(f"All {len(DRV)} drivers at their current positions. "
                       "Nothing dispatched yet.")
    with head2:
        if st.button("Reset demo", icon=":material/restart_alt:", width="stretch"):
            reset_demo()
            st.rerun()

    running = bool(d) and d["phase"] != "delivered"

    @st.fragment(run_every=1.0 if running else None)
    def tracking_fragment():
        advance()
        flush_toasts()
        dd = st.session_state.delivery
        ph = dd["phase"] if dd else "idle"

        # The map animates itself in the browser; this only re-sends the plan,
        # which changes on phase transitions rather than every tick.
        live_map(map_plan(), height=520)

        if dd:
            render_stepper(PHASE_STAGE[ph])
            st.divider()

        sheet_l, sheet_r = st.columns([1, 1], gap="medium")

        with sheet_l:
            if not dd:
                st.subheader("No active trip")
                st.info(f"{len(AVAIL)} drivers are on shift waiting for a job. "
                        "Start one from **New rescue**.")
            elif ph == "broadcast":
                st.subheader("Waiting for a driver")
                st.caption("Offer is out to the three nearest capable volunteers. "
                           "Tap Accept for any of them, or wait — the nearest takes "
                           "it automatically.")
                for o in dd["offers"]:
                    with st.container(border=True):
                        oc1, oc2 = st.columns([3, 1])
                        with oc1:
                            st.markdown(f'**{o["name"]}**')
                            st.caption(f'{o["distance_km"]:.1f} km away · '
                                       f'{VEH.get(o["vehicle_type"], "")} · '
                                       f'{o["max_capacity_kg"]} kg · {o["rating"]}★')
                        with oc2:
                            if o["status"] == "pending":
                                if st.button("Accept", key=f'acc_{o["driver_id"]}',
                                             type="primary", width="stretch"):
                                    do_accept(o["driver_id"])
                                    st.rerun(scope="fragment")
                            else:
                                st.badge("Accepted" if o["status"] == "won" else "Taken",
                                         color="green" if o["status"] == "won" else "gray")
            else:
                w = dd["winner"]
                headline = {
                    "to_pickup": f'{w["name"]} is on the way to you',
                    "at_pickup": f'{w["name"]} is outside',
                    "to_shelter": "Food is in transit",
                    "delivered": "Rescue complete",
                }[ph]
                st.subheader(headline)

                with st.container(border=True):
                    a1, a2 = st.columns([1, 3])
                    with a1:
                        st.title(initials(w["name"]))
                    with a2:
                        st.markdown(f'**{w["name"]}**  ·  {w["rating"]}★')
                        st.caption(f'{VEH.get(w["vehicle_type"], "")} '
                                   f'{w.get("vehicle_reg", "")} · {w["neighbourhood"]} · '
                                   f'carries {w["max_capacity_kg"]} kg')
                    with st.container(horizontal=True):
                        st.badge(VEH.get(w["vehicle_type"], ""),
                                 icon=VEH_ICON.get(w["vehicle_type"]), color="gray")
                        st.badge(f'{dd["safety"]["weight_kg"]} kg load', color="gray")
                        if w.get("has_thermal_bag"):
                            st.badge("Thermal bag", icon=":material/thermostat:", color="green")
                        st.badge(f'FSA {dd["safety"]["fsa_window_minutes"]} min',
                                 icon=":material/timer:", color="orange")

                instr = {
                    "to_pickup": ("Head to the pickup",
                                  f'{dd["restaurant"]["name"]} — {dd["restaurant"]["address"]}'),
                    "at_pickup": ("Collect the food",
                                  "Take the load from the kitchen team and confirm below."),
                    "to_shelter": ("Deliver the load",
                                   f'{dd["shelter"]["name"]} — {dd["shelter"]["address"]}'),
                    "delivered": ("Job done", "You are back on the available roster."),
                }[ph]
                st.markdown(f"**{instr[0]}**")
                st.caption(instr[1])

                if ph == "at_pickup":
                    b1, b2 = st.columns(2)
                    if b1.button("Food handed over", type="primary",
                                 icon=":material/check:", width="stretch", key="handover"):
                        do_handover()
                        st.rerun(scope="fragment")
                    if b2.button("Confirm collected", icon=":material/inventory_2:",
                                 width="stretch", key="drv_collect"):
                        do_handover()
                        st.rerun(scope="fragment")
                elif ph == "to_pickup":
                    if st.button("Report arrival early", icon=":material/flag:",
                                 width="stretch", key="drv_arrive"):
                        do_arrive()
                        st.rerun(scope="fragment")
                elif ph == "delivered" and st.session_state.history:
                    h = st.session_state.history[0]
                    st.success(f'**{h["meals"]} meals logged** — {h["summary"]} · '
                               f'{h["co2"]} CO₂e avoided')

        with sheet_r:
            st.subheader("What the agent is doing")
            render_narration(height=420)

    tracking_fragment()


def render_active_deliveries():
    st.header("Active deliveries")
    d = st.session_state.delivery
    if d and d["phase"] in ("to_pickup", "at_pickup", "to_shelter"):
        w = d["winner"]
        with st.container(border=True):
            st.subheader(d["food_text"])
            st.caption(f'{d["restaurant"]["name"]} → {d["shelter"]["name"]}')
            with st.container(horizontal=True):
                st.badge(w["name"], icon=":material/person:", color="gray")
                st.badge(VEH.get(w["vehicle_type"], ""),
                         icon=VEH_ICON.get(w["vehicle_type"]), color="gray")
                st.badge(f'{d["safety"]["weight_kg"]} kg', color="gray")
                st.badge(d["phase"].replace("_", " "), color="primary")
            if st.button("Track", type="primary", icon=":material/near_me:"):
                st.switch_page(page_live_tracking)
    else:
        st.info("Nothing in flight. A rescue appears here from dispatch until it is delivered.")


def render_shelters():
    st.header("Shelters & larders")
    st.caption(f"{len(SHELTERS)} registered locations across Edinburgh")
    f1, f2 = st.columns(2)
    hot_only = f1.toggle("Accepts hot food")
    veg_only = f2.toggle("Vegetarian options")

    rows = [s for s in SHELTERS
            if (not hot_only or s.get("accepts_hot_food"))
            and (not veg_only or any("veg" in t for t in s.get("dietary_tags", [])))]
    cols = st.columns(3)
    for i, s in enumerate(rows):
        demand = s.get("demand_score", 0)
        color = "red" if demand >= 4.5 else ("orange" if demand >= 4 else "green")
        cap, cur = s.get("capacity_meals", 0), s.get("current_intake_meals", 0)
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{s['name']}**")
                st.caption(s.get("address", ""))
                st.badge(f"demand {demand}/5", color=color)
                if s.get("dietary_tags"):
                    with st.container(horizontal=True):
                        for t in s["dietary_tags"]:
                            st.badge(t.replace("_", " "), color="gray")
                st.progress(min(1.0, (cur / cap) if cap else 0),
                            text=f'{cur}/{cap} meals · '
                                 f'{s.get("opens_24h", "")}–{s.get("closes_24h", "")}')


def render_drivers():
    st.header("Volunteer drivers")
    choice = st.segmented_control("filter", ["All", "Available", "Busy", "Off duty"],
                                  default="All", label_visibility="collapsed")
    key = {"Available": "available", "Busy": "busy", "Off duty": "off_duty"}.get(choice)
    rows = [d for d in DRV if key is None or d.get("status") == key]
    st.caption(f"{len(rows)} of {len(DRV)} drivers")
    cols = st.columns(3)
    for i, d in enumerate(rows):
        status = d.get("status", "")
        badge_color = "green" if status == "available" else (
            "primary" if status == "busy" else "gray")
        kit = ("Thermal bag" if d.get("has_thermal_bag")
               else ("Cool box" if d.get("has_cool_box") else "No thermal kit"))
        with cols[i % 3]:
            with st.container(border=True):
                top1, top2 = st.columns([3, 1])
                with top1:
                    st.markdown(f"**{d['name']}**")
                    st.caption(f'{d.get("neighbourhood", "")} · {d.get("rating", "")}★')
                with top2:
                    st.badge(status.replace("_", " "), color=badge_color)
                with st.container(horizontal=True):
                    st.badge(VEH.get(d.get("vehicle_type"), ""),
                             icon=VEH_ICON.get(d.get("vehicle_type")), color="gray")
                    st.badge(f'Max {d.get("max_capacity_kg", "?")} kg', color="gray")
                    st.badge(kit, color="gray")


def render_history():
    h1, h2 = st.columns([4, 1.1])
    with h1:
        st.header("Session history")
        st.caption("Rescues completed in this session")
    with h2:
        if st.button("Clear history", icon=":material/delete:", width="stretch"):
            st.session_state.history = []
            st.session_state.stats = {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0}
            st.rerun()

    if not st.session_state.history:
        st.info("No rescues logged yet this session.")
        return

    st.dataframe(
        [{"Time": h["time"], "ID": h["id"], "Food": h["food"], "Route": h["route"],
          "Meals": h["meals"], "CO₂ avoided": h["co2"]} for h in st.session_state.history],
        hide_index=True, width="stretch")


def render_alerts():
    a1, a2 = st.columns([4, 1.1])
    with a1:
        st.header("Alerts")
        st.caption("Every status change the network broadcast, newest first")
    with a2:
        if st.button("Clear", icon=":material/delete:", width="stretch"):
            st.session_state.notifs = []
            st.session_state.toast_cursor = 0
            st.rerun()

    if not st.session_state.notifs:
        st.info("Nothing yet. Alerts appear the moment a driver is offered a job.")
        return

    for n in st.session_state.notifs:
        ACCENTS.get(n["accent"], ACCENTS["brand"])["alert"](
            f'**{n["title"]}** · {n["role"]}  \n{n["body"]}  \n:gray[{n["time"]}]')


# ---------------------------------------------------------------- header + nav
SHELTERS = load("shelters.json")
RESTAURANTS = load("restaurants.json")
DRV = drivers_live()
AVAIL = [d for d in DRV if d.get("status") == "available"]

st.title("🍽 RescueAgent")
st.caption("Edinburgh food rescue network")

with st.sidebar:
    st.subheader("Settings")
    st.session_state.use_bedrock = st.toggle(
        "AI reasoning", value=st.session_state.use_bedrock,
        help="Off = deterministic tool pipeline only, no network call. "
             "Useful if the venue wifi is bad.")
    st.caption("Each leg plays at about its real driving time, so a run lasts "
               "roughly 3–8 minutes depending on the distance. Use the buttons "
               "on Live tracking to move a stage on early.")

page_dashboard = st.Page(render_dashboard, title="Dashboard",
                         icon=":material/dashboard:", default=True)
page_new_rescue = st.Page(render_new_rescue, title="New rescue",
                          icon=":material/add_circle:")
page_live_tracking = st.Page(render_live_tracking, title="Live tracking",
                             icon=":material/near_me:")
page_active_deliveries = st.Page(render_active_deliveries, title="Active deliveries",
                                 icon=":material/local_shipping:")
page_shelters = st.Page(render_shelters, title="Shelters", icon=":material/storefront:")
page_drivers = st.Page(render_drivers, title="Drivers", icon=":material/group:")
page_history = st.Page(render_history, title="History", icon=":material/history:")
page_alerts = st.Page(render_alerts, title="Alerts", icon=":material/notifications:")

pg = st.navigation([
    page_dashboard, page_new_rescue, page_live_tracking, page_active_deliveries,
    page_shelters, page_drivers, page_history, page_alerts,
], position="top")
pg.run()

if pg.title != "Live tracking":
    flush_toasts()
