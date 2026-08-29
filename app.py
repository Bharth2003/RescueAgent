"""
RescueAgent — Edinburgh Food Rescue
===================================
Streamlit dashboard with real-time driver tracking, built entirely from
native Streamlit components (st.navigation, st.metric, st.badge,
st.chat_message, st.container) — no injected HTML/CSS for layout or chrome.
Theming (dark, brand orange) lives in .streamlit/config.toml.

Run:  streamlit run app.py
"""

import json
import os
import time
from datetime import datetime

import folium
import streamlit as st
from folium.features import DivIcon
from streamlit_folium import st_folium

from strands import Agent
from strands.models import BedrockModel

from routing import get_route, haversine_km, point_at
from tools import (accept_rescue, analyze_food_safety, broadcast_rescue,
                   dispatch_driver, estimate_weight_kg, find_eligible_shelter,
                   release_driver, route_lookup)

# ---------------------------------------------------------------- config
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BEDROCK_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"
BEDROCK_REGION = "eu-west-2"

# demo clock — full journey plays in ~1 min 30 s (slow enough to read on a projector)
LEG1_SECONDS = 30.0     # driver -> restaurant
LEG2_SECONDS = 40.0     # restaurant -> shelter
OFFER_SECONDS = 8.0     # auto-accept if nobody taps Accept
PICKUP_SECONDS = 9.0    # auto-handover if nobody taps the button

VEH = {"van": "VAN", "car": "CAR", "motorbike": "MOTO", "e-bike": "E-BIKE",
       "bicycle": "BIKE", "scooter": "SCOOTER"}
STAGES = ["Assigned", "En route to pickup", "At restaurant", "Delivering", "Delivered"]

# hex colors used only for the folium map (leaflet needs real colors)
MAP_BRAND, MAP_GREEN, MAP_VIOLET, MAP_INK = "#FF7A2F", "#34D399", "#A78BFA", "#0E1013"

# accent key -> (st.badge color, toast icon, st alert function)
ACCENTS = {
    "brand": {"badge": "primary", "toast": "🛵", "alert": st.info},
    "green": {"badge": "green", "toast": "✅", "alert": st.success},
    "amber": {"badge": "orange", "toast": "⏳", "alert": st.warning},
    "red": {"badge": "red", "toast": "⚠️", "alert": st.error},
    "violet": {"badge": "violet", "toast": "🏠", "alert": st.info},
}

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
        "steps": [],               # agent reasoning trace
        "notifs": [],              # alert feed
        "toast_cursor": 0,         # notifications already toasted
        "history": [],
        "stats": {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0},
        "restaurant": None,
        "food_text": "",
        "route_source": "standby",
        "use_bedrock": True,
        "agent_reply": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def step(kind, title, body="", accent="brand"):
    st.session_state.steps.append({"kind": kind, "title": title, "body": body,
                                   "accent": accent, "t": now_str()})


def notify(title, body, role, accent="brand"):
    st.session_state.notifs.insert(0, {"title": title, "body": body, "role": role,
                                       "accent": accent, "time": now_str()})


def flush_toasts():
    """Toast anything added since the last run."""
    pending = st.session_state.notifs[:max(0, len(st.session_state.notifs) - st.session_state.toast_cursor)]
    for n in reversed(pending):
        st.toast(f"**{n['title']}**  \n{n['body']}", icon=ACCENTS.get(n["accent"], ACCENTS["brand"])["toast"])
    st.session_state.toast_cursor = len(st.session_state.notifs)


# ---------------------------------------------------------------- flow (business logic — unchanged)
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
    step("user → agent", "Rescue request received", f'"{text}"\nfrom {r["name"]}', "brand")

    safety = json.loads(analyze_food_safety(text))
    step("tool · analyze_food_safety", "Food classified",
         f'temperature: {"hot" if safety["needs_hot"] else "cold"}\n'
         f'category: {safety["category"]}\n'
         f'weight_kg: {safety["weight_kg"]}\n'
         f'fsa_window_min: {safety["fsa_window_minutes"]}', "amber")

    shelter = json.loads(find_eligible_shelter(
        safety["needs_hot"], safety["needs_meat"], safety["needs_drinks"],
        r["lat"], r["lng"]))
    if shelter.get("error"):
        step("tool · find_eligible_shelter", "No eligible shelter", shelter["error"], "red")
        notify("Match failed", shelter["error"], "system", "red")
        return
    free = shelter["capacity_meals"] - shelter["current_intake_meals"]
    step("tool · find_eligible_shelter", "Shelter matched",
         f'{shelter["id"]} — {shelter["name"]}\n'
         f'accepts_hot: {shelter["accepts_hot_food"]} · accepts_meat: {shelter["accepts_meat"]}\n'
         f'demand {shelter["demand_score"]}/5 · {free} meals of headroom', "violet")

    bc = json.loads(broadcast_rescue(shelter["id"], safety["needs_hot"], safety["needs_meat"],
                                     safety["weight_kg"], r["lat"], r["lng"], 3))
    if bc.get("error"):
        step("tool · broadcast_rescue", "No capable driver on shift", bc["error"], "red")
        notify("Dispatch failed", bc["error"], "system", "red")
        return

    offers = [{**o, "status": "pending"} for o in bc["offered"]]
    step("tool · broadcast_rescue", f'Broadcast to {len(offers)} nearest capable drivers',
         "\n".join(f'{o["name"]:<18}{VEH.get(o["vehicle_type"], ""):<8}'
                   f'{o["distance_km"]:.1f} km · {o["max_capacity_kg"]} kg' for o in offers), "green")

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
            step("agent → user", "Summary", reply.strip()[:900], "brand")
        except Exception as e:
            step("agent → user", "Bedrock unavailable",
                 f"{e}\nfalling back to the deterministic tool pipeline", "red")


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

    step("driver → agent", "Offer accepted",
         f'{won["name"]} accepted\nother {len(d["offers"]) - 1} offers withdrawn', "green")
    notify("Driver assigned",
           f'{won["name"]} accepted and is heading to {d["restaurant"]["name"]}.',
           "manager", "green")

    r, s = d["restaurant"], d["shelter"]
    leg1 = get_route((won["current_lat"], won["current_lng"]), (r["lat"], r["lng"]), won["vehicle_type"])
    leg2 = get_route((r["lat"], r["lng"]), (s["lat"], s["lng"]), won["vehicle_type"])
    d["legs"] = [leg1, leg2]
    st.session_state.route_source = leg1["source"]
    step("tool · route_lookup", "Road route resolved",
         f'leg 1 pickup: {leg1["km"]:.1f} km · ~{leg1["minutes"]} min\n'
         f'leg 2 delivery: {leg2["km"]:.1f} km · ~{leg2["minutes"]} min\n'
         f'source: {leg1["source"]}', "brand")

    d["phase"] = "to_pickup"
    d["leg"] = 0
    d["t0"] = time.time()


def do_arrive():
    d = st.session_state.delivery
    d["phase"] = "at_pickup"
    d["t0"] = time.time()
    step("event · driver_arrived", "Driver at pickup point",
         f'{d["winner"]["name"]} arrived at {d["restaurant"]["name"]}\n'
         f'awaiting handover confirmation', "green")
    notify(f'{d["winner"]["name"]} has arrived',
           f'Waiting at {d["restaurant"]["name"]} — hand over the food and confirm.',
           "manager", "brand")


def do_handover():
    d = st.session_state.delivery
    if not d or d["phase"] != "at_pickup":
        return
    d["phase"] = "to_shelter"
    d["leg"] = 1
    d["t0"] = time.time()
    bag = " into thermal bag" if d["winner"].get("has_thermal_bag") else ""
    step("event · food_collected", "Handover confirmed",
         f'{d["safety"]["weight_kg"]} kg loaded{bag}\n'
         f'FSA window: {d["safety"]["fsa_window_minutes"]} min', "amber")
    notify("Food collected", f'{d["winner"]["name"]} is en route to {d["shelter"]["name"]}.',
           "driver", "green")


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
    step("event · delivered", "Delivery complete",
         f'{meals} meals logged at {s["name"]}\n{co2} kg CO₂e avoided', "green")
    notify("Delivered", f'{s["name"]} signed for {meals} meals. {w["name"]} is back on shift.',
           "manager", "green")
    d["phase"] = "delivered"
    load.clear()


def advance():
    """Move the state machine on according to the wall clock."""
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
        p = min(1.0, el / LEG1_SECONDS)
        if p >= 1.0:
            do_arrive()
            return 1.0
        return p
    if ph == "at_pickup":
        if el >= PICKUP_SECONDS:
            do_handover()
        return 0.0
    if ph == "to_shelter":
        p = min(1.0, el / LEG2_SECONDS)
        if p >= 1.0:
            do_deliver()
            return 1.0
        return p
    return 1.0


def reset_demo():
    d = st.session_state.delivery
    if d and d.get("winner") and d["phase"] not in ("delivered",):
        release_driver(d["winner"]["driver_id"])
    st.session_state.delivery = None
    st.session_state.steps = []
    st.session_state.route_source = "standby"
    load.clear()


# ---------------------------------------------------------------- map
def build_map(progress):
    d = st.session_state.delivery
    m = folium.Map(location=[55.9490, -3.1900], zoom_start=12,
                   tiles="OpenStreetMap", control_scale=True,
                   zoom_control=True, attr=None)

    winner_id = d["winner"]["driver_id"] if d and d.get("winner") else None
    dim = bool(d)
    for dr in drivers_live():
        if dr.get("current_lat") is None or dr["id"] == winner_id:
            continue
        on = dr.get("status") == "available"
        colour = MAP_GREEN if on else "#B9AE9C"
        folium.CircleMarker(
            [dr["current_lat"], dr["current_lng"]], radius=5 if on else 4,
            color=MAP_INK, weight=1.5, fill=True, fill_color=colour,
            fill_opacity=0.5 if dim else 1.0,
            tooltip=f'{dr["name"]} · {VEH.get(dr["vehicle_type"], "")} · {dr["status"].replace("_", " ")}',
        ).add_to(m)

    if not d:
        return m

    r, s = d["restaurant"], d["shelter"]
    if d["legs"]:
        folium.PolyLine(d["legs"][0]["coords"], color="#8A8072", weight=4,
                        opacity=0.7, dash_array="7,8").add_to(m)
        folium.PolyLine(d["legs"][1]["coords"], color=MAP_BRAND, weight=5, opacity=0.95).add_to(m)

    def square(latlng, colour, label, tip):
        folium.Marker(
            latlng, tooltip=tip,
            icon=DivIcon(icon_size=(34, 34), icon_anchor=(17, 17), html=(
                f'<div style="width:32px;height:32px;border-radius:7px;background:{colour};'
                f'border:2px solid {MAP_INK};box-shadow:0 3px 12px rgba(0,0,0,.6);display:grid;'
                f'place-items:center;font:700 10px/1 sans-serif;color:{MAP_INK}">{label}</div>')),
        ).add_to(m)

    square([r["lat"], r["lng"]], MAP_BRAND, "P", r["name"])
    square([s["lat"], s["lng"]], MAP_VIOLET, "S", s["name"])

    w = d.get("winner")
    if w:
        if d["phase"] == "to_pickup" and d["legs"]:
            pos = point_at(d["legs"][0]["coords"], progress)
        elif d["phase"] == "at_pickup":
            pos = (r["lat"], r["lng"])
        elif d["phase"] == "to_shelter" and d["legs"]:
            pos = point_at(d["legs"][1]["coords"], progress)
        elif d["phase"] == "delivered":
            pos = (s["lat"], s["lng"])
        else:
            pos = (w["current_lat"], w["current_lng"])
        folium.Marker(
            list(pos), tooltip=f'{w["name"]} · {VEH.get(w["vehicle_type"], "")}',
            icon=DivIcon(icon_size=(40, 40), icon_anchor=(20, 20), html=(
                f'<div style="width:30px;height:30px;border-radius:50%;background:{MAP_BRAND};'
                f'border:2.5px solid {MAP_INK};box-shadow:0 3px 12px rgba(0,0,0,.6);display:grid;'
                f'place-items:center;font:700 8px/1 sans-serif;color:{MAP_INK}">'
                f'{VEH.get(w["vehicle_type"], "DRV")}</div>')),
        ).add_to(m)

    pts = [[r["lat"], r["lng"]], [s["lat"], s["lng"]]]
    if w:
        pts.append([w["current_lat"], w["current_lng"]])
    m.fit_bounds(pts, padding=(40, 40))
    return m


# ---------------------------------------------------------------- native render helpers
def render_timeline(stage_idx):
    cols = st.columns(len(STAGES))
    for i, (col, label) in enumerate(zip(cols, STAGES)):
        mark = "✅" if i <= stage_idx else "⚪"
        col.caption(f"{mark} {label}")


_KIND_AVATAR = {"user → agent": "🧑", "agent → user": "🤖", "driver → agent": "🛵"}


def _avatar_for(kind):
    if kind in _KIND_AVATAR:
        return _KIND_AVATAR[kind]
    if kind.startswith("tool"):
        return "🛠️"
    if kind.startswith("event"):
        return "⚡"
    return "💬"


def render_brain():
    st.subheader("Agent reasoning")
    st.caption("strands · tool calls and returns")
    with st.container(height=460, border=True):
        if not st.session_state.steps:
            st.info(f"Idle. All {len(DRV)} drivers are plotted on the map with their current "
                    "positions. Send a rescue to start the trace.")
            return
        for s in st.session_state.steps:
            with st.chat_message(name=s["kind"], avatar=_avatar_for(s["kind"])):
                st.caption(f'{s["kind"]} · {s["t"]}')
                st.markdown(f"**{s['title']}**")
                if s["body"]:
                    st.code(s["body"], language=None)


# ---------------------------------------------------------------- pages
def render_dashboard():
    c1, c2 = st.columns([4, 1])
    with c1:
        st.header("Dashboard")
        st.caption("Session totals. Counters move only as the agent completes work.")
    with c2:
        if st.button("New rescue", type="primary", width="stretch"):
            st.switch_page(page_new_rescue)

    s = st.session_state.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rescues completed", s["rescues"], border=True)
    m2.metric("Meals delivered", s["meals"], border=True)
    m3.metric("Food rescued", f'{s["kg"]:.1f} kg', border=True)
    m4.metric("CO₂ avoided", f'{s["co2"]:.1f} kg', border=True)

    left, right = st.columns([1.6, 1])
    with left:
        active = 1 if st.session_state.delivery and st.session_state.delivery["phase"] != "delivered" else 0
        with st.container(border=True):
            st.caption("Network")
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Partner restaurants", f"{len(RESTAURANTS):,}")
            n2.metric("Shelters & larders", len(SHELTERS))
            n3.metric("Drivers on shift", f"{len(AVAIL)}/{len(DRV)}")
            n4.metric("In flight", active)
            st.divider()
            st.write(
                "A manager describes surplus food in plain English. The agent classifies it "
                "against FSA thermal rules, matches a shelter that accepts it, then broadcasts "
                "the job to the three nearest capable drivers. The first to accept gets the route."
            )
    with right:
        with st.container(border=True):
            st.caption("Recent this session")
            if st.session_state.history:
                for i, h in enumerate(st.session_state.history[:3]):
                    st.markdown(f"**{h['food'][:70]}**")
                    st.caption(f"{h['time']} · {h['shelter']}")
                    if i < 2:
                        st.divider()
            else:
                st.info("Nothing rescued yet. Open **New Rescue** and describe what is left over.")


def render_new_rescue():
    st.header("New rescue")
    st.caption("Pick the pickup kitchen, describe the surplus. The agent handles safety, shelter and driver.")

    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.markdown("**1 · Pickup kitchen**")
        if st.session_state.restaurant:
            r = st.session_state.restaurant
            cc1, cc2 = st.columns([4, 1])
            with cc1:
                st.markdown(f"**{r['name']}**")
                st.caption(r["address"])
            if cc2.button("Change", width="stretch"):
                st.session_state.restaurant = None
                st.rerun()
        else:
            q = st.text_input("search", placeholder="Start typing a restaurant name or street…",
                              label_visibility="collapsed")
            ql = q.strip().lower()
            if ql:
                starts, contains = [], []
                for r in RESTAURANTS:
                    n = r["name"].lower()
                    if n.startswith(ql):
                        starts.append(r)
                    elif ql in n or ql in r.get("address", "").lower():
                        contains.append(r)
                hits = starts + contains
                st.caption(f'{len(hits)} of {len(RESTAURANTS)} match "{q.strip()}"'
                           if hits else "No restaurant in the OpenStreetMap set matches that.")
                with st.container(height=300, border=True):
                    for r in hits[:60]:
                        cuisine = f' · {r["cuisine"].replace("_", " ")}' if r.get("cuisine") else ""
                        if st.button(f'{r["name"]}  —  {r.get("address", "")}{cuisine}',
                                     key=f'pick_{r["id"]}', width="stretch"):
                            st.session_state.restaurant = r
                            st.rerun()
            else:
                st.caption(f"{len(RESTAURANTS):,} restaurants in the OSM set. "
                           f"Type any part of a name — Awaafi, Dishoom, Gorgie.")

        st.markdown("**2 · Surplus food**")
        st.session_state.food_text = st.text_area(
            "food", value=st.session_state.food_text, height=112,
            placeholder="e.g. 5 kg hot chicken biryani and 2 kg garlic naan, needs collecting within the hour",
            label_visibility="collapsed")

        p1, p2, p3 = st.columns(3)
        if p1.button("Hot curry, 6 kg", width="stretch"):
            st.session_state.food_text = ("6 kg hot chicken curry and rice, cooked 40 minutes ago, "
                                          "needs collecting soon")
            st.rerun()
        if p2.button("Chilled sandwiches", width="stretch"):
            st.session_state.food_text = "3 kg chilled sandwiches and salad boxes from the counter"
            st.rerun()
        if p3.button("Bakery, end of day", width="stretch"):
            st.session_state.food_text = "12 loaves and 20 pastries, ambient, end of day"
            st.rerun()

        if st.button("Send to agent", type="primary", width="stretch"):
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
        reqs.append("Status: available")

        with st.container(border=True):
            st.caption("Pre-flight read")
            st.dataframe(
                {"Property": ["Temperature class", "Category", "Estimated weight", "FSA handover window"],
                 "Value": ["Hot / cooked" if sf["needs_hot"] else "Cold / ambient",
                           sf["category"].title(), f'{sf["weight_kg"]} kg',
                           f'{sf["fsa_window_minutes"]} min']},
                hide_index=True, width="stretch")
            st.caption(sf["safety_note"])

        with st.container(border=True):
            st.caption("Driver requirements implied")
            with st.container(horizontal=True):
                for x in reqs:
                    st.badge(x, color="gray")


def render_live_tracking():
    d = st.session_state.delivery
    head1, head2 = st.columns([4, 1.1])
    with head1:
        st.header("Live tracking")
        if d:
            st.caption(f'{d["restaurant"]["name"]} → {d["shelter"]["name"]} · {d["phase"].replace("_", " ")}')
        else:
            st.caption(f"All {len(DRV)} drivers shown at their current positions. Nothing dispatched yet.")
    with head2:
        if st.button("Reset demo", width="stretch"):
            reset_demo()
            st.rerun()

    @st.fragment(run_every=1.0 if (d and d["phase"] not in ("delivered",)) else None)
    def tracking_fragment():
        dd = st.session_state.delivery
        progress = advance()
        flush_toasts()

        col_brain, col_map = st.columns([1, 2.1], gap="small")
        with col_brain:
            render_brain()
        with col_map:
            if not dd:
                hud = ("Fleet standby", len(AVAIL), "drivers on shift",
                       f"{len(DRV)} plotted across Edinburgh", 0)
            elif dd["phase"] == "broadcast":
                hud = ("Dispatching", len(dd["offers"]), "offers open",
                       "Waiting for a driver to accept", 6)
            elif dd["phase"] == "at_pickup":
                hud = ("At the kitchen", 0, "min away",
                       f'{dd["winner"]["name"]} is waiting for handover', 50)
            elif dd["phase"] == "delivered":
                hud = ("Delivered", "✓", "", "Route complete", 100)
            else:
                leg = dd["legs"][dd["leg"]]
                remaining_min = leg["minutes"] * (1 - progress)
                remaining_km = leg["km"] * (1 - progress)
                pct = int((progress * 50) if dd["leg"] == 0 else (50 + progress * 50))
                hud = ("ETA to pickup" if dd["leg"] == 0 else "ETA to shelter",
                       "<1" if remaining_min < 1 else f"{int(round(remaining_min))}", "min",
                       f'{remaining_km:.1f} km remaining of {leg["km"]:.1f} km', pct)

            st.metric(hud[0], f"{hud[1]} {hud[2]}".strip(), border=True)
            st.progress(min(100, max(0, hud[4])) / 100, text=hud[3])
            st_folium(build_map(progress), height=420, width="stretch",
                      returned_objects=[], key="tracking_map")

        ph = dd["phase"] if dd else "idle"
        mgr_active = ph in ("at_pickup", "delivered", "to_pickup")
        drv_active = ph in ("broadcast", "to_shelter")
        m_col, d_col = st.columns(2, gap="small")

        with m_col:
            with st.container(border=True):
                top1, top2 = st.columns([3, 1])
                top1.markdown("**Restaurant manager**")
                with top2:
                    st.badge(ph.replace("_", " "), color="primary" if mgr_active else "gray")
                texts = {
                    "idle": ("No open rescue", "Send a request from New Rescue to begin."),
                    "broadcast": ("Looking for a driver",
                                  "Offer is out to the three nearest capable volunteers."),
                    "to_pickup": (f'{dd["winner"]["name"]} is on the way to you' if dd and dd.get("winner") else "Driver on the way",
                                  f'Destination after pickup: {dd["shelter"]["name"]}.' if dd else ""),
                    "at_pickup": (f'{dd["winner"]["name"]} is outside' if dd and dd.get("winner") else "Driver has arrived",
                                  "Hand the food over and confirm below to release the driver."),
                    "to_shelter": ("Food is in transit",
                                   f'Heading to {dd["shelter"]["name"]}.' if dd else ""),
                    "delivered": ("Rescue complete",
                                  f'{dd["shelter"]["name"]} has signed for the load.' if dd else ""),
                }[ph]
                stage_idx = {"idle": -1, "broadcast": 0, "to_pickup": 1, "at_pickup": 2,
                             "to_shelter": 3, "delivered": 4}[ph]
                st.markdown(f"**{texts[0]}**")
                st.caption(texts[1])
                render_timeline(stage_idx)
                if ph == "at_pickup":
                    if st.button("Food handed over — release driver", type="primary",
                                 width="stretch", key="handover"):
                        do_handover()
                        st.rerun(scope="fragment")
                if ph == "delivered" and st.session_state.history:
                    h = st.session_state.history[0]
                    st.success(f'**{h["meals"]} meals logged** — {h["summary"]} · {h["co2"]} CO₂e avoided')

        with d_col:
            with st.container(border=True):
                top1, top2 = st.columns([3, 1])
                top1.markdown("**Driver app**")
                with top2:
                    st.badge("offer open" if ph == "broadcast" else ph.replace("_", " "),
                            color="primary" if drv_active else "gray")
                if not dd:
                    st.info(f"No open offers. {len(AVAIL)} drivers are on shift waiting for a job.")
                elif ph == "broadcast":
                    for o in dd["offers"]:
                        oc1, oc2 = st.columns([4, 1])
                        with oc1:
                            st.markdown(f'**{o["name"]}**')
                            st.caption(f'{VEH.get(o["vehicle_type"], "")} · {o["distance_km"]:.1f} km away · '
                                       f'{o["max_capacity_kg"]} kg · {o["neighbourhood"]} · {o["rating"]}★')
                        with oc2:
                            if o["status"] == "pending":
                                if st.button("Accept", key=f'acc_{o["driver_id"]}', width="stretch"):
                                    do_accept(o["driver_id"])
                                    st.rerun(scope="fragment")
                            else:
                                st.badge("Accepted" if o["status"] == "won" else "Taken",
                                        color="green" if o["status"] == "won" else "gray")
                    st.caption("Tap Accept on any driver, or wait — the nearest one takes it automatically.")
                else:
                    w = dd["winner"]
                    instr = {"to_pickup": ("Head to the pickup",
                                           f'{dd["restaurant"]["name"]} — {dd["restaurant"]["address"]}'),
                             "at_pickup": ("You have arrived", "Collect the food from the kitchen team."),
                             "to_shelter": ("Deliver the load",
                                            f'{dd["shelter"]["name"]} — {dd["shelter"]["address"]}'),
                             "delivered": ("Job done", "You are back on the available roster.")}[ph]
                    bag = " · thermal bag" if w.get("has_thermal_bag") else ""
                    ic1, ic2 = st.columns([1, 4])
                    with ic1:
                        st.markdown(f"### {''.join(x[0] for x in w['name'].split()[:2])}")
                    with ic2:
                        st.markdown(f'**{w["name"]}**')
                        st.caption(f'{VEH.get(w["vehicle_type"], "")} · {w.get("vehicle_reg", "")} · '
                                   f'{w["neighbourhood"]} · {w["max_capacity_kg"]} kg{bag} · {w["rating"]}★ rating')
                    st.markdown(f"**{instr[0]}**")
                    st.caption(instr[1])
                    if ph == "at_pickup":
                        if st.button("Confirm food collected", width="stretch", key="drv_collect"):
                            do_handover()
                            st.rerun(scope="fragment")
                    elif ph == "to_pickup":
                        if st.button("Report arrival early", width="stretch", key="drv_arrive"):
                            do_arrive()
                            st.rerun(scope="fragment")

    tracking_fragment()


def render_active_deliveries():
    st.header("Active deliveries")
    d = st.session_state.delivery
    if d and d["phase"] in ("to_pickup", "at_pickup", "to_shelter"):
        w = d["winner"]
        with st.container(border=True):
            st.markdown(f"**{d['food_text']}**")
            st.caption(f'{d["restaurant"]["name"]} → {d["shelter"]["name"]}')
            with st.container(horizontal=True):
                st.badge(w["name"], color="gray")
                st.badge(VEH.get(w["vehicle_type"], ""), color="gray")
                st.badge(f'{d["safety"]["weight_kg"]} kg', color="gray")
                st.badge(d["phase"].replace("_", " "), color="primary")
            if st.button("Track", type="primary"):
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
        pct = (cur / cap) if cap else 0
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{s['name']}**")
                st.caption(s.get("address", ""))
                st.badge(f"demand {demand}/5", color=color)
                if s.get("dietary_tags"):
                    with st.container(horizontal=True):
                        for t in s["dietary_tags"]:
                            st.badge(t.replace("_", " "), color="gray")
                st.progress(min(1.0, pct),
                            text=f'{cur}/{cap} meals · {s.get("opens_24h", "")}–{s.get("closes_24h", "")}')


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
        badge_color = "green" if status == "available" else ("primary" if status == "busy" else "gray")
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
                    st.badge(VEH.get(d.get("vehicle_type"), ""), color="gray")
                    st.badge(f'Max {d.get("max_capacity_kg", "?")} kg', color="gray")
                    st.badge(kit, color="gray")
                    st.badge(f'ETA {d.get("eta_minutes", "?")} min', color="gray")


def render_history():
    h1, h2 = st.columns([4, 1])
    with h1:
        st.header("Session history")
        st.caption("Rescues completed in this session")
    with h2:
        if st.button("Clear history", width="stretch"):
            st.session_state.history = []
            st.session_state.stats = {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0}
            st.rerun()

    if not st.session_state.history:
        st.info("No rescues logged yet this session.")
        return

    st.dataframe(
        [{"Time": h["time"], "ID": h["id"], "Food": h["food"], "Route": h["route"],
          "Meals": h["meals"], "CO₂ avoided": h["co2"]} for h in st.session_state.history],
        hide_index=True, width="stretch",
    )


def render_alerts():
    a1, a2 = st.columns([4, 1])
    with a1:
        st.header("Alerts")
        st.caption("Every status change the network broadcast, newest first")
    with a2:
        if st.button("Clear", width="stretch"):
            st.session_state.notifs = []
            st.session_state.toast_cursor = 0
            st.rerun()

    if not st.session_state.notifs:
        st.info("Nothing yet. Alerts appear the moment a driver is offered a job.")
        return

    for n in st.session_state.notifs:
        alert_fn = ACCENTS.get(n["accent"], ACCENTS["brand"])["alert"]
        alert_fn(f'**{n["title"]}** · {n["role"]}  \n{n["body"]}  \n:gray[{n["time"]}]')


# ---------------------------------------------------------------- header + navigation
SHELTERS = load("shelters.json")
RESTAURANTS = load("restaurants.json")
DRV = drivers_live()
AVAIL = [d for d in DRV if d.get("status") == "available"]

st.title("🍽 RescueAgent")
st.caption("Edinburgh Food Rescue Network")
with st.container(horizontal=True):
    st.badge(f"qwen3-235b · {BEDROCK_REGION}", color="gray")
    st.badge(f"routing: {st.session_state.route_source}", color="gray")
    st.badge(f"alerts: {len(st.session_state.notifs)}", color="gray")

with st.sidebar:
    st.subheader("Demo controls")
    st.session_state.use_bedrock = st.toggle(
        "Call Bedrock agent", value=st.session_state.use_bedrock,
        help="Off = deterministic tool pipeline only, no network call. Useful if the venue wifi is bad.")
    st.caption(f"Leg 1 {LEG1_SECONDS:.0f}s · Leg 2 {LEG2_SECONDS:.0f}s · "
               f"auto-accept {OFFER_SECONDS:.0f}s")

page_dashboard = st.Page(render_dashboard, title="Dashboard", icon=":material/home:", default=True)
page_new_rescue = st.Page(render_new_rescue, title="New Rescue", icon=":material/add_circle:")
page_live_tracking = st.Page(render_live_tracking, title="Live Tracking", icon=":material/near_me:")
page_active_deliveries = st.Page(render_active_deliveries, title="Active Deliveries",
                                 icon=":material/local_shipping:")
page_shelters = st.Page(render_shelters, title="Shelters", icon=":material/storefront:")
page_drivers = st.Page(render_drivers, title="Drivers", icon=":material/pedal_bike:")
page_history = st.Page(render_history, title="History", icon=":material/history:")
page_alerts = st.Page(render_alerts, title="Alerts", icon=":material/notifications:")

pg = st.navigation([
    page_dashboard, page_new_rescue, page_live_tracking, page_active_deliveries,
    page_shelters, page_drivers, page_history, page_alerts,
], position="top")
pg.run()

if pg.title != "Live Tracking":
    flush_toasts()
