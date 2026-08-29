"""
RescueAgent — Edinburgh Food Rescue
===================================
Streamlit dashboard with real-time driver tracking.

What changed vs v2.0
--------------------
* Warm light theme, Bricolage Grotesque + Instrument Sans, denser layout.
* Live tracking is a real map: OSRM road geometry, a driver marker that moves
  along the route, ETA countdown, distance remaining and a status timeline.
* Rescues are broadcast to the 3 nearest CAPABLE drivers; first to accept wins.
* Manager and driver both act, side by side under a shared map.
* st.toast notification on every status change, including driver arrival.
* Bedrock / Strands wiring unchanged.

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

BRAND, GREEN, AMBER, RED, VIOLET, MUTED = "#FF7A2F", "#34D399", "#FBBF24", "#F87171", "#A78BFA", "#6B737E"
INK, PANEL, LINE, DIM = "#0E1013", "#171A20", "#23272F", "#9BA3AE"
VEH = {"van": "VAN", "car": "CAR", "motorbike": "MOTO", "e-bike": "E-BIKE",
       "bicycle": "BIKE", "scooter": "SCOOTER"}
STAGES = ["Assigned", "En route to pickup", "At restaurant", "Delivering", "Delivered"]

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
        "page": "Dashboard",
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


def step(kind, title, body="", color=BRAND):
    st.session_state.steps.append({"kind": kind, "title": title, "body": body,
                                   "color": color, "t": now_str()})


def notify(title, body, role, accent=BRAND):
    st.session_state.notifs.insert(0, {"title": title, "body": body, "role": role,
                                       "accent": accent, "time": now_str()})


def flush_toasts():
    """Toast anything added since the last run."""
    pending = st.session_state.notifs[:max(0, len(st.session_state.notifs) - st.session_state.toast_cursor)]
    for n in reversed(pending):
        icon = "✅" if n["accent"] == GREEN else ("⚠️" if n["accent"] == RED else "🛵")
        st.toast(f"**{n['title']}**  \n{n['body']}", icon=icon)
    st.session_state.toast_cursor = len(st.session_state.notifs)


# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stButton, input, textarea {
    font-family:'Instrument Sans', system-ui, sans-serif !important;
}
.stApp { background:#0E1013; color:#E9EBEE; }
header[data-testid="stHeader"] { background:#12151A; border-bottom:1px solid #23272F; }
section[data-testid="stSidebar"] { background:#12151A; border-right:1px solid #23272F; }
.block-container { padding-top:1rem; padding-bottom:3rem; max-width:1620px; }
h1,h2,h3 { font-family:'Bricolage Grotesque',sans-serif !important; letter-spacing:-0.6px; color:#E9EBEE !important; }
h1 { font-size:27px !important; font-weight:700 !important; margin-bottom:2px !important; }
h2 { font-size:19px !important; font-weight:700 !important; }
h3 { font-size:16px !important; font-weight:700 !important; }
p, span, label, .stCaption { color:#9BA3AE; }

/* header strip */
.ra-top { display:flex; align-items:center; justify-content:space-between;
  background:#12151A; border:1px solid #23272F; border-radius:9px;
  padding:11px 18px; margin-bottom:14px; }
.ra-brand { font-family:'Bricolage Grotesque',sans-serif; font-weight:700;
  font-size:17px; letter-spacing:-0.4px; color:#E9EBEE; }
.ra-meta { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#6B737E; }

/* cards */
.ra-card { background:#171A20; border:1px solid #23272F; border-radius:9px;
  padding:16px 18px; margin-bottom:11px; }
.ra-card.active-mgr { border-color:#FF7A2F; }
.ra-card.active-drv { border-color:#4C5663; }
.ra-label { font-size:10.5px; font-weight:600; letter-spacing:.11em;
  text-transform:uppercase; color:#6B737E; }
.ra-num { font-family:'Bricolage Grotesque',sans-serif; font-size:36px;
  font-weight:600; letter-spacing:-1.6px; line-height:1.05; margin-top:5px; color:#E9EBEE; }
.ra-num small { font-size:17px; letter-spacing:0; color:#6B737E; }
.ra-chip { display:inline-block; border:1px solid #2A303A; background:#12151A;
  border-radius:5px; padding:3px 8px; font-size:11px; font-weight:600;
  color:#9BA3AE; margin:0 5px 5px 0; }
.ra-ph { width:72px; height:72px; border-radius:8px; flex:none;
  background:repeating-linear-gradient(135deg,#1B1F26 0 6px,#23272F 6px 12px);
  display:grid; place-items:center; font-family:'IBM Plex Mono',monospace;
  font-size:8.5px; color:#5B626C; text-align:center; line-height:1.3; }

/* reasoning panel */
.ra-brain { background:#12151A; border:1px solid #23272F; border-radius:9px;
  padding:16px 18px; color:#E9EBEE; height:520px; overflow-y:auto; }
.ra-brain .hd { font-size:10.5px; font-weight:600; letter-spacing:.11em;
  text-transform:uppercase; color:#6B737E; }
.ra-brain .sub { font-family:'IBM Plex Mono',monospace; font-size:10.5px;
  color:#4C5663; margin:4px 0 14px; }
.ra-stepbox { background:#171A20; border-radius:0 7px 7px 0; padding:10px 12px;
  margin-bottom:9px; }
.ra-stepbox .k { font-family:'IBM Plex Mono',monospace; font-size:10.5px;
  letter-spacing:.05em; }
.ra-stepbox .ti { font-size:13px; font-weight:600; margin-top:5px; color:#E9EBEE; }
.ra-stepbox .bd { font-family:'IBM Plex Mono',monospace; font-size:11px;
  color:#7A828D; margin-top:5px; line-height:1.55; white-space:pre-wrap; }
.ra-empty { border:1px dashed #2A303A; border-radius:8px; padding:20px 16px;
  color:#6B737E; font-size:12.5px; line-height:1.6; }

/* hud */
.ra-hud { background:#12151A; border:1px solid #2A303A; border-radius:8px;
  padding:12px 14px; margin-bottom:8px; }
.ra-hud .v { font-family:'Bricolage Grotesque',sans-serif; font-size:28px;
  font-weight:600; letter-spacing:-1.3px; line-height:1; color:#E9EBEE; }
.ra-bar { height:4px; border-radius:2px; background:#23272F; overflow:hidden; margin-top:9px; }
.ra-bar > i { display:block; height:100%; background:#FF7A2F; }

/* buttons */
.stButton > button { border-radius:7px !important; font-weight:600 !important;
  border:1px solid #2A303A !important; background:#171A20 !important;
  color:#C7CDD6 !important; padding:.5rem 1rem !important; font-size:13px !important; }
.stButton > button:hover { border-color:#39414D !important; color:#fff !important; }
.stButton > button[kind="primary"] { background:#FF7A2F !important; color:#0E1013 !important;
  border:none !important; font-weight:700 !important; }
.stButton > button[kind="primary"]:hover { background:#FF8E4D !important; }

/* sidebar radio nav */
div[role="radiogroup"] { gap:3px !important; }
div[role="radiogroup"] label { background:#171A20 !important; border:1px solid #23272F !important;
  border-radius:7px !important; padding:7px 12px !important; margin:0 !important; }
div[role="radiogroup"] label p { font-size:13px !important; font-weight:600 !important;
  color:#9BA3AE !important; }

/* inputs */
textarea, input { background:#12151A !important; border-radius:8px !important;
  border:1px solid #2A303A !important; color:#E9EBEE !important; font-size:14.5px !important; }
textarea:focus, input:focus { border-color:#FF7A2F !important; }
[data-testid="stMetricValue"] { font-family:'Bricolage Grotesque',sans-serif; color:#E9EBEE; }
[data-testid="stToast"] { background:#1B1F26 !important; border:1px solid #2A303A !important;
  color:#E9EBEE !important; }
hr { border-color:#23272F !important; }
footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- header
SHELTERS = load("shelters.json")
RESTAURANTS = load("restaurants.json")
DRV = drivers_live()
AVAIL = [d for d in DRV if d.get("status") == "available"]

st.markdown(
    f'<div class="ra-top">'
    f'<div style="display:flex;align-items:center;gap:24px">'
    f'<span class="ra-brand">RescueAgent</span>'
    f'<span style="font-size:12.5px;color:#8A8072">Edinburgh Food Rescue Network</span></div>'
    f'<div style="display:flex;gap:18px">'
    f'<span class="ra-meta">qwen3-235b · {BEDROCK_REGION}</span>'
    f'<span class="ra-meta">routing: {st.session_state.route_source}</span>'
    f'<span class="ra-meta">alerts: {len(st.session_state.notifs)}</span></div></div>',
    unsafe_allow_html=True)

PAGES = ["Dashboard", "New Rescue", "Live Tracking", "Active Deliveries",
         "Shelters", "Drivers", "History", "Alerts"]
st.session_state.page = st.radio("nav", PAGES, horizontal=True,
                                 index=PAGES.index(st.session_state.page),
                                 label_visibility="collapsed")
page = st.session_state.page

with st.sidebar:
    st.markdown("### Demo controls")
    st.session_state.use_bedrock = st.toggle(
        "Call Bedrock agent", value=st.session_state.use_bedrock,
        help="Off = deterministic tool pipeline only, no network call. Useful if the venue wifi is bad.")
    st.caption(f"Leg 1 {LEG1_SECONDS:.0f}s · Leg 2 {LEG2_SECONDS:.0f}s · "
               f"auto-accept {OFFER_SECONDS:.0f}s")


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
    step("user → agent", "Rescue request received", f'"{text}"\nfrom {r["name"]}', BRAND)

    safety = json.loads(analyze_food_safety(text))
    step("tool · analyze_food_safety", "Food classified",
         f'temperature: {"hot" if safety["needs_hot"] else "cold"}\n'
         f'category: {safety["category"]}\n'
         f'weight_kg: {safety["weight_kg"]}\n'
         f'fsa_window_min: {safety["fsa_window_minutes"]}', AMBER)

    shelter = json.loads(find_eligible_shelter(
        safety["needs_hot"], safety["needs_meat"], safety["needs_drinks"],
        r["lat"], r["lng"]))
    if shelter.get("error"):
        step("tool · find_eligible_shelter", "No eligible shelter", shelter["error"], RED)
        notify("Match failed", shelter["error"], "system", RED)
        return
    free = shelter["capacity_meals"] - shelter["current_intake_meals"]
    step("tool · find_eligible_shelter", "Shelter matched",
         f'{shelter["id"]} — {shelter["name"]}\n'
         f'accepts_hot: {shelter["accepts_hot_food"]} · accepts_meat: {shelter["accepts_meat"]}\n'
         f'demand {shelter["demand_score"]}/5 · {free} meals of headroom', VIOLET)

    bc = json.loads(broadcast_rescue(shelter["id"], safety["needs_hot"], safety["needs_meat"],
                                     safety["weight_kg"], r["lat"], r["lng"], 3))
    if bc.get("error"):
        step("tool · broadcast_rescue", "No capable driver on shift", bc["error"], RED)
        notify("Dispatch failed", bc["error"], "system", RED)
        return

    offers = [{**o, "status": "pending"} for o in bc["offered"]]
    step("tool · broadcast_rescue", f'Broadcast to {len(offers)} nearest capable drivers',
         "\n".join(f'{o["name"]:<18}{VEH.get(o["vehicle_type"], ""):<8}'
                   f'{o["distance_km"]:.1f} km · {o["max_capacity_kg"]} kg' for o in offers), GREEN)

    st.session_state.delivery = {
        "restaurant": r, "shelter": shelter, "safety": safety, "food_text": text,
        "offers": offers, "winner": None, "phase": "broadcast",
        "t0": time.time(), "legs": [], "leg": 0,
    }
    notify("New rescue offer",
           f'{safety["weight_kg"]} kg from {r["name"]} → {shelter["name"]}. '
           f'Offered to {len(offers)} drivers.', "driver", BRAND)

    if st.session_state.use_bedrock:
        try:
            with st.spinner("Agent reasoning over the request…"):
                reply = str(get_agent()(
                    f'Surplus food at {r["name"]} ({r["address"]}), '
                    f'coordinates {r["lat"]}, {r["lng"]}. The manager says: "{text}"'))
            st.session_state.agent_reply = reply
            step("agent → user", "Summary", reply.strip()[:900], BRAND)
        except Exception as e:
            step("agent → user", "Bedrock unavailable",
                 f"{e}\nfalling back to the deterministic tool pipeline", RED)

    st.session_state.page = "Live Tracking"


def do_accept(driver_id):
    d = st.session_state.delivery
    if not d or d["phase"] != "broadcast":
        return
    won = next((o for o in d["offers"] if o["driver_id"] == driver_id), None)
    if not won:
        return
    res = json.loads(accept_rescue(driver_id))
    if res.get("error"):
        notify("Offer already taken", res["error"], "driver", RED)
        return

    for o in d["offers"]:
        o["status"] = "won" if o["driver_id"] == driver_id else "lost"
    d["winner"] = won

    step("driver → agent", "Offer accepted",
         f'{won["name"]} accepted\nother {len(d["offers"]) - 1} offers withdrawn', GREEN)
    notify("Driver assigned",
           f'{won["name"]} accepted and is heading to {d["restaurant"]["name"]}.',
           "manager", GREEN)

    r, s = d["restaurant"], d["shelter"]
    leg1 = get_route((won["current_lat"], won["current_lng"]), (r["lat"], r["lng"]), won["vehicle_type"])
    leg2 = get_route((r["lat"], r["lng"]), (s["lat"], s["lng"]), won["vehicle_type"])
    d["legs"] = [leg1, leg2]
    st.session_state.route_source = leg1["source"]
    step("tool · route_lookup", "Road route resolved",
         f'leg 1 pickup: {leg1["km"]:.1f} km · ~{leg1["minutes"]} min\n'
         f'leg 2 delivery: {leg2["km"]:.1f} km · ~{leg2["minutes"]} min\n'
         f'source: {leg1["source"]}', BRAND)

    d["phase"] = "to_pickup"
    d["leg"] = 0
    d["t0"] = time.time()


def do_arrive():
    d = st.session_state.delivery
    d["phase"] = "at_pickup"
    d["t0"] = time.time()
    step("event · driver_arrived", "Driver at pickup point",
         f'{d["winner"]["name"]} arrived at {d["restaurant"]["name"]}\n'
         f'awaiting handover confirmation', GREEN)
    notify(f'{d["winner"]["name"]} has arrived',
           f'Waiting at {d["restaurant"]["name"]} — hand over the food and confirm.',
           "manager", BRAND)


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
         f'FSA window: {d["safety"]["fsa_window_minutes"]} min', AMBER)
    notify("Food collected", f'{d["winner"]["name"]} is en route to {d["shelter"]["name"]}.',
           "driver", GREEN)


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
         f'{meals} meals logged at {s["name"]}\n{co2} kg CO₂e avoided', GREEN)
    notify("Delivered", f'{s["name"]} signed for {meals} meals. {w["name"]} is back on shift.',
           "manager", GREEN)
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
        colour = GREEN if on else "#B9AE9C"
        folium.CircleMarker(
            [dr["current_lat"], dr["current_lng"]], radius=5 if on else 4,
            color="#0E1013", weight=1.5, fill=True, fill_color=colour,
            fill_opacity=0.5 if dim else 1.0,
            tooltip=f'{dr["name"]} · {VEH.get(dr["vehicle_type"], "")} · {dr["status"].replace("_", " ")}',
        ).add_to(m)

    if not d:
        return m

    r, s = d["restaurant"], d["shelter"]
    if d["legs"]:
        folium.PolyLine(d["legs"][0]["coords"], color="#8A8072", weight=4,
                        opacity=0.7, dash_array="7,8").add_to(m)
        folium.PolyLine(d["legs"][1]["coords"], color=BRAND, weight=5, opacity=0.95).add_to(m)

    def square(latlng, colour, label, tip):
        folium.Marker(
            latlng, tooltip=tip,
            icon=DivIcon(icon_size=(34, 34), icon_anchor=(17, 17), html=(
                f'<div style="width:32px;height:32px;border-radius:7px;background:{colour};'
                f'border:2px solid #0E1013;box-shadow:0 3px 12px rgba(0,0,0,.6);display:grid;'
                f'place-items:center;font:700 10px/1 sans-serif;color:#0E1013">{label}</div>')),
        ).add_to(m)

    square([r["lat"], r["lng"]], BRAND, "P", r["name"])
    square([s["lat"], s["lng"]], VIOLET, "S", s["name"])

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

        if pos is None:
            pos = (w["current_lat"], w["current_lng"])

        lat, lng = pos
        folium.Marker(
            [lat, lng], tooltip=f'{w["name"]} · {VEH.get(w["vehicle_type"], "")}',
            icon=DivIcon(icon_size=(40, 40), icon_anchor=(20, 20), html=(
                f'<div style="width:30px;height:30px;border-radius:50%;background:{BRAND};'
                f'border:2.5px solid #0E1013;box-shadow:0 3px 12px rgba(0,0,0,.6);display:grid;'
                f'place-items:center;font:700 8px/1 sans-serif;color:#0E1013">'
                f'{VEH.get(w["vehicle_type"], "DRV")}</div>')),
        ).add_to(m)

    pts = [[r["lat"], r["lng"]], [s["lat"], s["lng"]]]
    if w:
        pts.append([w["current_lat"], w["current_lng"]])
    m.fit_bounds(pts, padding=(40, 40))
    return m


def timeline_html(stage_idx):
    cells = []
    for i, label in enumerate(STAGES):
        done = i <= stage_idx
        cells.append(
            f'<div style="flex:1;min-width:0">'
            f'<div style="display:flex;align-items:center">'
            f'<span style="width:12px;height:12px;border-radius:50%;flex:none;'
            f'background:{BRAND if done else "#fff"};border:2px solid {BRAND if done else "#DED5C6"}"></span>'
            f'<span style="flex:1;height:2px;background:{BRAND if i < stage_idx else "#EDE6DA"}"></span></div>'
            f'<div style="font-size:10.5px;font-weight:600;line-height:1.25;padding-right:8px;'
            f'margin-top:6px;color:{"#4A4238" if done else "#A79C8C"}">{label}</div></div>')
    return f'<div style="display:flex;margin:16px 0 2px">{"".join(cells)}</div>'


def brain_html():
    if not st.session_state.steps:
        return (f'<div class="ra-brain"><div class="hd">Agent reasoning</div>'
                f'<div class="sub">strands · tool calls and returns</div>'
                f'<div class="ra-empty">Idle. All {len(DRV)} drivers are plotted on the map with '
                f'their current positions. Send a rescue to start the trace.</div></div>')
    rows = []
    for s in st.session_state.steps:
        body = f'<div class="bd">{s["body"]}</div>' if s["body"] else ""
        rows.append(
            f'<div class="ra-stepbox" style="border-left:2px solid {s["color"]}">'
            f'<div style="display:flex;gap:9px"><span class="k" style="color:{s["color"]};'
            f'filter:brightness(1.6);flex:1;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap">{s["kind"]}</span>'
            f'<span class="k" style="color:#6F6557;flex:none">{s["t"]}</span></div>'
            f'<div class="ti">{s["title"]}</div>{body}</div>')
    return (f'<div class="ra-brain"><div class="hd">Agent reasoning</div>'
            f'<div class="sub">strands · tool calls and returns</div>{"".join(rows)}</div>')


# ---------------------------------------------------------------- pages
if page == "Dashboard":
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("# Dashboard")
        st.caption("Session totals. Counters move only as the agent completes work.")
    with c2:
        if st.button("New rescue", type="primary", use_container_width=True):
            st.session_state.page = "New Rescue"
            st.rerun()

    s = st.session_state.stats
    cols = st.columns(4)
    for col, (label, val) in zip(cols, [
            ("Rescues completed", f'{s["rescues"]}'),
            ("Meals delivered", f'{s["meals"]}'),
            ("Food rescued", f'{s["kg"]:.1f}<small> kg</small>'),
            ("CO₂ avoided", f'<span style="color:{GREEN}">{s["co2"]:.1f}<small> kg</small></span>')]):
        col.markdown(f'<div class="ra-card"><div class="ra-label">{label}</div>'
                     f'<div class="ra-num">{val}</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with left:
        active = 1 if st.session_state.delivery and st.session_state.delivery["phase"] not in ("delivered",) else 0
        net = [(f"{len(RESTAURANTS):,}", "Partner restaurants", BRAND),
               (f"{len(SHELTERS)}", "Shelters & larders", AMBER),
               (f'{len(AVAIL)}<span style="font-size:15px;color:#A79C8C">/{len(DRV)}</span>',
                "Drivers on shift", GREEN),
               (f"{active}", "In flight", RED)]
        inner = "".join(
            f'<div style="border-left:2px solid {c};padding-left:13px">'
            f'<div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:26px;'
            f'font-weight:600;letter-spacing:-.8px">{v}</div>'
            f'<div style="font-size:12.5px;color:#7A7065;margin-top:2px">{l}</div></div>'
            for v, l, c in net)
        st.markdown(
            f'<div class="ra-card"><div class="ra-label">Network</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:14px">{inner}</div>'
            f'<div style="margin-top:18px;padding-top:14px;border-top:1px solid #F0EAE0;'
            f'font-size:13.5px;color:#6E6559;line-height:1.6;max-width:64ch">'
            f'A manager describes surplus food in plain English. The agent classifies it against '
            f'FSA thermal rules, matches a shelter that accepts it, then broadcasts the job to the '
            f'three nearest capable drivers. The first to accept gets the route.</div></div>',
            unsafe_allow_html=True)
    with right:
        if st.session_state.history:
            rows = "".join(
                f'<div style="border-top:1px solid #F0EAE0;padding-top:11px;margin-top:11px">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#A65524">{h["time"]}</div>'
                f'<div style="font-size:13.5px;font-weight:600;margin-top:3px">{h["food"][:70]}</div>'
                f'<div style="font-size:12.5px;color:#8A8072">{h["shelter"]}</div></div>'
                for h in st.session_state.history[:3])
        else:
            rows = ('<div style="border:1px dashed #DED5C6;border-radius:8px;padding:22px 16px;'
                    'color:#9A9082;font-size:13px;line-height:1.55;margin-top:12px">'
                    'Nothing rescued yet. Open <b>New Rescue</b> and describe what is left over.</div>')
        st.markdown(f'<div class="ra-card"><div class="ra-label">Recent this session</div>{rows}</div>',
                    unsafe_allow_html=True)


elif page == "New Rescue":
    st.markdown("# New rescue")
    st.caption("Pick the pickup kitchen, describe the surplus. The agent handles safety, shelter and driver.")

    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.markdown('<div class="ra-label">1 · Pickup kitchen</div>', unsafe_allow_html=True)
        if st.session_state.restaurant:
            r = st.session_state.restaurant
            cc1, cc2 = st.columns([4, 1])
            cc1.markdown(f'**{r["name"]}**  \n<span style="font-size:12.5px;color:#7A7065">{r["address"]}</span>',
                         unsafe_allow_html=True)
            if cc2.button("Change", use_container_width=True):
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
                                     key=f'pick_{r["id"]}', use_container_width=True):
                            st.session_state.restaurant = r
                            st.rerun()
            else:
                st.caption(f"{len(RESTAURANTS):,} restaurants in the OSM set. "
                           f"Type any part of a name — Awaafi, Dishoom, Gorgie.")

        st.markdown('<div class="ra-label" style="margin-top:14px">2 · Surplus food</div>',
                    unsafe_allow_html=True)
        st.session_state.food_text = st.text_area(
            "food", value=st.session_state.food_text, height=112,
            placeholder="e.g. 5 kg hot chicken biryani and 2 kg garlic naan, needs collecting within the hour",
            label_visibility="collapsed")

        p1, p2, p3 = st.columns(3)
        if p1.button("Hot curry, 6 kg", use_container_width=True):
            st.session_state.food_text = ("6 kg hot chicken curry and rice, cooked 40 minutes ago, "
                                          "needs collecting soon")
            st.rerun()
        if p2.button("Chilled sandwiches", use_container_width=True):
            st.session_state.food_text = "3 kg chilled sandwiches and salad boxes from the counter"
            st.rerun()
        if p3.button("Bakery, end of day", use_container_width=True):
            st.session_state.food_text = "12 loaves and 20 pastries, ambient, end of day"
            st.rerun()

        if st.button("Send to agent", type="primary", use_container_width=True):
            launch_rescue()
            st.rerun()
        st.caption("Broadcasts to the 3 nearest capable drivers. First to accept takes the job.")

    with right:
        sf = json.loads(analyze_food_safety(st.session_state.food_text or ""))
        reqs = [f'Capacity ≥ {sf["weight_kg"]} kg']
        reqs += ["Thermal bag", "Accepts hot"] if sf["needs_hot"] else ["Accepts cold"]
        if sf["needs_meat"]:
            reqs.append("Accepts meat")
        reqs.append("Status: available")
        rows = [("Temperature class", "Hot / cooked" if sf["needs_hot"] else "Cold / ambient"),
                ("Category", sf["category"].title()),
                ("Estimated weight", f'{sf["weight_kg"]} kg'),
                ("FSA handover window", f'{sf["fsa_window_minutes"]} min')]
        body = "".join(
            f'<div style="display:flex;justify-content:space-between;gap:12px;font-size:13.5px;'
            f'border-bottom:1px solid #F4EFE6;padding:8px 0"><span style="color:#7A7065">{k}</span>'
            f'<span style="font-weight:600">{v}</span></div>' for k, v in rows)
        st.markdown(f'<div class="ra-card"><div class="ra-label">Pre-flight read</div>{body}'
                    f'<div style="margin-top:12px;font-size:12.5px;color:#7A7065;line-height:1.55">'
                    f'{sf["safety_note"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ra-card"><div class="ra-label">Driver requirements implied</div>'
                    f'<div style="margin-top:10px">'
                    f'{"".join(f"<span class=ra-chip>{x}</span>" for x in reqs)}</div></div>',
                    unsafe_allow_html=True)


elif page == "Live Tracking":
    d = st.session_state.delivery
    head1, head2 = st.columns([4, 1.1])
    with head1:
        st.markdown("# Live tracking")
        if d:
            st.caption(f'{d["restaurant"]["name"]} → {d["shelter"]["name"]} · {d["phase"].replace("_", " ")}')
        else:
            st.caption(f"All {len(DRV)} drivers shown at their current positions. Nothing dispatched yet.")
    with head2:
        if st.button("Reset demo", use_container_width=True):
            reset_demo()
            st.rerun()

    @st.fragment(run_every=1.0 if (d and d["phase"] not in ("delivered",)) else None)
    def tracking_fragment():
        dd = st.session_state.delivery
        progress = advance()
        flush_toasts()

        col_brain, col_map = st.columns([1, 2.1], gap="small")
        with col_brain:
            st.markdown(brain_html(), unsafe_allow_html=True)
        with col_map:
            # HUD
            if not dd:
                hud = ("Fleet standby", f"{len(AVAIL)}", "drivers on shift",
                       f"{len(DRV)} plotted across Edinburgh", 0)
            elif dd["phase"] == "broadcast":
                hud = ("Dispatching", f'{len(dd["offers"])}', "offers open",
                       "Waiting for a driver to accept", 6)
            elif dd["phase"] == "at_pickup":
                hud = ("At the kitchen", "0", "min away",
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

            st.markdown(
                f'<div class="ra-hud"><div class="ra-label">{hud[0]}</div>'
                f'<div style="display:flex;align-items:baseline;gap:7px;margin-top:3px">'
                f'<span class="v">{hud[1]}</span>'
                f'<span style="font-size:12.5px;color:#7A7065;font-weight:600">{hud[2]}</span></div>'
                f'<div style="margin-top:6px;font-size:12.5px;color:#6E6559">{hud[3]}</div>'
                f'<div class="ra-bar"><i style="width:{hud[4]}%"></i></div></div>',
                unsafe_allow_html=True)
            st_folium(build_map(progress), height=460, use_container_width=True,
                      returned_objects=[], key="tracking_map")

        # --- role strips -------------------------------------------------
        ph = dd["phase"] if dd else "idle"
        mgr_active = ph in ("at_pickup", "delivered", "to_pickup")
        drv_active = ph in ("broadcast", "to_shelter")
        m_col, d_col = st.columns(2, gap="small")

        with m_col:
            cls = "ra-card active-mgr" if mgr_active else "ra-card"
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
            st.markdown(
                f'<div class="{cls}"><div style="display:flex;justify-content:space-between">'
                f'<span class="ra-label">Restaurant manager</span>'
                f'<span style="font-size:11px;font-weight:700;letter-spacing:.06em;'
                f'text-transform:uppercase;color:{BRAND if mgr_active else MUTED}">'
                f'{ph.replace("_", " ")}</span></div>'
                f'<div style="font-size:15px;font-weight:600;margin-top:10px">{texts[0]}</div>'
                f'<div style="font-size:13px;color:#7A7065;margin-top:3px">{texts[1]}</div>'
                f'{timeline_html(stage_idx)}</div>', unsafe_allow_html=True)
            if ph == "at_pickup":
                if st.button("Food handed over — release driver", type="primary",
                             use_container_width=True, key="handover"):
                    do_handover()
                    st.rerun(scope="fragment")
            if ph == "delivered" and st.session_state.history:
                h = st.session_state.history[0]
                st.success(f'**{h["meals"]} meals logged** — {h["summary"]} · {h["co2"]} CO₂e avoided')

        with d_col:
            cls = "ra-card active-drv" if drv_active else "ra-card"
            st.markdown(
                f'<div class="{cls}"><div style="display:flex;justify-content:space-between">'
                f'<span class="ra-label">Driver app</span>'
                f'<span style="font-size:11px;font-weight:700;letter-spacing:.06em;'
                f'text-transform:uppercase;color:{"#221E19" if drv_active else MUTED}">'
                f'{"offer open" if ph == "broadcast" else ph.replace("_", " ")}</span></div>',
                unsafe_allow_html=True)
            if not dd:
                st.markdown(f'<div style="border:1px dashed #DED5C6;border-radius:8px;padding:22px 16px;'
                            f'color:#9A9082;font-size:13px">No open offers. {len(AVAIL)} drivers are '
                            f'on shift waiting for a job.</div></div>', unsafe_allow_html=True)
            elif ph == "broadcast":
                st.markdown("</div>", unsafe_allow_html=True)
                for o in dd["offers"]:
                    oc1, oc2 = st.columns([4, 1])
                    oc1.markdown(
                        f'<div style="font-size:14.5px;font-weight:600">{o["name"]} '
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10.5px;'
                        f'background:#F1EAE0;border-radius:4px;padding:2px 6px;color:#6E6559">'
                        f'{VEH.get(o["vehicle_type"], "")}</span></div>'
                        f'<div style="font-size:12px;color:#7A7065">{o["distance_km"]:.1f} km away · '
                        f'{o["max_capacity_kg"]} kg · {o["neighbourhood"]} · {o["rating"]}★</div>',
                        unsafe_allow_html=True)
                    if o["status"] == "pending":
                        if oc2.button("Accept", key=f'acc_{o["driver_id"]}', use_container_width=True):
                            do_accept(o["driver_id"])
                            st.rerun(scope="fragment")
                    else:
                        oc2.markdown("**Accepted**" if o["status"] == "won" else "Taken")
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
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:13px;border:1px solid #EFE8DD;'
                    f'border-radius:8px;padding:12px 14px;margin-top:10px">'
                    f'<div style="width:40px;height:40px;border-radius:50%;background:#221E19;color:#fff;'
                    f'display:grid;place-items:center;font-weight:600;flex:none">'
                    f'{"".join(x[0] for x in w["name"].split()[:2])}</div>'
                    f'<div style="flex:1"><div style="font-size:15px;font-weight:600">{w["name"]}</div>'
                    f'<div style="font-size:12.5px;color:#7A7065">{VEH.get(w["vehicle_type"], "")} · '
                    f'{w.get("vehicle_reg", "")} · {w["neighbourhood"]} · {w["max_capacity_kg"]} kg{bag}</div></div>'
                    f'<div style="text-align:right"><div style="font-family:\'Bricolage Grotesque\','
                    f'sans-serif;font-size:19px;font-weight:600">{w["rating"]}</div>'
                    f'<div style="font-size:10.5px;color:#9A9082">rating</div></div></div>'
                    f'<div style="margin-top:11px;font-size:13.5px;font-weight:600;color:#4A4238">{instr[0]}</div>'
                    f'<div style="font-size:12.5px;color:#7A7065;margin-top:2px">{instr[1]}</div></div>',
                    unsafe_allow_html=True)
                if ph == "at_pickup":
                    if st.button("Confirm food collected", use_container_width=True, key="drv_collect"):
                        do_handover()
                        st.rerun(scope="fragment")
                elif ph == "to_pickup":
                    if st.button("Report arrival early", use_container_width=True, key="drv_arrive"):
                        do_arrive()
                        st.rerun(scope="fragment")

    tracking_fragment()


elif page == "Active Deliveries":
    st.markdown("# Active deliveries")
    d = st.session_state.delivery
    if d and d["phase"] in ("to_pickup", "at_pickup", "to_shelter"):
        w = d["winner"]
        st.markdown(
            f'<div class="ra-card" style="display:flex;gap:18px;align-items:center">'
            f'<div class="ra-ph">food<br>photo</div>'
            f'<div style="flex:1"><div style="font-size:16px;font-weight:600">{d["food_text"]}</div>'
            f'<div style="font-size:13px;color:#7A7065;margin-top:3px">'
            f'{d["restaurant"]["name"]} → {d["shelter"]["name"]}</div>'
            f'<div style="margin-top:8px"><span class="ra-chip">{w["name"]}</span>'
            f'<span class="ra-chip">{VEH.get(w["vehicle_type"], "")}</span>'
            f'<span class="ra-chip">{d["safety"]["weight_kg"]} kg</span></div></div>'
            f'<div style="text-align:right"><div style="font-size:11px;font-weight:700;'
            f'letter-spacing:.07em;text-transform:uppercase;color:{BRAND}">'
            f'{d["phase"].replace("_", " ")}</div></div></div>', unsafe_allow_html=True)
        if st.button("Track", type="primary"):
            st.session_state.page = "Live Tracking"
            st.rerun()
    else:
        st.markdown('<div class="ra-card" style="border-style:dashed;color:#9A9082;padding:44px 24px">'
                    'Nothing in flight. A rescue appears here from dispatch until it is delivered.</div>',
                    unsafe_allow_html=True)


elif page == "Shelters":
    st.markdown("# Shelters & larders")
    st.caption(f"{len(SHELTERS)} registered locations across Edinburgh")
    f1, f2, _ = st.columns([1, 1, 3])
    hot_only = f1.toggle("Accepts hot food")
    veg_only = f2.toggle("Vegetarian options")

    rows = [s for s in SHELTERS
            if (not hot_only or s.get("accepts_hot_food"))
            and (not veg_only or any("veg" in t for t in s.get("dietary_tags", [])))]
    cols = st.columns(3)
    for i, s in enumerate(rows):
        demand = s.get("demand_score", 0)
        colour = RED if demand >= 4.5 else (AMBER if demand >= 4 else GREEN)
        cap, cur = s.get("capacity_meals", 0), s.get("current_intake_meals", 0)
        pct = int(cur / cap * 100) if cap else 0
        tags = "".join(f'<span class="ra-chip">{t.replace("_", " ")}</span>'
                       for t in s.get("dietary_tags", []))
        cols[i % 3].markdown(
            f'<div class="ra-card"><div style="display:flex;justify-content:space-between;gap:10px">'
            f'<div style="font-size:15.5px;font-weight:600;line-height:1.3">{s["name"]}</div>'
            f'<div style="text-align:right;flex:none"><div style="font-family:\'Bricolage Grotesque\','
            f'sans-serif;font-size:16px;font-weight:600;color:{colour}">{demand}/5</div>'
            f'<div style="font-size:10px;color:#9A9082;text-transform:uppercase">demand</div></div></div>'
            f'<div style="font-size:12.5px;color:#7A7065;margin-top:5px">{s.get("address", "")}</div>'
            f'<div style="margin-top:9px">{tags}</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#7A7065;'
            f'margin-top:8px"><span>{s.get("opens_24h", "")}–{s.get("closes_24h", "")}</span>'
            f'<span>{cur}/{cap} meals</span></div>'
            f'<div class="ra-bar"><i style="width:{pct}%;background:{colour}"></i></div></div>',
            unsafe_allow_html=True)


elif page == "Drivers":
    st.markdown("# Volunteer drivers")
    choice = st.radio("filter", ["All", "Available", "Busy", "Off duty"],
                      horizontal=True, label_visibility="collapsed")
    key = {"Available": "available", "Busy": "busy", "Off duty": "off_duty"}.get(choice)
    rows = [d for d in DRV if key is None or d.get("status") == key]
    st.caption(f"{len(rows)} of {len(DRV)} drivers")
    cols = st.columns(3)
    for i, d in enumerate(rows):
        status = d.get("status", "")
        dot = GREEN if status == "available" else (BRAND if status == "busy" else "#B9AE9C")
        kit = ("Thermal bag" if d.get("has_thermal_bag")
               else ("Cool box" if d.get("has_cool_box") else "No thermal kit"))
        cols[i % 3].markdown(
            f'<div class="ra-card"><div style="display:flex;align-items:center;gap:12px">'
            f'<div style="width:38px;height:38px;border-radius:50%;background:#F1EAE0;'
            f'display:grid;place-items:center;font-family:\'Bricolage Grotesque\',sans-serif;'
            f'font-weight:600;flex:none">{"".join(x[0] for x in d["name"].split()[:2])}</div>'
            f'<div style="flex:1"><div style="font-size:15px;font-weight:600">{d["name"]}</div>'
            f'<div style="font-size:12px;color:#7A7065"><span style="display:inline-block;width:7px;'
            f'height:7px;border-radius:50%;background:{dot};margin-right:5px"></span>'
            f'{status.replace("_", " ")} · {d.get("neighbourhood", "")}</div></div>'
            f'<div style="text-align:right"><div style="font-family:\'Bricolage Grotesque\',sans-serif;'
            f'font-size:17px;font-weight:600">{d.get("rating", "")}</div>'
            f'<div style="font-size:10px;color:#9A9082">rating</div></div></div>'
            f'<div style="margin-top:11px"><span class="ra-chip">{VEH.get(d.get("vehicle_type"), "")}</span>'
            f'<span class="ra-chip">Max {d.get("max_capacity_kg", "?")} kg</span>'
            f'<span class="ra-chip">{kit}</span>'
            f'<span class="ra-chip">ETA {d.get("eta_minutes", "?")} min</span></div></div>',
            unsafe_allow_html=True)


elif page == "History":
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown("# Session history")
        st.caption("Rescues completed in this session")
    with h2:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.stats = {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0}
            st.rerun()

    if not st.session_state.history:
        st.markdown('<div class="ra-card" style="border-style:dashed;color:#9A9082;padding:44px 24px">'
                    'No rescues logged yet this session.</div>', unsafe_allow_html=True)
    for h in st.session_state.history:
        st.markdown(
            f'<div class="ra-card" style="display:flex;gap:18px;align-items:center">'
            f'<div class="ra-ph">food<br>photo</div>'
            f'<div style="flex:1"><div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#A65524">{h["time"]} · {h["id"]}</div>'
            f'<div style="font-size:15.5px;font-weight:600;margin-top:3px">{h["food"]}</div>'
            f'<div style="font-size:13px;color:#7A7065">{h["route"]}</div>'
            f'<div style="font-size:12.5px;color:#8A8072;margin-top:5px">{h["summary"]}</div></div>'
            f'<div style="text-align:right;flex:none">'
            f'<div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:22px;'
            f'font-weight:600">{h["meals"]}</div><div style="font-size:10.5px;color:#9A9082">meals</div>'
            f'<div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:15px;font-weight:600;'
            f'color:{GREEN};margin-top:6px">{h["co2"]}</div>'
            f'<div style="font-size:10.5px;color:#9A9082">CO₂ saved</div></div></div>',
            unsafe_allow_html=True)


elif page == "Alerts":
    a1, a2 = st.columns([4, 1])
    with a1:
        st.markdown("# Alerts")
        st.caption("Every status change the network broadcast, newest first")
    with a2:
        if st.button("Clear", use_container_width=True):
            st.session_state.notifs = []
            st.session_state.toast_cursor = 0
            st.rerun()

    if not st.session_state.notifs:
        st.markdown('<div class="ra-card" style="border-style:dashed;color:#9A9082;padding:44px 24px">'
                    'Nothing yet. Alerts appear the moment a driver is offered a job.</div>',
                    unsafe_allow_html=True)
    for n in st.session_state.notifs:
        st.markdown(
            f'<div class="ra-card" style="border-left:3px solid {n["accent"]};display:flex;'
            f'gap:16px;align-items:center;padding:14px 18px">'
            f'<div style="flex:1"><div style="font-size:14.5px;font-weight:600">{n["title"]} '
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
            f'text-transform:uppercase;color:{n["accent"]};border:1px solid {n["accent"]}33;'
            f'border-radius:4px;padding:2px 6px">{n["role"]}</span></div>'
            f'<div style="font-size:13px;color:#7A7065;margin-top:3px">{n["body"]}</div></div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#A79C8C">'
            f'{n["time"]}</div></div>', unsafe_allow_html=True)

if page != "Live Tracking":
    flush_toasts()
