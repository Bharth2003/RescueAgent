"""Cross-session broker for the two-account (Manager + Driver) demo.

Streamlit keeps ``st.session_state`` per browser session, so a Manager window
and a Driver window cannot see each other's actions on their own. This module
holds a single, process-global rescue that both windows read and write, guarded
by a lock, so a dispatch in the Manager window shows up as a live offer (and a
toast) in the Driver window within a second, and both watch the same journey.

The app wraps ``Broker`` in ``st.cache_resource`` so every session in the one
Streamlit server process shares the exact same instance. State is intentionally
in-memory: it resets when the server restarts, which is all a demo needs.

The whole rescue state machine lives here (behind the lock) so that, even with
both windows polling once a second, each transition — auto-accept, arrival,
handover, delivery — fires exactly once.
"""

import copy
import threading
import time
from datetime import datetime

from routing import get_route, haversine_km
from tools import accept_rescue, release_driver

# Journey clock. Each leg plays for a fixed, watchable ~20-30s on screen (the
# demo is narrated over roughly three minutes), independent of real distance.
LEG_MIN_SECONDS = 20.0
LEG_MAX_SECONDS = 30.0
OFFER_SECONDS = 75.0     # auto-accept fallback if the driver never taps Accept
PICKUP_SECONDS = 60.0    # auto-handover fallback if nobody confirms collection


def _leg_seconds(leg):
    return max(LEG_MIN_SECONDS, min(LEG_MAX_SECONDS, leg["minutes"] * 60.0))


def _now_str():
    return datetime.now().strftime("%H:%M:%S")


class Broker:
    """Single shared rescue plus a cross-window notification feed."""

    def __init__(self):
        self._lock = threading.RLock()
        self._reset_state(full=True)

    # ------------------------------------------------------------------ state
    def _reset_state(self, full=False):
        """Clear the active rescue. ``history`` and cumulative ``stats`` survive
        between rescues (they are the session's impact log); ``full`` also wipes
        those, used on __init__ and an explicit reset()."""
        prev = getattr(self, "state", None)
        self.state = {
            "delivery": None,   # the one shared delivery dict
            "steps": [],        # plain-English agent narration
            "events": [],       # notification feed, oldest first
            "seq": (prev["seq"] + 1) if prev else 0,
            "history": [] if (full or not prev) else prev.get("history", []),
            "stats": {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0}
                     if (full or not prev) else prev.get("stats",
                         {"rescues": 0, "meals": 0, "kg": 0.0, "co2": 0.0}),
        }

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.state)

    def _bump(self):
        self.state["seq"] += 1

    def _say(self, role, text, meta="", accent="brand"):
        self.state["steps"].append({"role": role, "text": text, "meta": meta,
                                    "accent": accent, "t": _now_str()})

    def _event(self, title, body, target, accent="brand", icon="🛰"):
        """Queue a notification. ``target`` is 'manager', 'driver' or 'both'."""
        self.state["events"].append({"title": title, "body": body, "target": target,
                                     "accent": accent, "icon": icon, "time": _now_str()})

    # -------------------------------------------------------------- dispatch
    def start_rescue(self, restaurant, shelter, safety, food_text, offers):
        """Manager dispatched a rescue: open offers to the nearest drivers."""
        with self._lock:
            self._reset_state()
            self.state["delivery"] = {
                "id": f"rescue-{int(time.time())}",
                "restaurant": restaurant, "shelter": shelter, "safety": safety,
                "food_text": food_text,
                "offers": [{**o, "status": "pending"} for o in offers],
                "winner": None, "phase": "broadcast",
                "t0": time.time(), "legs": [], "leg": 0,
            }
            roster = ", ".join(f'{o["name"]} ({o["distance_km"]:.1f} km)' for o in offers)
            self._say("agent",
                      f'Offer is out to the **{len(offers)} nearest capable drivers** — '
                      f'{roster}. First to accept takes it.',
                      "broadcast_rescue", "green")
            self._event("New rescue offer",
                        f'{safety["weight_kg"]} kg from {restaurant["name"]} → '
                        f'{shelter["name"]}. Tap to accept.',
                        target="driver", accent="brand", icon="🛵")
            self._event("Dispatch sent",
                        f'Offered to {len(offers)} drivers. Waiting for one to accept.',
                        target="manager", accent="brand", icon="📡")
            self._bump()

    # --------------------------------------------------------------- accept
    def accept(self, driver_id):
        """A driver accepted. First to accept wins; the rest are withdrawn."""
        with self._lock:
            return self._accept_locked(driver_id, manual=True)

    def _accept_locked(self, driver_id, manual):
        d = self.state["delivery"]
        if not d or d["phase"] != "broadcast":
            return {"error": "Offer is no longer open."}
        won = next((o for o in d["offers"] if o["driver_id"] == driver_id), None)
        if not won:
            return {"error": "Unknown offer."}
        import json
        res = json.loads(accept_rescue(driver_id))
        if res.get("error"):
            return res
        for o in d["offers"]:
            o["status"] = "won" if o["driver_id"] == driver_id else "lost"
        d["winner"] = won

        r, s = d["restaurant"], d["shelter"]
        leg1 = get_route((won["current_lat"], won["current_lng"]),
                         (r["lat"], r["lng"]), won["vehicle_type"])
        leg2 = get_route((r["lat"], r["lng"]), (s["lat"], s["lng"]),
                         won["vehicle_type"])
        d["legs"] = [leg1, leg2]
        d["phase"] = "to_pickup"
        d["leg"] = 0
        d["t0"] = time.time()

        how = "auto-assigned to the nearest driver" if not manual else "accepted the job"
        self._say("driver",
                  f'**{won["name"]}** {how} — {won["distance_km"]:.1f} km away in '
                  f'{won.get("neighbourhood", "")}. Heading to {r["name"]} now.',
                  f'accept_rescue · {won["driver_id"]}', "green")
        self._event("Driver assigned",
                    f'{won["name"]} accepted and is heading to {r["name"]}.',
                    target="manager", accent="green", icon="✅")
        self._event("Job accepted",
                    f'You have the pickup at {r["name"]}. Route is live.',
                    target="driver", accent="green", icon="✅")
        self._bump()
        return {"ok": True, "winner": won}

    # ------------------------------------------------------ manual advances
    def arrive_now(self):
        with self._lock:
            d = self.state["delivery"]
            if d and d["phase"] == "to_pickup":
                self._arrive_locked()
                self._bump()

    def handover_now(self):
        with self._lock:
            d = self.state["delivery"]
            if d and d["phase"] == "at_pickup":
                self._handover_locked()
                self._bump()

    # ------------------------------------------------------------ the clock
    def tick(self):
        """Drive the state machine off the wall clock. Safe to call from every
        session on every refresh — the lock + phase checks make each transition
        fire exactly once. Returns leg progress 0..1 for the active leg."""
        with self._lock:
            d = self.state["delivery"]
            if not d:
                return 0.0
            el = time.time() - d["t0"]
            ph = d["phase"]
            if ph == "broadcast":
                if el >= OFFER_SECONDS and d["offers"]:
                    self._accept_locked(d["offers"][0]["driver_id"], manual=False)
                return 0.0
            if ph == "to_pickup":
                p = min(1.0, el / _leg_seconds(d["legs"][0]))
                if p >= 1.0:
                    self._arrive_locked()
                    self._bump()
                    return 1.0
                return p
            if ph == "at_pickup":
                if el >= PICKUP_SECONDS:
                    self._handover_locked()
                    self._bump()
                return 0.0
            if ph == "to_shelter":
                p = min(1.0, el / _leg_seconds(d["legs"][1]))
                if p >= 1.0:
                    self._deliver_locked()
                    self._bump()
                    return 1.0
                return p
            return 1.0

    # ------------------------------------------------------ transitions
    def _arrive_locked(self):
        d = self.state["delivery"]
        d["phase"] = "at_pickup"
        d["t0"] = time.time()
        w, r = d["winner"], d["restaurant"]
        self._say("driver",
                  f'**{w["name"]} has arrived at {r["name"]}** and is waiting for the '
                  f'handover.', "leg 1 complete", "green")
        self._event("Driver arrived",
                    f'{w["name"]} is at {r["name"]} — hand over the food.',
                    target="both", accent="brand", icon="📍")

    def _handover_locked(self):
        d = self.state["delivery"]
        d["phase"] = "to_shelter"
        d["leg"] = 1
        d["t0"] = time.time()
        w, sf, s = d["winner"], d["safety"], d["shelter"]
        self._say("driver",
                  f'Handover confirmed — **{w["name"]} collected {sf["weight_kg"]} kg** '
                  f'and is driving to **{s["name"]}**, {d["legs"][1]["km"]:.1f} km away. '
                  f'Serve within {sf["fsa_window_minutes"]} minutes.',
                  f'FSA clock started · {sf["category"]}', "amber")
        self._event("Food collected",
                    f'{w["name"]} loaded {sf["weight_kg"]} kg and is en route to {s["name"]}.',
                    target="both", accent="green", icon="📦")

    def _deliver_locked(self):
        d = self.state["delivery"]
        w, s, sf = d["winner"], d["shelter"], d["safety"]
        meals = max(2, int(round(sf["weight_kg"] * 2.4)))
        co2 = round(sf["weight_kg"] * 2.5, 1)
        total_km = d["legs"][0]["km"] + d["legs"][1]["km"]
        release_driver(w["driver_id"])
        d["phase"] = "delivered"
        d["result"] = {"meals": meals, "co2": co2, "total_km": total_km}
        # log the completed rescue and roll up the session's cumulative impact
        st = self.state["stats"]
        st["rescues"] += 1
        st["meals"] += meals
        st["kg"] = round(st["kg"] + sf["weight_kg"], 1)
        st["co2"] = round(st["co2"] + co2, 1)
        self.state["history"].insert(0, {
            "id": d["id"], "time": _now_str(),
            "food": d["food_text"],
            "route": f'{d["restaurant"]["name"]} → {s["name"]}',
            "driver": w["name"], "meals": meals, "kg": sf["weight_kg"],
            "co2": co2, "km": round(total_km, 1),
        })
        self._say("shelter",
                  f'**Delivered.** {s["name"]} signed for {sf["weight_kg"]} kg — about '
                  f'**{meals} meals** — and {w["name"]} is back on shift. '
                  f'**{co2} kg CO₂e** kept out of the air.',
                  f'release_driver · {w["driver_id"]}', "green")
        self._event("Delivered",
                    f'{s["name"]} signed for {meals} meals. {w["name"]} is back on shift.',
                    target="both", accent="green", icon="🎉")

    # -------------------------------------------------------------- reset
    def reset(self):
        with self._lock:
            d = self.state["delivery"]
            if d and d.get("winner") and d["phase"] != "delivered":
                release_driver(d["winner"]["driver_id"])
            self._reset_state()
            self._bump()
