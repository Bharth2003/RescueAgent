# How RescueAgent works — code to deployment

A walkthrough of every moving part, written to be read top to bottom. If you only read
one section, read [4 — How the live map actually works](#4--how-the-live-map-actually-works).

- [1 — The mental model](#1--the-mental-model)
- [2 — Streamlit fundamentals you need](#2--streamlit-fundamentals-you-need)
- [3 — The agent layer](#3--the-agent-layer)
- [4 — How the live map actually works](#4--how-the-live-map-actually-works)
- [5 — The delivery state machine](#5--the-delivery-state-machine)
- [6 — Notifications](#6--notifications)
- [7 — File by file](#7--file-by-file)
- [8 — Adding a feature, worked example](#8--adding-a-feature-worked-example)
- [9 — Running it locally](#9--running-it-locally)
- [10 — Deployment](#10--deployment)
- [11 — Debugging](#11--debugging)

---

## 1 — The mental model

Three layers, each of which can be understood alone:

```
Streamlit UI (app.py)          what the human sees, and the clock that ticks the demo
      |
Strands agent (tools.py)       decides: is it safe, which shelter, which drivers
      |
data/*.json + OSRM             the world the agent reads and writes
```

The important design decision: **the agent decides, the tools enforce.** A language model
is good at reading "6 kg hot chicken curry, cooked 40 minutes ago" and turning it into
structured facts. It is not something you want deciding whether a driver without a
thermal bag may legally carry hot food. So every hard rule lives in Python inside
`tools.py`, and the model only ever picks from lists the tools already filtered. The
model cannot hallucinate its way past a food safety rule because the unsafe options were
removed before it saw them.

---

## 2 — Streamlit fundamentals you need

Four concepts explain almost everything in `app.py`.

### 2.1 The script reruns, top to bottom

Streamlit has no event handlers. Every interaction — a click, a keystroke in a text
input — reruns your entire script from line 1. `if st.button("Accept"):` is not a
callback; it is `True` on exactly the one rerun caused by that click, and `False` on
every other rerun.

This is why the code is written as "compute current state, then draw it" rather than
"draw once, then mutate". Nothing is drawn incrementally.

### 2.2 `st.session_state` is the only thing that survives a rerun

Local variables die at the end of every rerun. Anything that must persist — the active
delivery, the agent trace, the alert feed, session stats — lives in `st.session_state`,
a dict that outlives reruns for as long as the browser tab is open.

```python
def init_state():
    defaults = {"page": "Dashboard", "delivery": None, "steps": [], ...}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)   # setdefault, so a rerun never wipes live data
```

`setdefault` matters. `st.session_state["steps"] = []` at the top of the script would
erase the trace on every rerun.

### 2.3 `st.rerun()` forces an immediate rerun

After mutating state in response to a click, you often want the page redrawn from the new
state right away. `st.rerun()` aborts the current run and starts a fresh one.
`st.rerun(scope="fragment")` reruns only the fragment, which is what the Accept and
handover buttons use inside Live Tracking.

### 2.4 `@st.fragment(run_every=...)` is what makes the map live

This is the piece people usually miss. A fragment is a function whose body can rerun
**independently of the rest of the page**, and `run_every=1.0` makes it rerun itself once
a second with no user interaction:

```python
@st.fragment(run_every=1.0 if (d and d["phase"] not in ("delivered",)) else None)
def tracking_fragment():
    progress = advance()      # move the clock
    flush_toasts()            # fire any new notifications
    ...                       # redraw HUD, map, role strips
tracking_fragment()
```

Note `run_every=... else None`. Once the delivery is finished the fragment stops polling,
so a completed demo does not sit there burning a rerun every second.

Without fragments you would need `st.rerun()` in a loop, which reruns the *whole* script —
the sidebar, all eight pages of logic, every data load — once a second. The fragment
scopes that cost to just the map panel.

### 2.5 `@st.cache_data` and `@st.cache_resource`

`restaurants.json` is 470 KB. Parsing it on every one-second rerun would be wasteful:

```python
@st.cache_data                  # for data: cached by function + arguments
def load(name): ...

@st.cache_resource              # for objects that must be shared, not copied
def get_agent(): ...
```

`cache_data` returns a copy each time (safe for data). `cache_resource` returns the *same
object* (right for an agent or a DB connection — you want one, not one per session).
`load.clear()` invalidates the cache, which the code calls after `drivers.json` is
written so the roster reflects the new status.

---

## 3 — The agent layer

### 3.1 Wiring the agent

```python
from strands import Agent
from strands.models import BedrockModel

Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[analyze_food_safety, find_eligible_shelter, broadcast_rescue,
           accept_rescue, route_lookup, release_driver, dispatch_driver],
    model=BedrockModel(model_id="qwen.qwen3-235b-a22b-2507-v1:0",
                       region_name="eu-west-2"),
)
```

That is the whole integration. Strands handles the tool-calling loop: it describes your
Python functions to the model, receives the model's chosen call, executes it, feeds the
result back, and repeats until the model produces a final answer.

### 3.2 What makes a function a tool

```python
from strands import tool

@tool
def analyze_food_safety(food_description: str) -> str:
    """Classify surplus food and attach the FSA handover window."""
    ...
    return json.dumps({...})
```

Three things carry meaning to the model:

1. **The decorator** registers it.
2. **The type hints** become the parameter schema. `weight_kg: float` tells the model to
   send a number. Untyped parameters give the model nothing to aim at.
3. **The docstring is the prompt.** This is the only description the model gets of what
   the tool does and when to reach for it. A vague docstring is the single most common
   cause of a tool never being called.

Return a JSON string, not a dict. It gives you one predictable shape crossing the
boundary, and it is what shows up verbatim in the reasoning trace.

### 3.3 The system prompt as a policy, not a personality

```
1. `analyze_food_safety` to classify the food and get the FSA handover window.
2. `find_eligible_shelter` with the safety flags and the restaurant coordinates.
3. `broadcast_rescue` to offer the job to the nearest capable drivers.

Do not invent names or addresses — only use what the tools return.
```

Ordered steps, and an explicit ban on inventing facts. Ordering matters because
`find_eligible_shelter` needs the safety flags that `analyze_food_safety` produces —
saying so in the prompt is cheaper and more reliable than hoping the model infers it.

### 3.4 The filtering that keeps it safe

`broadcast_rescue` is where the real logic lives:

```python
for d in drivers:
    if d.get("status") != "available":            continue
    if d.get("max_capacity_kg", 0) < weight_kg:   continue
    accepts = d.get("accepts", [])
    if needs_hot and not ("hot" in accepts and d.get("has_thermal_bag")): continue
    if needs_meat and "meat" not in accepts:      continue
    candidates.append({**d, "distance_km": haversine_km(...)})

candidates.sort(key=lambda d: d["distance_km"])
offered = candidates[:3]
```

Availability, then capacity, then handling capability, then distance. Hot food without a
thermal bag is impossible, not discouraged. Then the nearest three get the offer, and
`accept_rescue` is first-come-first-served:

```python
if d.get("status") != "available":
    return json.dumps({"error": "Offer already taken."})
d["status"] = "busy"
```

### 3.5 The reasoning trace

Every tool call appends to `st.session_state.steps`:

```python
step("tool · analyze_food_safety", "Food classified",
     f'temperature: {"hot" if safety["needs_hot"] else "cold"}\n'
     f'weight_kg: {safety["weight_kg"]}\n'
     f'fsa_window_min: {safety["fsa_window_minutes"]}', AMBER)
```

Rendered by `brain_html()` into the dark panel beside the map. This is the "to and fro"
the judging criteria ask for — visible input, visible tool, visible return.

### 3.6 The offline switch

```python
st.session_state.use_bedrock = st.toggle("AI reasoning", value=True)
# On: Strands Agents SDK → Amazon Bedrock Qwen 3 235B
#     (qwen.qwen3-235b-a22b-2507-v1:0, eu-west-2)
```

Off, `launch_rescue()` calls the tools directly and skips the model. Same classification,
same shelter match, same broadcast, same map — no network. Judges without AWS
credentials still see the app work, and bad venue wifi cannot kill your demo.

---

## 4 — How the live map actually works

Four separate problems, solved in order.

### 4.1 Why `st.map` could not do this

`st.map` draws dots at lat/lng. It cannot draw a line, cannot style a marker, cannot
follow a road. That is why the original tracking looked broken — it was showing scattered
points with no route. The fix was to move to `folium` (a Python wrapper over Leaflet)
rendered through `streamlit-folium`:

```python
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[55.9490, -3.1900], zoom_start=12, tiles="OpenStreetMap")
st_folium(m, height=460, use_container_width=True,
          returned_objects=[], key="tracking_map")
```

`returned_objects=[]` matters for performance: by default `st_folium` ships map state
(bounds, last click) back to Python on every interaction, which triggers a rerun. An
empty list says "send nothing back", making the map a pure output.

`tiles="OpenStreetMap"` is deliberate — it needs no API key. CARTO's dark basemap now
requires one and renders an "API KEY REQUIRED" watermark instead of streets.

### 4.2 Getting road geometry — OSRM

A straight line between two points is not a route. Roads come from OSRM, a free routing
engine, in `routing.py`:

```python
OSRM_URL = ("https://router.project-osrm.org/route/v1/driving/"
            "{fl},{fa};{tl},{ta}?overview=full&geometries=geojson")
```

Two details that bite people:

- **OSRM takes `lng,lat`. Leaflet takes `lat,lng`.** Hence the flip on the way in and
  `[(c[1], c[0]) for c in coordinates]` on the way out. Swapped coordinates put your
  driver in the North Sea.
- **`overview=full`** returns the complete polyline. The default is simplified and looks
  like it cuts corners through buildings.

Then a three-tier fallback, because demo day networks fail:

```python
1. try OSRM live (4s timeout)   -> cache the result to data/routes_cache.json
2. else use the cache
3. else a synthetic curved line (source="estimated")
```

`scripts/precompute_routes.py` warms the cache in advance so tier 2 is always available.

### 4.3 Moving the driver — interpolation by distance

The polyline is a list of points, unevenly spaced. To find "where is the driver at 40% of
the journey", you must measure by **distance travelled**, not by point index — otherwise
the marker crawls through dense city segments and teleports across sparse ones.

```python
def point_at(coords, t):
    target = path_km(coords) * t          # how far along, in km
    acc = 0.0
    for i in range(1, len(coords)):
        seg = haversine_km(coords[i-1], coords[i])
        if acc + seg >= target:           # target falls inside this segment
            f = (target - acc) / seg      # fraction into it
            return (coords[i-1][0] + (coords[i][0] - coords[i-1][0]) * f,
                    coords[i-1][1] + (coords[i][1] - coords[i-1][1]) * f)
        acc += seg
    return coords[-1]
```

Linear interpolation inside the containing segment. That single function is the whole
animation: the marker's position is always `point_at(leg["coords"], progress)`.

Real GPS would replace this function and nothing else.

### 4.4 Where `progress` comes from

Wall-clock time, not a counter:

```python
LEG1_SECONDS = 30.0
elapsed = time.time() - d["t0"]
progress = min(1.0, elapsed / LEG1_SECONDS)
```

Deriving from `time.time()` rather than incrementing a counter means the animation stays
correct no matter how irregular the reruns are. A dropped rerun causes one skipped frame,
not permanent drift.

### 4.5 Drawing the frame

```python
folium.PolyLine(d["legs"][0]["coords"], color="#8A8072", weight=4, dash_array="7,8")
folium.PolyLine(d["legs"][1]["coords"], color=BRAND, weight=5)
```

Dashed grey for the approach, solid orange for the delivery. Markers use `DivIcon`, which
lets you inject arbitrary HTML — that is how the square P/S badges and the round driver
puck are drawn without any image assets:

```python
folium.Marker(latlng, icon=DivIcon(icon_size=(34, 34), icon_anchor=(17, 17), html=(
    f'<div style="width:32px;height:32px;border-radius:7px;background:{colour};'
    f'border:2px solid #0E1013;display:grid;place-items:center;'
    f'font:700 10px/1 sans-serif;color:#0E1013">{label}</div>')))
```

`icon_anchor` is half of `icon_size` so the marker centres on its coordinate instead of
hanging below it.

Finally `m.fit_bounds(pts, padding=(40, 40))` keeps kitchen, shelter and driver all in
frame without hand-tuning zoom.

### 4.6 Idle state

Before dispatch, all 40 drivers are plotted with `CircleMarker` — green available, grey
busy. Once one accepts, the rest dim (`fill_opacity=0.5`) and the assigned driver is
excluded from the fleet layer so the animated marker is unambiguous. That was your
requirement: all drivers visible when idle, one tracked when assigned.

---

## 5 — The delivery state machine

One dict in session state holds everything:

```python
st.session_state.delivery = {
    "restaurant": {...}, "shelter": {...}, "safety": {...},
    "offers": [...], "winner": None,
    "phase": "broadcast",     # the state
    "t0": time.time(),        # when this phase started
    "legs": [leg1, leg2], "leg": 0,
}
```

```
broadcast ──accept (tap or 8s auto)──> to_pickup ──30s──> at_pickup
   ↑                                                          │
   └── no capable driver → idle          confirm or 9s auto ───┘
                                                              ↓
                                        to_shelter ──40s──> delivered
```

`advance()` is the whole clock. It reads `phase`, compares `time.time() - t0` against the
duration for that phase, calls the transition function when it expires, and returns
`progress` for the map:

```python
def advance():
    el = time.time() - d["t0"]
    if d["phase"] == "broadcast":
        if el >= OFFER_SECONDS: do_accept(d["offers"][0]["driver_id"])
        return 0.0
    if d["phase"] == "to_pickup":
        p = min(1.0, el / LEG1_SECONDS)
        if p >= 1.0: do_arrive(); return 1.0
        return p
    ...
```

Every transition does three things: update state, append a trace step, raise a
notification. `do_arrive()` is the one the judges watch:

```python
def do_arrive():
    d["phase"] = "at_pickup"
    d["t0"] = time.time()
    step("event · driver_arrived", "Driver at pickup point", ...)
    notify(f'{d["winner"]["name"]} has arrived',
           f'Waiting at {d["restaurant"]["name"]} — hand over the food and confirm.',
           "manager", BRAND)
```

Each phase has both a manual trigger (a button) and a timeout. Tap Accept and it advances
instantly; say nothing and it advances anyway. You cannot get stranded mid-demo.

---

## 6 — Notifications

`st.toast` cannot be called from a transition function that runs during a rerun and have
it survive — so notifications are stored, then flushed:

```python
def notify(title, body, role, accent=BRAND):
    st.session_state.notifs.insert(0, {...})       # newest first

def flush_toasts():
    n_new = len(st.session_state.notifs) - st.session_state.toast_cursor
    for n in reversed(st.session_state.notifs[:n_new]):
        st.toast(f"**{n['title']}**  \n{n['body']}", icon=...)
    st.session_state.toast_cursor = len(st.session_state.notifs)
```

`toast_cursor` is a high-water mark. Without it, every rerun re-toasts the entire history
once a second. The same list also renders as the persistent Alerts page.

---

## 7 — File by file

| File | Responsibility |
|---|---|
| `app.py` | All UI, the state machine, the map builder. The only file that imports `streamlit`. |
| `tools.py` | The 7 `@tool` functions and the JSON read/write helpers. No Streamlit import — testable standalone. |
| `routing.py` | OSRM calls, route cache, `haversine_km`, `path_km`, `point_at`. Pure functions, no I/O beyond the cache file. |
| `.streamlit/config.toml` | Theme. Streamlit reads it automatically; no code references it. |
| `data/*.json` | The world. Read on load, written by `accept_rescue` / `release_driver`. |
| `scripts/precompute_routes.py` | Warms the route cache. Run once before a demo. |
| `docs/architecture.svg` | The diagram Devpost requires. |

That separation is deliberate: `tools.py` and `routing.py` have no Streamlit dependency,
so you can exercise the entire agent from a plain Python REPL:

```python
python -c "from tools import analyze_food_safety; print(analyze_food_safety('6 kg hot curry'))"
```

---

## 8 — Adding a feature, worked example

Say you want a live FSA countdown — minutes left before the food is no longer safe.

**1. Where does the number come from?** `d["safety"]["fsa_window_minutes"]` and the moment
of collection. Store that moment when the handover happens, in `do_handover()`:

```python
d["collected_at"] = time.time()
```

**2. Compute it where it is displayed** — inside the fragment, so it recomputes every
second:

```python
if d.get("collected_at"):
    used = (time.time() - d["collected_at"]) / 60
    left = d["safety"]["fsa_window_minutes"] - used
```

**3. Render it** next to the ETA HUD, colour-coded:

```python
colour = GREEN if left > 30 else (AMBER if left > 10 else RED)
st.markdown(f'<div class="ra-hud"><div class="ra-label">FSA window</div>'
            f'<div class="v" style="color:{colour}">{int(left)}</div></div>',
            unsafe_allow_html=True)
```

**4. Raise an alert on the threshold.** Guard with a flag so it fires once, not every
second:

```python
if left < 10 and not d.get("warned"):
    d["warned"] = True
    notify("FSA window closing", f"{int(left)} min left", "manager", RED)
```

That pattern — store the origin time, derive in the fragment, guard one-shot side effects
with a flag — covers most features you will want to add.

---

## 9 — Running it locally

```bash
cd rescueagent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
aws configure                                          # key, secret, region eu-west-2
```

In the AWS console: **Bedrock → Model access** in `eu-west-2`, request access to
`qwen.qwen3-235b-a22b-2507-v1:0`. Access is per-region — enabling it in `us-east-1` does
nothing for you.

Warm the route cache (optional, recommended before a demo):

```bash
python scripts/precompute_routes.py --light
```

Run:

```bash
streamlit run app.py        # http://localhost:8501
```

Check the agent alone, without the UI:

```bash
python test_agent.py
```

---

## 10 — Deployment

### 10.1 GitHub first

Devpost requires a public repo with a detectable licence.

```bash
cd rescueagent
git init
git add .
git commit -m "RescueAgent — Strands agent for Edinburgh food rescue"
git branch -M main
git remote add origin https://github.com/<you>/rescueagent.git
git push -u origin main
```

Confirm on the repo page that the sidebar **About** section shows "MIT licence". If it
does not, the `LICENSE` file is not at the repo root — GitHub only detects it there.

Add a `.gitignore` before committing:

```
.venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
```

`secrets.toml` must never be committed. That is your AWS keys.

### 10.2 Streamlit Community Cloud

Free, and the fastest way to get the live demo link that scores you extra points.

1. Go to **share.streamlit.io** and sign in with GitHub.
2. **New app** → pick your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:

```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_DEFAULT_REGION = "eu-west-2"
```

4. **Deploy.** First build takes a few minutes while it installs `requirements.txt`.

`_wire_aws_credentials()` in `app.py` copies those secrets into environment variables
before the agent is built, which is how boto3 finds them — the cloud container has no
`~/.aws/credentials`.

**Create a dedicated IAM user for this**, not your root keys, with one narrow policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
    "Resource": "arn:aws:bedrock:eu-west-2::foundation-model/qwen.qwen3-235b-a22b-2507-v1:0"
  }]
}
```

If those keys leak, the worst case is someone runs one model in one region.

### 10.3 Things that break on first deploy

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: streamlit_folium` | Missing from `requirements.txt`. |
| App loads, agent errors on send | Secrets not set, or model access not requested in `eu-west-2`. |
| Routes all show `source: estimated` | Cloud egress to OSRM blocked or rate-limited. Commit `data/routes_cache.json` so tier 2 works. |
| Driver status resets | `accept_rescue` writes to `data/drivers.json`; cloud filesystems are ephemeral. Fine for a demo, but do not rely on persistence. |
| Slow, stuttering map | Another `st_folium` without `returned_objects=[]`. |

### 10.4 Note on the tracking demo

Streamlit redraws the map about once a second. If you are presenting on a projector and
want smoother motion, record that portion from the HTML prototype, which animates at
about 60 fps, and use Streamlit for the agent and Bedrock story. Both show the same
logic.

---

## 11 — Debugging

**Print the state.** Add to the sidebar while developing:

```python
with st.sidebar:
    st.write(st.session_state.delivery)
```

**Watch the terminal.** `streamlit run` prints every exception and traceback there.

**Test tools without the model.** They are plain functions returning JSON strings:

```python
python -c "from tools import broadcast_rescue; print(broadcast_rescue('shelter_01', True, True, 6.0, 55.9533, -3.1883))"
```

**If a tool is never called**, the docstring is almost always the problem. The model
chooses tools by reading docstrings and type hints. Rewrite it to say plainly what the
tool does and when to use it.

**If the map is blank**, check the tile URL in the browser's network tab. A 200 response
is not proof the tile has map content — a keyed CDN returns 200 with a watermark image.

**If the marker jumps**, you probably swapped lat/lng, or you are indexing points instead
of interpolating by distance.
