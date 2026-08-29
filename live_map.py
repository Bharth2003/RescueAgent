"""Uber-style live tracking map, built as a Streamlit custom component (CCv2).

Why a custom component instead of st_folium
-------------------------------------------
st_folium rebuilds the whole Leaflet map on every rerun, so a 1 s tracking
fragment made the map visibly blink and the driver marker jump. Here the map is
created once and kept in a WeakMap keyed by its host element; Python hands the
component the *whole journey plan* (route geometry, start time, duration) and
the browser animates the driver at 60 fps with requestAnimationFrame. Reruns
only change the plan on real phase transitions, so the map never remounts and
the motion stays smooth.

Tiles are plain OpenStreetMap with a CSS filter for the dark look — no API key.
"""

import streamlit as st

_HTML = """
<div id="ra-wrap">
  <div id="ra-map"></div>
  <div id="ra-hud" class="ra-glass">
    <div id="ra-hud-label">Standby</div>
    <div id="ra-hud-value">—</div>
    <div id="ra-hud-sub"></div>
  </div>
  <div id="ra-legend" class="ra-glass"></div>
  <button id="ra-recenter" type="button">Re-centre</button>
</div>
"""

_CSS = """
#ra-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 380px;
  border-radius: 16px;
  overflow: hidden;
  background: #10131a;
  font-family: var(--st-font, system-ui, sans-serif);
}
#ra-map { position: absolute; inset: 0; }

/* dark basemap without needing a tile API key */
#ra-map .leaflet-tile-pane {
  filter: invert(1) grayscale(.86) brightness(.94) contrast(1.06);
}

.ra-glass {
  position: absolute;
  z-index: 600;
  background: rgba(16, 19, 26, .86);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, .10);
  border-radius: 14px;
  color: #F2F4F7;
  box-shadow: 0 8px 28px rgba(0, 0, 0, .45);
}
#ra-hud { left: 14px; top: 14px; padding: 12px 18px; min-width: 172px; }
#ra-hud-label {
  font-size: 11.5px; font-weight: 700; letter-spacing: .10em;
  text-transform: uppercase; color: #9AA3B2;
}
#ra-hud-value { font-size: 34px; font-weight: 700; line-height: 1.12; margin-top: 3px; }
#ra-hud-value span { font-size: 16px; font-weight: 600; color: #9AA3B2; margin-left: 3px; }
#ra-hud-sub { font-size: 13px; color: #B6BECC; margin-top: 3px; }

#ra-legend { right: 14px; top: 14px; padding: 9px 14px; font-size: 12.5px; line-height: 1.85; }
#ra-legend i {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 8px; vertical-align: middle;
}

#ra-recenter {
  position: absolute; right: 14px; bottom: 16px; z-index: 600;
  background: rgba(16, 19, 26, .86); color: #F2F4F7; cursor: pointer;
  border: 1px solid rgba(255, 255, 255, .14); border-radius: 11px;
  padding: 9px 15px; font-size: 13px; font-weight: 600;
  backdrop-filter: blur(10px); opacity: 0; transition: opacity .18s;
}
#ra-recenter.show { opacity: 1; }

/* driver puck */
.ra-puck { position: relative; width: 42px; height: 42px; }
.ra-puck .ring {
  position: absolute; inset: 0; border-radius: 50%;
  background: var(--brand, #FF7A2F); opacity: .30;
  animation: ra-pulse 2s ease-out infinite;
}
@keyframes ra-pulse {
  0%   { transform: scale(.55); opacity: .55; }
  70%  { transform: scale(1.5);  opacity: 0;   }
  100% { transform: scale(1.5);  opacity: 0;   }
}
.ra-puck .body {
  position: absolute; left: 7px; top: 7px; width: 28px; height: 28px;
  border-radius: 50%; background: var(--brand, #FF7A2F);
  border: 2.5px solid #10131a; box-shadow: 0 3px 12px rgba(0, 0, 0, .6);
  display: grid; place-items: center;
  transition: transform .35s linear;
}
.ra-puck .body svg { width: 15px; height: 15px; display: block; }

/* pickup / dropoff pins */
.ra-pin {
  width: 30px; height: 30px; border-radius: 9px;
  border: 2.5px solid #10131a; box-shadow: 0 3px 12px rgba(0, 0, 0, .55);
  display: grid; place-items: center;
  font: 700 11px/1 system-ui, sans-serif; color: #10131a;
}
.ra-flag { font-size: 12px; }
"""

_JS = """
const REG = new WeakMap()

function leaflet() {
  if (window.L) return Promise.resolve(window.L)
  if (!window.__raLeaflet) {
    window.__raLeaflet = new Promise((resolve, reject) => {
      const css = document.createElement("link")
      css.rel = "stylesheet"
      css.href = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"
      document.head.appendChild(css)
      const s = document.createElement("script")
      s.src = "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"
      s.onload = () => resolve(window.L)
      s.onerror = reject
      document.head.appendChild(s)
    })
  }
  return window.__raLeaflet
}

const R = 6371
const rad = (d) => d * Math.PI / 180
function segKm(a, b) {
  const dLat = rad(b[0] - a[0]), dLng = rad(b[1] - a[1])
  const h = Math.sin(dLat / 2) ** 2 +
            Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(h))
}
// cumulative distance table so motion is even along the road, not per-vertex
function measure(coords) {
  const cum = [0]
  for (let i = 1; i < coords.length; i++) cum.push(cum[i - 1] + segKm(coords[i - 1], coords[i]))
  return cum
}
function atFraction(coords, cum, f) {
  const total = cum[cum.length - 1]
  if (!total) return { pt: coords[0], bearing: 0, idx: 0 }
  const target = total * Math.max(0, Math.min(1, f))
  let lo = 0, hi = cum.length - 1
  while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (cum[mid] <= target) lo = mid; else hi = mid }
  const a = coords[lo], b = coords[Math.min(lo + 1, coords.length - 1)]
  const span = cum[lo + 1] - cum[lo]
  const t = span ? (target - cum[lo]) / span : 0
  const pt = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
  const bearing = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI
  return { pt, bearing, idx: lo }
}

const CAR = '<svg viewBox="0 0 24 24" fill="#10131a"><path d="M5 11l1.5-4.5A2 2 0 018.4 5h7.2a2 2 0 011.9 1.5L19 11v7h-2.5v-1.5h-9V18H5v-7zm2.6-.5h8.8l-1-3H8.6l-1 3zM7.75 15a1.25 1.25 0 100-2.5 1.25 1.25 0 000 2.5zm8.5 0a1.25 1.25 0 100-2.5 1.25 1.25 0 000 2.5z"/></svg>'

export default function (component) {
  const { data, parentElement } = component
  const host = parentElement.querySelector("#ra-map")
  if (!host) return

  const hudLabel = parentElement.querySelector("#ra-hud-label")
  const hudValue = parentElement.querySelector("#ra-hud-value")
  const hudSub   = parentElement.querySelector("#ra-hud-sub")
  const legend   = parentElement.querySelector("#ra-legend")
  const recenter = parentElement.querySelector("#ra-recenter")
  const brand = data.brand || "#FF7A2F"

  leaflet().then((L) => {
    let S = REG.get(host)

    if (!S) {
      const map = L.map(host, {
        zoomControl: false, attributionControl: false,
        zoomSnap: 0.25, preferCanvas: true,
      }).setView([55.9490, -3.1900], 12.5)
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map)
      L.control.attribution({ prefix: false, position: "bottomleft" })
        .addAttribution("&copy; OpenStreetMap").addTo(map)

      S = {
        map, raf: null, sig: null, plan: null, follow: true,
        layers: L.layerGroup().addTo(map),
        lastLine: 0,
      }
      REG.set(host, S)

      // let the user take over the camera; offer a way back
      map.on("dragstart zoomstart", () => {
        if (!S.programmatic) { S.follow = false; recenter.classList.add("show") }
      })
      recenter.onclick = () => {
        S.follow = true
        recenter.classList.remove("show")
        if (S.fitAll) S.fitAll()
      }
      const ro = new ResizeObserver(() => map.invalidateSize())
      ro.observe(host)
      setTimeout(() => map.invalidateSize(), 80)
    }

    const L2 = L
    const sig = JSON.stringify([data.phase, data.t0, data.dur,
                                (data.route || []).length, (data.done_route || []).length,
                                (data.fleet || []).length])

    if (S.sig !== sig) {
      S.sig = sig
      S.layers.clearLayers()
      S.plan = data
      S.cum = null
      S.driver = null

      // idle: show the fleet spread across the city
      for (const f of (data.fleet || [])) {
        L2.circleMarker([f[0], f[1]], {
          radius: f[2] ? 5 : 4, weight: 1.5, color: "#10131a",
          fillColor: f[2] ? "#34D399" : "#7D8794",
          fillOpacity: data.route && data.route.length ? 0.35 : 0.95,
        }).addTo(S.layers)
      }

      // the leg already completed, drawn faint
      if ((data.done_route || []).length > 1) {
        L2.polyline(data.done_route, {
          color: "#5C6675", weight: 4, opacity: .55, dashArray: "6,9",
        }).addTo(S.layers)
      }

      // active leg: dark casing + bright line, with a "travelled" overlay on top
      if ((data.route || []).length > 1) {
        L2.polyline(data.route, { color: "#0B0E14", weight: 11, opacity: .85,
                                  lineJoin: "round", lineCap: "round" }).addTo(S.layers)
        S.routeLine = L2.polyline(data.route, { color: brand, weight: 5.5, opacity: .95,
                                                lineJoin: "round", lineCap: "round" }).addTo(S.layers)
        S.trailLine = L2.polyline([], { color: "#FFFFFF", weight: 5.5, opacity: .30,
                                        lineJoin: "round", lineCap: "round" }).addTo(S.layers)
        S.cum = measure(data.route)
      }

      const pin = (p, color, label, title) => {
        if (!p) return null
        return L2.marker([p.lat, p.lng], {
          title: title || "",
          icon: L2.divIcon({
            className: "", iconSize: [30, 30], iconAnchor: [15, 15],
            html: `<div class="ra-pin" style="background:${color}">${label}</div>`,
          }),
        }).addTo(S.layers)
      }
      pin(data.pickup, brand, "P", data.pickup && data.pickup.name)
      pin(data.shelter, "#A78BFA", "S", data.shelter && data.shelter.name)

      if ((data.route || []).length > 1) {
        S.driver = L2.marker(data.route[0], {
          zIndexOffset: 1000,
          icon: L2.divIcon({
            className: "", iconSize: [42, 42], iconAnchor: [21, 21],
            html: `<div class="ra-puck" style="--brand:${brand}">
                     <div class="ring"></div>
                     <div class="body">${CAR}</div>
                   </div>`,
          }),
        }).addTo(S.layers)
        S.puckBody = null
      }

      // what the camera should frame when "Re-centre" is pressed
      S.fitAll = () => {
        const pts = []
        if ((data.route || []).length) pts.push(...data.route)
        if (data.pickup) pts.push([data.pickup.lat, data.pickup.lng])
        if (data.shelter) pts.push([data.shelter.lat, data.shelter.lng])
        if (!pts.length && (data.fleet || []).length) {
          for (const f of data.fleet) pts.push([f[0], f[1]])
        }
        if (pts.length > 1) {
          S.programmatic = true
          S.map.fitBounds(L2.latLngBounds(pts), { padding: [55, 55], animate: true })
          setTimeout(() => { S.programmatic = false }, 700)
        }
      }
      S.follow = true
      recenter.classList.remove("show")
      S.fitAll()

      legend.innerHTML = (data.legend || [])
        .map((x) => `<div><i style="background:${x[1]}"></i>${x[0]}</div>`).join("")
      legend.style.display = (data.legend || []).length ? "block" : "none"
    }

    // ---- animation loop: owns the motion, independent of Python reruns ----
    if (S.raf) cancelAnimationFrame(S.raf)

    const fmtMin = (m) => (m < 1 ? "<1" : String(Math.round(m)))

    const frame = () => {
      const p = S.plan || {}
      const hasRoute = (p.route || []).length > 1 && S.cum && S.driver
      // `hold` parks the driver at a fixed fraction (waiting at the kitchen,
      // or delivered) instead of animating along the leg.
      const held = p.hold !== null && p.hold !== undefined
      let f = null
      if (hasRoute) {
        if (held) f = p.hold
        else if (p.dur > 0) f = Math.max(0, Math.min(1, ((Date.now() / 1000) - p.t0) / p.dur))
      }

      if (hasRoute && f !== null) {
        const { pt, bearing } = atFraction(p.route, S.cum, f)

        S.driver.setLatLng(pt)
        if (!S.puckBody) {
          const el = S.driver.getElement()
          if (el) S.puckBody = el.querySelector(".body")
        }
        if (S.puckBody) S.puckBody.style.transform = `rotate(${bearing}deg)`

        // grow the travelled overlay (throttled - it is a whole polyline rewrite)
        const now = performance.now()
        if (S.trailLine && now - S.lastLine > 90) {
          S.lastLine = now
          const cut = []
          const total = S.cum[S.cum.length - 1]
          const target = total * f
          for (let i = 0; i < p.route.length; i++) {
            if (S.cum[i] <= target) cut.push(p.route[i]); else break
          }
          cut.push(pt)
          S.trailLine.setLatLngs(cut)
        }

        if (S.follow && !held) {
          S.programmatic = true
          S.map.panTo(pt, { animate: true, duration: .55, easeLinearity: .5, noMoveStart: true })
          setTimeout(() => { S.programmatic = false }, 60)
        }
      }

      if (hasRoute && !held && p.dur > 0) {
        const totalKm = S.cum[S.cum.length - 1]
        hudLabel.textContent = p.label || "En route"
        hudValue.innerHTML = fmtMin((p.eta_min || 0) * (1 - f)) + "<span>min</span>"
        hudSub.textContent = (totalKm * (1 - f)).toFixed(1) + " km left of " +
                             totalKm.toFixed(1) + " km"
      } else {
        hudLabel.textContent = p.label || "Standby"
        hudValue.innerHTML = p.value || "&mdash;"
        hudSub.textContent = p.sub || ""
      }
      S.raf = requestAnimationFrame(frame)
    }
    frame()
  })

  return () => {
    const S = REG.get(host)
    if (S && S.raf) cancelAnimationFrame(S.raf)
  }
}
"""

_component = st.components.v2.component(
    "rescueagent_live_map", html=_HTML, css=_CSS, js=_JS, isolate_styles=False
)


def live_map(plan: dict, *, height: int = 460, key: str = "ra_live_map"):
    """Render the tracking map.

    `plan` carries the entire journey so the browser can animate it unaided:
    phase, t0 (epoch seconds the phase began), dur, eta_min, the active `route`
    polyline, the already-finished `done_route`, pickup/shelter pins, and the
    idle `fleet` dots.
    """
    return _component(data=plan, height=height, key=key)
