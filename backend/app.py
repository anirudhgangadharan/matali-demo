"""
Matali Demo Backend — D-3
FastAPI + WebSocket. Palantir-grade ops console.

Changes from D1.5:
- Real HMM (backend/hmm.py) replaces hardcoded SDM/rejection list
- Hospital coords now match the math source (hmm.py default_hospitals)
- Pre-seeded GIMS dispatch_count=4 → SDM legitimately wins for red TBI
- Slowed rejection cadence (1.0s/hosp) so judges can read each
- Transcript timing driven by actual MP3 duration (no drift vs caller audio)
- Scenario picker scaffolding (still default = red TBI for now; T5 wires UI)
"""
import asyncio
import time
import uuid
import wave
import contextlib
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from hmm import (
    EmergencyCall, default_hospitals, select_hospital, reason_text,
    haversine_km, estimate_eta,
)

app = FastAPI(title="Matali Ops")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
AUDIO_DIR = FRONTEND / "audio"
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
AUDIO_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


@app.get("/")
async def root():
    return RedirectResponse("/caller")

@app.get("/caller")
async def caller_page():
    return FileResponse(str(FRONTEND / "caller.html"))

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(str(FRONTEND / "dashboard.html"))


# ─────────────────────────────────────────────────────────
# Audio duration probe — drives transcript timing
# ─────────────────────────────────────────────────────────
def mp3_duration_seconds(path: Path) -> float:
    """Approximate MP3 duration. Uses mutagen if available, else file-size heuristic."""
    try:
        from mutagen.mp3 import MP3
        return float(MP3(str(path)).info.length)
    except Exception:
        # Heuristic: gTTS/edge-tts MP3s are ~24kbps avg → 3 KB ≈ 1 second
        size = path.stat().st_size
        return max(size / 3000.0, 1.0)


# ─────────────────────────────────────────────────────────
# Scenarios — each fully drives one demo run
# ─────────────────────────────────────────────────────────
SCENARIOS = {
    "red_tbi": {
        "label": "Red · Head trauma · NH-67",
        "accident": {"coords": [15.4150, 75.3000], "label": "NH-67 mid-stretch"},
        "transcript_hi": ("हेलो! यहाँ accident हो गया है! गदग highway पर! "
                          "आदमी bike से गिरा है, सिर से खून निकल रहा है, "
                          "होश में नहीं है! जल्दी आओ भाई!"),
        "transcript_en": ("Hello! There's been an accident! On Gadag highway! "
                          "Man fell from his bike, blood from his head, "
                          "not conscious! Come fast brother!"),
        "extracted": {
            "injury_type": "head", "severity": "red",
            "mechanism": "motorcycle fall",
            "vitals": ["active bleeding", "altered consciousness"],
        },
        "audio_user": "caller_scripted.mp3",
        "audio_firstaid": "ai_firstaid.mp3",
        "firstaid_hi": ("गर्दन मत हिलाओ। साफ कपड़े से सिर के घाव पर दबाव डालो। "
                        "अगर उल्टी हो तो सिर एक तरफ कर दो। लाइन पर बने रहो।"),
        "firstaid_en": ("Don't move neck. Firm pressure on head wound with clean cloth. "
                        "Turn head sideways if vomiting. Stay on the line."),
        # Pre-seeded hospital state — gives HMM a realistic starting load
        "preseed": [("GIMS", 4, [0.40, 0.40, 0.20])],
        "ambulance_speed": 60,
        "expected_winner": "SDM",
    },
}


# ─────────────────────────────────────────────────────────
# Connection manager
# ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.dashboards: list[WebSocket] = []
        self.callers: list[WebSocket] = []

    async def connect(self, ws: WebSocket, kind: str):
        await ws.accept()
        (self.dashboards if kind == "dashboard" else self.callers).append(ws)

    def disconnect(self, ws: WebSocket, kind: str):
        pool = self.dashboards if kind == "dashboard" else self.callers
        if ws in pool:
            pool.remove(ws)

    async def broadcast(self, msg: dict):
        payload = {**msg, "t": time.time()}
        dead = []
        for ws in self.dashboards:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.dashboards.remove(ws)

    async def send_to_caller(self, ws: WebSocket, msg: dict):
        try:
            await ws.send_json({**msg, "t": time.time()})
        except Exception:
            pass


mgr = ConnectionManager()


async def log_event(channel: str, text: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    await mgr.broadcast({
        "stage": "log",
        "ts": ts,
        "channel": channel,
        "text": text,
        "level": level,
    })


def hospital_to_payload(h) -> dict:
    """Convert HMM Hospital to JSON-safe dict for the dashboard."""
    return {
        "id": h.id, "name": h.name, "short": h.short,
        "coords": [h.lat, h.lng], "tier": h.tier, "beds": h.beds,
        "dispatch_count": h.dispatch_count,
    }


# ─────────────────────────────────────────────────────────
# Pipeline — driven by scenario name
# ─────────────────────────────────────────────────────────
async def run_pipeline(caller_ws: WebSocket, scenario_id: str = "red_tbi"):
    scen = SCENARIOS.get(scenario_id, SCENARIOS["red_tbi"])

    t0 = time.time()
    def el(): return round(time.time() - t0, 2)

    incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

    # Build hospitals + apply preseed for this scenario
    hospitals = default_hospitals()
    for hid, dispatch_count, sv in scen.get("preseed", []):
        for h in hospitals:
            if h.id == hid:
                h.dispatch_count = dispatch_count
                h.state_vector = np.array(sv)

    # Probe scripted-audio durations once per run
    user_audio_path = AUDIO_DIR / scen["audio_user"]
    user_audio_duration = mp3_duration_seconds(user_audio_path) if user_audio_path.exists() else 10.5

    firstaid_audio_path = AUDIO_DIR / scen["audio_firstaid"]
    firstaid_audio_duration = mp3_duration_seconds(firstaid_audio_path) if firstaid_audio_path.exists() else 5.0

    # ── Call received ────────────────────────────────────
    await mgr.broadcast({
        "stage": "call_received",
        "elapsed": el(),
        "incident_id": incident_id,
        "scenario": scenario_id,
        "scenario_label": scen["label"],
        "accident": scen["accident"],
        "hospitals": [hospital_to_payload(h) for h in hospitals],
    })
    await log_event("COMMS", f"Incoming call · 1800-MATALI · scenario={scenario_id}", "info")
    await log_event("GEO", f"Caller coords acquired: {scen['accident']['coords']}", "info")
    await asyncio.sleep(0.3)

    # ── AI greets caller ─────────────────────────────────
    greet = {
        "stage": "ai_speak", "clip": "ai_greeting",
        "hi": "मातली सुन रहा है। क्या हुआ, बताइए।",
        "en": "Matali listening. Please tell me what happened.",
    }
    await mgr.broadcast(greet)
    await mgr.send_to_caller(caller_ws, greet)
    await log_event("TTS.hi-IN", "Greeting delivered to caller", "info")
    await asyncio.sleep(2.4)

    # ── ASR streams the user's Hindi (timing driven by audio duration) ──
    await log_event("ASR", "Voice activity detected · streaming transcription", "info")
    await mgr.send_to_caller(caller_ws, {
        "stage": "user_speaking",
        "audio_clip": scen["audio_user"],
    })

    # Tell dashboard to start displaying transcript progressively
    words = scen["transcript_hi"].split()
    # Reserve ~0.4s for caller audio startup latency on the wire; fit the rest to audio length.
    streaming_window = max(user_audio_duration - 0.4, 4.0)
    per_word = streaming_window / len(words)
    partial = ""
    for w in words:
        partial = (partial + " " + w).strip()
        await mgr.broadcast({"stage": "transcript_partial", "hi": partial})
        await asyncio.sleep(per_word)

    await mgr.broadcast({
        "stage": "transcript_final",
        "hi": scen["transcript_hi"],
        "en": scen["transcript_en"],
        "elapsed": el(),
    })
    await log_event("ASR", f"Utterance final · conf=0.94 · {len(scen['transcript_hi'])} chars", "ok")
    await asyncio.sleep(0.4)

    # ── NLU structured extraction ───────────────────────
    await log_event("NLU", "Running injury/severity/location extraction (LLM)", "info")
    await asyncio.sleep(0.7)
    extracted = {
        "location": scen["accident"]["label"],
        "coords": scen["accident"]["coords"],
        "injury_type": scen["extracted"]["injury_type"],
        "severity":    scen["extracted"]["severity"],
        "mechanism":   scen["extracted"]["mechanism"],
        "vitals":      scen["extracted"]["vitals"],
    }
    await mgr.broadcast({"stage": "extracted", "data": extracted, "elapsed": el()})
    await log_event(
        "NLU",
        f"injury={extracted['injury_type']} · severity={extracted['severity'].upper()} · mech={extracted['mechanism']}",
        "ok",
    )
    await asyncio.sleep(0.3)

    # ── Run real HMM ────────────────────────────────────
    call_obj = EmergencyCall(
        call_id=1,
        lat=scen["accident"]["coords"][0],
        lng=scen["accident"]["coords"][1],
        injury_type=extracted["injury_type"],
        severity=extracted["severity"],
    )
    hmm_result = select_hospital(call_obj, hospitals=hospitals)

    # ── Four parallel threads fire ──────────────────────
    await mgr.broadcast({"stage": "threads_start", "elapsed": el()})
    await log_event("ORCHESTRATOR", "Firing 4 parallel threads", "warn")

    async def thread_route():
        await mgr.broadcast({"stage": "thread_running", "thread": "route"})
        await log_event("HMM", f"Computing utility vectors for {len(hospitals)} candidates...", "info")
        await asyncio.sleep(0.6)  # was 0.4

        winner = hmm_result["selected"]
        rejected_ranked = [u for u in hmm_result["all_ranked"]
                           if u["hospital_id"] != winner["hospital_id"]]
        # Iterate worst→best so the dim animation feels like elimination
        rejected_ranked = list(reversed(rejected_ranked))

        for u in rejected_ranked:
            await asyncio.sleep(1.2)  # was 0.35 — slowed for readability + narration time
            reason = reason_text(u, winner)
            await mgr.broadcast({
                "stage": "hospital_rejected",
                "hospital_id": u["hospital_id"],
                "reason": reason,
                "utility": u["utility"],
            })
            await log_event("HMM", f"{u['hospital_id']} rejected · {reason}", "info")

        await asyncio.sleep(0.7)  # was 0.3 — beat before winner announcement

        await mgr.broadcast({
            "stage": "hospital_selected",
            "hospital_id": winner["hospital_id"],
            "utility": winner["utility"],
            "p_available": winner["p_available"],
            "low_confidence": hmm_result["low_confidence"],
        })
        await log_event(
            "HMM",
            f"{winner['hospital_id']} selected · U={winner['utility']:+.3f} · "
            f"P(Avail)={winner['p_available']:.2f}",
            "ok",
        )
        await asyncio.sleep(0.2)

        # Compute ambulance speed-aware ETA from accident → selected hospital
        eta_min = winner["eta_min"]
        await mgr.broadcast({
            "stage": "thread_done", "thread": "route", "elapsed": el(),
            "result": {
                "hospital":     winner["hospital"],
                "hospital_id":  winner["hospital_id"],
                "eta_min":      round(eta_min, 1),
                "utility":      winner["utility"],
                "p_available":  winner["p_available"],
                "capability":   winner["capability"],
                "tier":         winner["tier"],
                "reason":       f"Tier-{winner['tier']} · C={winner['capability']:.2f} · P(Avail)={winner['p_available']:.2f}",
                "low_confidence": hmm_result["low_confidence"],
            },
        })

    async def thread_firstaid():
        await mgr.broadcast({"stage": "thread_running", "thread": "firstaid"})
        await log_event("LLM", "Generating first-aid protocol...", "info")
        await asyncio.sleep(1.2)
        result = {
            "hi": scen["firstaid_hi"],
            "en": scen["firstaid_en"],
        }
        await mgr.broadcast({
            "stage": "thread_done", "thread": "firstaid",
            "elapsed": el(), "result": result,
        })
        await log_event("LLM", "First-aid protocol delivered", "ok")

    async def thread_dispatch():
        await mgr.broadcast({"stage": "thread_running", "thread": "dispatch"})
        await log_event("EMRI.108", "Dispatching nearest unit to scene...", "info")
        await asyncio.sleep(1.9)
        await mgr.broadcast({
            "stage": "thread_done", "thread": "dispatch", "elapsed": el(),
            "result": {
                "unit_id": "KA-18-7291",
                "eta_pickup_min": 9,
                "crew": "EMT + Paramedic",
                "dispatched_from": "Gadag 108 Base",
            },
        })
        await log_event("EMRI.108", "KA-18-7291 acknowledged · ETA 9 min to scene", "ok")

    async def thread_eralert():
        await mgr.broadcast({"stage": "thread_running", "thread": "eralert"})
        await log_event("HL7", "Opening handoff channel to receiving ER...", "info")
        await asyncio.sleep(1.6)
        await mgr.broadcast({
            "stage": "thread_done", "thread": "eralert", "elapsed": el(),
            "result": {
                "handoff_sent": True,
                "trauma_team_activated": True,
                "blood_preordered": "O-neg · 2 units",
                "ct_scanner_reserved": True,
            },
        })
        await log_event("HL7", "Trauma team activated · CT reserved · 2U O-neg", "ok")

    await asyncio.gather(
        thread_route(), thread_firstaid(), thread_dispatch(), thread_eralert()
    )

    # ── AI speaks acknowledgment ────────────────────────
    await asyncio.sleep(0.3)
    ack = {
        "stage": "ai_speak", "clip": "ai_ack",
        "hi": "समझ गया। ambulance भेज दी है। घायल के पास रहिए।",
        "en": "Understood. Ambulance dispatched. Stay with the victim.",
    }
    await mgr.broadcast(ack)
    await mgr.send_to_caller(caller_ws, ack)
    await log_event("TTS.hi-IN", "Acknowledgment delivered to caller", "info")
    await asyncio.sleep(3.0)

    # ── AI speaks first-aid ─────────────────────────────
    fa = {
        "stage": "ai_speak", "clip": "ai_firstaid",
        "hi": scen["firstaid_hi"],
        "en": scen["firstaid_en"],
    }
    await mgr.broadcast(fa)
    await mgr.send_to_caller(caller_ws, fa)
    await log_event("TTS.hi-IN", "First-aid instructions delivered", "info")
    # Wait for the firstaid clip to finish playing (no abrupt cut-off)
    await asyncio.sleep(firstaid_audio_duration + 0.3)

    # ── Complete ────────────────────────────────────────
    await mgr.broadcast({"stage": "complete", "elapsed": el()})
    await log_event("SYS", "Incident handoff complete · session logged", "ok")


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await mgr.connect(ws, "dashboard")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(ws, "dashboard")


@app.websocket("/ws/caller")
async def ws_caller(ws: WebSocket):
    await mgr.connect(ws, "caller")
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "call_start":
                scenario_id = msg.get("scenario", "red_tbi")
                asyncio.create_task(run_pipeline(ws, scenario_id))
    except WebSocketDisconnect:
        mgr.disconnect(ws, "caller")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
