# Matali Demo — D1.5 

End-to-end Hindi emergency-call pipeline. Mocked transcript + NLU + HMM.
Real map (Gadag geography, Leaflet + CartoDB dark tiles).
Real AI voice (Microsoft Edge TTS, hi-IN neural).
Real websockets. Real parallel orchestration.

## Setup (Windows)

```powershell
cd matali-demo
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Hindi audio (one-time, needs internet)

```powershell
python generate_audio.py
```

Creates 4 MP3s in `frontend/audio/`:
- `ai_greeting.mp3` — AI answering call
- `ai_ack.mp3` — AI confirming dispatch
- `ai_firstaid.mp3` — AI reading first-aid instructions
- `caller_scripted.mp3` — scripted bystander voice

## Run

```powershell
python backend\app.py
```

## Test

Open two browser tabs:

- **Caller:** http://localhost:8000/caller
- **Dashboard:** http://localhost:8000/dashboard

Click the phone button on the caller page. Watch dashboard animate:

1. Transcript streams word-by-word
2. Structured extraction appears
3. Four threads fire in parallel, each completes with realistic delay
4. Summary ribbon shows total time

First-aid instructions appear on caller phone UI.

## Structure

```
matali-demo/
├── backend/app.py          # FastAPI + WebSocket + mock pipeline
├── frontend/
│   ├── caller.html         # Phone UI — mic button
│   └── dashboard.html      # Operator view — 4 parallel threads
└── requirements.txt
```

## D1.5 Checkpoint

- [x] Palantir-grade ops console (IBM Plex Mono/Sans, signal-orange accent, 1px borders)
- [x] Leaflet map with Gadag geography, 5 real hospitals, CartoDB dark tiles
- [x] Hospital rejection stream (pins dim as HMM rules them out)
- [x] Selected hospital animates route from accident site
- [x] Hindi transcript streaming (Devanagari) + English translation
- [x] Two-way AI voice — greeting, acknowledgment, first-aid (Edge TTS hi-IN neural)
- [x] Timestamped event log ribbon (COMMS/ASR/NLU/HMM/LLM/EMRI/HL7/SYS channels)
- [x] JSON-style extraction payload panel
- [x] 4 parallel threads with running/done states
- [x] Summary ribbon: total time, hospital, ambulance ETA, 108 baseline, time saved

## Next (D2)

Swap scripted Hindi transcript for real Moonshine ASR on live mic audio.
