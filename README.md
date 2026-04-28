# Matali Demo — D3

End-to-end Hindi emergency-call pipeline. Real HMM routing (matali.py ported to backend). Real map (Gadag geography, Leaflet + CartoDB dark tiles). Real AI voice (Microsoft Edge TTS, hi-IN neural). Real websockets. Real parallel orchestration.

## Setup (Windows)

```bash
cd matali-demo
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Hindi audio (one-time, needs internet)

```bash
python generate_audio.py
```

Creates 4 MP3s in `frontend/audio/`:
- `ai_greeting.mp3` — AI answering call
- `ai_ack.mp3` — AI confirming dispatch
- `ai_firstaid.mp3` — AI reading first-aid instructions
- `caller_scripted.mp3` — scripted bystander voice

## Run

```bash
python backend\app.py
```

## Test

Open two browser tabs:
- Caller: http://localhost:8000/caller
- Dashboard: http://localhost:8000/dashboard

Click the phone button on the caller page. Watch dashboard animate:
1. Transcript streams word-by-word, synced to audio
2. Structured extraction appears
3. Four threads fire in parallel, each completes with realistic delay
4. HMM evaluates all 5 hospitals, dims rejections with real utility scores
5. Summary ribbon shows winning hospital, real ETA, and time saved vs 108 baseline

First-aid instructions appear on caller phone UI.

## Structure

```
matali-demo/
├── backend/
│   ├── app.py          # FastAPI + WebSocket + pipeline orchestration
│   └── hmm.py          # Hidden Markov Model — load-aware hospital routing
├── frontend/
│   ├── caller.html     # Phone UI
│   └── dashboard.html  # Operator console
├── generate_audio.py
└── requirements.txt
```

## D3 Checkpoint

- Real HMM (`backend/hmm.py`) — load-dependent dynamic transition matrix, forward projection, capability scoring, severity-weighted utility function
- SDM Medical Dharwad selected for red TBI via honest utility math — not hardcoded
- Rejection sequence shows real utility scores; judge can ask "what was DH's score?" and get a real number
- Pacing fixed — each hospital rejection readable at 1.2s, route line draws over 2s
- Transcript synced to audio duration (no drift)
- Non-route thread cards fade during HMM decision window
- Summary ribbon shows real ETA from HMM, not placeholder

## Key HMM parameters

| Parameter | Value |
|---|---|
| Accident location | NH-67 mid-stretch (15.4150, 75.3000) |
| Ambulance speed | 60 km/h |
| GIMS pre-load | δ=4 (saturated from prior calls) |
| SDM utility (red TBI) | negative but highest — least-bad routing |
| λ (red severity) | 0.04 per minute |
