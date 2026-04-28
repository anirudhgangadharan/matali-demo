"""
Generate Hindi TTS MP3s for the Matali demo.
Run ONCE on user's machine: python generate_audio.py
Creates 4 files in frontend/audio/

Uses Microsoft Edge TTS — free, no API key, high quality neural voices.
"""
import asyncio
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("Installing edge-tts...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
    import edge_tts

OUT = Path(__file__).resolve().parent / "frontend" / "audio"
OUT.mkdir(parents=True, exist_ok=True)

# Voice reference:
#   hi-IN-SwaraNeural  — calm female  (AI operator)
#   hi-IN-MadhurNeural — male          (anxious bystander caller)

LINES = [
    ("ai_greeting.mp3",    "hi-IN-SwaraNeural",
     "मातली सुन रहा है। क्या हुआ, बताइए।",
     "+0%", "+0Hz"),
    ("ai_ack.mp3",         "hi-IN-SwaraNeural",
     "समझ गया। ambulance भेज दी है। घायल के पास रहिए।",
     "+0%", "+0Hz"),
    ("ai_firstaid.mp3",    "hi-IN-SwaraNeural",
     "गर्दन मत हिलाओ। साफ कपड़े से सिर के घाव पर दबाव डालो। "
     "अगर उल्टी हो तो सिर एक तरफ कर दो। लाइन पर बने रहो।",
     "-5%", "+0Hz"),
    ("caller_scripted.mp3","hi-IN-MadhurNeural",
     "हेलो! यहाँ accident हो गया है! गदग highway पर! "
     "आदमी bike से गिरा है, सिर से खून निकल रहा है, होश में नहीं है! जल्दी आओ भाई!",
     "+15%", "+0Hz"),
]

async def gen_one(fn, voice, text, rate, pitch):
    path = OUT / fn
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await comm.save(str(path))
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ {fn:<25} {voice:<22} {size_kb:>6.1f} KB")

async def main():
    print(f"Generating 4 MP3 files → {OUT}\n")
    for args in LINES:
        await gen_one(*args)
    print(f"\nDone. Files ready in frontend/audio/")

if __name__ == "__main__":
    asyncio.run(main())
