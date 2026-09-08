"""Voice narration for RescueAgent — Kokoro TTS, ``am_michael`` (warm US male).

The agent's plain-English narration is spoken aloud so a projector demo has a
voice, not just text. Synthesis runs locally with kokoro-onnx (no network, no
API key) and the clip is autoplayed in the browser as an inline data URI.

Everything here degrades gracefully: if the package or the model files are not
present, every entry point becomes a no-op and the app runs exactly as before.
The model weights are large (~325 MB) and are NOT committed — fetch them once
with ``python scripts/fetch_voice_model.py`` (or set KOKORO_MODEL_PATH /
KOKORO_VOICES_PATH to an existing copy).
"""

import base64
import hashlib
import io
import os
import re
import threading

DEFAULT_VOICE = "am_michael"      # warm US male
SAMPLE_RATE = 24000

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.environ.get("KOKORO_MODEL_PATH"),
    os.path.join(_HERE, ".models", "kokoro", "kokoro-v1.0.onnx"),
    os.path.join(_HERE, "models", "kokoro-v1.0.onnx"),
]
_VOICES_CANDIDATES = [
    os.environ.get("KOKORO_VOICES_PATH"),
    os.path.join(_HERE, ".models", "kokoro", "voices-v1.0.bin"),
    os.path.join(_HERE, "models", "voices-v1.0.bin"),
]

_lock = threading.Lock()
_engine = None            # cached Kokoro instance (or False if unavailable)
_wav_cache = {}           # (text, voice, speed) -> base64 wav string


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def model_paths():
    """Return (model_path, voices_path) if both are present, else (None, None)."""
    return _first_existing(_MODEL_CANDIDATES), _first_existing(_VOICES_CANDIDATES)


def is_available():
    """True when kokoro-onnx is importable and the model files exist."""
    if _engine is False:
        return False
    m, v = model_paths()
    if not (m and v):
        return False
    try:
        import kokoro_onnx  # noqa: F401
    except Exception:
        return False
    return True


def _get_engine():
    """Lazily build and cache the Kokoro engine. Returns None if unavailable."""
    global _engine
    if _engine is not None:
        return _engine or None
    with _lock:
        if _engine is not None:
            return _engine or None
        m, v = model_paths()
        if not (m and v):
            _engine = False
            return None
        try:
            from kokoro_onnx import Kokoro
            _engine = Kokoro(m, v)
        except Exception:
            _engine = False
            return None
    return _engine


def _clean(text):
    """Strip markdown/emoji so the spoken line sounds natural."""
    t = text or ""
    t = re.sub(r"[*_`#>]", "", t)                 # markdown emphasis / code
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)     # links -> label
    t = t.replace("→", " to ").replace("·", ". ").replace("—", ", ")
    t = re.sub(r"CO₂e?", "C O 2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def synth_base64(text, voice=DEFAULT_VOICE, speed=1.0):
    """Synthesize ``text`` and return a base64-encoded WAV, or None on failure."""
    clean = _clean(text)
    if not clean:
        return None
    key = (clean, voice, round(speed, 2))
    if key in _wav_cache:
        return _wav_cache[key]
    engine = _get_engine()
    if engine is None:
        return None
    try:
        import soundfile as sf
        samples, sr = engine.create(clean, voice=voice, speed=speed, lang="en-us")
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
    _wav_cache[key] = b64
    return b64


def audio_html(text, voice=DEFAULT_VOICE, speed=1.0, nonce=""):
    """Return an autoplaying <audio> element for ``text``, or "" if unavailable.

    ``nonce`` makes the element unique so Streamlit remounts it (and the browser
    replays it) only when the spoken line actually changes.
    """
    b64 = synth_base64(text, voice=voice, speed=speed)
    if not b64:
        return ""
    tag = hashlib.md5((str(nonce) + text).encode()).hexdigest()[:10]
    return (
        f'<audio id="ra-voice-{tag}" autoplay style="display:none">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
    )
