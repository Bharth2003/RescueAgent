"""Download the Kokoro TTS model used for voice narration (am_michael).

The weights are ~325 MB and are not committed. Run once:

    python scripts/fetch_voice_model.py

Files land in ``.models/kokoro/`` where voice.py looks for them by default.
Override the location with KOKORO_MODEL_PATH / KOKORO_VOICES_PATH.
"""

import os
import sys
import urllib.request

BASE = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/")
FILES = {"kokoro-v1.0.onnx": BASE + "kokoro-v1.0.onnx",
         "voices-v1.0.bin": BASE + "voices-v1.0.bin"}

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".models", "kokoro")


def _progress(block, block_size, total):
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, url in FILES.items():
        out = os.path.join(DEST, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"{name}: already present")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, out, _progress)
        print(f"\n  saved to {out}")
    print("done — voice narration is ready.")


if __name__ == "__main__":
    main()
