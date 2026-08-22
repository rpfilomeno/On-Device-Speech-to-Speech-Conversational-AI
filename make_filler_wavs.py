"""Generate TTS wav files of filler/backchannel words into recordings/.

Usage:
  uv run python make_filler_wavs.py
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.config import settings
from src.utils.generator import VoiceGenerator

# (subdir, words)
GROUPS = [
    ("negative", [
        "Um", "Uh", "Ah", "Er", "Hmm", "So", "Well", "Okay",
        "You know", "I mean", "Kind of", "Sort of", "I guess", "I think",
    ]),
    ("positive", [
        "Absolutely!", "Definitely!", "Certainly!", "Exactly!", "Totally!",
        "Right!", "Sure!", "Honestly", "Really", "Naturally", "Clearly",
        "Actually", "Basically", "You know", "You see",
    ]),
    ("neutral", [
        "Um", "Uh", "Er", "Ah", "Hmm", "Well", "So", "Okay", "Right",
    ]),
]


def slugify(text: str) -> str:
    return text.lower().rstrip("!").replace(" ", "_")


def main():
    gen = VoiceGenerator()
    gen.initialize(settings.POCKET_TTS_URL, settings.POCKET_TTS_VOICE)

    for subdir, words in GROUPS:
        out_dir = Path("recordings") / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for word in words:
            out_path = out_dir / f"{slugify(word)}.wav"
            if out_path.exists():
                print(f"skip (exists): {out_path}")
                continue
            try:
                # ponytail: stream=False breaks on this server's raw-PCM responses;
                # _stream_audio already handles both RIFF and bare PCM
                audio = np.concatenate(list(gen.generate(word, stream=True)))
                if not len(audio):
                    raise ValueError("empty audio")
            except Exception as e:
                print(f"FAILED '{word}': {e}")
                continue
            sf.write(out_path, audio, 24000)
            print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
