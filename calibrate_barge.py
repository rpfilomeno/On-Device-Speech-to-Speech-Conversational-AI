"""Barge-in calibration harness.

Plays a known TTS sentence through the speakers, monitors barge-in, and
reports whether it fired. Use it to find config values that kill false
positives while keeping real barges.

Usage:
  uv run calibrate_barge.py                        # stay SILENT during playback (false-positive check)
  uv run calibrate_barge.py --speak                # SPEAK a barge word during playback (true-barge check)
  uv run calibrate_barge.py --count 5              # repeat N times, report x/N barges
  uv run calibrate_barge.py --timeout 0.3 --margin 4.0 --threshold 0.03
  uv run calibrate_barge.py --mic-test             # record your voice + transcribe (ASR sanity)
  uv run calibrate_barge.py --transcribe           # transcribe captured barge speech on barge
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

from src.utils.config import settings
from src.utils.generator import VoiceGenerator
from src.utils.speech import (
    TurnAudioPlayer,
    _sd_device_index,
    init_asr_model,
    record_audio,
    transcribe_audio,
)

SENTENCE = (
    "Testing barge in detection. Speak now if you want to interrupt me, "
    "otherwise stay completely silent until the test is over."
)


def print_config():
    print(
        "  INTERRUPTION_THRESHOLD = "
        f"{settings.INTERRUPTION_THRESHOLD}\n"
        f"  SPEECH_CHECK_TIMEOUT   = {settings.SPEECH_CHECK_TIMEOUT}s\n"
        f"  BARGE_IN_NOISE_MARGIN  = {settings.BARGE_IN_NOISE_MARGIN}"
    )


def get_generator():
    gen = VoiceGenerator()
    gen.initialize(settings.POCKET_TTS_URL, settings.POCKET_TTS_VOICE)
    if not gen.is_initialized():
        print("VoiceGenerator not initialized")
        sys.exit(1)
    return gen


def play_and_watch(player, stream, tail):
    for seg in stream:
        if player.is_interrupted():
            return
        player.push(seg)
        time.sleep(0.005)
    t0 = time.time()
    while player.is_playing() and not player.is_interrupted() and time.time() - t0 < 60:
        time.sleep(0.05)
    t0 = time.time()
    while not player.is_interrupted() and time.time() - t0 < tail:
        time.sleep(0.05)


def run_barge_test(gen, speak: bool, tail: float):
    player = TurnAudioPlayer(monitor_input=True)
    try:
        stream = gen.generate(SENTENCE, stream=True)
    except Exception as e:
        from src.utils.config import log_error

        log_error(e)
        print(f"TTS failed: {e}")
        return None, None
    try:
        print(
            f"\n>>> TTS sentence playing. "
            f"{'SPEAK a word NOW to test true barge.' if speak else 'STAY SILENT. This checks false positives.'}"
        )
        play_and_watch(player, stream, tail)
        barged = player.is_interrupted()
        capture = player.take_capture()
        print(f"<<< Result: {'BARGE' if barged else 'no barge'}")
        return barged, capture
    finally:
        player.stop()


def transcribe(model, audio, rate):
    return transcribe_audio(model, audio, sampling_rate=rate)


def levels_test(duration: float):
    """Print the mic's block-levels so we can see ambient vs speech amplitude."""
    idx = _sd_device_index(settings.MIC_DEVICE, want_input=True)
    levels = []

    def cb(indata, frames, t, status):
        levels.append(float(np.abs(indata[:, 0]).mean()))

    print("\n[levels] Live mic amplitude (mean/max per 0.5s). "
          "Say a loud word like STOP! a few times.")
    print(f"[levels] INTERRUPTION_THRESHOLD={settings.INTERRUPTION_THRESHOLD}  "
          f"noise_floor*BARGE_IN_NOISE_MARGIN will be the effective trigger.\n")
    with sd.InputStream(
        samplerate=settings.RATE, channels=1, callback=cb,
        device=idx, blocksize=1024,
    ):
        t0 = time.time()
        while time.time() - t0 < duration:
            win, levels[:] = list(levels), []
            if win:
                print(
                    f"  {time.time() - t0:5.1f}s  mean={float(np.mean(win)):.4f}  "
                    f"max={float(np.max(win)):.4f}",
                    flush=True,
                )
            time.sleep(0.5)
    print("[levels] done")


def echo_test(gen, duration: float):
    """Play TTS through the speaker while logging mic levels, to see the echo floor."""
    print("\n[echo] Playing TTS while logging mic levels. "
          "STAY SILENT during playback, then say STOP! at the end.")
    audio = np.concatenate(list(gen.generate(SENTENCE, stream=True)))
    spk = _sd_device_index(settings.SPEAKER_DEVICE, want_input=False)
    try:
        sd.play(audio, samplerate=24000, device=spk)
        levels_test(duration)
    finally:
        sd.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", action="store_true", help="speak a barge word during playback")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--tail", type=float, default=2.0, help="extra seconds to listen after playback")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--mic", type=str, default=None)
    ap.add_argument("--speaker", type=str, default=None)
    ap.add_argument("--mic-test", action="store_true")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--levels", type=float, default=None, help="print mic amplitude for N seconds")
    ap.add_argument("--echo", type=float, default=None, help="play TTS and log mic levels for N seconds")
    args = ap.parse_args()

    if args.threshold is not None:
        settings.INTERRUPTION_THRESHOLD = args.threshold
    if args.timeout is not None:
        settings.SPEECH_CHECK_TIMEOUT = args.timeout
    if args.margin is not None:
        settings.BARGE_IN_NOISE_MARGIN = args.margin
    if args.mic is not None:
        settings.MIC_DEVICE = args.mic
    if args.speaker is not None:
        settings.SPEAKER_DEVICE = args.speaker

    print("Barge-in calibration harness")
    print_config()

    if args.levels:
        levels_test(args.levels)
        return

    if args.echo:
        echo_test(get_generator(), args.echo)
        return

    model = None
    if args.mic_test or args.transcribe:
        print("\nLoading ASR model (one-time download on first run)...")
        model = init_asr_model()

    if args.mic_test:
        print("\n[mic-test] Speak a clear sentence into the mic during the whole recording window:")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        audio = record_audio(duration=5)
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        print(f"[mic-test] captured 5s: rms={rms:.4f}  peak={peak:.4f}")
        from pathlib import Path

        import soundfile as sf

        out = Path("output") / "calibration_mic.wav"
        out.parent.mkdir(exist_ok=True)
        sf.write(out, audio, settings.RATE)
        print(f"[mic-test] saved to {out}")
        text = transcribe(model, audio, settings.RATE)
        print(f"[mic-test] ASR heard: {text!r}")
        return

    gen = get_generator()
    n_barged = 0
    for i in range(args.count):
        print(f"\n--- Test {i + 1}/{args.count} ---")
        barged, capture = run_barge_test(gen, args.speak, args.tail)
        if barged is None:
            sys.exit(1)
        n_barged += int(barged)
        if barged and capture is not None and model is not None:
            text = transcribe(model, capture, settings.RATE)
            print(f"  captured speech: {text!r}")
        elif barged and capture is None:
            print("  (no capture)")
        time.sleep(2)

    mode = "false positives" if not args.speak else "true barges"
    print(f"\n=== {n_barged}/{args.count} barges ({mode}) ===")


if __name__ == "__main__":
    main()
