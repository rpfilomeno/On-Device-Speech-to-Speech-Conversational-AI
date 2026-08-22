import os
from pathlib import Path
from typing import Optional
import pyaudio
import numpy as np
import threading
import time
from queue import Queue
import sounddevice as sd
from .config import settings, log_error

CHUNK = settings.CHUNK
FORMAT = pyaudio.paFloat32
CHANNELS = settings.CHANNELS
RATE = settings.RATE
SILENCE_THRESHOLD = settings.SILENCE_THRESHOLD
SPEECH_CHECK_THRESHOLD = settings.SPEECH_CHECK_THRESHOLD
MAX_SILENCE_DURATION = settings.MAX_SILENCE_DURATION


def _abs_model_dir(dir_path: str) -> Path:
    path = Path(dir_path)
    return path if path.is_absolute() else settings.BASE_DIR / path


def _register_cuda_dlls():
    """Make the CUDA/cuDNN DLLs bundled with torch (torch/lib) and any NVIDIA
    pip wheels visible to onnxruntime's CUDA provider. No-op if absent."""
    import os
    import sysconfig

    dirs = [Path(sysconfig.get_paths()["purelib"]) / "torch" / "lib"]
    dirs += sorted((Path(sysconfig.get_paths()["purelib"])).glob("nvidia/*/bin"))
    for pkg_dir in dirs:
        try:
            if pkg_dir.is_dir():
                os.add_dll_directory(str(pkg_dir))
                os.environ["PATH"] = str(pkg_dir) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass


def _onnx_providers() -> list[str] | None:
    """Preferred ONNX execution providers: CUDA first, CPU always last."""
    try:
        import onnxruntime as rt

        wanted = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in rt.get_available_providers()
        ]
        return wanted or None
    except Exception:
        return None


def _offline_safe_load(fn):
    """Call an onnx-asr loader with HF offline mode temporarily lifted so the
    first call can download model files into their local dir (config.py pins
    HF_HUB_OFFLINE=1 globally).

    Patches both the env var and huggingface_hub's cached constant, since the
    library snapshots the flag at import time.
    """
    import os

    saved_env = os.environ.pop("HF_HUB_OFFLINE", None)
    hf_constants = None
    saved_flag = None
    try:
        from huggingface_hub import constants

        hf_constants = constants
        saved_flag = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = False
    except Exception:
        pass

    try:
        return fn()
    finally:
        if saved_env is not None:
            os.environ["HF_HUB_OFFLINE"] = saved_env
        if hf_constants is not None and saved_flag is not None:
            hf_constants.HF_HUB_OFFLINE = saved_flag


def _pyaudio_input_index(p, name: str):
    """Resolve a device name to its PyAudio input-device index (None = default)."""
    if not name:
        return None
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0) or 0) > 0 and info.get("name") == name:
                return i
    except Exception:
        pass
    return None


def _sd_device_index(name: str, want_input: bool):
    """Resolve a device name to its sounddevice index (None = default)."""
    if not name:
        return None
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("name") != name:
                continue
            if want_input and d.get("max_input_channels", 0) > 0:
                return i
            if not want_input and d.get("max_output_channels", 0) > 0:
                return i
    except Exception:
        pass
    return None


def list_audio_devices() -> tuple[list[str], list[str]]:
    """Return (input device names, output device names) for the /config dialog."""
    inputs, outputs = [], []
    try:
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0) or 0) > 0:
                    inputs.append(info["name"])
        finally:
            p.terminate()
    except Exception:
        pass
    try:
        for d in sd.query_devices():
            if d.get("max_output_channels", 0) > 0:
                outputs.append(d["name"])
    except Exception:
        pass
    return inputs, outputs


def init_asr_model():
    """Initializes the Parakeet TDT ASR model via onnx-asr (int8 ONNX weights).

    Downloads to ASR_MODEL_DIR on first use, then loads fully offline. Prefers
    CUDA; falls back to CPU if the GPU session can't be created (e.g. missing
    cuDNN DLLs).
    """
    import onnx_asr

    local_dir = _abs_model_dir(settings.ASR_MODEL_DIR)
    _register_cuda_dlls()
    providers = _onnx_providers()
    try:
        model = _offline_safe_load(lambda: onnx_asr.load_model(
            settings.ASR_MODEL_NAME, str(local_dir),
            quantization="int8", providers=providers,
        ))
    except Exception as e:
        log_error(e)
        print(f"GPU ASR load failed ({e}); retrying on CPU...")
        model = _offline_safe_load(lambda: onnx_asr.load_model(
            settings.ASR_MODEL_NAME, str(local_dir),
            quantization="int8", providers=["CPUExecutionProvider"],
        ))
        providers = ["CPUExecutionProvider"]

    device = "GPU" if providers and providers[0] == "CUDAExecutionProvider" else "CPU"
    print(f"ASR ready: {settings.ASR_MODEL_NAME} int8 ({device})")
    return model


def init_vad_pipeline():
    """Initializes Silero VAD via onnx-asr.

    Downloads to VAD_MODEL_DIR on first use, then loads fully offline.
    """
    import onnx_asr

    vad_dir = _abs_model_dir(settings.VAD_MODEL_DIR)
    _register_cuda_dlls()
    pipeline = _offline_safe_load(
        lambda: onnx_asr.load_vad("silero", str(vad_dir))
    )
    print("VAD ready: silero (onnx-asr)")
    return pipeline


def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detects speech segments using Silero VAD.

    Args:
        pipeline: VAD pipeline from init_vad_pipeline().
        audio_data (np.ndarray): Mono float32 audio.
        sample_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        np.ndarray or None: Concatenated speech segments as float32 audio, or None if no speech is detected.
    """
    if sample_rate is None:
        sample_rate = settings.RATE
    if audio_data is None:
        return None

    audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    if audio.size < sample_rate // 10:
        return None

    segments = next(pipeline.segment_batch(
        audio[None, :],
        np.array([audio.size], dtype=np.int64),
        sample_rate,
        min_speech_duration_ms=settings.VAD_MIN_DURATION_ON * 1000,
        min_silence_duration_ms=settings.VAD_MIN_DURATION_OFF * 1000,
    ))

    parts = [audio[start:end] for start, end in segments]
    if parts:
        return np.concatenate(parts)
    return None


def record_audio(duration=None):
    """Records audio for a specified duration.

    Args:
        duration (int, optional): Recording duration in seconds. Defaults to settings.RECORD_DURATION.

    Returns:
        np.ndarray: Recorded audio data as a numpy array.
    """
    if duration is None:
        duration = settings.RECORD_DURATION

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=settings.CHANNELS,
        rate=settings.RATE,
        input=True,
        frames_per_buffer=settings.CHUNK,
        input_device_index=_pyaudio_input_index(p, settings.MIC_DEVICE),
    )

    print("\nRecording...")
    frames = []

    for i in range(0, int(settings.RATE / settings.CHUNK * duration)):
        data = stream.read(settings.CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames, axis=0)
    return audio_data


def record_continuous_audio(max_wait=None):
    """Continuously monitors audio and detects speech segments.

    Args:
        max_wait (float, optional): Maximum time in seconds to wait for speech before returning None.

    Returns:
        np.ndarray or None: Recorded audio data as a numpy array, or None if no speech is detected.
    """
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
        input_device_index=_pyaudio_input_index(p, settings.MIC_DEVICE),
    )

    frames = []
    buffer_frames = []
    buffer_size = int(RATE * 0.5 / CHUNK)
    silence_frames = 0
    max_silence_frames = int(RATE * settings.END_SILENCE_SECONDS / CHUNK)
    recording = False
    start_time = time.time()

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            buffer_frames.append(audio_chunk)
            if len(buffer_frames) > buffer_size:
                buffer_frames.pop(0)

            audio_level = np.abs(np.concatenate(buffer_frames)).mean()

            if audio_level > SILENCE_THRESHOLD:
                if not recording:
                    print("\nPotential speech detected...")
                    recording = True
                    frames.extend(buffer_frames)
                frames.append(audio_chunk)
                silence_frames = 0
            elif recording:
                frames.append(audio_chunk)
                silence_frames += 1

                if silence_frames >= max_silence_frames:
                    print("Processing speech segment...")
                    break
            elif max_wait is not None and (time.time() - start_time) >= max_wait:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if frames:
        return np.concatenate(frames)
        
    return None


def check_for_speech(timeout=0.1):
    """Checks if speech is detected in a non-blocking way.

    Args:
        timeout (float, optional): Duration to check for speech in seconds. Defaults to 0.1.

    Returns:
        tuple: A tuple containing a boolean indicating if speech was detected and the audio data as a numpy array, or (False, None) if no speech is detected.
    """
    p = pyaudio.PyAudio()

    frames = []
    is_speech = False

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=_pyaudio_input_index(p, settings.MIC_DEVICE),
        )

        for _ in range(int(RATE * timeout / CHUNK)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)

            audio_level = np.abs(audio_chunk).mean()
            if audio_level > SPEECH_CHECK_THRESHOLD:
                is_speech = True
                break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if is_speech and frames:
        return True, np.concatenate(frames)
    return False, None


BARGE_COMMANDS = {"stop", "wait"}


def classify_barge(text: str) -> str:
    """Classify captured speech for the barge gate.

    Returns:
        "command" — contains a stop/wait command (halt playback only).
        "turn"    — more than 3 words, no command (halt and reuse as next input).
        ""        — not a barge (resume playback).
    """
    words = [w.lower().strip(".,!?;:'\"()[]-") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return ""
    if any(w in BARGE_COMMANDS for w in words):
        return "command"
    if len(words) > 3:
        return "turn"
    return ""


class TurnAudioPlayer:
    """Output (and optional barge-in input) streams kept open for an entire turn.

    Chunks are queued FIFO and drained gaplessly by a single output callback, so
    sentence boundaries never pay stream open/close overhead or click. The input
    stream stays open too, so gaps between chunks are still monitored for
    barge-in and capture the interrupting speech, instead of re-initializing
    PyAudio on every loop iteration.
    """

    _DONE = object()
    _ROLL_BLOCKS = max(4, int(settings.RATE * 0.5 / 1024))
    _CAP_BLOCKS = max(16, int(settings.RATE * 3 / 1024))

    def __init__(
        self,
        sample_rate: int = 24000,
        stop_events=None,
        monitor_input: bool = True,
    ):
        self._sample_rate = int(sample_rate)
        self._stop_events = stop_events or []
        self._q = Queue()
        self._buf = None
        self._pos = 0
        self._playing = False
        self._interrupt = threading.Event()
        self._capture = None
        self._roll = []
        self._last_hot = None
        self._noise_floor = 0.0
        self._paused = False
        self._last_speech_time = 0.0
        self._drain_event = threading.Event()
        self._stopped = False

        mic_device = _sd_device_index(settings.MIC_DEVICE, want_input=True)
        spk_device = _sd_device_index(settings.SPEAKER_DEVICE, want_input=False)

        self._out: Optional[sd.OutputStream] = None
        self._in: Optional[sd.InputStream] = None
        self._out = sd.OutputStream(
            channels=1, callback=self._output_callback,
            samplerate=self._sample_rate, device=spk_device, blocksize=1024,
        )
        self._out.start()
        if monitor_input:
            # Optional: a mic problem must never kill playback, only disable barge-in.
            try:
                self._in = sd.InputStream(
                    channels=1, callback=self._input_callback,
                    samplerate=settings.RATE, device=mic_device, blocksize=1024,
                )
                self._in.start()
            except Exception as e:
                log_error(e)
                self._in = None

    # ---- callbacks (PortAudio threads) ----

    def _input_callback(self, indata, frames, time_info, status):
        # Only detect barge-in while audio is actually playing; ambient noise
        # during the LLM "thinking" phase must not kill the turn.
        if status or not self._playing:
            return
        chunk = indata[:, 0]
        self._roll.append(chunk.copy())
        if len(self._roll) > self._ROLL_BLOCKS:
            self._roll.pop(0)
        cap = self._capture
        level = float(np.abs(chunk).mean())
        # Noise-floor EMA on quiet blocks, so the trigger adapts to ambient level.
        # Starts low (5% of the first block) and rises slowly, so a loud barge
        # word never becomes its own noise floor.
        self._noise_floor = min(
            self._noise_floor * 0.95 + level * 0.05, level * 2
        )
        trigger = max(
            settings.INTERRUPTION_THRESHOLD,
            self._noise_floor * settings.BARGE_IN_NOISE_MARGIN,
        )
        if level > trigger:
            # Debounce: require SPEECH_CHECK_TIMEOUT of sustained level, so
            # short transients (clicks, claps) never barge in.
            self._last_speech_time = time.perf_counter()
            if self._last_hot is None:
                self._last_hot = time.perf_counter()
            if (
                time.perf_counter() - self._last_hot >= settings.SPEECH_CHECK_TIMEOUT
                and not self._interrupt.is_set()
            ):
                self._interrupt.set()
                self._capture = cap = list(self._roll)
        else:
            self._last_hot = None
        if cap is not None and len(cap) < self._CAP_BLOCKS:
            cap.append(chunk.copy())

    def _output_callback(self, outdata, frames, time_info, status):
        if self._stopped or any(e.is_set() for e in self._stop_events):
            outdata.fill(0)
            raise sd.CallbackStop
        if self._paused or self._interrupt.is_set():
            # Barge candidate: hold playback silently while the main loop
            # transcribes the capture and decides whether to halt.
            outdata.fill(0)
            return
        out = []
        need = frames
        while need > 0:
            if self._buf is not None and self._pos < len(self._buf):
                take = min(need, len(self._buf) - self._pos)
                out.append(self._buf[self._pos : self._pos + take])
                self._pos += take
                need -= take
            else:
                try:
                    item = self._q.get_nowait()
                except Exception:
                    break
                if item is TurnAudioPlayer._DONE:
                    self._drain_event.set()
                    break
                self._buf = item
                self._pos = 0
        data = np.concatenate(out) if out else np.empty(0, dtype=np.float32)
        n = min(frames, len(data))
        outdata[:n, 0] = data[:n]
        if n < frames:
            outdata[n:, 0] = 0

    # ---- worker thread API ----

    def push(self, chunk: np.ndarray):
        """Queue a chunk for gapless playback; returns immediately."""
        if self._stopped or self._interrupt.is_set():
            return
        if not self._playing:
            self._playing = True
            self._roll = []
        self._q.put(chunk)

    def is_interrupted(self) -> bool:
        """True once barge-in speech or a stop event has halted playback."""
        return self._interrupt.is_set() or any(e.is_set() for e in self._stop_events)

    def is_playing(self) -> bool:
        """True while audio is still queued or actively playing."""
        if not self._q.empty():
            return True
        return self._buf is not None and self._pos < len(self._buf)

    def take_capture(self) -> Optional[np.ndarray]:
        """Return audio captured from barge-in onset (for re-transcription)."""
        cap, self._capture = self._capture, None
        if cap:
            return np.concatenate(cap)
        return None

    def pause(self):
        """Hold playback (silence) without stopping the stream or losing buffers."""
        self._paused = True

    def resume(self):
        """Continue playback after a rejected barge candidate."""
        self._paused = False
        self._interrupt.clear()
        self._capture = None

    def wait_for_quiet(self, min_quiet=None, max_wait=4.0):
        """Block until the mic has been quiet for min_quiet seconds.

        Used by the barge gate so the full interrupted phrase is captured
        instead of a truncated prefix. Capture keeps growing meanwhile.
        """
        if min_quiet is None:
            min_quiet = settings.BARGE_QUIET_TIME
        t0 = time.time()
        while time.time() - t0 < max_wait:
            if time.perf_counter() - self._last_speech_time >= min_quiet:
                return True
            time.sleep(0.05)
        return False

    def flush(self):
        """Wait until everything pushed so far has been played."""
        if self._stopped or self.is_interrupted():
            return
        self._drain_event.clear()
        self._q.put(TurnAudioPlayer._DONE)
        while not self._drain_event.is_set() and not self.is_interrupted():
            self._drain_event.wait(timeout=0.05)

    def stop(self):
        self._stopped = True
        for s in (self._out, self._in):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._out = self._in = None


def transcribe_audio(model, audio_data, sampling_rate=None):
    """Transcribes audio using the onnx-asr model from init_asr_model().

    Args:
        model: ASR model (onnx-asr).
        audio_data (np.ndarray): Mono float32/int16-range audio to transcribe.
        sampling_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        str: Transcribed text.
    """
    if sampling_rate is None:
        sampling_rate = settings.RATE

    if model is None or audio_data is None:
        return ""

    audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return ""

    return model.recognize(audio, sample_rate=sampling_rate)
