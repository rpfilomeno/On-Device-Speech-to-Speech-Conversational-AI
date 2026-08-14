import os
from pathlib import Path
from typing import Optional
import pyaudio
import numpy as np
import torch
from torch.nn.functional import pad
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


def ensure_model_downloaded(repo_id: str, local_dir: str, token: Optional[str] = None) -> Path:
    """Ensures that a Hugging Face model repository is downloaded into a local directory.

    If the local directory does not exist or is empty, downloads from HF Hub.

    Args:
        repo_id (str): Hugging Face repository ID (e.g., 'openai/whisper-base.en').
        local_dir (str): Local directory path to store model files.
        token (str, optional): Hugging Face token for gated/private repositories.

    Returns:
        Path: Path object pointing to the local directory.
    """
    target_path = Path(local_dir)
    if not target_path.is_absolute():
        target_path = settings.BASE_DIR / target_path

    if not target_path.exists() or not any(target_path.iterdir()):
        print(f"\nModel files not found at '{target_path}'. Downloading '{repo_id}' from Hugging Face...")
        target_path.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import snapshot_download

        orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        orig_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(target_path),
            }
            if token:
                download_kwargs["token"] = token

            snapshot_download(**download_kwargs)
            print(f"Successfully downloaded '{repo_id}' to '{target_path}'.")
        finally:
            if orig_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
            if orig_tf_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = orig_tf_offline

    return target_path


def init_whisper_model(
    model_id: Optional[str] = None,
    model_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Initializes the Whisper processor and model, downloading to local_dir first if missing.

    Args:
        model_id (str, optional): Hugging Face repo ID. Defaults to settings.WHISPER_MODEL_ID.
        model_dir (str, optional): Local target directory. Defaults to settings.WHISPER_MODEL_DIR.
        hf_token (str, optional): Hugging Face API token.

    Returns:
        tuple: (WhisperProcessor, WhisperForConditionalGeneration)
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    if model_id is None:
        model_id = settings.WHISPER_MODEL_ID
    if model_dir is None:
        model_dir = settings.WHISPER_MODEL_DIR

    local_path = ensure_model_downloaded(model_id, model_dir, token=hf_token)

    whisper_processor = WhisperProcessor.from_pretrained(
        str(local_path), local_files_only=True
    )
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        str(local_path), local_files_only=True
    )

    return whisper_processor, whisper_model


def init_vad_pipeline(hf_token):
    """Initializes the Voice Activity Detection pipeline.

    Args:
        hf_token (str): Hugging Face API token.

    Returns:
        pyannote.audio.pipelines.VoiceActivityDetection: VAD pipeline.
    """
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    vad_dir = ensure_model_downloaded(
        repo_id=settings.VAD_MODEL_ID,
        local_dir=settings.VAD_MODEL_DIR,
        token=hf_token,
    )

    model = Model.from_pretrained(
        str(vad_dir), token=hf_token, local_files_only=True
    )
    assert model is not None

    pipeline = VoiceActivityDetection(segmentation=model)

    HYPER_PARAMETERS = {
        "min_duration_on": settings.VAD_MIN_DURATION_ON,
        "min_duration_off": settings.VAD_MIN_DURATION_OFF,
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    return pipeline


def detect_speech_segments(pipeline, audio_data, sample_rate=None):
    """Detects speech segments in audio using pyannote VAD.

    Args:
        pipeline (pyannote.audio.pipelines.VoiceActivityDetection): VAD pipeline.
        audio_data (np.ndarray or torch.Tensor): Audio data.
        sample_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        torch.Tensor or None: Concatenated speech segments as a torch tensor, or None if no speech is detected.
    """
    if sample_rate is None:
        sample_rate = settings.RATE

    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)

    if not isinstance(audio_data, torch.Tensor):
        audio_data = torch.from_numpy(audio_data)

    if audio_data.shape[1] < sample_rate:
        padding_size = sample_rate - audio_data.shape[1]
        audio_data = pad(audio_data, (0, padding_size))

    vad = pipeline({"waveform": audio_data, "sample_rate": sample_rate})

    speech_segments = []
    for speech in vad.get_timeline().support():
        start_sample = int(speech.start * sample_rate)
        end_sample = int(speech.end * sample_rate)
        if start_sample < audio_data.shape[1]:
            end_sample = min(end_sample, audio_data.shape[1])
            segment = audio_data[0, start_sample:end_sample]
            speech_segments.append(segment)

    if speech_segments:
        return torch.cat(speech_segments)
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
    max_silence_frames = int(RATE / CHUNK * 1)
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
        if float(np.abs(chunk).mean()) > settings.INTERRUPTION_THRESHOLD:
            if not self._interrupt.is_set():
                self._interrupt.set()
                self._capture = cap = list(self._roll)
            if cap is not None and len(cap) < self._CAP_BLOCKS:
                cap.append(chunk.copy())
        elif cap is not None and len(cap) < self._CAP_BLOCKS:
            cap.append(chunk.copy())

    def _output_callback(self, outdata, frames, time_info, status):
        if self._stopped or self._interrupt.is_set() or any(e.is_set() for e in self._stop_events):
            outdata.fill(0)
            raise sd.CallbackStop
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


def transcribe_audio(processor, model, audio_data, sampling_rate=None):
    """Transcribes audio using Whisper.

    Args:
        processor (transformers.WhisperProcessor): Whisper processor.
        model (transformers.WhisperForConditionalGeneration): Whisper model.
        audio_data (np.ndarray or torch.Tensor): Audio data to transcribe.
        sampling_rate (int, optional): Sample rate of the audio. Defaults to settings.RATE.

    Returns:
        str: Transcribed text.
    """
    if sampling_rate is None:
        sampling_rate = settings.RATE

    if audio_data is None:
        return ""

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()

    input_features = processor(
        audio_data, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
