# AGENTS.md

Real-time local speech-to-speech chatbot: VAD → Whisper → LLM (Ollama/LM Studio) → streamed text chunking → Kokoro TTS. Models auto-pick GPU if `torch.cuda.is_available()`, else CPU. Requires a microphone, speakers, and a running LLM server.

## Commands

- Setup: `uv sync` (source of truth is `pyproject.toml`; `requirements.txt` is a stale fallback). Copy `.env.template` → `.env` and fill in `HUGGINGFACE_TOKEN`.
- Run: `uv run speech_to_speech.py` (Textual TUI). **Voice input starts OFF** — VAD/Whisper load on `/voice on`. `t` focuses the text input, Esc unfocuses it. Quit via `System` → `(Q)uit` or `/quit` (both confirm); `q`/`Ctrl+Q` do NOT quit. `hello.py` is the `uv init` scaffold — ignore it.
- Tests: runnable self-checks in `tests/`, no pytest. `uv run python tests/test_barge_in.py`, `tests/test_barge_command.py`, `tests/test_history_trim.py`, `tests/test_text_history_trim.py`, and `tests/test_text_chunker.py` are plain asserts; `tests/test_idle.py` uses stdlib unittest. Some are slow to start (importing `speech_to_speech.py` pulls in transformers + textual).
- Typecheck: pyrefly (dev dep) is the configured typechecker and the OpenCode LSP (`opencode.json`). No lint tool, no CI.

## Runtime requirements (not in code)

- An LLM server must be running at `LM_STUDIO_URL` (default `http://localhost:1234/v1`) with `LLM_MODEL` pulled, e.g. `ollama run qwen2.5:0.5b-instruct-q8_0`. The app will hang/fail without it.
- Linux: eSpeak NG must be installed (`apt install espeak-ng`) for fallback TTS.
- `calibrate_barge.py` is a barge-in tuning harness (`--speak`, `--count`, `--threshold/--timeout/--margin`, `--mic-test`, `--levels`, `--echo`) — run it instead of editing thresholds blind.

## Gotchas

- `src/utils/config.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at module import, before `load_dotenv()`. Models auto-download once to `data/models/<dir>` (gated pyannote needs `HUGGINGFACE_TOKEN`), then the app is fully offline. Deleting a model dir triggers a re-download.
- `settings` is a module-level singleton (`config.py:136`) constructed at import time; env vars are only read then. Editing `.env` requires restarting the process.
- `settings.json` (repo root) overrides env for `MIC_DEVICE`, `SPEAKER_DEVICE`, `TARGET_SIZE`, `PLAYBACK_DELAY`; set via `/config` or `save_settings()`. It is committed; don't treat it as scratch. Currently it only pins the two devices.
- `log_error(exc)` (call from inside an except block) writes full tracebacks to `output/error/`.
- The text variant (`text_to_speech.py` / `src/text_to_speech/state.py`) resets `SPEAKER_DEVICE` to the `.env`/default at import, ignoring the `settings.json` device override.
- `configure_logging()` sets root logger to ERROR — debug prints won't show unless you lower it.

## Layout

- Entrypoints are scripts at repo root; all logic lives in `src/utils/`:
  - `speech.py` — mic/audio I/O, GPU/CPU device pick, VAD + Whisper model init, transcription, `TurnAudioPlayer` (barge-in + debounce), `classify_barge`
  - `llm.py` — Ollama/OpenAI-compatible streaming client; `parse_stream_chunk(chunk)` is the SSE parser
  - `text_chunker.py` — `TextChunker`, greedy streaming splitter that fills each chunk up to `TARGET_SIZE` words at the furthest natural break
  - `audio_queue.py` — `AudioGenerationQueue`, async TTS playback queue with interruption
  - `generator.py` — `VoiceGenerator` TTS client (Kokoro via PocketTTS or eSpeak fallback)
  - `memory.py` — Qdrant (`/memory on`) vs in-RAM memory backend, with a worker queue
  - `audio.py` / `audio_io.py` / `audio_utils.py` — lower-level playback/save helpers
  - `twitch_bot.py` — optional Twitch chat integration (no-op without Twitch creds in `.env`)
- `speech_to_speech.py` holds the TUI, slash-command dispatch, and idle loop (some tests import it directly). `calibrate_barge.py` is the barge calibration harness.
- `text_to_speech.py` is a **thin entrypoint** for the text-only variant (no mic/VAD/Whisper; text in → Kokoro TTS out). Its logic lives in `src/text_to_speech/`: `state.py` (event bus + shared mutable state), `logging.py` (error routing), `history.py` (context-budget trimming), `pipeline.py`, `tui.py`. It never imports `speech_to_speech.py`.
- Directories `data/models/`, `output/`, `recordings/` are gitignored; `settings.setup_directories()` creates them at startup.
