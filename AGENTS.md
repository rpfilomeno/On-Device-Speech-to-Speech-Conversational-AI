# AGENTS.md

Real-time local speech-to-speech chatbot: VAD → Whisper → LLM (Ollama/LM Studio) → streamed text chunking → Kokoro TTS. All model inference runs on CPU. Requires a microphone, speakers, and a running LLM server.

## Commands

- Setup: `uv sync` (source of truth is `pyproject.toml`; `requirements.txt` is a stale fallback). Copy `.env.template` → `.env` and fill in `HUGGINGFACE_TOKEN`.
- Run: `uv run speech_to_speech.py` (Textual TUI; `t` focuses the text input, Esc unfocuses it. Quitting is menu-only: top menu bar `System` → `(Q)uit` → confirm dialog. `q`/`Ctrl+Q` do NOT quit.) Or `uv run text_to_speech.py` (single-turn TTS demo). `hello.py` is the `uv init` scaffold — ignore it.
- No tests, linters, typecheckers, or CI exist. `tests/` is empty. Don't assume pytest is configured.

## Runtime requirements (not in code)

- An LLM server must be running at `LM_STUDIO_URL` (default `http://localhost:1234/v1`) with `LLM_MODEL` pulled, e.g. `ollama run qwen2.5:0.5b-instruct-q8_0`. The app will hang/fail without it.
- Linux: eSpeak NG must be installed (`apt install espeak-ng`) for fallback TTS.

## Gotchas

- `src/utils/config.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at module import. Models auto-download once to `data/models/<dir>` (gated pyannote needs `HUGGINGFACE_TOKEN`), then the app is fully offline. Deleting a model dir triggers a re-download.
- `config.py` reads `LMM_TEMPERATURE` (typo, `settings.py:34`), so `.env`'s `LLM_TEMPERATURE` is silently ignored. `LLM_TEMPERATURE` is a required-ish field with no default.
- `Settings` is a module-level singleton (`settings = Settings()`) constructed at import time; env vars are only read then. Editing `.env` requires restarting the process.

## Layout

- Entrypoints are scripts at repo root; all logic lives in `src/utils/`:
  - `speech.py` — mic/audio I/O, VAD + Whisper model init, transcription, interrupt handling
  - `llm.py` — Ollama/OpenAI-compatible streaming client; `parse_stream_chunk(chunk)` is the SSE parser
  - `text_chunker.py` — `TextChunker`, priority-based streaming splitter for latency reduction
  - `audio_queue.py` — `AudioGenerationQueue`, async TTS playback queue with interruption
  - `generator.py` — `VoiceGenerator` TTS client (Kokoro via PocketTTS or eSpeak fallback)
  - `twitch_bot.py` — optional Twitch chat integration (no-op without Twitch creds in `.env`)
- Directories `data/models/`, `output/`, `recordings/` are gitignored; `settings.setup_directories()` creates them at startup.
