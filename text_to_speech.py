"""Standalone text-to-speech chatbot: no mic, VAD, or Whisper.

Text input → LLM (Ollama/LM Studio) → streamed chunking → Kokoro TTS.
A self-contained variant of speech_to_speech.py with all voice-input paths
removed (no /voice, no /config, no barge-in). Twitch chat and idle auto-talk
are retained since they are text-driven.

Logic lives in the src.text_to_speech package; this file is just the entrypoint.

Run: uv run text_to_speech.py
"""

from src.text_to_speech.tui import TextSpeechTUI


def main():
    TextSpeechTUI().run()


if __name__ == "__main__":
    main()
