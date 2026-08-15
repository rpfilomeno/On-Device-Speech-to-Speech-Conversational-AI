"""Runnable check for the barge-command gate.

Run: uv run python tests/test_barge_command.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.speech import classify_barge


def main():
    assert classify_barge("stop") == "command", "single 'stop' -> command"
    assert classify_barge("wait") == "command", "single 'wait' -> command"
    assert classify_barge("stop right now") == "command", "'stop' contained -> command"
    assert classify_barge("hey wait a moment please") == "command", "'wait' -> command"
    assert classify_barge("Stop, right now.") == "command", "punctuation stripped -> command"
    assert classify_barge("please stop talking") == "command", "'stop' beats length"

    assert classify_barge("hello there my friend") == "turn", "4 words -> turn"
    assert classify_barge("a b c d") == "turn", "4 words -> turn"
    assert classify_barge("can you tell me more") == "turn", "5 words -> turn"

    assert classify_barge("") == "", "empty -> no barge"
    assert classify_barge("you") == "", "single word, not a command -> no barge"
    assert classify_barge("hello there") == "", "2 words, no command -> no barge"
    assert classify_barge("oh really") == "", "2 words, no command -> no barge"
    assert classify_barge("thank you") == "", "2 words, no command -> no barge"
    assert classify_barge("walking around") == "", "2 words, no command -> no barge"
    assert classify_barge("waiting here") == "", "'waiting' is not 'wait' -> no barge"
    assert classify_barge("stopwatch") == "", "exact word match only -> no barge"

    print("barge-command gate: OK")


if __name__ == "__main__":
    main()
