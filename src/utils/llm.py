import re
import requests
import json
import time
import urllib.parse
from src.utils.config import settings, log_error

_last_parse_error_time: float | None = None


def filter_response(response: str) -> str:
    """Removes markdown formatting, parenthetical text, and unicode characters from a string.

    Args:
        response (str): The string to filter.

    Returns:
        str: The filtered string.
    """
    response = re.sub(r"\*\*|__|~~|`", "", response)
    response = re.sub(r"\(.*?\)", "", response)
    response = re.sub(r"[()\[\]{}]", "", response)
    response = re.sub(r"[\U00010000-\U0010ffff]", "", response, flags=re.UNICODE)
    response = re.sub(r"<.*?>", "", response)
    # remove text like *sample*
    response = re.sub(r"\*.*?\*", "", response)
    return response


def warmup_llm(session: requests.Session, llm_model: str, llm_url: str):
    """Sends a warmup request to the LLM server.

    Args:
        session (requests.Session): The requests session to use.
        llm_model (str): The name of the LLM model.
        llm_url (str): The URL of the LLM server.
    """
    try:
        url = llm_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            endpoint = f"{url}/chat/completions"
        else:
            endpoint = url

        res = session.post(
            endpoint,
            json={
                "model": llm_model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": False,
            },
            timeout=5,
        )
        if res.status_code != 200:
            print(f"LLM warmup status code: {res.status_code}")
    except requests.RequestException as e:
        log_error(e)
        print(f"Warmup failed: {str(e)}")


def get_ai_response(
    session: requests.Session,
    messages: list,
    llm_model: str,
    llm_url: str,
    max_tokens: int,
    temperature: float = 0.7,
    stream: bool = False,
):
    """Sends a request to the LLM (LM Studio / OpenAI compatible API) and returns a streaming iterator.

    Args:
        session (requests.Session): The requests session to use.
        messages (list): The list of messages to send to the LLM.
        llm_model (str): The name of the LLM model.
        llm_url (str): The URL of the LLM server.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float, optional): The temperature to use for generation. Defaults to 0.7.
        stream (bool, optional): Whether to stream the response. Defaults to False.

    Returns:
        iterator: An iterator over the streaming response.
    """
    try:
        url = llm_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            endpoint = f"{url}/chat/completions"
        else:
            endpoint = url

        response = session.post(
            endpoint,
            json={
                "model": llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            },
            timeout=3600,
            stream=stream,
        )
        response.raise_for_status()

        def streaming_iterator():
            """Iterates over the streaming response."""
            try:
                for line in response.iter_lines():
                    if line:
                        yield line
                    else:
                        yield b""
            except Exception as e:
                log_error(e)
                print(f"\nError: {str(e)}")
                yield b""

        return streaming_iterator()

    except Exception as e:
        log_error(e)
        print(f"\nError: {str(e)}")


def parse_stream_chunk(chunk: bytes) -> dict | None:
    """Parses a chunk of data from the LLM stream.

    Args:
        chunk (bytes): The chunk of data to parse.

    Returns:
        dict: A dictionary containing the parsed data.
    """
    if not chunk:
        return {"keep_alive": True}

    try:
        text = chunk.decode("utf-8").strip()
        if text.startswith("data: "):
            text = text[6:]
        if text == "[DONE]":
            return {"choices": [{"finish_reason": "stop", "delta": {}}]}
        if text.startswith("{"):
            data = json.loads(text)
            content = ""
            msg = data.get("message") if isinstance(data, dict) else None
            if isinstance(msg, dict):
                content = msg.get("content", "")
            elif (
                isinstance(data, dict)
                and isinstance(data.get("choices"), list)
                and data["choices"]
            ):
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    content = choice.get("delta", {}).get("content", "") or choice.get(
                        "message", {}
                    ).get("content", "")

            if content:
                return {"choices": [{"delta": {"content": filter_response(content)}}]}
        return None

    except Exception as e:
        global _last_parse_error_time
        if _last_parse_error_time is None or time.time() - _last_parse_error_time > 10:
            log_error(e)
            _last_parse_error_time = time.time()
        if str(e) != "Expecting value: line 1 column 2 (char 1)":
            print(f"Error parsing stream chunk: {str(e)}")
        return None


def fetch_context_window(
    session: requests.Session, llm_model: str, llm_url: str
) -> int | None:
    """Query the LM Studio REST API for the model's actually-loaded context length.

    LM Studio serves its REST API from the host root (not under /v1), so a
    trailing /v1 on LM_STUDIO_URL is stripped. Returns None on any failure
    (server down, non-LM Studio backend like Ollama, model not loaded) so the
    caller can fall back to a configured default.
    """
    url = llm_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    endpoint = f"{url}/api/v0/models/{urllib.parse.quote(llm_model)}"
    try:
        res = session.get(endpoint, timeout=5)
        res.raise_for_status()
        ctx = res.json().get("loaded_context_length")
        if ctx:
            return int(ctx)
    except Exception:
        pass
    return None
