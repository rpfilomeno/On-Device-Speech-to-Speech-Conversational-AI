import logging
import os
import queue
import threading

import requests

from .config import log_error, settings


class Mem0Memory:
    """Long-term conversational memory backed by mem0 (file-based).

    mem0 runs fully local: the vector store is an embedded qdrant collection on
    disk under settings.MEM0_DIR and the change history is a SQLite file next to
    it — no server. Fact extraction uses the LLM server's OpenAI-compatible API,
    embeddings use its /v1/embeddings endpoint. Memories persist across restarts.
    """

    USER_ID = "user"

    def __init__(self):
        # mem0's OpenAI clients refuse to init without any key set, even when
        # pointed at LM Studio.
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
        from mem0 import Memory

        # Probe the embedding dimension (model-dependent) so the collection is
        # created with the right size.
        res = requests.post(
            f"{settings.LM_STUDIO_URL.rstrip('/')}/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "input": "ping"},
            timeout=10,
        )
        res.raise_for_status()
        dims = len(res.json()["data"][0]["embedding"])

        self.client = Memory.from_config(
            {
                "llm": {
                    "provider": "lmstudio",
                    "config": {
                        "model": settings.LLM_MODEL,
                        "lmstudio_base_url": settings.LM_STUDIO_URL,
                        # mem0 defaults to json_object, which some LM Studio
                        # builds reject ("type must be json_schema or text").
                        "lmstudio_response_format": {"type": "text"},
                    },
                },
                "embedder": {
                    "provider": "lmstudio",
                    "config": {"model": settings.EMBEDDING_MODEL},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(settings.MEM0_DIR),
                        "collection_name": "conversation_memory",
                        "embedding_model_dims": dims,
                    },
                },
                "history_db_path": str(settings.MEM0_DIR / "history.db"),
            }
        )

    def store(self, role: str, content: str):
        """Persist one turn (role is 'user' or 'assistant'); mem0 extracts facts via the LLM."""
        self.client.add([{"role": role, "content": content}], user_id=self.USER_ID)

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return the most relevant extracted memories for the query."""
        # ponytail: mem0's stubs mistype search(); it actually returns list[dict]
        results = self.client.search(query, filters={f"user_id": self.USER_ID}, limit=limit)  # pyrefly: ignore[bad-assignment]
        return [r["memory"] for r in results if isinstance(r, dict) and r.get("memory")]  # pyrefly: ignore[bad-index, missing-attribute]


class MemoryWorker:
    """Runs all mem0 work (store + recall) on a background thread so the
    conversation loop is never blocked by embedding/LLM calls.

    Exposes the same store/search interface as Mem0Memory:
      - store() is fire-and-forget (jobs are dropped, never block, if the worker
        is backed up).
      - search() submits a recall job and blocks until the worker returns it.
    """

    def __init__(self, memory: Mem0Memory, max_queue: int = 50):
        self._memory = memory
        self._jobs: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._jobs.get()
            try:
                if job["op"] == "store":
                    self._memory.store(job["role"], job["content"])
                else:
                    job["result"].extend(self._memory.search(job["query"], job["limit"]))
            except Exception as e:
                log_error(e)
                if job["op"] == "recall":
                    job["error"] = str(e)
                else:
                    logging.getLogger("memory").warning("Memory store failed: %s", e)
            finally:
                if "done" in job:
                    job["done"].set()

    def store(self, role: str, content: str):
        """Queue a persist job; dropped (not blocking) if the worker is backed up."""
        try:
            self._jobs.put_nowait({"op": "store", "role": role, "content": content})
        except queue.Full:
            pass

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Queue a recall job and wait for the worker's result."""
        done = threading.Event()
        job = {
            "op": "recall",
            "query": query,
            "limit": limit,
            "result": [],
            "error": None,
            "done": done,
        }
        try:
            self._jobs.put(job, timeout=1)
        except queue.Full:
            return []
        done.wait(timeout=10)
        if job["error"]:
            raise RuntimeError(job["error"])
        return job["result"]
