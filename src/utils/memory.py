import logging
import queue
import threading
import time
from uuid import uuid4

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Only recall memories that are at least this semantically similar to the query.
MIN_SCORE = 0.4


class _BaseMemory:
    """Shared embedding source for both the Qdrant and RAM implementations."""

    def __init__(self, embed_url: str, embed_model: str):
        self.embed_url = embed_url.rstrip("/")
        self.embed_model = embed_model

    def store(self, role: str, content: str):
        raise NotImplementedError

    def search(self, query: str, limit: int = 5) -> list[str]:
        raise NotImplementedError

    def _embed(self, text: str) -> list[float]:
        res = requests.post(
            f"{self.embed_url}/embeddings",
            json={"model": self.embed_model, "input": text},
            timeout=10,
        )
        res.raise_for_status()
        return res.json()["data"][0]["embedding"]


class Memory(_BaseMemory):
    """Long-term conversational memory backed by Qdrant.

    Each user/assistant turn is embedded (via the LLM server's OpenAI-compatible
    /v1/embeddings endpoint) and stored in a Qdrant collection. On every new input
    the most similar past turns are recalled and injected into the prompt, so the
    bot remembers across sessions and restarts.
    """

    def __init__(self, host: str, embed_url: str, embed_model: str, collection: str = "conversation_memory"):
        super().__init__(embed_url, embed_model)
        self.client = QdrantClient(url=host, timeout=10)
        self.collection = collection
        self._ready = False

    def check(self):
        """Raise if the Qdrant server is unreachable (e.g. for a startup probe)."""
        self.client.get_collections()

    def _ensure_ready(self):
        """Create the collection if it doesn't exist yet (once per process)."""
        if self._ready:
            return
        if not self.client.collection_exists(self.collection):
            # Need the embedding dimension before we can create the collection.
            dim = len(self._embed("ping"))
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._ready = True

    def store(self, role: str, content: str):
        """Persist one turn (role is 'user' or 'assistant')."""
        self._ensure_ready()
        vector = self._embed(content)
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=uuid4().int & ((1 << 63) - 1),
                    vector=vector,
                    payload={"role": role, "content": content, "ts": time.time()},
                )
            ],
        )

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return the contents of the most similar stored turns."""
        self._ensure_ready()
        vector = self._embed(query)
        result = self.client.query_points(
            self.collection, query=vector, limit=limit, score_threshold=MIN_SCORE
        )
        return [
            p.payload.get("content")
            for p in result.points
            if p.payload and p.payload.get("content")
        ]


class RamMemory(_BaseMemory):
    """In-process fallback: same embedding-based recall as Memory, but nothing is
    persisted (lost when the app exits). Used when Qdrant is unavailable."""

    def __init__(self, embed_url: str, embed_model: str):
        super().__init__(embed_url, embed_model)
        self._points: list[dict] = []

    def store(self, role: str, content: str):
        """Keep one turn in the in-process list (role is 'user' or 'assistant')."""
        self._points.append(
            {
                "vector": self._embed(content),
                "role": role,
                "content": content,
                "ts": time.time(),
            }
        )

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return the contents of the most similar stored turns (cosine)."""
        query_vec = np.asarray(self._embed(query))
        query_norm = np.linalg.norm(query_vec)
        results = []
        for point in self._points:
            vec = np.asarray(point["vector"])
            score = float(np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec)))
            if score >= MIN_SCORE:
                results.append((score, point["content"]))
        results.sort(key=lambda r: -r[0])
        return [content for _, content in results[:limit]]


class MemoryWorker:
    """Runs all embedding work (store + recall) on a background thread so the
    conversation loop is never blocked by embedding HTTP calls.

    Exposes the same store/search interface as Memory/RamMemory:
      - store() is fire-and-forget (jobs are dropped, never block, if the worker
        is backed up).
      - search() submits a recall job and blocks until the worker returns it.
    """

    def __init__(self, memory: _BaseMemory, max_queue: int = 50):
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
