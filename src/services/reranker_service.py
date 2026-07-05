# src/services/reranker_service.py

import os
import httpx
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv

from modules.qdrant_hybrid_retrieval import RetrievedChunk

load_dotenv()

RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "http://localhost:8002/v1")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
RERANKER_TIMEOUT_SECONDS = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "30"))
RERANKER_MAX_RETRIES = int(os.getenv("RERANKER_MAX_RETRIES", "3"))
RERANKER_RETRY_BACKOFF = float(os.getenv("RERANKER_RETRY_BACKOFF", "0.5"))


@dataclass(slots=True)
class RerankedChunk:
    text: str
    source_file: str
    page: int
    query: str
    id: str
    retrieval_score: float
    reranker_score: float


class RerankerService:
    """
    Cross-encoder reranking via vLLM score endpoint.
    """

    def __init__(self):
        # httpx.AsyncClient inherently uses a connection pool.
        # This is safe and performant for concurrent server use.
        self.client = httpx.AsyncClient(
            timeout=RERANKER_TIMEOUT_SECONDS,
        )
        print(f"[RerankerService] Ready (top_n={RERANKER_TOP_N})")

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RerankedChunk]:

        if not candidates:
            return []

        # Deduplicate chunks by ID to save massive GPU compute.
        # Dict comprehension inherently keeps only the first seen unique ID.
        unique_candidates_map = {}
        for chunk in candidates:
            if chunk.id not in unique_candidates_map:
                unique_candidates_map[chunk.id] = chunk
                
        unique_candidates = list(unique_candidates_map.values())

        scores = await self._score_with_retry(
            query=query,
            candidates=unique_candidates,
        )

        # Prevent silent truncation if the API returns mismatched data
        if len(scores) != len(unique_candidates):
            raise ValueError(
                f"[RerankerService] API returned {len(scores)} scores "
                f"for {len(unique_candidates)} candidates."
            )

        reranked = [
            RerankedChunk(
                text=chunk.text,
                source_file=chunk.source_file,
                page=chunk.page,
                query=chunk.query,
                id=chunk.id,
                retrieval_score=chunk.score,
                reranker_score=float(score),
            )
            for chunk, score in zip(unique_candidates, scores)
        ]

        reranked.sort(
            key=lambda x: x.reranker_score,
            reverse=True,
        )

        return reranked[:RERANKER_TOP_N]

    async def _score_with_retry(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[float]:

        payload = {
            "model": RERANKER_MODEL,
            "text_1": query,
            "text_2": [chunk.text for chunk in candidates],
        }

        last_error = None
        for attempt in range(RERANKER_MAX_RETRIES):
            try:
                response = await self.client.post(
                    f"{RERANKER_BASE_URL}/score",
                    json=payload,
                )
                
                response.raise_for_status() 
                
                data = response.json()
                return [
                    float(item["score"])
                    for item in data["data"]
                ]

            except Exception as e:
                last_error = e
                wait_time = RERANKER_RETRY_BACKOFF * (2 ** attempt)
                print(
                    f"[RerankerService] retry {attempt + 1}/"
                    f"{RERANKER_MAX_RETRIES} failed: {e} "
                    f"→ retrying in {wait_time:.2f}s"
                )
                if attempt < RERANKER_MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)

        raise RuntimeError(
            "[RerankerService] reranking failed after "
            f"{RERANKER_MAX_RETRIES} retries."
        ) from last_error

    async def close(self):
        """Cleanly closes HTTP connections."""
        print("[RerankerService] Shutting down reranker...")
        await self.client.close()