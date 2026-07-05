# src/services/dense_embedding_service.py

import os
import asyncio
import math
from dataclasses import dataclass

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8001/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "not-needed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# safety + performance tuning
EMBED_TIMEOUT_SECONDS = float(os.getenv("EMBED_TIMEOUT_SECONDS", "20"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
EMBED_RETRY_BACKOFF = float(os.getenv("EMBED_RETRY_BACKOFF", "0.6"))


@dataclass  
class QueryVectors:
    query: str
    dense: list[float]


class EmbeddingService:
    """
    Dense embedding service backed by vLLM OpenAI-compatible endpoint.
    Designed for concurrent RAG workloads.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=EMBED_BASE_URL,
            api_key=EMBED_API_KEY,
            timeout=EMBED_TIMEOUT_SECONDS,
        )
        print(f"[EmbeddingService] Ready (timeout={EMBED_TIMEOUT_SECONDS}s)")

    # public API
    async def embed(self, queries: list[str]) -> list[QueryVectors]:
        """
        Embeds all queries in one batch call via vLLM.
        Returns dense vectors per query.
        """
        response = await self._embed_with_retry(queries)
        return [
            QueryVectors(
                query=q,
                dense=response.data[i].embedding,
            )    
            for i, q in enumerate(queries)
        ]
    
    # network call
    async def _embed_with_retry(self, queries: list[str]):
        last_error = None

        for attempt in range(EMBED_MAX_RETRIES):
            try:
                return await self.client.embeddings.create(
                    model=EMBED_MODEL,
                    input=queries,
                )
            
            except Exception as e:
                last_error = e
                wait_time = EMBED_RETRY_BACKOFF * (2 ** attempt)

                print(
                    f"[EmbeddingService] retry {attempt + 1}/"
                    f"{EMBED_MAX_RETRIES} failed: {e} "
                    f"→ retrying in {wait_time:.2f}s"
                )

                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"[EmbeddingService] embedding failed after "
            f"{EMBED_MAX_RETRIES} retries"
        ) from last_error
    
    async def close(self):
        print("[EmbeddingService] Shutting down bge-m3 dense embedder...")
        await self.client.close()