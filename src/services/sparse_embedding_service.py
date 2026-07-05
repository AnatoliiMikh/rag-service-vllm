# src/services/sparse_embedding_service.py

import asyncio
from dataclasses import dataclass

from fastembed import SparseTextEmbedding


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class QuerySparseVectors:
    query: str
    sparse: SparseVector


class SparseEmbeddingService:
    """
    FastEmbed BM25 sparse encoder.
    Batch-first, single execution path.
    """

    def __init__(self):
        print("[SparseEmbeddingService] Loading fastembed BM25...")
        self.encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        print("[SparseEmbeddingService] Ready")

    def _sync_encode_batch(self, queries: list[str]) -> list[SparseVector]:
        results = list(self.encoder.embed(queries))

        return [
            SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]

    async def embed(self, queries: list[str]) -> list[QuerySparseVectors]:
        """
        Batch-encodes queries into sparse vectors.

        Runs CPU work in a single thread to avoid async overhead.
        """

        try:
            sparse_vectors = await asyncio.to_thread(
                self._sync_encode_batch,
                queries
            )

        except Exception as e:
            raise RuntimeError(
                "[SparseEmbeddingService] Failed to encode sparse vectors"
            ) from e

        return [
            QuerySparseVectors(
                query=q,
                sparse=sparse_vectors[i],
            )
            for i, q in enumerate(queries)
        ]