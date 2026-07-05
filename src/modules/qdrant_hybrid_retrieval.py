# src/modules/qdrant_hybrid_retrieval.py

import os
import asyncio

from dataclasses import dataclass

from dotenv import load_dotenv

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from services.dense_embedding_service import QueryVectors
from services.sparse_embedding_service import QuerySparseVectors

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "data_ds_bsc")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))

QDRANT_TIMEOUT_SECONDS = float(
    os.getenv("QDRANT_TIMEOUT_SECONDS", "20")
)

QDRANT_MAX_RETRIES = int(
    os.getenv("QDRANT_MAX_RETRIES", "3")
)

QDRANT_RETRY_BACKOFF = float(
    os.getenv("QDRANT_RETRY_BACKOFF", "0.5")
)

DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "sparse"

# your ingestion currently stores content here
PAYLOAD_CONTENT_FIELD = "summary"


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    source_file: str
    page: int
    score: float
    query: str
    id: str


class HybridRetrievalModule:
    """
    Qdrant-native hybrid retrieval.

    Dense retrieval:
        BGE-M3 dense vectors

    Sparse retrieval:
        FastEmbed BM25 sparse vectors

    Fusion:
        Qdrant native Reciprocal Rank Fusion (RRF)

    Returns:
        Strongly typed RetrievedChunk objects.
    """

    def __init__(self):

        self.client = AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )

        print(
            "[HybridRetrievalModule] Ready "
            f"(collection={QDRANT_COLLECTION})"
        )

    async def retrieve(
        self,
        dense_vectors: list[QueryVectors],
        sparse_vectors: list[QuerySparseVectors],
    ) -> list[list[RetrievedChunk]]:
        """
        Executes hybrid retrieval for all expanded queries.

        Each query expansion is searched independently.

        Returns:
            [
                [RetrievedChunk, ...],
                [RetrievedChunk, ...],
                ...
            ]
        """

        if len(dense_vectors) != len(sparse_vectors):
            raise ValueError(
                "Dense and sparse query counts do not match."
            )

        tasks = [
            self._retrieve_single(
                dense_qv,
                sparse_qv,
            )
            for dense_qv, sparse_qv in zip(
                dense_vectors,
                sparse_vectors,
            )
        ]

        return await asyncio.gather(*tasks)

    async def _retrieve_single(
        self,
        dense_qv: QueryVectors,
        sparse_qv: QuerySparseVectors,
    ) -> list[RetrievedChunk]:

        response = await self._query_with_retry(
            dense_qv,
            sparse_qv,
        )
        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    text=payload.get(PAYLOAD_CONTENT_FIELD, ""),
                    source_file=payload.get("source", ""),
                    page=payload.get("page", 0),
                    score=float(point.score),
                    query=dense_qv.query,
                    id=str(point.id),
                )
            )

        return chunks

    async def _query_with_retry(
        self,
        dense_qv: QueryVectors,
        sparse_qv: QuerySparseVectors,
    ):
        last_error = None
        for attempt in range(QDRANT_MAX_RETRIES):
            try:
                return await self.client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_qv.sparse.indices,
                                values=sparse_qv.sparse.values,
                            ),
                            using=SPARSE_VECTOR_NAME,
                            limit=RETRIEVAL_TOP_K,
                        ),
                        models.Prefetch(
                            query=dense_qv.dense,
                            using=DENSE_VECTOR_NAME,
                            limit=RETRIEVAL_TOP_K,
                        ),
                    ],
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF,
                    ),
                    limit=RETRIEVAL_TOP_K,
                    with_payload=True,
                )

            except Exception as e:
                last_error = e
                wait_time = (
                    QDRANT_RETRY_BACKOFF * (2 ** attempt)
                )
                print(
                    f"[HybridRetrievalModule] "
                    f"retry {attempt + 1}/"
                    f"{QDRANT_MAX_RETRIES} failed: {e}"
                )
                if attempt < QDRANT_MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)

        raise RuntimeError(
            "[HybridRetrievalModule] "
            "Qdrant hybrid retrieval failed "
            f"after {QDRANT_MAX_RETRIES} retries."
        ) from last_error

    async def close(self):
        """
        Cleanly closes Qdrant connections.
        Called during application shutdown.
        """
        print("[HybridRetrievalModule] Shutting down qdrant connections...")
        await self.client.close()