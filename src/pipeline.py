# src/pipeline.py

import asyncio
from typing import AsyncGenerator

from services.llm_service import LLMService
from services.dense_embedding_service import EmbeddingService
from services.sparse_embedding_service import SparseEmbeddingService
from services.reranker_service import RerankerService
from modules.qdrant_hybrid_retrieval import HybridRetrievalModule
from modules.context_builder import build_context


class RAGPipeline:
    def __init__(
        self,
        llm: LLMService,
        embedder: EmbeddingService,
        sparse: SparseEmbeddingService,
        reranker: RerankerService,
        retrieval: HybridRetrievalModule,
    ):
        self._llm = llm
        self._embedder = embedder
        self._sparse = sparse
        self._reranker = reranker
        self._retrieval = retrieval

    async def run(
        self,
        message: str,
        history: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Full async RAG pipeline. Yields tokens as generated.

        Args:
            message: current user question
            history: flat list alternating user/assistant oldest first
                     [user_msg1, asst_msg1, user_msg2, ...]
        """
        if history is None:
            history = []

        # Convert flat history list to OpenAI format
        history_messages = []
        roles = ["user", "assistant"]
        for i, content in enumerate(history):
            history_messages.append({"role": roles[i % 2], "content": content})

        # Step 1 - query expansion (sequential, needed before retrieval)
        queries = await self._llm.expand_query(message)

        # Step 2 - dense + sparse embedding concurrently
        dense_vectors, sparse_vectors = await asyncio.gather(
            self._embedder.embed(queries),
            self._sparse.embed(queries),
        )

        # Step 3 - hybrid Qdrant search (all queries concurrently inside)
        candidate_lists = await self._retrieval.retrieve(
            dense_vectors,
            sparse_vectors,
        )

        # Step 4 - flatten all candidates for reranking
        all_candidates = [
            chunk
            for candidate_list in candidate_lists
            for chunk in candidate_list
        ]
        if not all_candidates: 
            print(f"[RAGPipeline] No retrieval candidates found for {message}.")
            all_candidates = []

        # Step 5 - rerank
        chunks = await self._reranker.rerank(message, all_candidates)

        # Step 6 - assemble prompt
        prompt_messages = build_context(message, history_messages, chunks)

        # Step 7 - stream tokens
        async for token in self._llm.generate_stream(prompt_messages):
            yield token
    
    async def close(self):
        """Clean shutdown of all connections."""
        await self._pipeline._retrieval.close()
        await self._pipeline._reranker.close()
        await self._pipeline._llm.close()
        await self._pipeline._embedder.close()