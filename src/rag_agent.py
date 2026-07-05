# src/rag_agent.py

import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

from services.llm_service import LLMService
from services.dense_embedding_service import EmbeddingService
from services.sparse_embedding_service import SparseEmbeddingService
from services.reranker_service import RerankerService
from modules.qdrant_hybrid_retrieval import HybridRetrievalModule
from pipeline import RAGPipeline

load_dotenv()

END_TOKEN = "<end>"

_shared_sparse: SparseEmbeddingService | None = None


class LLMAgent:
    """
    Async RAG pipeline interface.

    Initialize once at application startup:
        agent = await LLMAgent.create()

    Token streaming:
        async for token in agent.request(message, history):
            print(token, end="", flush=True)

    Args:
        message: current user question
        history: flat list alternating user/assistant oldest first
                 [user_msg1, asst_msg1, user_msg2, asst_msg2, ...]
    """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline
        self._semaphore = asyncio.Semaphore(1)

    @classmethod
    async def create(cls) -> "LLMAgent":
        """
        Async factory. Initializes all services.
        SparseEmbeddingService is a shared singleton (model loaded once).
        Call once at application startup.
        """
        global _shared_sparse
        print("[LLMAgent] Initializing...")

        llm = LLMService()
        embedder = EmbeddingService()
        reranker = RerankerService()
        retrieval = HybridRetrievalModule()

        if _shared_sparse is None:
            _shared_sparse = SparseEmbeddingService()

        pipeline = RAGPipeline(
            llm=llm,
            embedder=embedder,
            sparse=_shared_sparse,
            reranker=reranker,
            retrieval=retrieval,
        )

        print("[LLMAgent] Ready.")
        return cls(pipeline)

    async def request(
        self,
        message: str,
        history: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Runs RAG pipeline and streams tokens.
        Semaphore ensures one active generation per agent instance.
        Create multiple LLMAgent instances for true concurrency.
        """
        async with self._semaphore:
            async for token in self._pipeline.run(message, history):
                yield token

    async def close(self):
        """Clean shutdown of all connections."""
        await self._pipeline.close()