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


class LLMAgent:
    """
        Unified entry point for the high-concurrency RAG pipeline.

        This class encapsulates all generation services. It acts
        as a stateless, thread-safe manager designed to be instantiated exactly once
        at application startup using its asynchronous factory method.

        Example:
            # At FastAPI application startup (lifespan hook):
            app.state.agent = await LLMAgent.create()

            # Shutdown:
            await app.state.agent.close()

            Remark: close() could still work incorrectly
        """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    @classmethod
    async def create(cls) -> "LLMAgent":
        """
        Async factory. Initializes all services.
        Because this is called once at server boot, all initialized services 
        naturally act as application-wide singletons.
        """
        print("[LLMAgent] Initializing...")

        llm = LLMService()
        embedder = EmbeddingService()
        sparse = SparseEmbeddingService()
        reranker = RerankerService()
        retrieval = HybridRetrievalModule()

        pipeline = RAGPipeline(
            llm=llm,
            embedder=embedder,
            sparse=sparse,
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
        Processes a user question through the RAG pipeline and streams response tokens.

        This method is inherently thread-safe and safe for concurrent execution 
        across hundreds of overlapping client tasks.

        Args:
            message: The current, raw user query text.
            history: A flat list alternating between user and assistant messages,
                ordered oldest to newest (e.g., [user_1, assistant_1, user_2]).
                Defaults to None if starting a new conversation session.

        Yields:
            str: Individual text tokens as they stream from the LLM.
        """
        async for token in self._pipeline.run(message, history):
                    yield token
                    
    async def close(self):
        """Clean shutdown of all connections."""
        print("[LLMAgent] Shutting down services...")
        await self._pipeline.close()