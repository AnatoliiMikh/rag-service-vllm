# src/rag_agent.py

import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

from services.llm_service import LLMService
from services.dense_embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.bm25_service import BM25Service
from modules.retrieval import RetrievalModule
from pipeline import RAGPipeline

load_dotenv()

END_TOKEN = "<end>"


class LLMAgent:
    """
    Async RAG pipeline interface for BE.

    Initialize once:
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
        self._semaphore = asyncio.Semaphore(1)  # one request at a time per instance

    @classmethod
    async def create(cls) -> "LLMAgent":
        """
        Async factory. Initializes all services and returns ready agent.
        Call once at application startup.
        """
        print("[LLMAgent] Initializing...")

        llm      = LLMService()
        embedder = EmbeddingService()
        reranker = RerankerService()
        retrieval = RetrievalModule()

        all_chunks = await retrieval.load_all_chunks()
        bm25 = BM25Service(all_chunks)

        pipeline = RAGPipeline(
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            bm25=bm25,
            retrieval=retrieval,
        )

        print("[LLMAgent] Ready.")
        return cls(pipeline)

    async def request(
        self,
        message: str,
        history: list[str] = [],
    ) -> AsyncGenerator[str, None]:
        """
        Runs RAG pipeline and streams tokens.
        Semaphore ensures one active generation per agent instance.
        BE can create multiple LLMAgent instances for true concurrency.
        """
        async with self._semaphore:
            async for token in self._pipeline.run(message, history):
                yield token