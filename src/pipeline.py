# src/pipeline.py

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.bm25_service import BM25Service
from modules.retrieval import RetrievalModule
from modules.context_builder import build_context


class RAGPipeline:
    def __init__(
        self,
        llm: LLMService,
        embedder: EmbeddingService,
        reranker: RerankerService,
        bm25: BM25Service,
        retrieval: RetrievalModule,
    ):
        self._llm = llm
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = bm25
        self._retrieval = retrieval

    def run(self, messages: list[str]) -> str:
        """
        Receives pre-formatted list of strings from BE:
            messages[0] = current user message
            messages[1:] = conversation history (optional)

        Returns full answer string.
        """
        user_message = messages[0]
        history_strings = messages[1:] if len(messages) > 1 else []

        # Convert history strings to OpenAI format
        # BE passes alternating user/assistant messages oldest first
        history = []
        roles = ["user", "assistant"]
        for i, content in enumerate(history_strings):
            history.append({"role": roles[i % 2], "content": content})

        # Query expansion
        queries = self._llm.expand_query(user_message)

        # Dense retrieval
        vectors = self._embedder.embed(queries)
        dense_lists = self._retrieval.retrieve(vectors)

        # BM25 retrieval
        bm25_lists = [self._bm25.search(q) for q in queries]

        # RRF + cross-encoder rerank
        chunks = self._reranker.rerank(user_message, dense_lists, bm25_lists)

        # Assemble prompt
        prompt_messages = build_context(user_message, history, chunks)

        # Generate answer
        return self._llm.generate(prompt_messages)