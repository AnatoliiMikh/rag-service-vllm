# src/server.py

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import grpc
import grpc.aio

import rag_service_pb2
import rag_service_pb2_grpc

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.bm25_service import BM25Service
from modules.retrieval import RetrievalModule
from pipeline import RAGPipeline

GRPC_PORT = os.getenv("GRPC_PORT", "50051")


class MessageServiceServicer(rag_service_pb2_grpc.MessageServiceServicer):

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    async def GenerateReply(self, request):
        """
        Receives NewMessageRequest with list of messages (user's prompt + chat history).
        Returns NewMessageResponse with full answer string.
        """
        try:
            messages = list(request.messages)

            if not messages:
                return rag_service_pb2.NewMessageResponse(
                    error="No messages provided.",
                )

            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                self._pipeline.run,
                messages,
            )

            return rag_service_pb2.NewMessageResponse(
                answer=answer,
            )
        
        # Might be changed to native grpc errors handling
        except Exception as e:
            import traceback
            traceback.print_exc()
            return rag_service_pb2.NewMessageResponse(
                error=str(e),
            )


async def main():
    print("[Server] Initializing services...")

    llm = LLMService()
    embedder = EmbeddingService()
    reranker = RerankerService()
    retrieval = RetrievalModule()

    all_chunks = retrieval.load_all_chunks()
    bm25 = BM25Service(all_chunks)

    pipeline = RAGPipeline(
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        bm25=bm25,
        retrieval=retrieval,
    )

    server = grpc.aio.server()
    rag_service_pb2_grpc.add_MessageServiceServicer_to_server(
        MessageServiceServicer(pipeline),
        server,
    )

    listen_addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    await server.start()
    print(f"[Server] Listening on {listen_addr}")
    print("[Server] Ready.")

    async def shutdown():
        print("[Server] Shutting down...")
        await server.stop(grace=5)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(
        __import__("signal").SIGTERM,
        lambda: asyncio.create_task(shutdown()),
    )

    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(main())