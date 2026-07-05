# src/modules/context_builder.py

from services.reranker_service import RerankedChunk

SYSTEM_PROMPT = """\
You are a precise and helpful university information assistant.
Your answers are based STRICTLY on the provided context chunks below.

Rules you must follow:
- Answer in the same language the user wrote in.
- If the context does not contain enough information to answer, say exactly:
  "I cannot find this information in the available documents."
- Never speculate or use knowledge outside the provided context.
- Always cite your sources at the end of your answer in this format:
  [Source: <source_file>, Page <page>]
  Do not cite documents that were not used.
- Be concise and direct. Do not repeat the question back.
"""


def build_context(
    message: str,
    history: list[dict],
    chunks: list[RerankedChunk],
    max_history_messages: int = 4,
) -> list[dict]:
    """
    Assembles final prompt as OpenAI message list.
    Combines system instructions, retrieved chunks with citations,
    chat history, and current user message.
    """
    context_str = "\n\n".join(
        f"[Source: {c.source_file} | Page {c.page}]\n{c.text}"
        for i, c in enumerate(chunks)
    )

    system_message = {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n\n### RETRIEVED CONTEXT\n{context_str}",
    }

    user_message = {
        "role": "user",
        "content": message,
    }

    safe_history = history[-max_history_messages:] if history else []

    return [system_message] + safe_history + [user_message]