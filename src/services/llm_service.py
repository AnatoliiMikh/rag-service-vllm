# src/services/llm_service.py

import os
import json
import asyncio
from typing import AsyncGenerator
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "0.5"))

EXPANSION_PROMPT = """\
You are a search query rewriter for a university information retrieval system.

Given the user's question, generate exactly 3 alternative search queries.
Each query must:
- Be semantically diverse (not just paraphrases)
- Target a different aspect or perspective of the question
- Be self-contained and specific
- Be in the same language as the original question

Respond with ONLY a JSON array of 3 strings. No explanation, no markdown.

Example input: "What GPA do I need to apply?"
Example output: ["minimum GPA requirement for admission", "academic score threshold for university application", "grade point average eligibility criteria for enrollment"]

User question: {message}
"""

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0, 
        )
        print("[LLMService] Ready.")

    async def expand_query(self, message: str) -> list[str]:
        """
        Generates 3 diverse query variations via vLLM.
        Falls back to [message x3] on malformed output or network failure..
        """
        last_error = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": EXPANSION_PROMPT.format(message=message),
                    }],
                    temperature=0.7,
                    max_tokens=256,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = response.choices[0].message.content.strip()
                try:
                    queries = json.loads(raw)
                    if (isinstance(queries, list) 
                        and len(queries) == 3 
                        and all(isinstance(q, str) for q in queries)
                    ):
                        return queries
                except json.JSONDecodeError:
                    pass
                print(f"[LLMService] expand_query: malformed output, falling back.\nRaw: {raw}")
                return [message, message, message]

            except Exception as e:
                last_error = e
                wait_time = LLM_RETRY_BACKOFF * (2 ** attempt)
                print(f"[LLMService] expand_query retry {attempt + 1}/{LLM_MAX_RETRIES}: {e}")
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)

        print(f"[LLMService] expand_query failed after {attempt} retries, falling back.")
        return [message, message, message]

    async def generate_stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """
        Streams tokens from vLLM one by one.
        """
        stream = await self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def close(self):
        print("[LLMService] Shutting down llm_service...")
        await self.client.close()