# src/services/llm_service.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

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

_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class LLMService:
    def __init__(self):
        print("[LLMService] Connecting to vLLM...")
        # verify connection
        models = _client.models.list()
        print(f"[LLMService] Connected. Available models: {[m.id for m in models.data]}")
        print("[LLMService] Ready.")

    def expand_query(self, message: str) -> list[str]:
        """
        Calls vLLM to generate 3 diverse query variations.
        Falls back to [message x3] on malformed output.
        """
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": EXPANSION_PROMPT.format(message=message)}],
            temperature=0.7,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        raw = response.choices[0].message.content.strip()

        try:
            queries = json.loads(raw)
            if isinstance(queries, list) and len(queries) == 3:
                return queries
        except json.JSONDecodeError:
            pass

        print(f"[LLMService] expand_query: malformed output, falling back.\nRaw: {raw}")
        return [message, message, message]

    def generate(self, messages: list[dict]) -> str:
        """
        Sends assembled prompt to vLLM.
        Returns full answer string.
        """
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content.strip()