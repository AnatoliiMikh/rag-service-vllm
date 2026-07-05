# test_agent.py

import asyncio
import sys
import time
import statistics
from dotenv import load_dotenv

load_dotenv("/home/user1/.env")

sys.path.insert(0, "src")
from rag_agent import LLMAgent


async def run_query(
    agent: LLMAgent, 
    message: str, 
    history: list[str] | None = None, 
    silent: bool = True
):
    start = time.perf_counter()
    
    token_count = 0
    full_response = ""
    
    if not silent:
        print(f"\nUser: {message}\nAI: ", end="")

    # Capture the text while streaming
    async for token in agent.request(message, history):
        token_count += 1
        full_response += token
        if not silent:
            print(token, end="", flush=True)

    if not silent:
        print("\n")

    end = time.perf_counter()

    return {
        "latency": end - start,
        "tokens": token_count,
        "response": full_response, # Return the text for history tracking
    }


async def worker(agent: LLMAgent, id: int, message: str, results: list):
    # Keep stress test workers silent so they don't spam the Jupyter output
    result = await run_query(agent, f"[User {id}] {message}", silent=True)
    results.append(result)


async def concurrency_test(agent: LLMAgent, concurrency: int):
    print(f"\n--- Concurrency test: {concurrency} users ---")

    results = []
    tasks = [
        worker(agent, i, "What are the admission requirements?", results)
        for i in range(concurrency)
    ]

    await asyncio.gather(*tasks)

    latencies = [r["latency"] for r in results]

    print("--- RESULTS ---")
    print(f"Avg latency: {statistics.mean(latencies):.3f}s")
    print(f"Max latency: {max(latencies):.3f}s")
    print(f"Min latency: {min(latencies):.3f}s")
    
    # Safer P95 calculation
    sorted_lats = sorted(latencies)
    p95_idx = int(len(sorted_lats) * 0.95)
    # Ensure index doesn't go out of bounds on tiny concurrency
    p95_idx = min(p95_idx, len(sorted_lats) - 1) 
    print(f"P95 approx:  {sorted_lats[p95_idx]:.3f}s")


async def sequential_test(agent: LLMAgent):
    print("\n--- Sequential correctness test ---")
    
    # Build a persistent history array to prove stateless RAG works
    chat_history = []

    q1_text = "What are the admission requirements?"
    q1 = await run_query(agent, q1_text, history=None, silent=False)
    
    chat_history.extend([q1_text, q1["response"]])

    q2_text = "What about language requirements specifically?"
    q2 = await run_query(agent, q2_text, history=chat_history, silent=False)

    print(f"Q1 latency: {q1['latency']:.3f}s")
    print(f"Q2 latency: {q2['latency']:.3f}s")


async def burst_test(agent: LLMAgent):
    print("\n--- Burst test (stress spike) ---")

    tasks = [
        run_query(agent, f"Explain topic {i}", silent=True)
        for i in range(20)
    ]

    results = await asyncio.gather(*tasks)
    latencies = [r["latency"] for r in results]

    print(f"Max latency under burst: {max(latencies):.3f}s")


async def main():
    agent = await LLMAgent.create()

    # 1. correctness (Loud, prints text)
    await sequential_test(agent)

    # 2. low concurrency (Silent, prints metrics)
    await concurrency_test(agent, concurrency=5)

    # 3. medium concurrency (Silent, prints metrics)
    await concurrency_test(agent, concurrency=20)

    # 4. burst stress (Silent, prints metrics)
    await burst_test(agent)

    await agent.close()

asyncio.run(main())