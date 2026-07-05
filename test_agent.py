# test_agent.py

import asyncio
import sys
import time
import statistics

sys.path.insert(0, "src")

from rag_agent import LLMAgent


async def run_query(agent: LLMAgent, message: str):
    start = time.perf_counter()

    token_count = 0
    async for token in agent.request(message):
        token_count += 1

    end = time.perf_counter()

    return {
        "latency": end - start,
        "tokens": token_count,
    }


async def worker(agent: LLMAgent, id: int, message: str, results: list):
    result = await run_query(agent, f"[User {id}] {message}")
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

    print("\n--- RESULTS ---")
    print(f"Avg latency: {statistics.mean(latencies):.3f}s")
    print(f"Max latency: {max(latencies):.3f}s")
    print(f"Min latency: {min(latencies):.3f}s")
    print(f"P95 approx: {sorted(latencies)[int(len(latencies)*0.95)-1]:.3f}s")


async def sequential_test(agent: LLMAgent):
    print("\n--- Sequential correctness test ---")

    q1 = await run_query(agent, "What are the admission requirements?")
    q2 = await run_query(agent, "What about language requirements specifically?")

    print(f"Q1 latency: {q1['latency']:.3f}s")
    print(f"Q2 latency: {q2['latency']:.3f}s")


async def burst_test(agent: LLMAgent):
    print("\n--- Burst test (stress spike) ---")

    tasks = [
        run_query(agent, f"Explain topic {i}")
        for i in range(20)
    ]

    results = await asyncio.gather(*tasks)

    latencies = [r["latency"] for r in results]

    print(f"Max latency under burst: {max(latencies):.3f}s")


async def main():
    agent = await LLMAgent.create()

    # 1. correctness
    await sequential_test(agent)

    # 2. low concurrency
    await concurrency_test(agent, concurrency=5)

    # 3. medium concurrency
    await concurrency_test(agent, concurrency=20)

    # 4. burst stress
    await burst_test(agent)

    await agent.close()


asyncio.run(main())