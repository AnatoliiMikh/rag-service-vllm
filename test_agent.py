# test_agent.py

import argparse

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

    test_questions = [
        "What are the admission requirements?",
        "What courses are in semester 1?",
        "What specializations are available?",
        "What is the language of instruction?",
        "How long is the program?",
    ]
    tasks = [
        run_query(agent, test_questions[i % len(test_questions)], silent=True)
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    latencies = [r["latency"] for r in results]

    print(f"Max latency under burst: {max(latencies):.3f}s")


async def interactive_chat(agent: LLMAgent):
    print("\n--- Interactive CLI Chat ---")
    print("Type 'exit' or 'quit' to stop. Press Ctrl+C to abort.")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break
            if not user_input.strip():
                continue

            # Run query and print to console
            result = await run_query(agent, user_input, history=chat_history, silent=False)
            
            # Append to history for context
            chat_history.extend([user_input, result["response"]])
            
            # Optional: keep history array from growing infinitely (Sliding window of last 4 turns)
            if len(chat_history) > 8:
                chat_history = chat_history[-8:]
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

async def interactive_chat(agent: LLMAgent):
    print("\n--- Interactive CLI Chat ---")
    print("Type 'exit' or 'quit' to stop. Press Ctrl+C to abort.")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break
            if not user_input.strip():
                continue

            # Run query and print to console
            result = await run_query(agent, user_input, history=chat_history, silent=False)
            
            # Append to history for context
            chat_history.extend([user_input, result["response"]])
            
            # Optional: keep history array from growing infinitely (Sliding window of last 4 turns)
            if len(chat_history) > 8:
                chat_history = chat_history[-8:]
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

async def main():
    parser = argparse.ArgumentParser(description="University RAG Pipeline CLI")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat session")
    parser.add_argument("--test", action="store_true", help="Run the full performance test suite")
    args = parser.parse_args()

    # If no arguments provided, print help and exit
    if not args.chat and not args.test:
        parser.print_help()
        return

    agent = await LLMAgent.create()

    try:
        if args.chat:
            await interactive_chat(agent)
            
        if args.test:
            await sequential_test(agent)
            await concurrency_test(agent, concurrency=5)
            await concurrency_test(agent, concurrency=20)
            await burst_test(agent)
    finally:
        # Guarantee teardown happens even if you Ctrl+C out of the chat
        await agent.close()

if __name__ == "__main__":
    asyncio.run(main())