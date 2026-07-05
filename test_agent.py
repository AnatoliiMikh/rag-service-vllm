# test_agent.py

import asyncio
import sys
sys.path.insert(0, 'src')

from rag_agent import LLMAgent


async def test(agent: LLMAgent, message: str, history: list[str] = []):
    print(f"\nQuestion: {message}")
    if history:
        print(f"History entries: {len(history)}")
    print("=" * 50)
    print("Answer: ", end="", flush=True)

    async for token in agent.request(message, history):
        print(token, end="", flush=True)

    print("\n" + "=" * 50)


async def main():
    agent = await LLMAgent.create()

    await test(agent, "What are the admission requirements?")

    await test(
        agent,
        message="What about language requirements specifically?",
        history=[
            "What are the admission requirements?",
            "The program requires a general university entrance qualification...",
        ]
    )

    # Test concurrent requests - two agents, two users simultaneously
    print("\n--- Concurrent test: 2 users simultaneously ---")
    agent2 = await LLMAgent.create()

    await asyncio.gather(
        test(agent, "What courses are in semester 1?"),
        test(agent2, "What specializations are available?"),
    )


asyncio.run(main())