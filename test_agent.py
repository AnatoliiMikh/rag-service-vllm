# test_agent.py

import asyncio
import sys
sys.path.insert(0, 'src')

from rag_agent import LLMAgent


async def test(agent: LLMAgent, message: str, history: list[str] = []):
    print(f"\nQuestion: {message}")
    if history:
        print(f"History: {len(history)} messages")
    print("=" * 50)
    print("Answer: ", end="", flush=True)

    async for token in agent.request(message, history):
        print(token, end="", flush=True)

    print("\n" + "=" * 50)


async def main():
    agent = await LLMAgent.create()

    # Single turn
    await test(agent, "What are the admission requirements?")

    # With history
    await test(
        agent,
        message="What about language requirements specifically?",
        history=[
            "What are the admission requirements?",
            "The program requires a general university entrance qualification...",
        ]
    )

    # Concurrent - two agents, two users simultaneously
    print("\n--- Concurrent test: 2 users simultaneously ---")
    agent2 = await LLMAgent.create()

    await asyncio.gather(
        test(agent, "What courses are in semester 1?"),
        test(agent2, "What specializations are available?"),
    )

    await agent.close()
    await agent2.close()


asyncio.run(main())