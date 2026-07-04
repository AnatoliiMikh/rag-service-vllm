# test_grpc.py

import sys
sys.path.insert(0, 'src')

import grpc
import rag_service_pb2
import rag_service_pb2_grpc


def test(user_message: str, history: list[str] = []):
    channel = grpc.insecure_channel('localhost:50051')
    stub = rag_service_pb2_grpc.MessageServiceStub(channel)

    # messages[0] = current user message, messages[1:] = history
    messages = [user_message] + history

    request = rag_service_pb2.NewMessageRequest(
        conversation_id=1,
        messages=messages,
    )

    print(f"Sending: {user_message}")
    if history:
        print(f"History: {history}")
    print("=" * 50)

    response = stub.GenerateReply(request)

    if response.error:
        print(f"[ERROR] {response.error}")
    else:
        print(response.answer)

    channel.close()


if __name__ == "__main__":
    # Test without history
    test("What are the admission requirements?")

    print("\n" + "=" * 50 + "\n")

    # Test with history
    test(
        user_message="What about language requirements specifically?",
        history=[]
    )