using Grpc.Net.Client;
using LLMWorkerAPI;

var channel = GrpcChannel.ForAddress("http://172.20.0.2:50051");
var client = new MessageService.MessageServiceClient(channel);

var request = new NewMessageRequest();
request.Messages.Add("What are the admission requirements?");

var response = await client.GenerateReplyAsync(request);

if (!string.IsNullOrEmpty(response.Error))
{
    Console.WriteLine(response.Error);
}
else
{
    Console.WriteLine(response.Answer);
}