from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-sonnet-20240229",  # safest model to try
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.content)
