import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-6"


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def chat(messages, system=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
    }
    if system:
        params["system"] = system
    with anthropic.messages.stream(**params) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        return stream.get_final_message().content[0].text


system_prompt = "You are a helpful assistant that responds only in pirate speak."
messages = []

print("Chat with Claude (type 'quit' to exit)")
print("----------------------------------------")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    add_user_message(messages, user_input)
    print("Claude: ", end="", flush=True)
    reply = chat(messages)
    add_assistant_message(messages, reply)
    print()
