import asyncio
import os

from autogen_core.models import (AssistantMessage, ModelFamily, ModelInfo,
                                 SystemMessage, UserMessage)
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    model_client_ollama = OpenAIChatCompletionClient(
        model="qwen2",
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family=ModelFamily.UNKNOWN,
        )
    )

    messages = []
    messages.append(SystemMessage(content="You're a helpful personal assistant"))

    while True:
        user_message = await asyncio.to_thread(input, "User: ")
        if user_message == "exit":
            break

        messages.append(UserMessage(content=user_message, source="user"))
        try:
            response = await model_client_ollama.create(messages=messages)
            print(f"{response.content}\n{response.usage}")
            messages.append(AssistantMessage(content=response.content, source="assistant"))
        except Exception as e:
            print(f"An error occurred: {e}")

asyncio.run(main())