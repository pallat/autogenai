import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
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

    assistant_agent = AssistantAgent(
        name="assistant",
        model_client=model_client_ollama,
        system_message="You're a helpful agent"
    )

    messages = []  # Defined messages list

    while True:
        user_message = await asyncio.to_thread(input, "User: ")
        if user_message == "exit":
            break

        cancellation_token = CancellationToken()
        message = TextMessage(content=user_message, source="user")

        try:
            response = await assistant_agent.on_messages(
                messages=[message],
                cancellation_token=cancellation_token,
            )
            print(f"{response.chat_message.content}\n{response.chat_message.models_usage}")
            messages.append(AssistantMessage(content=response.chat_message.content, source="assistant"))
        except Exception as e:
            print(f"An error occurred: {e}")

asyncio.run(main())