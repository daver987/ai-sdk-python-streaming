import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.tools import get_current_weather

load_dotenv(".env")

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Request(BaseModel):
    messages: list[ClientMessage]


available_tools = {
    "get_current_weather": get_current_weather,
}


def do_stream(messages: list[ChatCompletionMessageParam]):
    return client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather at a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "The latitude of the location",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "The longitude of the location",
                            },
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            }
        ],
    )


def stream_text(messages: list[ChatCompletionMessageParam], protocol: str = "data"):
    draft_tool_calls = []
    draft_tool_calls_index = -1

    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather at a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "The latitude of the location",
                            },
                            "longitude": {
                                "type": "number",
                                "description": "The longitude of the location",
                            },
                        },
                        "required": ["latitude", "longitude"],
                    },
                },
            }
        ],
    )

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                    )

                for tool_call in draft_tool_calls:
                    tool_result = available_tools[tool_call["name"]](**json.loads(tool_call["arguments"]))

                    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                        result=json.dumps(tool_result),
                    )

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    tool_id = tool_call.id if tool_call and hasattr(tool_call, "id") else None

                    if tool_call and tool_call.function:
                        tool_name = tool_call.function.name if hasattr(tool_call.function, "name") else None
                        tool_args = tool_call.function.arguments if hasattr(tool_call.function, "arguments") else None
                    else:
                        tool_name = None
                        tool_args = None

                    if tool_id is not None:
                        draft_tool_calls_index += 1
                        draft_tool_calls.append({"id": tool_id, "name": tool_name, "arguments": ""})
                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += tool_args if tool_args else ""

            else:
                yield f"0:{json.dumps(choice.delta.content)}\n"

        if chunk.choices == [] and hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens if hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None else 0
            completion_tokens = usage.completion_tokens if hasattr(usage, "completion_tokens") and usage.completion_tokens is not None else 0

            yield ('e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},' '"completionTokens":{completion}}},"isContinued":false}}\n').format(
                reason="tool-calls" if len(draft_tool_calls) > 0 else "stop",
                prompt=prompt_tokens,
                completion=completion_tokens,
            )


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query("data")):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)

    response = StreamingResponse(stream_text(openai_messages, protocol))
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response
