import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import boto3

log = logging.getLogger(__name__)


class BedrockError(Exception):
    pass


@dataclass(frozen=True)
class InvokeResult:
    text: str
    input_tokens: int
    output_tokens: int


class BedrockClient:
    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, system_prompt: str, user_message: str) -> InvokeResult:
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_message}]},
            ],
            "system": [{"type": "text", "text": system_prompt}],
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31",
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            text = result["content"][0]["text"]
            usage = result.get("usage") or {}
            if "usage" not in result:
                log.warning("Bedrock response missing usage metadata")
            return InvokeResult(
                text=text,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            )
        except Exception as e:
            log.exception("Bedrock invoke failed")
            raise BedrockError(str(e)) from e

    def invoke_with_tools(
        self,
        system_prompt: str,
        initial_messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_iterations: int = 5,
    ) -> InvokeResult:
        messages = list(initial_messages)
        total_input = total_output = 0

        for _ in range(max_iterations):
            body = {
                "messages": messages,
                "system": [{"type": "text", "text": system_prompt}],
                "tools": tools,
                "max_tokens": 2048,
                "anthropic_version": "bedrock-2023-05-31",
            }
            try:
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                )
                resp = json.loads(response["body"].read())
            except Exception as e:
                log.exception("Bedrock invoke_with_tools failed")
                raise BedrockError(str(e)) from e

            usage = resp.get("usage") or {}
            total_input += int(usage.get("input_tokens", 0))
            total_output += int(usage.get("output_tokens", 0))

            if resp["stop_reason"] == "end_turn":
                text = next(b["text"] for b in resp["content"] if b["type"] == "text")
                return InvokeResult(text=text, input_tokens=total_input, output_tokens=total_output)
            elif resp["stop_reason"] != "tool_use":
                raise BedrockError(f"Unexpected stop_reason: {resp['stop_reason']!r}")

            messages.append({"role": "assistant", "content": resp["content"]})
            tool_results = []
            for block in resp["content"]:
                if block["type"] == "tool_use":
                    result_str = tool_executor(block["name"], block["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result_str,
                    })
            messages.append({"role": "user", "content": tool_results})

        raise BedrockError(f"Max iterations ({max_iterations}) reached without end_turn")
