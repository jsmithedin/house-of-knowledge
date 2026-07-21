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


def _build_tool_config(tools: list[dict]) -> dict:
    """Convert internal tool defs (Anthropic-style input_schema) to Converse toolConfig."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {"json": tool["input_schema"]},
                }
            }
            for tool in tools
        ]
    }


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Accept Converse or legacy Anthropic message blocks."""
    normalized = []
    for msg in messages:
        content = []
        for block in msg["content"]:
            if "text" in block:
                content.append({"text": block["text"]})
            elif block.get("type") == "text":
                content.append({"text": block["text"]})
            elif "toolResult" in block:
                content.append(block)
            else:
                content.append(block)
        normalized.append({"role": msg["role"], "content": content})
    return normalized


def _extract_text(content: list[dict]) -> str:
    for block in content:
        if "text" in block:
            return block["text"]
    raise BedrockError("No text block in model response")


def _usage_tokens(response: dict) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))


class BedrockClient:
    """Bedrock Runtime client using the unified Converse API."""

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        inference_config: dict | None = None,
    ) -> InvokeResult:
        cfg = inference_config if inference_config is not None else {"maxTokens": 2048}
        try:
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": user_message}]},
                ],
                inferenceConfig=cfg,
            )
        except Exception as e:
            log.exception("Bedrock converse failed (model_id=%s)", self.model_id)
            raise BedrockError(str(e)) from e

        stop_reason = response.get("stopReason")
        if stop_reason != "end_turn":
            raise BedrockError(f"Unexpected stopReason: {stop_reason!r}")

        message = response["output"]["message"]
        input_tokens, output_tokens = _usage_tokens(response)
        return InvokeResult(
            text=_extract_text(message["content"]),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def invoke_with_tools(
        self,
        system_prompt: str,
        initial_messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        max_iterations: int = 5,
        on_iteration: Callable[[int, str, dict], None] | None = None,
        inference_config: dict | None = None,
    ) -> InvokeResult:
        messages = _normalize_messages(initial_messages)
        tool_config = _build_tool_config(tools)
        cfg = inference_config if inference_config is not None else {"maxTokens": 2048}
        total_input = total_output = 0

        for i in range(max_iterations):
            try:
                response = self._client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    toolConfig=tool_config,
                    inferenceConfig=cfg,
                )
            except Exception as e:
                log.exception(
                    "Bedrock converse with tools failed (model_id=%s)", self.model_id
                )
                raise BedrockError(str(e)) from e

            input_tokens, output_tokens = _usage_tokens(response)
            total_input += input_tokens
            total_output += output_tokens

            stop_reason = response.get("stopReason")
            output_message = response["output"]["message"]
            messages.append(output_message)

            if on_iteration is not None:
                on_iteration(i + 1, stop_reason, output_message)

            if stop_reason == "end_turn":
                return InvokeResult(
                    text=_extract_text(output_message["content"]),
                    input_tokens=total_input,
                    output_tokens=total_output,
                )
            if stop_reason != "tool_use":
                raise BedrockError(f"Unexpected stopReason: {stop_reason!r}")

            tool_results = []
            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue
                tool_use = block["toolUse"]
                result_str = tool_executor(tool_use["name"], tool_use["input"])
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": result_str}],
                        "status": "success",
                    }
                })
            if not tool_results:
                raise BedrockError("stopReason tool_use but no toolUse blocks in response")
            messages.append({"role": "user", "content": tool_results})

        raise BedrockError(f"Max iterations ({max_iterations}) reached without end_turn")
