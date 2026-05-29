import json
import logging
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
