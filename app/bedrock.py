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
                {"role": "user", "content": [{"text": user_message}]},
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"maxTokens": 2048, "temperature": 0.3},
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            text = result["output"]["message"]["content"][0]["text"]
            usage = result.get("usage") or {}
            if "usage" not in result:
                log.warning("Bedrock response missing usage metadata")
            return InvokeResult(
                text=text,
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
            )
        except Exception as e:
            log.exception("Bedrock invoke failed")
            raise BedrockError(str(e)) from e
