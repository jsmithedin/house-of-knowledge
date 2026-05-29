import json
import logging

import boto3

log = logging.getLogger(__name__)


class BedrockError(Exception):
    pass


class BedrockClient:
    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, system_prompt: str, user_message: str) -> str:
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
            return result["output"]["message"]["content"][0]["text"]
        except Exception as e:
            log.exception("Bedrock invoke failed")
            raise BedrockError(str(e)) from e
