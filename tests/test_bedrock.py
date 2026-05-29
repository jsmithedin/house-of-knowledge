import json
from unittest.mock import MagicMock, patch
from app.bedrock import BedrockClient, BedrockError


def test_invoke_success():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({"output": {"message": {"content": [{"text": "Answer"}]}}}).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user message")
        assert result == "Answer"


def test_invoke_failure_raises():
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = Exception("timeout")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke("system", "user")
            assert False, "Should have raised"
        except BedrockError:
            pass
