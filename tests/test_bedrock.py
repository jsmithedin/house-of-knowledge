import json
from unittest.mock import MagicMock, patch

from app.bedrock import BedrockClient, BedrockError, InvokeResult


def test_invoke_success_with_usage():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {
            "output": {"message": {"content": [{"text": "Answer"}]}},
            "usage": {"inputTokens": 120, "outputTokens": 45},
        }
    ).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user message")
        assert isinstance(result, InvokeResult)
        assert result.text == "Answer"
        assert result.input_tokens == 120
        assert result.output_tokens == 45


def test_invoke_success_missing_usage_defaults_zero():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {"output": {"message": {"content": [{"text": "Answer"}]}}}
    ).encode()

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": mock_body}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user")
        assert result.text == "Answer"
        assert result.input_tokens == 0
        assert result.output_tokens == 0


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
