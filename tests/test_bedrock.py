import json
from unittest.mock import MagicMock, patch

from app.bedrock import BedrockClient, BedrockError, InvokeResult


def test_invoke_success_with_usage():
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {
            "content": [{"text": "Answer"}],
            "usage": {"input_tokens": 120, "output_tokens": 45},
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
        {"content": [{"text": "Answer"}]}
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


def _make_body(content, stop_reason, input_tokens=10, output_tokens=5):
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode()
    return mock_body


def test_invoke_with_tools_no_tool_calls():
    """Model answers immediately without calling any tool."""
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": _make_body(
        content=[{"type": "text", "text": "Direct answer."}],
        stop_reason="end_turn",
        input_tokens=20,
        output_tokens=8,
    )}
    executor = MagicMock()

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        result = bc.invoke_with_tools(
            system_prompt="You are helpful.",
            initial_messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            tools=[],
            tool_executor=executor,
        )

    assert result.text == "Direct answer."
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    executor.assert_not_called()
    assert mock_client.invoke_model.call_count == 1


def test_invoke_with_tools_one_tool_call():
    """Model makes one tool call then gives a final answer."""
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = [
        {"body": _make_body(
            content=[{"type": "tool_use", "id": "t1", "name": "search_knowledge_base", "input": {"query": "Ivaran"}}],
            stop_reason="tool_use",
            input_tokens=30,
            output_tokens=10,
        )},
        {"body": _make_body(
            content=[{"type": "text", "text": "Ivaran was revealed as Bergst."}],
            stop_reason="end_turn",
            input_tokens=60,
            output_tokens=15,
        )},
    ]
    executor = MagicMock(return_value="[Session 3] Ivaran is Bergst.")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        result = bc.invoke_with_tools(
            system_prompt="You are helpful.",
            initial_messages=[{"role": "user", "content": [{"type": "text", "text": "Who is Ivaran?"}]}],
            tools=[],
            tool_executor=executor,
        )

    assert result.text == "Ivaran was revealed as Bergst."
    assert result.input_tokens == 90
    assert result.output_tokens == 25
    executor.assert_called_once_with("search_knowledge_base", {"query": "Ivaran"})
    assert mock_client.invoke_model.call_count == 2


def test_invoke_with_tools_second_call_has_tool_result_in_messages():
    """Verifies the tool result is sent back in the correct message format."""
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = [
        {"body": _make_body(
            content=[{"type": "tool_use", "id": "t99", "name": "search_knowledge_base", "input": {"query": "Q"}}],
            stop_reason="tool_use",
        )},
        {"body": _make_body(
            content=[{"type": "text", "text": "Done."}],
            stop_reason="end_turn",
        )},
    ]

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        bc.invoke_with_tools(
            system_prompt="sys",
            initial_messages=[{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
            tools=[],
            tool_executor=lambda name, inp: "tool output",
        )

    second_call_body = json.loads(mock_client.invoke_model.call_args_list[1][1]["body"])
    messages = second_call_body["messages"]
    assert len(messages) == 3
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "t99"
    assert messages[2]["content"][0]["content"] == "tool output"


def test_invoke_with_tools_multiple_tool_calls_in_one_turn():
    """Model requests two tools in a single response — both are executed."""
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = [
        {"body": _make_body(
            content=[
                {"type": "tool_use", "id": "a1", "name": "search_knowledge_base", "input": {"query": "Aldric"}},
                {"type": "tool_use", "id": "a2", "name": "search_knowledge_base", "input": {"query": "guild"}},
            ],
            stop_reason="tool_use",
        )},
        {"body": _make_body(content=[{"type": "text", "text": "Answer."}], stop_reason="end_turn")},
    ]
    executor = MagicMock(return_value="result")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        bc.invoke_with_tools("sys", [{"role": "user", "content": [{"type": "text", "text": "?"}]}], [], executor)

    assert executor.call_count == 2
    executor.assert_any_call("search_knowledge_base", {"query": "Aldric"})
    executor.assert_any_call("search_knowledge_base", {"query": "guild"})


def test_invoke_with_tools_max_iterations_raises():
    """If the model never stops calling tools, BedrockError is raised."""
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": _make_body(
        content=[{"type": "tool_use", "id": "x", "name": "search_knowledge_base", "input": {"query": "loop"}}],
        stop_reason="tool_use",
    )}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"type": "text", "text": "?"}]}],
                [],
                lambda n, i: "r",
                max_iterations=3,
            )
            assert False, "Should have raised"
        except BedrockError as e:
            assert "Max iterations" in str(e)

    assert mock_client.invoke_model.call_count == 3


def test_invoke_with_tools_boto3_exception_raises_bedrock_error():
    """boto3 failure wraps in BedrockError."""
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = Exception("network error")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"type": "text", "text": "?"}]}],
                [],
                lambda n, i: "r",
            )
            assert False, "Should have raised"
        except BedrockError:
            pass


def test_invoke_with_tools_unexpected_stop_reason_raises():
    """An unexpected stop_reason (e.g. max_tokens) raises BedrockError immediately."""
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {"body": _make_body(
        content=[],
        stop_reason="max_tokens",
    )}

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="claude-sonnet-4-6", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"type": "text", "text": "?"}]}],
                [],
                lambda n, i: "r",
            )
            assert False, "Should have raised"
        except BedrockError as e:
            assert "max_tokens" in str(e)

    assert mock_client.invoke_model.call_count == 1
