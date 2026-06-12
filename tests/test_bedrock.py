from unittest.mock import MagicMock, patch

from app.bedrock import BedrockClient, BedrockError, InvokeResult


def _converse_response(text, stop_reason="end_turn", input_tokens=10, output_tokens=5):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": stop_reason,
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def _tool_use_response(tool_use_id, name, tool_input, input_tokens=10, output_tokens=5):
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def test_invoke_success_with_usage():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response(
        "Answer", input_tokens=120, output_tokens=45
    )

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user message")

    assert isinstance(result, InvokeResult)
    assert result.text == "Answer"
    assert result.input_tokens == 120
    assert result.output_tokens == 45
    mock_client.converse.assert_called_once()
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["modelId"] == "amazon.nova-lite-v1:0"
    assert call_kwargs["system"] == [{"text": "system"}]


def test_invoke_success_missing_usage_defaults_zero():
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Answer"}]}},
        "stopReason": "end_turn",
    }

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        result = bc.invoke("system", "user")

    assert result.text == "Answer"
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_invoke_unexpected_stop_reason_raises():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response("partial", stop_reason="max_tokens")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke("system", "user")
            assert False, "Should have raised"
        except BedrockError as e:
            assert "max_tokens" in str(e)


def test_invoke_failure_raises():
    mock_client = MagicMock()
    mock_client.converse.side_effect = Exception("timeout")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke("system", "user")
            assert False, "Should have raised"
        except BedrockError:
            pass


def test_invoke_with_tools_no_tool_calls():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response(
        "Direct answer.", input_tokens=20, output_tokens=8
    )
    executor = MagicMock()

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="anthropic.claude-haiku-4-5-20251001-v1:0", region="eu-west-2")
        result = bc.invoke_with_tools(
            system_prompt="You are helpful.",
            initial_messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            tools=[],
            tool_executor=executor,
        )

    assert result.text == "Direct answer."
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    executor.assert_not_called()
    assert mock_client.converse.call_count == 1


def test_invoke_with_tools_one_tool_call():
    mock_client = MagicMock()
    mock_client.converse.side_effect = [
        _tool_use_response("t1", "search_knowledge_base", {"query": "Ivaran"}, 30, 10),
        _converse_response("Ivaran was revealed as Bergst.", input_tokens=60, output_tokens=15),
    ]
    executor = MagicMock(return_value="[Session 3] Ivaran is Bergst.")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="anthropic.claude-haiku-4-5-20251001-v1:0", region="eu-west-2")
        result = bc.invoke_with_tools(
            system_prompt="You are helpful.",
            initial_messages=[{"role": "user", "content": [{"text": "Who is Ivaran?"}]}],
            tools=[],
            tool_executor=executor,
        )

    assert result.text == "Ivaran was revealed as Bergst."
    assert result.input_tokens == 90
    assert result.output_tokens == 25
    executor.assert_called_once_with("search_knowledge_base", {"query": "Ivaran"})
    assert mock_client.converse.call_count == 2


def test_invoke_with_tools_second_call_has_tool_result_in_messages():
    mock_client = MagicMock()
    mock_client.converse.side_effect = [
        _tool_use_response("t99", "search_knowledge_base", {"query": "Q"}),
        _converse_response("Done."),
    ]

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="anthropic.claude-haiku-4-5-20251001-v1:0", region="eu-west-2")
        bc.invoke_with_tools(
            system_prompt="sys",
            initial_messages=[{"role": "user", "content": [{"text": "Q"}]}],
            tools=[],
            tool_executor=lambda name, inp: "tool output",
        )

    # messages list is mutated in place after the second call; index 2 is stable.
    messages = mock_client.converse.call_args_list[1][1]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    tool_result = messages[2]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "t99"
    assert tool_result["content"][0]["text"] == "tool output"
    assert tool_result["status"] == "success"


def test_invoke_with_tools_builds_tool_config_from_input_schema():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response("Answer.")
    tools = [{
        "name": "search_knowledge_base",
        "description": "Search notes",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        bc.invoke_with_tools(
            system_prompt="sys",
            initial_messages=[{"role": "user", "content": [{"text": "?"}]}],
            tools=tools,
            tool_executor=lambda n, i: "r",
        )

    tool_config = mock_client.converse.call_args[1]["toolConfig"]
    spec = tool_config["tools"][0]["toolSpec"]
    assert spec["name"] == "search_knowledge_base"
    assert spec["inputSchema"]["json"]["required"] == ["query"]


def test_invoke_with_tools_multiple_tool_calls_in_one_turn():
    mock_client = MagicMock()
    mock_client.converse.side_effect = [
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "a1",
                                "name": "search_knowledge_base",
                                "input": {"query": "Aldric"},
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "a2",
                                "name": "search_knowledge_base",
                                "input": {"query": "guild"},
                            }
                        },
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        },
        _converse_response("Answer."),
    ]
    executor = MagicMock(return_value="result")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        bc.invoke_with_tools(
            "sys",
            [{"role": "user", "content": [{"text": "?"}]}],
            [],
            executor,
        )

    assert executor.call_count == 2
    executor.assert_any_call("search_knowledge_base", {"query": "Aldric"})
    executor.assert_any_call("search_knowledge_base", {"query": "guild"})


def test_invoke_with_tools_max_iterations_raises():
    mock_client = MagicMock()
    mock_client.converse.return_value = _tool_use_response(
        "x", "search_knowledge_base", {"query": "loop"}
    )

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"text": "?"}]}],
                [],
                lambda n, i: "r",
                max_iterations=3,
            )
            assert False, "Should have raised"
        except BedrockError as e:
            assert "Max iterations" in str(e)

    assert mock_client.converse.call_count == 3


def test_invoke_with_tools_boto3_exception_raises_bedrock_error():
    mock_client = MagicMock()
    mock_client.converse.side_effect = Exception("network error")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"text": "?"}]}],
                [],
                lambda n, i: "r",
            )
            assert False, "Should have raised"
        except BedrockError:
            pass


def test_invoke_with_tools_unexpected_stop_reason_raises():
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": []}},
        "stopReason": "max_tokens",
    }

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        try:
            bc.invoke_with_tools(
                "sys",
                [{"role": "user", "content": [{"text": "?"}]}],
                [],
                lambda n, i: "r",
            )
            assert False, "Should have raised"
        except BedrockError as e:
            assert "max_tokens" in str(e)

    assert mock_client.converse.call_count == 1


def test_invoke_with_tools_accepts_legacy_anthropic_message_blocks():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response("Answer.")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        bc.invoke_with_tools(
            "sys",
            [{"role": "user", "content": [{"type": "text", "text": "legacy"}]}],
            [],
            lambda n, i: "r",
        )

    messages = mock_client.converse.call_args[1]["messages"]
    assert messages[0]["content"][0] == {"text": "legacy"}


def test_invoke_uses_default_inference_config_when_none():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response("Answer")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        bc.invoke("sys", "user")

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["inferenceConfig"] == {"maxTokens": 2048}


def test_invoke_uses_provided_inference_config():
    mock_client = MagicMock()
    mock_client.converse.return_value = _converse_response("Answer")

    with patch("app.bedrock.boto3.client", return_value=mock_client):
        bc = BedrockClient(model_id="amazon.nova-lite-v1:0", region="eu-west-2")
        bc.invoke("sys", "user", inference_config={"maxTokens": 2048, "temperature": 0.0})

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["inferenceConfig"] == {"maxTokens": 2048, "temperature": 0.0}
