from unittest.mock import MagicMock, patch

import lmfast


def test_train_functional():
    """Test lmfast.train() functional API."""
    with patch("lmfast.SLMTrainer") as MockTrainer:
        mock_instance = MockTrainer.return_value

        # Call train
        lmfast.train(
            model="test-model",
            dataset="test-data",
            output_dir="./out",
            max_steps=100,
            learning_rate=1e-4,
        )

        # Verify initialization
        MockTrainer.assert_called_once()
        _, kwargs = MockTrainer.call_args
        assert kwargs["model_config"].model_name == "test-model"
        assert kwargs["training_config"].max_steps == 100
        # Check if kwargs were passed to config
        # We didn't strictly implement kwarg passing to exact config classes in a granular way,
        # but the test checks if it didn't crash and initialized trainer.

        # Verify train called
        mock_instance.train.assert_called_once_with("test-data")
        mock_instance.save.assert_called_once_with("./out")


def test_serve_functional_http():
    """Test lmfast.serve() for HTTP."""
    with patch("lmfast.inference.server.SLMServer") as MockServer:
        mock_instance = MockServer.return_value

        lmfast.serve("test-model", port=9000)

        MockServer.assert_called_with("test-model", use_vllm=True)
        mock_instance.serve.assert_called_with(host="0.0.0.0", port=9000)


@patch("lmfast.mcp.server.LMFastMCPServer")
@patch.dict("sys.modules", {"mcp": MagicMock(), "mcp.server.fastmcp": MagicMock()})
def test_serve_functional_mcp(MockMCPServer):
    """Test lmfast.serve() for MCP."""

    # We need to ensure import lmfast.mcp.server works or is mocked
    # Since we use lazy import inside serve(), we need to patch the module it imports from
    # But since we patched the class directly in the decorator, it might handle it if we imported it at top level?
    # No, serve() does 'from lmfast.mcp.server import LMFastMCPServer'

    lmfast.serve("test-model", mcp=True, mcp_name="my-mcp")

    MockMCPServer.assert_called_with("test-model", name="my-mcp")
    MockMCPServer.return_value.run.assert_called_once()
