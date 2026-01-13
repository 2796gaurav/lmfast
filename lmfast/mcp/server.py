import logging

from lmfast.inference.server import SLMServer

logger = logging.getLogger(__name__)


class LMFastMCPServer:
    """
    Model Context Protocol (MCP) Server for LMFast models.
    Exposes the LLM as a tool and resource for MCP clients (Claude, Cursor, etc).
    """

    def __init__(self, model_path: str, name: str = "lmfast-server"):
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as e:
            raise ImportError(
                "mcp package is not installed. Run `pip install lmfast[mcp]`"
            ) from e

        self.mcp = FastMCP(name)
        self.server = SLMServer(model_path)

        self._register_tools()

    def _register_tools(self):
        """Register default tools."""

        @self.mcp.tool()
        def generate(prompt: str, max_tokens: int = 256) -> str:
            """
            Generate text using the loaded model.

            Args:
                prompt: The input text to generate from
                max_tokens: Maximum number of tokens to generate
            """
            return self.server.generate(prompt, max_new_tokens=max_tokens)

        @self.mcp.resource("model://info")
        def model_info() -> str:
            """Get information about the currently loaded model."""
            return f"Model: {self.server.model_path}"

    def run(self):
        """Run the MCP server (blocks)."""
        logger.info("Starting LMFast MCP Server...")
        self.mcp.run()
