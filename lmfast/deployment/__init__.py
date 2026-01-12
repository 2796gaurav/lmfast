"""
LMFast Deployment Module - Export models for various deployment targets.

Supports:
- Browser (WebLLM, ONNX, Transformers.js)
- Edge devices (GGUF, quantized)
- Cloud (Docker, serverless)

Example:
    >>> from lmfast.deployment import BrowserExporter, export_for_browser
    >>> export_for_browser("./my_model", "./browser", target="onnx")
"""

from lmfast.deployment.browser import (
    BrowserExporter,
    export_for_browser,
)

__all__ = [
    "BrowserExporter",
    "export_for_browser",
]
