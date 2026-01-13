"""
LMFast Inference Module.

Provides:
- SLMServer: High-performance inference server with OpenAI-compatible API
- Quantization: INT4, INT8, AWQ, GPTQ, GGUF export
- Speculative Decoding: 1.5-4x speedup with draft models

Example:
    >>> from lmfast.inference import SLMServer, SpeculativeDecoder
    >>>
    >>> # Standard inference
    >>> server = SLMServer("./my_model")
    >>> output = server.generate("Hello!")
    >>>
    >>> # Fast inference with speculative decoding
    >>> fast = SpeculativeDecoder("./my_model")
    >>> output = fast.generate("Hello!", max_tokens=100)
"""

from lmfast.inference.quantization import export_gguf, quantize_model
from lmfast.inference.server import SLMServer

# Try to import speculative decoding
try:
    from lmfast.inference.speculative import (
        SpeculativeDecoder,
        SpeculativeDecodingConfig,
        generate_fast,
    )

    _SPECULATIVE_AVAILABLE = True
except ImportError:
    _SPECULATIVE_AVAILABLE = False

__all__ = [
    "SLMServer",
    "quantize_model",
    "export_gguf",
]

if _SPECULATIVE_AVAILABLE:
    __all__.extend(
        [
            "SpeculativeDecoder",
            "SpeculativeDecodingConfig",
            "generate_fast",
        ]
    )
