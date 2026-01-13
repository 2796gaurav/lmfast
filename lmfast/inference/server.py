"""
High-performance inference server.

Provides fast inference with optional vLLM backend.
"""

import logging
import time
from typing import Any

import torch
from transformers import PreTrainedModel

from lmfast.core.config import InferenceConfig, SLMConfig
from lmfast.core.models import TokenizerType, load_model, load_tokenizer

logger = logging.getLogger(__name__)


class SLMServer:
    _model: PreTrainedModel | None
    _tokenizer: TokenizerType | None
    _vllm_engine: Any | None

    """
    High-throughput inference server for SLMs.

    Features:
    - vLLM backend for maximum throughput (optional)
    - Batched generation
    - OpenAI-compatible API serving
    - Automatic quantization

    Example:
        >>> server = SLMServer("./my_model")
        >>>
        >>> # Single generation
        >>> response = server.generate("Hello, how are you?")
        >>>
        >>> # Batch generation
        >>> responses = server.generate_batch(["Prompt 1", "Prompt 2"])
        >>>
        >>> # Start API server
        >>> server.serve(port=8000)
    """

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig | None = None,
        *,
        use_vllm: bool | None = None,
        use_unsloth: bool = True,
    ):
        """
        Initialize inference server.

        Args:
            model_path: Path to model or HuggingFace ID
            config: Inference configuration
            use_vllm: Override config to use/not use vLLM
            use_unsloth: Attempt to use Unsloth for optimization (if installed)
        """
        self.model_path = model_path
        self.config = config or InferenceConfig()

        if use_vllm is not None:
            self.config.use_vllm = use_vllm

        self.use_unsloth = use_unsloth
        self._model = None
        self._tokenizer = None
        self._vllm_engine = None
        self._is_vllm = False

        logger.info(f"SLMServer initialized for {model_path}")

    @property
    def model(self):
        """Get model, loading if necessary."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def _load(self) -> None:
        """Load model and tokenizer."""
        if self.config.use_vllm:
            self._load_vllm()

        if self._vllm_engine is None:
            self._load_transformers()

    def _load_vllm(self) -> None:
        """Try to load with vLLM."""
        try:
            from vllm import LLM

            logger.info("Loading model with vLLM...")

            self._vllm_engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype="auto",
                trust_remote_code=True,
            )
            self._is_vllm = True

            # Load tokenizer separately for compatibility
            tokenizer = load_tokenizer(self.model_path)
            self._tokenizer = tokenizer

            logger.info("vLLM engine loaded successfully")

        except ImportError:
            logger.info(
                "vLLM not installed. Falling back to transformers. "
                "Install with: pip install lmfast[inference]"
            )
        except Exception as e:
            logger.warning(f"vLLM loading failed: {e}. Falling back to transformers.")

    def _load_transformers(self) -> None:
        """Load with transformers."""
        logger.info("Loading model with transformers...")

        from lmfast.core.config import DType

        # Detect if model path indicates quantization
        load_in_4bit = self.config.quantization == "int4" or "int4" in self.model_path.lower()
        load_in_8bit = self.config.quantization == "int8" or "int8" in self.model_path.lower()

        model_config = SLMConfig(
            model_name=self.model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=DType.FLOAT16,
        )

        model, tokenizer = load_model(
            model_config,
            for_training=False,
            use_unsloth=self.use_unsloth,
        )
        self._model = model
        self._tokenizer = tokenizer

        if self._model is not None:
            self._model.eval()
        logger.info("Model loaded with transformers")

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            **kwargs: Additional generation kwargs

        Returns:
            Generated text
        """
        responses = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )
        return responses[0]

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            **kwargs: Additional generation kwargs

        Returns:
            List of generated texts
        """
        # Use config defaults
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k

        if self._is_vllm:
            return self._generate_vllm(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            return self._generate_transformers(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs,
            )

    def _generate_vllm(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        """Generate with vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if self._vllm_engine is None:
            raise RuntimeError("vLLM engine not loaded")
        outputs = self._vllm_engine.generate(prompts, sampling_params)

        return [output.outputs[0].text for output in outputs]

    def _generate_transformers(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs,
    ) -> list[str]:
        """Generate with transformers."""
        model = self.model
        tokenizer = self.tokenizer

        # Ensure left padding for inference
        tokenizer.padding_side = "left"

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode
        responses = []
        for i, output in enumerate(outputs):
            # Prompt length in tokens (including left padding)
            input_len = inputs["input_ids"][i].shape[0]

            # Extract generated tokens
            generated_tokens = output[input_len:]

            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    def serve(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Start an OpenAI-compatible API server.

        Args:
            host: Server host
            port: Server port
        """
        host = host or self.config.host
        port = port or self.config.port

        try:
            import uvicorn
            from fastapi import FastAPI
            from pydantic import BaseModel
        except ImportError:
            logger.error(
                "FastAPI/uvicorn not installed. " "Install with: pip install lmfast[inference]"
            )
            return

        # Ensure model is loaded
        _ = self.model

        # Create FastAPI app
        app = FastAPI(title="LMFast Inference Server")

        class CompletionRequest(BaseModel):
            prompt: str
            max_tokens: int = 256
            temperature: float = 0.7
            top_p: float = 0.9

        class CompletionResponse(BaseModel):
            text: str
            usage: dict

        @app.post("/v1/completions")
        async def create_completion(request: CompletionRequest) -> CompletionResponse:
            start_time = time.time()

            response = self.generate(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            return CompletionResponse(
                text=response,
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(response.split()),
                    "latency_ms": (time.time() - start_time) * 1000,
                },
            )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": self.model_path}

        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def benchmark(
        self,
        prompts: list[str],
        *,
        num_runs: int = 3,
    ) -> dict:
        """
        Benchmark inference performance.

        Args:
            prompts: Test prompts
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking with {len(prompts)} prompts, {num_runs} runs...")

        # Warmup
        self.generate(prompts[0])

        latencies = []
        for _ in range(num_runs):
            start = time.time()
            self.generate_batch(prompts)
            elapsed = time.time() - start
            latencies.append(elapsed)

        total_tokens = sum(len(p.split()) + self.config.max_new_tokens for p in prompts)

        return {
            "prompts": len(prompts),
            "runs": num_runs,
            "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "throughput_tokens_per_sec": total_tokens / (sum(latencies) / len(latencies)),
            "backend": "vllm" if self._is_vllm else "transformers",
        }
