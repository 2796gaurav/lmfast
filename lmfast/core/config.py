"""
Configuration classes for LMFast.

All configurations use Pydantic for validation and serialization.
Designed for Colab T4 GPU (12GB VRAM) optimization by default.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DType(str, Enum):
    """Supported data types for training and inference."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class ChatTemplate(str, Enum):
    """Supported chat templates for formatting."""

    ALPACA = "alpaca"
    CHATML = "chatml"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    CUSTOM = "custom"


class SLMConfig(BaseModel):
    """
    Base configuration for Small Language Models.

    Optimized defaults for Colab T4 (12GB VRAM).

    Example:
        >>> config = SLMConfig(model_name="HuggingFaceTB/SmolLM-135M")
        >>> print(config.model_size_estimate)
    """

    # Model identification
    model_name: str = Field(
        default="HuggingFaceTB/SmolLM-135M", description="HuggingFace model ID or local path"
    )
    revision: str = Field(default="main", description="Model revision/branch")

    # Sequence configuration
    max_seq_length: int = Field(
        default=2048, ge=128, le=131072, description="Maximum sequence length"
    )

    # Precision configuration
    dtype: DType = Field(
        default=DType.FLOAT16,
        description="Training/inference data type (float16 for T4, bfloat16 for A100+)",
    )
    load_in_4bit: bool = Field(default=True, description="Load model in 4-bit quantization (QLoRA)")
    load_in_8bit: bool = Field(default=False, description="Load model in 8-bit quantization")

    # Memory optimization
    use_gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing to save memory"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )

    # Device configuration
    device_map: str = Field(default="auto", description="Device mapping strategy")

    # Trust remote code (for custom architectures)
    trust_remote_code: bool = Field(default=False, description="Trust remote code from HuggingFace")

    @field_validator("load_in_8bit", mode="before")
    @classmethod
    def validate_quantization(cls, v, info):
        """Ensure only one quantization method is enabled."""
        if v and info.data.get("load_in_4bit", False):
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        return v

    @property
    def model_size_estimate(self) -> str:
        """Estimate model size based on name."""
        name_lower = self.model_name.lower()
        if "135m" in name_lower or "125m" in name_lower:
            return "~135M parameters"
        elif "360m" in name_lower or "350m" in name_lower:
            return "~360M parameters"
        elif "500m" in name_lower:
            return "~500M parameters"
        elif "1b" in name_lower or "1.1b" in name_lower:
            return "~1B parameters"
        elif "1.5b" in name_lower or "1.7b" in name_lower:
            return "~1.5-1.7B parameters"
        return "Unknown size"

    def is_colab_t4_compatible(self) -> bool:
        """Check if configuration is optimized for Colab T4."""
        return self.load_in_4bit and self.use_gradient_checkpointing and self.max_seq_length <= 4096

    def to_transformers_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for transformers.AutoModelForCausalLM.from_pretrained()."""
        kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }

        # Quantization config
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            kwargs["torch_dtype"] = self._get_torch_dtype()

        # Flash attention
        if self.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        return kwargs

    def _get_torch_dtype(self):
        """Get torch dtype from config."""
        import torch

        dtype_map = {
            DType.FLOAT32: torch.float32,
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float16)


class TrainingConfig(BaseModel):
    """
    Training configuration optimized for Colab T4.

    Defaults are tuned for 30-40 minute training sessions.

    Example:
        >>> config = TrainingConfig(max_steps=500, learning_rate=2e-4)
    """

    # Output
    output_dir: str = Field(
        default="./lmfast_output", description="Directory to save model and checkpoints"
    )

    # Training duration
    max_steps: int = Field(
        default=500, ge=1, description="Maximum training steps (optimized for T4)"
    )
    num_epochs: float | None = Field(
        default=None, description="Number of epochs (overrides max_steps if set)"
    )

    # Batch configuration
    batch_size: int = Field(default=4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Gradient accumulation steps"
    )

    # Learning rate
    learning_rate: float = Field(default=2e-4, gt=0, description="Peak learning rate")
    warmup_steps: int = Field(default=50, ge=0, description="Number of warmup steps")
    warmup_ratio: float | None = Field(
        default=None, description="Warmup ratio (overrides warmup_steps if set)"
    )
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")

    # LoRA configuration
    use_lora: bool = Field(default=True, description="Use LoRA for parameter-efficient training")
    lora_r: int = Field(default=16, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, ge=0, le=1, description="LoRA dropout")
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Modules to apply LoRA",
    )

    # Optimization
    optim: str = Field(
        default="adamw_8bit", description="Optimizer (adamw_8bit for memory efficiency)"
    )
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")

    # Checkpointing
    save_steps: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    save_total_limit: int = Field(default=3, ge=1, description="Maximum checkpoints to keep")

    # Logging
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")
    report_to: list[str] = Field(
        default_factory=lambda: ["none"], description="Reporting integrations"
    )

    # Mixed precision
    fp16: bool = Field(default=True, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision (for A100+)")

    # Reproducibility
    seed: int = Field(default=42, description="Random seed")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def estimated_training_time_minutes(self) -> int:
        """Rough estimate of training time on T4."""
        # Assuming ~2 seconds per step on T4 with default settings
        return int(self.max_steps * 2 / 60)

    def to_sft_config_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for trl.SFTConfig."""
        kwargs = {
            "output_dir": self.output_dir,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "weight_decay": self.weight_decay,
            "optim": self.optim,
            "max_grad_norm": self.max_grad_norm,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "report_to": self.report_to,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "seed": self.seed,
        }

        if self.num_epochs is not None:
            kwargs["num_train_epochs"] = self.num_epochs
            kwargs.pop("max_steps", None)

        if self.warmup_ratio is not None:
            kwargs["warmup_ratio"] = self.warmup_ratio
            kwargs.pop("warmup_steps", None)

        return kwargs

    def to_lora_config_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for peft.LoraConfig."""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.lora_target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }


class DistillationConfig(BaseModel):
    """
    Configuration for knowledge distillation.

    Transfer knowledge from a larger teacher model to a smaller student.

    Example:
        >>> config = DistillationConfig(
        ...     teacher_model="Qwen/Qwen2-1.5B",
        ...     temperature=2.0
        ... )
    """

    # Teacher model
    teacher_model: str = Field(description="Teacher model HuggingFace ID or path")
    teacher_dtype: DType = Field(default=DType.FLOAT16, description="Teacher model precision")
    teacher_load_in_4bit: bool = Field(
        default=True, description="Load teacher in 4-bit for memory efficiency"
    )

    # Distillation parameters
    temperature: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Distillation temperature (higher = softer)"
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between CE loss (1-alpha) and KL loss (alpha)",
    )

    # Progressive distillation
    progressive: bool = Field(
        default=False, description="Use progressive distillation (layer-by-layer)"
    )

    # Attention transfer
    attention_transfer: bool = Field(
        default=False, description="Transfer attention patterns from teacher"
    )


class InferenceConfig(BaseModel):
    """
    Configuration for model inference and serving.

    Example:
        >>> config = InferenceConfig(max_new_tokens=256)
    """

    # Generation parameters
    max_new_tokens: int = Field(default=256, ge=1, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, description="Repetition penalty")

    # Serving configuration
    use_vllm: bool = Field(default=True, description="Use vLLM for fast inference")
    tensor_parallel_size: int = Field(
        default=1, ge=1, description="Tensor parallel size for multi-GPU"
    )

    # Quantization for deployment
    quantization: Literal["awq", "gptq", "int8", "int4"] | None = Field(
        default=None, description="Quantization method for deployment"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")

    def to_generation_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for model.generate()."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0,
        }
