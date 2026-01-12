"""
Model loading and management utilities.

Provides unified interface for loading models from HuggingFace Hub
with automatic optimization for Colab T4 (12GB VRAM).
"""

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from slmflow.core.config import SLMConfig, TrainingConfig

logger = logging.getLogger(__name__)

# Type alias for tokenizers
TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast


def detect_environment() -> dict[str, Any]:
    """
    Detect the current runtime environment.

    Returns:
        Dictionary with environment info (is_colab, gpu_name, vram_gb, etc.)
    """
    env_info = {
        "is_colab": False,
        "is_kaggle": False,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_gb": 0,
        "recommended_dtype": "float16",
    }

    # Check if running in Colab
    try:
        import google.colab  # noqa

        env_info["is_colab"] = True
    except ImportError:
        pass

    # Check if running in Kaggle
    try:
        import kaggle_secrets  # noqa

        env_info["is_kaggle"] = True
    except ImportError:
        pass

    # Get GPU info
    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Recommend dtype based on GPU
        if "A100" in env_info["gpu_name"] or "H100" in env_info["gpu_name"]:
            env_info["recommended_dtype"] = "bfloat16"
        else:
            env_info["recommended_dtype"] = "float16"

    return env_info


def get_model_info(model_name: str) -> dict[str, Any]:
    """
    Get information about a model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID

    Returns:
        Dictionary with model information
    """
    from huggingface_hub import model_info as hf_model_info

    try:
        info = hf_model_info(model_name)
        return {
            "model_id": info.modelId,
            "downloads": info.downloads,
            "likes": info.likes,
            "tags": info.tags,
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "created_at": info.created_at,
            "last_modified": info.last_modified,
        }
    except Exception as e:
        logger.warning(f"Could not fetch model info for {model_name}: {e}")
        return {"model_id": model_name, "error": str(e)}


def load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    padding_side: str = "right",
    add_eos_token: bool = True,
    **kwargs,
) -> TokenizerType:
    """
    Load tokenizer for a model.

    Args:
        model_name: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code
        padding_side: Padding side ("left" or "right")
        add_eos_token: Whether to add EOS token
        **kwargs: Additional kwargs for AutoTokenizer.from_pretrained

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    # Set padding side
    tokenizer.padding_side = padding_side

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure EOS token behavior
    if add_eos_token and hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = True

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    return tokenizer


def load_model(
    config: SLMConfig | str,
    *,
    tokenizer: TokenizerType | None = None,
    for_training: bool = False,
    use_unsloth: bool = True,
    **kwargs,
) -> tuple[PreTrainedModel, TokenizerType]:
    """
    Load a model with optimized settings for Colab T4.

    Automatically:
    - Detects environment and applies optimal settings
    - Uses Unsloth for faster training if available
    - Applies 4-bit quantization for memory efficiency
    - Enables gradient checkpointing for training

    Args:
        config: SLMConfig or model name string
        tokenizer: Optional pre-loaded tokenizer
        for_training: Whether the model will be used for training
        use_unsloth: Try to use Unsloth for optimization
        **kwargs: Additional kwargs for model loading

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> config = SLMConfig(model_name="HuggingFaceTB/SmolLM-135M")
        >>> model, tokenizer = load_model(config, for_training=True)
    """
    # Handle string input
    if isinstance(config, str):
        config = SLMConfig(model_name=config)

    model_name = config.model_name
    logger.info(f"Loading model: {model_name}")

    # Detect environment
    env = detect_environment()
    logger.info(f"Environment: {env}")

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = load_tokenizer(
            model_name,
            trust_remote_code=config.trust_remote_code,
        )

    # Try Unsloth for optimized loading
    if use_unsloth and for_training:
        model, tokenizer = _load_with_unsloth(config, tokenizer)
        if model is not None:
            return model, tokenizer

    # Standard loading with transformers
    model = _load_with_transformers(config, **kwargs)

    # Apply gradient checkpointing for training
    if for_training and config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    return model, tokenizer


def _load_with_unsloth(
    config: SLMConfig,
    tokenizer: TokenizerType,
) -> tuple[PreTrainedModel | None, TokenizerType]:
    """Try to load model using Unsloth for optimization."""
    try:
        from unsloth import FastLanguageModel

        logger.info("Using Unsloth for optimized model loading")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=config.load_in_4bit,
        )

        logger.info("Model loaded successfully with Unsloth")
        return model, tokenizer

    except ImportError:
        logger.info("Unsloth not installed, falling back to standard loading")
        return None, tokenizer
    except Exception as e:
        logger.warning(f"Unsloth loading failed: {e}, falling back to standard loading")
        return None, tokenizer


def _load_with_transformers(
    config: SLMConfig,
    **kwargs,
) -> PreTrainedModel:
    """Load model using standard transformers."""
    model_kwargs = config.to_transformers_kwargs()
    model_kwargs.update(kwargs)

    logger.info(f"Loading with transformers: {model_kwargs}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
    except Exception as e:
        # Fallback: try without flash attention
        if "flash_attention" in str(e).lower():
            logger.warning("Flash Attention not available, falling back to standard attention")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                **model_kwargs,
            )
        else:
            raise

    return model


def prepare_model_for_training(
    model: PreTrainedModel,
    training_config: TrainingConfig,
    *,
    use_unsloth: bool = True,
) -> PreTrainedModel:
    """
    Prepare a model for training with LoRA adapters.

    Args:
        model: The base model
        training_config: Training configuration
        use_unsloth: Try to use Unsloth for optimization

    Returns:
        Model with LoRA adapters applied
    """
    if not training_config.use_lora:
        logger.info("LoRA disabled, returning base model")
        return model

    # Try Unsloth PEFT
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model = FastLanguageModel.get_peft_model(
                model,
                r=training_config.lora_r,
                lora_alpha=training_config.lora_alpha,
                lora_dropout=training_config.lora_dropout,
                target_modules=training_config.lora_target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=training_config.seed,
            )
            logger.info("LoRA applied via Unsloth")
            return model

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Unsloth PEFT failed: {e}, falling back to standard PEFT")

    # Standard PEFT
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(**training_config.to_lora_config_kwargs())
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied: {trainable_params:,} trainable params "
        f"({100 * trainable_params / all_params:.2f}% of {all_params:,} total)"
    )

    return model


def save_model(
    model: PreTrainedModel,
    tokenizer: TokenizerType,
    output_dir: str | Path,
    *,
    merge_adapters: bool = False,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    **kwargs,
) -> Path:
    """
    Save a model and tokenizer.

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save to
        merge_adapters: Whether to merge LoRA adapters into base model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: Model ID for Hub (required if push_to_hub=True)
        **kwargs: Additional save kwargs

    Returns:
        Path to saved model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")

    # Merge adapters if requested
    if merge_adapters:
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
                logger.info("LoRA adapters merged into base model")
        except Exception as e:
            logger.warning(f"Could not merge adapters: {e}")

    # Save model and tokenizer
    model.save_pretrained(output_dir, **kwargs)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")

    # Push to Hub if requested
    if push_to_hub:
        if hub_model_id is None:
            raise ValueError("hub_model_id required when push_to_hub=True")

        model.push_to_hub(hub_model_id, **kwargs)
        tokenizer.push_to_hub(hub_model_id)
        logger.info(f"Model pushed to Hub: {hub_model_id}")

    return output_dir
