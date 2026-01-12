"""
Memory and training optimizations for Colab T4.

Provides utilities to maximize training efficiency on limited GPU memory.
"""

import gc
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_memory_stats() -> dict[str, float]:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    return {
        "gpu_available": True,
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }


def clear_memory() -> None:
    """
    Aggressively clear GPU memory.

    Useful between training runs or when encountering OOM errors.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gc.collect()

    logger.info(f"Memory cleared. Current stats: {get_memory_stats()}")


def optimize_for_t4() -> None:
    """
    Apply optimizations specifically for Colab T4 GPU.

    This should be called at the start of training on T4.
    """
    logger.info("Applying T4 optimizations...")

    # Enable TF32 for faster matmuls (T4 supports this partially)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Optimize memory allocation
    if hasattr(torch.cuda, "memory"):
        # Use more aggressive memory management
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Disable debug mode for faster execution
    torch.autograd.set_detect_anomaly(False)

    # Clear any existing cached memory
    clear_memory()

    logger.info("T4 optimizations applied")


def enable_gradient_checkpointing(
    model: Any,
    *,
    use_reentrant: bool = False,
) -> Any:
    """
    Enable gradient checkpointing for memory efficiency.

    Trades compute for memory by not storing all activations.
    Typically reduces memory by 50%+ at cost of 20-30% slower training.

    Args:
        model: Model to enable checkpointing on
        use_reentrant: Whether to use reentrant checkpointing

    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        # Try Unsloth's optimized checkpointing first
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
            logger.info("Gradient checkpointing enabled (standard)")
        except TypeError:
            # Fallback for older versions
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (legacy)")
    else:
        logger.warning("Model does not support gradient checkpointing")

    return model


def estimate_memory_requirements(
    model_params: int,
    batch_size: int,
    seq_length: int,
    *,
    optimizer: str = "adamw_8bit",
    use_lora: bool = True,
    lora_r: int = 16,
    load_in_4bit: bool = True,
) -> dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model_params: Number of model parameters
        batch_size: Training batch size
        seq_length: Sequence length
        optimizer: Optimizer type
        use_lora: Whether using LoRA
        lora_r: LoRA rank
        load_in_4bit: Whether using 4-bit quantization

    Returns:
        Dictionary with memory estimates in GB
    """
    # Base model memory
    if load_in_4bit:
        model_memory = model_params * 0.5 / (1024**3)  # 4-bit = 0.5 bytes
    else:
        model_memory = model_params * 2 / (1024**3)  # FP16 = 2 bytes

    # LoRA adapter memory (usually small)
    if use_lora:
        lora_params = model_params * 0.01 * (lora_r / 16)  # Rough estimate
        lora_memory = lora_params * 2 / (1024**3)  # FP16
    else:
        lora_memory = 0

    # Optimizer states
    if "8bit" in optimizer:
        opt_memory = lora_params * 1 / (1024**3) if use_lora else model_memory * 0.5
    else:
        opt_memory = lora_params * 8 / (1024**3) if use_lora else model_memory * 2

    # Activation memory (rough estimate)
    activation_memory = batch_size * seq_length * 4096 * 2 / (1024**3)

    # Gradient memory
    gradient_memory = lora_memory if use_lora else model_memory

    total = model_memory + lora_memory + opt_memory + activation_memory + gradient_memory

    return {
        "model_gb": round(model_memory, 2),
        "lora_gb": round(lora_memory, 2),
        "optimizer_gb": round(opt_memory, 2),
        "activation_gb": round(activation_memory, 2),
        "gradient_gb": round(gradient_memory, 2),
        "total_estimated_gb": round(total, 2),
        "t4_compatible": total < 12,  # T4 has ~12GB usable
    }


def get_optimal_batch_size(
    model_params: int,
    seq_length: int,
    target_vram_gb: float = 11.0,
    *,
    use_lora: bool = True,
    load_in_4bit: bool = True,
) -> int:
    """
    Calculate optimal batch size for given constraints.

    Args:
        model_params: Number of model parameters
        seq_length: Sequence length
        target_vram_gb: Target VRAM usage
        use_lora: Whether using LoRA
        load_in_4bit: Whether using 4-bit quantization

    Returns:
        Recommended batch size
    """
    # Start with batch size 1 and increase
    for batch_size in [1, 2, 4, 8, 16, 32]:
        estimate = estimate_memory_requirements(
            model_params,
            batch_size,
            seq_length,
            use_lora=use_lora,
            load_in_4bit=load_in_4bit,
        )

        if estimate["total_estimated_gb"] > target_vram_gb:
            return max(1, batch_size // 2)

    return 32  # Max reasonable batch size


class MemoryTracker:
    """
    Track memory usage during training.

    Useful for debugging OOM issues and optimizing batch sizes.
    """

    def __init__(self):
        self.snapshots = []

    def snapshot(self, label: str) -> dict:
        """Take a memory snapshot."""
        stats = get_memory_stats()
        stats["label"] = label
        self.snapshots.append(stats)
        logger.info(f"Memory [{label}]: {stats.get('allocated_gb', 0):.2f} GB allocated")
        return stats

    def report(self) -> str:
        """Generate a memory usage report."""
        if not self.snapshots:
            return "No snapshots recorded"

        lines = ["Memory Usage Report", "=" * 40]
        for snap in self.snapshots:
            lines.append(
                f"{snap['label']}: "
                f"{snap.get('allocated_gb', 0):.2f} GB allocated, "
                f"{snap.get('reserved_gb', 0):.2f} GB reserved"
            )

        return "\n".join(lines)

    def peak(self) -> float:
        """Get peak memory usage in GB."""
        if not self.snapshots:
            return 0.0
        return max(s.get("allocated_gb", 0) for s in self.snapshots)
