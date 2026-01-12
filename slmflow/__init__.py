"""
SLMFlow: Democratized Small Language Model Training

Train, fine-tune, distill, and deploy sub-500M parameter models
on Colab T4 in 30-40 minutes with enterprise-grade features.

Example:
    >>> from slmflow import SLMTrainer, SLMConfig, TrainingConfig
    >>>
    >>> # Configure model
    >>> config = SLMConfig(model_name="HuggingFaceTB/SmolLM-135M")
    >>>
    >>> # Train
    >>> trainer = SLMTrainer(config)
    >>> trainer.train(dataset, output_dir="./my_model")
"""

__version__ = "0.1.0"
__author__ = "Gaurav Chauhan"

# Core imports
from slmflow.core.config import (
    DistillationConfig,
    InferenceConfig,
    SLMConfig,
    TrainingConfig,
)
from slmflow.core.models import (
    get_model_info,
    load_model,
    load_tokenizer,
    prepare_model_for_training,
    save_model,
)
from slmflow.training.data import (
    DataCollator,
    load_dataset,
    prepare_dataset,
)

# Training
from slmflow.training.trainer import SLMTrainer


# Lazy imports for optional modules
def __getattr__(name: str):
    """Lazy loading for optional modules."""
    if name == "DistillationTrainer":
        from slmflow.distillation.teacher_student import DistillationTrainer

        return DistillationTrainer

    if name == "SLMServer":
        from slmflow.inference.server import SLMServer

        return SLMServer

    if name == "GuardrailsConfig":
        from slmflow.guardrails.config import GuardrailsConfig

        return GuardrailsConfig

    if name == "SLMTracer":
        from slmflow.observability.tracing import SLMTracer

        return SLMTracer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Config
    "SLMConfig",
    "TrainingConfig",
    "DistillationConfig",
    "InferenceConfig",
    # Models
    "load_model",
    "load_tokenizer",
    "prepare_model_for_training",
    "save_model",
    "get_model_info",
    # Training
    "SLMTrainer",
    "load_dataset",
    "prepare_dataset",
    "DataCollator",
    # Optional (lazy loaded)
    "DistillationTrainer",
    "SLMServer",
    "GuardrailsConfig",
    "SLMTracer",
]
