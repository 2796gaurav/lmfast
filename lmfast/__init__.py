"""
LMFast: The FastAPI of Small Language Models

Train, fine-tune, distill, and deploy sub-3B parameter models
on Colab T4 in minutes with enterprise-grade features.

Key Features:
- One-line training API
- Colab T4 optimized (12GB VRAM)
- Built-in distillation & alignment
- Agentic framework with tools
- MCP-native integration
- RAG support
- Browser deployment (ONNX, WebLLM)
- Enterprise guardrails

Example:
    >>> import lmfast
    >>> lmfast.setup_colab_env()
    >>> lmfast.train("HuggingFaceTB/SmolLM-135M", dataset="my_data.json")
    >>> lmfast.serve("./my_model", mcp=True)
"""

__version__ = "0.3.2"
__author__ = "Gaurav Chauhan"

import logging
from typing import Any, Optional

# Core imports
from lmfast.core.config import (
    DistillationConfig,
    InferenceConfig,
    SLMConfig,
    TrainingConfig,
)
from lmfast.core.models import (
    get_model_info,
    load_model,
    load_tokenizer,
    prepare_model_for_training,
    save_model,
)
from lmfast.training.data import (
    DataCollator,
    load_dataset,
    prepare_dataset,
)

# Training
from lmfast.training.trainer import SLMTrainer

# Utils
from lmfast.utils.colab import setup_colab_env

# Lazy imports for optional modules
def __getattr__(name: str):
    """Lazy loading for optional modules."""
    if name == "align":
        from lmfast.alignment import align
        return align
        
    if name == "reason":
        from lmfast.reasoning import reason
        return reason

    if name == "pretrain":
        from lmfast.pretraining import pretrain
        return pretrain

    if name == "DistillationTrainer":
        from lmfast.distillation.teacher_student import DistillationTrainer
        return DistillationTrainer

    if name == "SLMServer":
        from lmfast.inference.server import SLMServer
        return SLMServer

    if name == "GuardrailsConfig":
        from lmfast.guardrails.config import GuardrailsConfig
        return GuardrailsConfig

    if name == "SLMTracer":
        from lmfast.observability.tracing import SLMTracer
        return SLMTracer
        
    if name == "LMFastMCPServer":
        from lmfast.mcp.server import LMFastMCPServer
        return LMFastMCPServer

    # Agents
    if name == "Agent":
        from lmfast.agents.core import Agent
        return Agent
    if name == "CodeAgent":
        from lmfast.agents.specialized import CodeAgent
        return CodeAgent
    if name == "DataAgent":
        from lmfast.agents.specialized import DataAgent
        return DataAgent
    if name == "ThinkingAgent":
        from lmfast.reasoning import ThinkingAgent
        return ThinkingAgent

    # RAG
    if name == "LightweightRAG":
        from lmfast.rag import LightweightRAG
        return LightweightRAG
    if name == "create_rag":
        from lmfast.rag import create_rag
        return create_rag

    # Deployment
    if name == "BrowserExporter":
        from lmfast.deployment import BrowserExporter
        return BrowserExporter
    if name == "export_for_browser":
        from lmfast.deployment import export_for_browser
        return export_for_browser

    # Fast inference
    if name == "SpeculativeDecoder":
        from lmfast.inference import SpeculativeDecoder
        return SpeculativeDecoder
    if name == "generate_fast":
        from lmfast.inference import generate_fast
        return generate_fast

    # Advanced distillation
    if name == "AdvancedDistillationTrainer":
        from lmfast.distillation import AdvancedDistillationTrainer
        return AdvancedDistillationTrainer
    if name == "distill_advanced":
        from lmfast.distillation import distill_advanced
        return distill_advanced

    if name == "generate_structured":
        from lmfast.utils.structured import generate_structured
        return generate_structured

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Functional API ---

def train(
    model: str = "HuggingFaceTB/SmolLM-135M",
    dataset: Any = None,
    output_dir: str = "./my_model",
    max_steps: int = 500,
    use_unsloth: bool = True,
    **kwargs
) -> SLMTrainer:
    """
    Train an SLM in one line.
    
    Args:
        model: Model ID or path (default: SmolLM-135M)
        dataset: HuggingFace dataset or path
        output_dir: Where to save the model
        max_steps: Training steps
        use_unsloth: Use Unsloth for acceleration (recommended for T4)
        **kwargs: Additional args passed to SLMConfig or TrainingConfig
    
    Returns:
        The trainer instance
    """
    model_config = SLMConfig(model_name=model)
    # Update model config from kwargs if applicable
    for k, v in kwargs.items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
            
    training_config = TrainingConfig(
        output_dir=output_dir, 
        max_steps=max_steps
    )
    # Update training config
    for k, v in kwargs.items():
        if hasattr(training_config, k):
            setattr(training_config, k, v)
    
    trainer = SLMTrainer(
        model_config=model_config, 
        training_config=training_config,
        use_unsloth=use_unsloth
    )
    
    if dataset:
        trainer.train(dataset)
        trainer.save(output_dir)
        
    return trainer


def serve(
    model: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    mcp: bool = False,
    mcp_name: str = "lmfast-server",
    use_vllm: bool = True,
    **kwargs
) -> None:
    """
    Serve a model via HTTP API or MCP.
    
    Args:
        model: Model path or ID
        port: HTTP port (if not MCP)
        mcp: If True, run as MCP server over stdio
        mcp_name: Name of MCP server
        use_vllm: Use vLLM backend
    """
    if mcp:
        from lmfast.mcp.server import LMFastMCPServer
        server = LMFastMCPServer(model, name=mcp_name)
        server.run()
    else:
        from lmfast.inference.server import SLMServer
        server = SLMServer(model, use_vllm=use_vllm)
        server.serve(host=host, port=port)


def distill(
    student: str,
    teacher: str,
    dataset: Any,
    output_dir: str = "./distilled_model",
    **kwargs
) -> Any:
    """
    Distill knowledge from a teacher to a student.
    """
    from lmfast.distillation.teacher_student import DistillationTrainer
    
    distill_config = DistillationConfig(teacher_model=teacher, **kwargs)
    trainer = DistillationTrainer(student_model=student, distillation_config=distill_config)
    trainer.distill(dataset)
    return trainer


__all__ = [
    # Version
    "__version__",
    # Functional API
    "train",
    "serve",
    "distill",
    "setup_colab_env",
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
    # Research / Advanced
    "align",
    "reason",
    "pretrain",
    # Inference (lazy loaded)
    "DistillationTrainer",
    "SLMServer",
    "GuardrailsConfig",
    "SLMTracer",
    "LMFastMCPServer",
    # Agents
    "Agent",
    "CodeAgent",
    "DataAgent",
    "ThinkingAgent",
    # RAG
    "LightweightRAG",
    "create_rag",
    # Deployment
    "BrowserExporter",
    "export_for_browser",
    # Fast Inference
    "SpeculativeDecoder",
    "generate_fast",
    # Advanced Distillation
    # Advanced Distillation
    "AdvancedDistillationTrainer",
    "distill_advanced",
    # Structured Output
    "generate_structured",
]

