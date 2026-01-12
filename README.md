# 🚀 LMFast

[![PyPI version](https://badge.fury.io/py/lmfast.svg)](https://badge.fury.io/py/lmfast)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Democratized Small Language Model Training** - Train, fine-tune, distill, and deploy sub-500M parameter models on **Colab T4 in 30-40 minutes** with enterprise-grade features.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **T4 Optimized** | Train on free Colab T4 (12GB) with QLoRA + gradient checkpointing |
| ⚡ **Fast Training** | Unsloth integration for 2-5x faster fine-tuning |
| 🧠 **Distillation** | Transfer knowledge from larger models to tiny ones |
| 🤖 **Agents** | Tool-using agents and orchestration framework |
| 📚 **RAG** | Lightweight document retrieval and indexing |
| 🌐 **Browser** | Deploy to browser via ONNX/WebLLM (no server costs) |
| 🛡️ **Guardrails** | PII detection, toxicity filtering, prompt injection protection |
| 📊 **Observability** | Langfuse integration, metrics, attention visualization |
| 🚀 **Fast Inference** | vLLM backend with OpenAI-compatible API |
| 📦 **Easy Export** | GGUF, INT4, AWQ, GPTQ quantization |
| 🧩 **MCP** | Native Model Context Protocol server support |

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install lmfast

# With all features
pip install lmfast[all]

# Specific extras
pip install lmfast[fast]        # Unsloth for faster training
pip install lmfast[guardrails]  # Safety features
pip install lmfast[observability]  # Monitoring
pip install lmfast[inference]   # vLLM serving
```

### Train in 5 Lines

```python
from lmfast import SLMTrainer, SLMConfig, TrainingConfig
from datasets import load_dataset

# Load data
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

# Train
trainer = SLMTrainer(
    SLMConfig(model_name="HuggingFaceTB/SmolLM-135M"),
    TrainingConfig(max_steps=500)
)
trainer.train(dataset)
trainer.save("./my_slm")
```

### CLI Usage

```bash
# Train a model
lmfast train --model HuggingFaceTB/SmolLM-135M --data yahma/alpaca-cleaned --output ./my_model

# Knowledge distillation
lmfast distill --teacher Qwen/Qwen2-1.5B --student HuggingFaceTB/SmolLM-135M --data my_data.json

# Start inference server
lmfast serve --model ./my_model --port 8000

# Export to GGUF
lmfast export --model ./my_model --output ./model.gguf --format gguf

# Interactive chat
lmfast generate --model ./my_model --interactive
```

---

## 📚 Documentation

### Training

```python
from lmfast import SLMTrainer, SLMConfig, TrainingConfig

# Configure for T4 GPU
model_config = SLMConfig(
    model_name="HuggingFaceTB/SmolLM-135M",
    max_seq_length=2048,
    load_in_4bit=True,  # QLoRA
)

training_config = TrainingConfig(
    max_steps=500,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32,
)

trainer = SLMTrainer(model_config, training_config)
trainer.train(dataset)
```

### Knowledge Distillation

```python
from lmfast.distillation import DistillationTrainer
from lmfast.core.config import DistillationConfig

config = DistillationConfig(
    teacher_model="Qwen/Qwen2-1.5B",
    temperature=2.0,
    alpha=0.5,
)

trainer = DistillationTrainer(
    student_model="HuggingFaceTB/SmolLM-135M",
    distillation_config=config,
)
trainer.distill(dataset)
```

### Guardrails

```python
from lmfast.guardrails import GuardrailsConfig, InputValidator, OutputFilter

config = GuardrailsConfig(
    enable_pii_detection=True,
    enable_toxicity_filter=True,
    enable_prompt_injection=True,
)

validator = InputValidator(config)
result = validator.validate(user_input)
if result.is_valid:
    # Process sanitized input
    output = model.generate(result.sanitized_input)
```

### Observability

```python
from lmfast.observability import SLMTracer, MetricsCollector

# Tracing (Langfuse integration)
tracer = SLMTracer(project_name="my_project")

with tracer.trace("inference") as span:
    span.set_attribute("model", "smollm-135m")
    response = model.generate(prompt)
    span.set_attribute("tokens", len(response))

# Metrics
collector = MetricsCollector()
collector.log("loss", 0.5, step=100)
collector.plot("loss")
```

### Fast Inference

```python
from lmfast.inference import SLMServer

# Create server
server = SLMServer("./my_model", use_vllm=True)

# Generate
response = server.generate("Hello, how are you?")

# Batch generation
responses = server.generate_batch(["Prompt 1", "Prompt 2"])

# Start OpenAI-compatible API
server.serve(port=8000)
```

---

## 🎯 Supported Models

| Model | Parameters | T4 Compatible | Notes |
|-------|------------|---------------|-------|
| SmolLM-135M | 135M | ✅ | Fastest training |
| SmolLM-360M | 360M | ✅ | Good balance |
| TinyLlama-1.1B | 1.1B | ✅ (with QLoRA) | More capable |
| Qwen2-0.5B | 500M | ✅ | Multilingual |
| Phi-3-mini | 3.8B | ⚠️ (tight) | Most capable |

---

## 📦 Package Structure

```
lmfast/
├── core/           # Config and model loading
├── training/       # Training and data processing
├── distillation/   # Knowledge distillation
├── guardrails/     # Safety and filtering
├── observability/  # Tracing and metrics
├── inference/      # Serving and quantization
└── cli/            # Command-line interface
```

---

## 🧪 Development

```bash
# Clone
git clone https://github.com/lmfast/lmfast
cd lmfast

# Create environment
conda env create -f environment.yml
conda activate lmfast

# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black lmfast/ tests/
ruff check lmfast/ tests/
```

---


## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for fast training
- [HuggingFace](https://huggingface.co) for transformers ecosystem
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [Langfuse](https://langfuse.com) for observability
