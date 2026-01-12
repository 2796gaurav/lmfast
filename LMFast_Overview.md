# 🚀 LMFast: The Complete Technical Reference Manual

**LMFast** is an end-to-end ecosystem designed to democratize the entire lifecycle of **Small Language Models (SLMs)**. It provides a software-defined bridge between raw hardware constraints (like the 12GB VRAM on a Google Colab T4) and enterprise-grade AI production.

---

## 1. The Core Infrastructure

LMFast is built on a modular architecture that separates model definitions, hardware-aware optimization, and high-level behavioral loops.

### 1.1 Hardware-Aware Config System (`lmfast.core.config`)
LMFast uses **Pydantic** for rigorous configuration validation. Every setup is optimized for **Colab T4** by default:
*   **`SLMConfig`**:
    *   **Automatic Quantization**: Defaults to 4-bit (QLoRA) using `nf4` and double quantization.
    *   **Memory Efficiency**: Native support for **Flash Attention 2** and Gradient Checkpointing.
    *   **SMOL Compatibility**: Explicitly optimized for SmolLM (135M/360M) and TinyLlama (1.1B).
*   **`TrainingConfig`**:
    *   **Memory Safeguards**: Integrated `effective_batch_size` calculation.
    *   **LoRA Control**: Target modules default to all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
    *   **Time Estimation**: Pre-training logic that estimates `estimated_training_time_minutes` before launch.

### 1.2 Unified Model Loader (`lmfast.core.models`)
The model loader behaves differently based on the environment:
*   **Environment Detection**: Automatically detects if running in **Colab** or **Kaggle** and switches CPU/GPU dtypes.
*   **Unsloth Integration**: If the `fast` extra is installed, `load_model` bypasses standard Transformers for a **2-5x speedup** in memory and compute.
*   **Graceful Fallbacks**: If Flash Attention 2 fails, the loader automatically falls back to standard attention implementations.

---

## 2. Advanced Training & Refinement

### 2.1 The SLMTrainer (`lmfast.training.trainer`)
A unified interface that handles the entire pipeline:
1.  **Preparation**: Injects LoRA adapters and optimizes GPU memory via `optimize_for_t4()`.
2.  **Dataset Packing**: Uses `trl.SFTTrainer` with sequence packing enabled for maximum throughput.
3.  **One-Line Interface**: `trainer.train(dataset)` handles tokenization, filtering, and model saving automatically.

### 2.2 Pedagogical Distillation (`lmfast.distillation`)
Turn high-parameter knowledge into low-latency responses:
*   **Combined Loss Function**: A balanced mix of **Standard Cross-Entropy (1-α)** and **KL-Divergence (α)**.
*   **Offline Logit Generation**: `generate_teacher_labels` allows you to pre-compute teacher outputs, letting you distill from a model that *doesn't* fit in memory alongside the student.
*   **Temperature Control**: Softens the teacher's probability distribution for smoother knowledge transfer.

### 2.3 Optimized Alignment (`lmfast.alignment`)
*   **ORPO Integration**: The preferred path for 2025. By penalizing the log-odds of rejected responses during the SFT phase, it removes the need for a separate "Reference Model" (saving ~6GB VRAM on T4).

---

## 3. The Agentic Layer & System 2 Reasoning

LMFast treats models as active agents, not just passive completion engines.

### 3.1 Thinking & Test-Time Compute (`lmfast.reasoning`)
Smaller models benefit from "thinking" before they speak. The `ThinkingAgent` provides:
*   **Best-of-N Search**: Generates $N$ parallel candidate solutions and selects the best one based on length or logic heuristics.
*   **COT Enforcement**: Automatically wraps queries in "Let's think step by step" prompts and extracts the final answer.

### 3.2 Tool-Equipped Agents (`lmfast.agents`)
*   **Native Tool Extraction**: Automatically converts Python functions into tool definitions via type-hint inspection.
*   **Tool Loops**: The `Agent` class handles the ReAct (Reason + Act) loop, parsing JSON tool calls even from sub-500M models.
*   **Specialized Personas**: Includes `CodeAgent` (with sandboxed python execution) and `DataAgent`.

---

## 4. Deployment & Elastic Inference

### 4.1 High-Throughput Server (`lmfast.inference.server`)
*   **vLLM Acceleration**: When available, LMFast switches to the PagedAttention-powered vLLM engine.
*   **OpenAI Compatibility**: Native `/v1/completions` endpoint via FastAPI and Uvicorn.
*   **Interactive Chat**: Dedicated CLI mode for low-latency testing of fine-tuned models.

### 4.2 Model Context Protocol (MCP) (`lmfast.mcp`)
LMFast is an **MCP-Native** library.
*   **Model-as-a-Tool**: Expose your fine-tuned model as a tool that can be used by Claude, Cursor, or any other MCP client.
*   **Standardized API**: Tools like `generate` and resources like `model://info` are exposed over stdio/SSE out-of-the-box.

### 4.3 Quantization & Export (`lmfast.inference.quantization`)
*   **Post-Training Quantization (PTQ)**: Easily convert models to `int4`, `int8`, `AWQ`, or `GPTQ`.
*   **GGUF Export**: Bridge your Colab training to local Apple Silicon or Windows GPUs via `llama.cpp` compatible GGUF files.

---

## 5. Enterprise-Grade Observability & Safety

### 5.1 Deep Observability (`lmfast.observability`)
*   **Tracing (Langfuse)**: Full OpenTelemetry-compatible tracing of every generation, including latency, token usage, and cost tracing.
*   **Attention Visualizer**: Generate saliency heatmaps showing exactly what the model "looked at" when predicting a specific token.
*   **Metrics Collector**: Track loss, gradient norms, and throughput over time with automatic Matplotlib plotting.

### 5.2 Input/Output Guardrails (`lmfast.guardrails`)
*   **PII Sanitization**: Powered by **Microsoft Presidio**. Detects and redacts emails, phones, and names with configurable actions (block/redact/mask).
*   **Risk Scoring**: Calculates a risk score for every input, allowing for strict or loose safety policies.
*   **Security Filters**: Pre-built regex patterns for prompt injection and jailbreak detection.

---

## 6. Project & Directory Blueprint

*   `lmfast/core`: Unified Pydantic Configs & Intelligent Multi-Env Loading.
*   `lmfast/training`: Memory-optimized SFT and hardware orchestration.
*   `lmfast/distillation`: Multi-loss logit transfer and offline label generation.
*   `lmfast/alignment`: Next-gen ORPO/DPO preference optimization.
*   `lmfast/agents`: Thinking loops, COT, and automated tool extraction.
*   `lmfast/guardrails`: PII redaction and injection protection.
*   `lmfast/observability`: Saliency maps, attention rollout, and Langfuse tracing.
*   `lmfast/inference`: vLLM, OpenAI-API, AWQ/GPTQ/GGUF export/serving.
*   `lmfast/mcp`: Standardized Model Context Protocol transport.
*   `lmfast/cli`: Full-featured Typer interface for all operations.

---

**LMFast** is the definitive toolkit for transforming Small Language Models into production-ready, agentic, and highly optimized AI assets.

---

## 2️⃣ LMFast's Unique Value Proposition (USP)

### 2.1 Core Philosophy – "FastAPI for LLMs – Zero Barrier, Maximum Power"

| Primary Differentiator | Description |
|------------------------|-------------|
| **Colab T4‑First Design** | Optimized from the ground up for Google Colab’s free T4 (12 GB VRAM). All pipelines automatically select the most memory‑efficient settings (QLoRA 4‑bit, gradient checkpointing, TF32, etc.). |
| **Unified Lifecycle Management** | One coherent high‑level API that covers the entire model lifecycle – pre‑training, fine‑tuning, distillation, alignment, agentic reasoning, MCP serving, and multi‑format export – without juggling separate libraries. |
| **Production‑Grade Observability** | Built‑in Langfuse tracing, attention visualisation, PII detection/redaction (Microsoft Presidio), and token‑cost tracking for self‑hosted models. |
| **MCP‑Native Architecture** | Models are first‑class MCP resources, enabling seamless tool‑calling and composability across cloud, edge, and local GPUs. |

### 2.2 Competitive Matrix

| Feature | **LMFast** | Unsloth | Axolotl | Torchtune | Ollama |
|---------|------------|---------|---------|-----------|-------|
| **T4 Optimization** | ✅ Native | ✅ | ⚠️ Good | ⚠️ Good | ❌ Inference Only |
| **Training Speed** | ✅ (via Unsloth) | ✅ | ✅ | ✅ | ⚠️ |
| **Distillation** | ✅ Built‑in | ❌ | ❌ | ❌ | ❌ |
| **Alignment (ORPO)** | ✅ Native | ❌ | ⚠️ External | ❌ | ❌ |
| **Agentic Framework** | ✅ (ReAct + CoT) | ✅ | ❌ | ❌ | ❌ |
| **MCP Integration** | ✅ Native | ✅ | ❌ | ❌ | ⚠️ |
| **Multi‑Format Export** | ✅ (GGUF, ONNX, vLLM, AWQ, GPTQ) | ⚠️ GGUF | ⚠️ Limited | ✅ | ✅ |
| **Guardrails** | ✅ (PII, Injection) | ❌ | ❌ | ❌ | ❌ |
| **Observability** | ✅ (Langfuse, Viz) | ❌ | ❌ | ❌ | ⚠️ Basic |
| **CLI** | ✅ Typer | ✅ | ✅ | ✅ | ✅ |
| **Notebook Support** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Browser Deployment** | ⏳ Roadmap | ❌ | ❌ | ❌ | ❌ |

---

## 3️⃣ Critical Feature Gaps & Roadmap (2026)

### 3.1 Immediate Priorities (Q1 2026)

#### A. Hybrid Architecture Support – **Hymba**
```python
# lmfast/architectures/hymba.py
from lmfast.core.config import SLMConfig
from typing import Literal

class HymbaConfig(SLMConfig):
    """Hybrid‑head architecture combining attention + SSM (Mamba‑2)."""
    architecture: Literal["hymba"] = "hymba"
    n_meta_tokens: int = 16
    ssm_state_size: int = 16
    use_cross_layer_kv_sharing: bool = True
    sliding_window_size: int = 2048

def optimize_hymba_training(model, config: HymbaConfig):
    """Specialised training optimisations for Hymba.
    1. Gradient checkpointing for SSM blocks
    2. Initialise meta‑tokens from vocab statistics
    3. Adaptive LR for attention vs SSM heads
    """
    # Hook into lmfast.training.optimizations internally.
    pass
```

#### B. Browser Deployment Stack – **WebLLM + ONNX**
```python
# lmfast/deployment/browser.py
from pathlib import Path

class BrowserExporter:
    """Export a trained SLM to WebLLM/ONNX for in‑browser inference.
    Generates .wasm binaries, splits model weights, and creates a demo app.
    """
    def __init__(self, model_path: str, target: str = "webllm", quantization: str = "int4",
                 context_length: int = 2048):
        self.model_path = Path(model_path)
        self.target = target
        self.quantization = quantization
        self.context_length = context_length

    def export(self, output_dir: str, split_model_size_mb: int = 100, include_webworker: bool = True):
        """Perform the heavy‑weight export.
        Returns a dict with artifact locations.
        """
        # Placeholder – real implementation will invoke ONNX Runtime and Emscripten.
        return {"wasm": "model.wasm", "weights": ["part0.bin", "part1.bin"]}

    def create_demo_app(self, output_dir: str, framework: str = "react"):
        """Generate a minimal HTML/JS demo that loads the exported artifacts.
        Supports React, Vue, or vanilla JS.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "index.html").write_text("<!-- Demo placeholder -->")
```

#### C. Advanced Quantization Suite – **Mixed‑Precision Quantizer**
```python
# lmfast/inference/quantization.py (additional class)
class MixedPrecisionQuantizer:
    """Selective quantisation preserving accuracy on critical layers.
    Uses activation variance to decide which layers stay in FP16.
    """
    def calibrate(self, model, calibration_data):
        """Run calibration data to compute per‑layer sensitivity scores.
        Returns a dict mapping layer names to scores.
        """
        # Placeholder – would collect activations and compute variance.
        return {}

    def quantize_selective(self, model, sensitivity_threshold: float = 1.5):
        """Quantise low‑sensitivity layers to INT4, keep high‑sensitivity layers FP16.
        Returns a dict with the quantised model and a decision map.
        """
        # Placeholder – actual quantisation logic goes here.
        return {
            "quantized_model": model,
            "layer_decisions": {},
            "estimated_size_reduction": "65%",
            "expected_accuracy_delta": "<0.5%",
        }
```

### 3.2 Mid‑Term Enhancements (Q2‑Q3 2026)
- **Distributed Multi‑GPU Trainer** (`lmfast/training/distributed.py`).
- **Reasoning Model Trainer** (`lmfast/reasoning/reasoning_trainer.py`).
- **Vision‑Language Trainer** (`lmfast/multimodal/vision_language.py`).

### 3.3 Advanced Capabilities (Q4 2026)
- **Federated Learning** (`lmfast/federated/trainer.py`).
- **Neurosymbolic Reasoning** (`lmfast/reasoning/neuro_symbolic.py`).
- **Model Merging & MoE Construction** (`lmfast/merging/model_merger.py`, `lmfast/architectures/moe.py`).

---

## 4️⃣ Marketing & Positioning Strategy

### 4.1 Target Personas
| Persona | Pain Points | LMFast Value |
|--------|-------------|--------------|
| **Pragmatic Builder** | Limited compute, high API costs | Train/offline a custom GPT‑class model on free Colab T4 |
| **Enterprise Optimizer** | Production latency & cost of massive LLMs | Replace >90 % of GPT‑4 calls with cheap self‑hosted SLMs |
| **Academic Researcher** | Need reproducible, low‑cost baselines | Full experiment tracking, export to GGUF/ONNX, open‑source stack |

### 4.2 Content Marketing Phases
1. **Technical Credibility** – Blog series, benchmark leaderboard, research integrations.  
2. **Community Building** – Model zoo, Discord/Slack, weekly office hours.  
3. **Enterprise Adoption** – Case studies, ROI calculator, paid support tier, certification program.

### 4.3 SEO & Discovery
| Keyword | Monthly SV | Difficulty |
|---------|------------|------------|
| "fine‑tune llama on colab" | 8.1K | Medium |
| "self‑hosted gpt alternative" | 4.2K | Medium |
| "small language models training" | 3.5K | Low |
| "qlora training tutorial" | 2.8K | Low |
| "local llm deployment" | 12K | High |

---

## 5️⃣ Technical Architecture Enhancements

### 5.1 Streaming & Real‑Time Inference
```python
# lmfast/inference/streaming.py
from lmfast.inference import SLMServer

class StreamingServer(SLMServer):
    """Extends SLMServer with SSE streaming support for OpenAI‑compatible endpoints."""
    def generate_stream(self, prompt: str, **kwargs):
        # Placeholder – would yield token chunks as they are produced.
        yield {"text": "..."}
```

### 5.2 Dynamic Batching for Production
```python
# lmfast/inference/batching.py
class ContinuousBatchingEngine:
    def __init__(self, model_path: str, max_batch_size: int = 32, max_wait_ms: int = 50):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        # Placeholder – initialise vLLM engine with continuous batching.
    def generate(self, prompts: list[str]):
        # Placeholder – batch incoming prompts and return results.
        return ["response" for _ in prompts]
```

### 5.3 Multi‑Tenancy & Routing
```python
# lmfast/serving/router.py
class ModelRouter:
    def __init__(self, routes: list[dict]):
        self.routes = routes
    def route_and_generate(self, prompt: str, fallback: str = None):
        # Simple heuristic: longer prompts → quality model, else speed model.
        target = self.routes[0]["path"] if len(prompt.split()) > 10 else self.routes[-1]["path"]
        from lmfast.inference.server import SLMServer
        server = SLMServer(target)
        return server.generate(prompt)
```

---

## 6️⃣ Ecosystem Integrations
- **LangChain** – `from langchain_lmfast import LMFast`
- **Hugging Face Spaces** – one‑click Gradio UI (`lmfast serve --ui gradio`).
- **Modal / Runpod** – `from lmfast.deployment import ServerlessDeployer`.
- **MCP Clients** – native tool/resource exposure via `lmfast.mcp.server.LMFastMCPServer`.

---

*All placeholder implementations are intentionally minimal; they will be fleshed out in upcoming releases.*
