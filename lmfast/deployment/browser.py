"""
Browser Deployment for LMFast Models.

Export models for in-browser inference using WebLLM, ONNX, or Transformers.js.

Supports:
- WebLLM (MLC format) - Best performance with WebGPU
- ONNX Runtime Web - Cross-platform compatibility  
- Transformers.js - Easy integration with HuggingFace

Example:
    >>> from lmfast.deployment import BrowserExporter
    >>> exporter = BrowserExporter("./my_model", target="onnx", quantization="int4")
    >>> artifacts = exporter.export("./browser_model", create_demo=True)
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


class BrowserExporter:
    """
    Export LMFast models for in-browser inference.
    
    Targets:
    - webllm: Best performance, requires WebGPU (Chrome 113+)
    - onnx: Cross-platform, works with ONNX Runtime Web
    - transformers_js: Easy HuggingFace integration
    
    Example:
        >>> exporter = BrowserExporter(
        ...     model_path="./my_model",
        ...     target="onnx",
        ...     quantization="int4"
        ... )
        >>> exporter.export("./browser_model", create_demo=True)
    """
    
    def __init__(
        self,
        model_path: str,
        target: Literal["webllm", "onnx", "transformers_js"] = "onnx",
        quantization: Literal["int4", "int8", "fp16"] = "int4",
        context_length: int = 2048
    ):
        """
        Initialize browser exporter.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            target: Export target format
            quantization: Quantization level
            context_length: Max context length for browser inference
        """
        self.model_path = Path(model_path)
        self.target = target
        self.quantization = quantization
        self.context_length = context_length
        
        # Validate
        if not self.model_path.exists() and "/" not in str(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
    
    def export(
        self,
        output_dir: str,
        split_size_mb: int = 100,
        create_demo: bool = True,
        demo_framework: Literal["vanilla", "react", "vue"] = "vanilla"
    ) -> Dict[str, Any]:
        """
        Export model for browser deployment.
        
        Args:
            output_dir: Output directory for exported files
            split_size_mb: Split weights into chunks of this size (MB)
            create_demo: Whether to generate a demo application
            demo_framework: Framework for demo app
            
        Returns:
            Dictionary with paths to exported artifacts
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {self.model_path} to {self.target} format...")
        
        if self.target == "onnx":
            artifacts = self._export_onnx(out_path)
        elif self.target == "webllm":
            artifacts = self._export_webllm(out_path, split_size_mb)
        elif self.target == "transformers_js":
            artifacts = self._export_transformers_js(out_path)
        else:
            raise ValueError(f"Unknown target: {self.target}")
        
        if create_demo:
            demo_path = self._create_demo(out_path, demo_framework, artifacts)
            artifacts["demo"] = str(demo_path)
        
        # Save export config
        config = {
            "model": str(self.model_path),
            "target": self.target,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "artifacts": {k: str(v) for k, v in artifacts.items()}
        }
        
        with open(out_path / "export_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Export complete! Artifacts saved to {output_dir}")
        return artifacts
    
    def _export_onnx(self, output_path: Path) -> Dict[str, Path]:
        """Export to ONNX format for ONNX Runtime Web."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "ONNX export requires optimum. "
                "Install with: pip install optimum[onnxruntime]"
            )
        
        onnx_path = output_path / "onnx"
        onnx_path.mkdir(exist_ok=True)
        
        logger.info("Converting to ONNX format...")
        
        # Suppress warnings during export
        import warnings
        from torch.jit import TracerWarning
        warnings.filterwarnings("ignore", category=TracerWarning)
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        
        # Load and export
        # Optimum uses 'dtype' internally, we can pass it via kwargs if needed
        # but the warning "torch_dtype is deprecated" often comes from underlying transformers call
        model = ORTModelForCausalLM.from_pretrained(
            str(self.model_path),
            export=True,
            provider="CPUExecutionProvider",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        
        # Save
        model.save_pretrained(onnx_path)
        tokenizer.save_pretrained(onnx_path)
        
        # Apply quantization if needed
        if self.quantization in ["int4", "int8"]:
            self._quantize_onnx(onnx_path)
        
        # Create web config
        web_config = {
            "model_path": "onnx/model.onnx",
            "tokenizer_path": "onnx/tokenizer.json",
            "context_length": self.context_length,
            "quantization": self.quantization
        }
        
        with open(output_path / "web_config.json", "w") as f:
            json.dump(web_config, f, indent=2)
        
        return {
            "onnx_model": onnx_path / "model.onnx",
            "tokenizer": onnx_path / "tokenizer.json",
            "config": output_path / "web_config.json"
        }
    
    def _quantize_onnx(self, onnx_path: Path):
        """Apply quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            logger.warning("onnxruntime quantization not available")
            return
        
        model_path = onnx_path / "model.onnx"
        quantized_path = onnx_path / "model_quantized.onnx"
        
        quant_type = QuantType.QInt8 if self.quantization == "int8" else QuantType.QUInt8
        
        quantize_dynamic(
            str(model_path),
            str(quantized_path),
            weight_type=quant_type
        )
        
        # Replace original with quantized
        shutil.move(quantized_path, model_path)
        logger.info(f"Applied {self.quantization} quantization to ONNX model")
    
    def _export_webllm(self, output_path: Path, split_size_mb: int) -> Dict[str, Path]:
        """Export for WebLLM (MLC format)."""
        mlc_path = output_path / "mlc"
        mlc_path.mkdir(exist_ok=True)
        
        logger.info("Exporting for WebLLM (MLC format)...")
        logger.warning(
            "Full WebLLM export requires MLC-LLM toolkit. "
            "Creating config for manual conversion."
        )
        
        # Create WebLLM config
        config = {
            "model_lib": "SmolLM-135M-Instruct-q4f16_1-MLC",
            "model_id": str(self.model_path),
            "quantization": self.quantization,
            "context_window_size": self.context_length,
            "sliding_window_size": -1,
            "attention_sink_size": -1,
            "prefill_chunk_size": 512
        }
        
        with open(mlc_path / "mlc-chat-config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create instructions
        instructions = f"""# WebLLM Export Instructions

To fully convert this model for WebLLM:

1. Install MLC-LLM: https://llm.mlc.ai/docs/install/
2. Run conversion:
   ```bash
   mlc_llm convert_weight {self.model_path} --quantization {self.quantization}
   ```
3. Copy output to {mlc_path}
4. Use the generated files in your WebLLM application

## Quick Start (pre-converted models)

If using a supported model, you can skip conversion:
```javascript
import {{ CreateMLCEngine }} from "@anthropic-ai/sdk";
const engine = await CreateMLCEngine("SmolLM-135M-Instruct-q4f16_1-MLC");
```
"""
        
        with open(mlc_path / "README.md", "w") as f:
            f.write(instructions)
        
        return {
            "config": mlc_path / "mlc-chat-config.json",
            "readme": mlc_path / "README.md"
        }
    
    def _export_transformers_js(self, output_path: Path) -> Dict[str, Path]:
        """Export for Transformers.js."""
        tfjs_path = output_path / "transformers_js"
        tfjs_path.mkdir(exist_ok=True)
        
        logger.info("Exporting for Transformers.js...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Save tokenizer in correct format
            tokenizer.save_pretrained(tfjs_path)
            
            # Create usage instructions
            usage = f"""# Transformers.js Usage

```javascript
import {{ pipeline }} from '@xenova/transformers';

// Load the model
const generator = await pipeline('text-generation', '{self.model_path}');

// Generate text
const output = await generator('Hello, how are you?', {{
    max_new_tokens: 100,
    temperature: 0.7
}});
console.log(output[0].generated_text);
```

## Notes
- Model will be downloaded and cached automatically
- Quantization is applied by Transformers.js based on device
- For best performance, use WebGPU-enabled browser
"""
            
            with open(tfjs_path / "README.md", "w") as f:
                f.write(usage)
                
        except Exception as e:
            logger.error(f"Transformers.js export failed: {e}")
            raise
        
        return {
            "tokenizer": tfjs_path / "tokenizer.json",
            "readme": tfjs_path / "README.md"
        }
    
    def _create_demo(
        self, 
        output_path: Path, 
        framework: str,
        artifacts: Dict[str, Path]
    ) -> Path:
        """Create a demo application."""
        demo_path = output_path / "demo"
        demo_path.mkdir(exist_ok=True)
        
        if framework == "vanilla":
            return self._create_vanilla_demo(demo_path, artifacts)
        elif framework == "react":
            return self._create_react_demo(demo_path, artifacts)
        elif framework == "vue":
            return self._create_vue_demo(demo_path, artifacts)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def _create_vanilla_demo(self, demo_path: Path, artifacts: Dict) -> Path:
        """Create vanilla HTML/JS demo."""
        
        if self.target == "onnx":
            demo_html = self._get_onnx_demo_html()
        else:
            demo_html = self._get_webllm_demo_html()
        
        index_path = demo_path / "index.html"
        index_path.write_text(demo_html)
        
        # Create simple CSS
        css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 800px; 
    margin: 0 auto; 
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}
.container {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
h1 { 
    color: #333; 
    margin-bottom: 0.5rem;
    font-size: 1.8rem;
}
.subtitle { color: #666; margin-bottom: 1.5rem; }
#status { 
    padding: 0.75rem 1rem; 
    border-radius: 8px; 
    margin-bottom: 1rem;
    font-weight: 500;
}
.loading { background: #fff3cd; color: #856404; }
.ready { background: #d4edda; color: #155724; }
.error { background: #f8d7da; color: #721c24; }
#chat { 
    height: 400px; 
    overflow-y: auto; 
    border: 1px solid #e0e0e0; 
    border-radius: 12px;
    padding: 1rem; 
    margin-bottom: 1rem;
    background: #fafafa;
}
.message { margin-bottom: 1rem; padding: 0.75rem; border-radius: 8px; }
.user { background: #e3f2fd; margin-left: 2rem; }
.assistant { background: #f5f5f5; margin-right: 2rem; }
.input-row { display: flex; gap: 0.5rem; }
#input { 
    flex: 1; 
    padding: 0.75rem 1rem; 
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 1rem;
    transition: border-color 0.2s;
}
#input:focus { outline: none; border-color: #667eea; }
#input:disabled { background: #f5f5f5; }
button {
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    transition: transform 0.2s, box-shadow 0.2s;
}
button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.footer { margin-top: 1.5rem; text-align: center; color: #666; font-size: 0.9rem; }
.footer a { color: #667eea; }
"""
        
        (demo_path / "style.css").write_text(css)
        
        logger.info(f"Demo created at {demo_path / 'index.html'}")
        return demo_path / "index.html"
    
    def _get_onnx_demo_html(self) -> str:
        """Get ONNX Runtime Web demo HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LMFast Browser Demo</title>
    <link rel="stylesheet" href="style.css">
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>🚀 LMFast Browser AI</h1>
        <p class="subtitle">Running 100% locally in your browser. No server required.</p>
        
        <div id="status" class="loading">Loading model...</div>
        
        <div id="chat"></div>
        
        <div class="input-row">
            <input id="input" type="text" placeholder="Type your message..." disabled>
            <button id="send" disabled>Send</button>
        </div>
        
        <div class="footer">
            Powered by <a href="https://github.com/2796gaurav/LMFast">LMFast</a> & ONNX Runtime Web
        </div>
    </div>
    
    <script>
        let session = null;
        
        async function loadModel() {
            try {
                const statusEl = document.getElementById('status');
                statusEl.textContent = 'Loading ONNX model...';
                
                session = await ort.InferenceSession.create('./onnx/model.onnx');
                
                statusEl.textContent = 'Model ready! Start chatting.';
                statusEl.className = 'ready';
                document.getElementById('input').disabled = false;
                document.getElementById('send').disabled = false;
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
                document.getElementById('status').className = 'error';
            }
        }
        
        async function generate(prompt) {
            // Simplified - actual implementation needs tokenization
            const chat = document.getElementById('chat');
            chat.innerHTML += '<div class="message assistant">AI response would appear here...</div>';
            chat.scrollTop = chat.scrollHeight;
        }
        
        document.getElementById('send').addEventListener('click', async () => {
            const input = document.getElementById('input');
            const prompt = input.value.trim();
            if (!prompt) return;
            
            const chat = document.getElementById('chat');
            chat.innerHTML += '<div class="message user">' + prompt + '</div>';
            input.value = '';
            
            await generate(prompt);
        });
        
        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') document.getElementById('send').click();
        });
        
        loadModel();
    </script>
</body>
</html>'''
    
    def _get_webllm_demo_html(self) -> str:
        """Get WebLLM demo HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LMFast Browser Demo</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>🚀 LMFast Browser AI</h1>
        <p class="subtitle">Running 100% locally with WebGPU. No server required.</p>
        
        <div id="status" class="loading">Initializing WebGPU...</div>
        
        <div id="chat"></div>
        
        <div class="input-row">
            <input id="input" type="text" placeholder="Type your message..." disabled>
            <button id="send" disabled>Send</button>
        </div>
        
        <div class="footer">
            Powered by <a href="https://github.com/2796gaurav/LMFast">LMFast</a> & WebLLM
        </div>
    </div>
    
    <script type="module">
        import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";
        
        let engine = null;
        const statusEl = document.getElementById('status');
        const inputEl = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const chatEl = document.getElementById('chat');
        
        async function loadModel() {
            try {
                statusEl.textContent = 'Loading model (this may take a minute)...';
                
                engine = await CreateMLCEngine(
                    "SmolLM-360M-Instruct-q4f16_1-MLC",
                    {
                        initProgressCallback: (progress) => {
                            statusEl.textContent = progress.text || 'Loading...';
                        }
                    }
                );
                
                statusEl.textContent = 'Model ready! Start chatting.';
                statusEl.className = 'ready';
                inputEl.disabled = false;
                sendBtn.disabled = false;
            } catch (e) {
                statusEl.textContent = 'Error: ' + e.message;
                statusEl.className = 'error';
                console.error(e);
            }
        }
        
        async function generate(prompt) {
            sendBtn.disabled = true;
            inputEl.disabled = true;
            
            const reply = await engine.chat.completions.create({
                messages: [{ role: "user", content: prompt }],
                max_tokens: 256,
            });
            
            chatEl.innerHTML += '<div class="message assistant">' + 
                reply.choices[0].message.content + '</div>';
            chatEl.scrollTop = chatEl.scrollHeight;
            
            sendBtn.disabled = false;
            inputEl.disabled = false;
            inputEl.focus();
        }
        
        sendBtn.addEventListener('click', async () => {
            const prompt = inputEl.value.trim();
            if (!prompt) return;
            
            chatEl.innerHTML += '<div class="message user">' + prompt + '</div>';
            inputEl.value = '';
            
            await generate(prompt);
        });
        
        inputEl.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendBtn.click();
        });
        
        loadModel();
    </script>
</body>
</html>'''
    
    def _create_react_demo(self, demo_path: Path, artifacts: Dict) -> Path:
        """Create React demo template."""
        readme = """# React Demo

Run the following to create a React app:

```bash
npx create-react-app lmfast-demo
cd lmfast-demo
npm install @anthropic-ai/sdk
```

Then copy the ONNX/WebLLM files to `public/` and use the provided code.
"""
        (demo_path / "README.md").write_text(readme)
        return demo_path / "README.md"
    
    def _create_vue_demo(self, demo_path: Path, artifacts: Dict) -> Path:
        """Create Vue demo template."""
        readme = """# Vue Demo

Run the following to create a Vue app:

```bash
npm create vue@latest lmfast-demo
cd lmfast-demo
npm install @anthropic-ai/sdk
```

Then copy the ONNX/WebLLM files to `public/` and use the provided code.
"""
        (demo_path / "README.md").write_text(readme)
        return demo_path / "README.md"


def export_for_browser(
    model_path: str,
    output_dir: str,
    target: str = "onnx",
    quantization: str = "int4",
    create_demo: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    One-line browser export.
    
    Args:
        model_path: Path to model
        output_dir: Output directory
        target: Export target (onnx, webllm, transformers_js)
        quantization: Quantization level
        create_demo: Create demo app
        
    Returns:
        Exported artifact paths
        
    Example:
        >>> from lmfast.deployment import export_for_browser
        >>> export_for_browser("./my_model", "./browser", target="onnx")
    """
    exporter = BrowserExporter(
        model_path=model_path,
        target=target,
        quantization=quantization,
        **kwargs
    )
    return exporter.export(output_dir, create_demo=create_demo)
