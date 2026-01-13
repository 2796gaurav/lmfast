"""
Model quantization for deployment.

Provides utilities to quantize models for efficient inference.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_model(
    model_path: str,
    output_path: str,
    *,
    method: str = "int4",
    calibration_dataset: str | None = None,
) -> Path:
    """
    Quantize a model for efficient inference.

    Supported methods:
    - int4: 4-bit quantization (smallest, slight accuracy loss)
    - int8: 8-bit quantization (balanced)
    - awq: Activation-aware Weight Quantization
    - gptq: GPTQ quantization

    Args:
        model_path: Path to model or HuggingFace ID
        output_path: Output path for quantized model
        method: Quantization method
        calibration_dataset: Dataset for calibration (for GPTQ/AWQ)

    Returns:
        Path to quantized model

    Example:
        >>> quantize_model("./my_model", "./my_model_int4", method="int4")
    """
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing {model_path} with {method} to {out_path}")

    if method == "int4":
        return _quantize_bitsandbytes(model_path, out_path, bits=4)
    elif method == "int8":
        return _quantize_bitsandbytes(model_path, out_path, bits=8)
    elif method == "awq":
        return _quantize_awq(model_path, out_path, calibration_dataset)
    elif method == "gptq":
        return _quantize_gptq(model_path, out_path, calibration_dataset)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def _quantize_bitsandbytes(
    model_path: str,
    output_path: Path,
    bits: int,
) -> Path:
    """Quantize using bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype="float16" if bits == 4 else None,
        bnb_4bit_use_double_quant=True if bits == 4 else None,
        bnb_4bit_quant_type="nf4" if bits == 4 else None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Quantized model saved to {output_path}")
    return output_path


def _quantize_awq(
    model_path: str,
    output_path: Path,
    calibration_dataset: str | None,
) -> Path:
    """Quantize using AWQ."""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("AWQ not installed. Install with: pip install autoawq") from e

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Quantize
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(tokenizer, quant_config=quant_config)

    # Save
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(output_path)

    logger.info(f"AWQ quantized model saved to {output_path}")
    return output_path


def _quantize_gptq(
    model_path: str,
    output_path: Path,
    calibration_dataset: str | None,
) -> Path:
    """Quantize using GPTQ."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError as e:
        raise ImportError("GPTQ support requires: pip install optimum auto-gptq") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gptq_config = GPTQConfig(
        bits=4,
        dataset=calibration_dataset or "c4",
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map="auto",
    )

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"GPTQ quantized model saved to {output_path}")
    return output_path


def export_gguf(
    model_path: str,
    output_path: str,
    *,
    quantization: str = "q4_k_m",
) -> Path:
    """
    Export model to GGUF format for llama.cpp.

    Args:
        model_path: Path to model
        output_path: Output path for GGUF file
        quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, etc.)

    Returns:
        Path to GGUF file

    Example:
        >>> export_gguf("./my_model", "./my_model.gguf", quantization="q4_k_m")
    """
    import subprocess
    import sys
    from shutil import which

    out_path = Path(output_path)

    # Check for llama.cpp convert script in PATH
    convert_script = which("convert-hf-to-gguf.py")

    # If not in PATH, look for it in local llama.cpp dir or clone it
    if convert_script is None:
        llama_cpp_dir = Path("llama.cpp")
        convert_script_path = llama_cpp_dir / "convert_hf_to_gguf.py"  # New name
        convert_script_path_old = llama_cpp_dir / "convert-hf-to-gguf.py"  # Old name

        if not llama_cpp_dir.exists():
            logger.info("llama.cpp not found. Cloning from GitHub...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                    check=True,
                    capture_output=True,
                )
                logger.info("Cloned llama.cpp successfully.")

                # Install requirements
                logger.info("Installing llama.cpp requirements...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(llama_cpp_dir / "requirements.txt"),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to setup llama.cpp: {e}")

        if convert_script_path.exists():
            convert_script = str(convert_script_path)
        elif convert_script_path_old.exists():
            convert_script = str(convert_script_path_old)
        else:
            raise RuntimeError("Could not find convert_hf_to_gguf.py in llama.cpp directory")

    logger.info(f"Using conversion script: {convert_script}")

    # Run conversion
    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile",
        str(out_path),
        "--outtype",
        quantization,
    ]

    try:
        logger.info(f"Running GGUF conversion command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"GGUF conversion failed output: {e.stdout}")
        logger.error(f"GGUF conversion failed errors: {e.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {e.stderr}")

    logger.info(f"GGUF model exported to {output_path}")
    return out_path


def _export_gguf_online(
    model_path: str,
    output_path: Path,
    quantization: str,
) -> Path:
    """Export to GGUF using online conversion (HuggingFace Spaces)."""
    logger.info(
        "Using online GGUF conversion. " "For better performance, install llama.cpp locally."
    )

    # This is a placeholder - in production you'd use a conversion API
    # or guide the user to use llama.cpp directly
    raise NotImplementedError(
        "Online GGUF conversion not implemented. "
        "Please install llama.cpp and run convert-hf-to-gguf.py manually."
    )


def get_model_size(model_path: str) -> dict:
    """
    Get model size information.

    Args:
        model_path: Path to model

    Returns:
        Dictionary with size information
    """
    from pathlib import Path

    path = Path(model_path)

    if not path.exists():
        # Try to get from HuggingFace
        try:
            from huggingface_hub import model_info

            info = model_info(model_path)
            return {
                "source": "huggingface",
                "model_id": model_path,
                "size_bytes": sum(
                    s.size
                    for s in (info.siblings or [])
                    if s.size is not None
                    and s.rfilename is not None
                    and s.rfilename.endswith((".bin", ".safetensors"))
                ),
            }
        except Exception:
            return {"error": "Model not found"}

    # Local model
    total_size = 0
    file_count = 0

    for file in path.rglob("*"):
        if file.is_file() and file.suffix in (".bin", ".safetensors"):
            total_size += file.stat().st_size
            file_count += 1

    return {
        "source": "local",
        "path": str(path),
        "size_bytes": total_size,
        "size_gb": total_size / (1024**3),
        "file_count": file_count,
    }
