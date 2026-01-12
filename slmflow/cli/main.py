"""
SLMFlow Command Line Interface.

Provides commands for training, distillation, inference, and export.
"""

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Create CLI app
app = typer.Typer(
    name="slmflow",
    help="SLMFlow: Democratized Small Language Model Training",
    add_completion=False,
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("slmflow")


@app.command()
def train(
    model: str = typer.Option(
        "HuggingFaceTB/SmolLM-135M",
        "--model",
        "-m",
        help="Model name or HuggingFace ID",
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Training data (HuggingFace dataset or local path)",
    ),
    output: str = typer.Option(
        "./slmflow_output",
        "--output",
        "-o",
        help="Output directory",
    ),
    max_steps: int = typer.Option(
        500,
        "--max-steps",
        help="Maximum training steps",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        "-b",
        help="Batch size per device",
    ),
    learning_rate: float = typer.Option(
        2e-4,
        "--lr",
        help="Learning rate",
    ),
    lora_r: int = typer.Option(
        16,
        "--lora-r",
        help="LoRA rank",
    ),
    text_field: str = typer.Option(
        "text",
        "--text-field",
        help="Dataset field containing text",
    ),
):
    """
    Train or fine-tune a Small Language Model.

    Example:
        slmflow train --model HuggingFaceTB/SmolLM-135M --data yahma/alpaca-cleaned
    """
    from slmflow import SLMConfig, SLMTrainer, TrainingConfig
    from slmflow.training.data import load_dataset

    console.print("\n[bold blue]SLMFlow Training[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Data: {data}")
    console.print(f"Output: {output}")
    console.print()

    # Load dataset
    with console.status("Loading dataset..."):
        dataset = load_dataset(data)
        console.print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Configure
    model_config = SLMConfig(model_name=model)
    train_config = TrainingConfig(
        output_dir=output,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
    )

    # Train
    trainer = SLMTrainer(model_config, train_config)

    console.print("\nStarting training...")
    metrics = trainer.train(dataset, text_field=text_field)

    console.print("\n[bold green]✓ Training complete![/bold green]")
    console.print(f"Final loss: {metrics.get('train_loss', 'N/A')}")
    console.print(f"Model saved to: {output}")


@app.command()
def distill(
    teacher: str = typer.Option(
        ...,
        "--teacher",
        "-t",
        help="Teacher model (larger model)",
    ),
    student: str = typer.Option(
        "HuggingFaceTB/SmolLM-135M",
        "--student",
        "-s",
        help="Student model (smaller model)",
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Training data",
    ),
    output: str = typer.Option(
        "./slmflow_distilled",
        "--output",
        "-o",
        help="Output directory",
    ),
    temperature: float = typer.Option(
        2.0,
        "--temperature",
        help="Distillation temperature",
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        help="Balance between CE and KL loss",
    ),
):
    """
    Distill knowledge from a larger teacher to a smaller student model.

    Example:
        slmflow distill --teacher Qwen/Qwen2-1.5B --student HuggingFaceTB/SmolLM-135M --data my_data.json
    """
    from slmflow.core.config import DistillationConfig, TrainingConfig
    from slmflow.distillation import DistillationTrainer
    from slmflow.training.data import load_dataset

    console.print("\n[bold blue]SLMFlow Knowledge Distillation[/bold blue]")
    console.print(f"Teacher: {teacher}")
    console.print(f"Student: {student}")
    console.print()

    # Load dataset
    with console.status("Loading dataset..."):
        dataset = load_dataset(data)
        console.print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Configure
    distill_config = DistillationConfig(
        teacher_model=teacher,
        temperature=temperature,
        alpha=alpha,
    )
    train_config = TrainingConfig(output_dir=output)

    # Distill
    trainer = DistillationTrainer(
        student_model=student,
        distillation_config=distill_config,
        training_config=train_config,
    )

    console.print("\nStarting distillation...")
    trainer.distill(dataset, output_dir=output)

    console.print("\n[bold green]✓ Distillation complete![/bold green]")
    console.print(f"Model saved to: {output}")


@app.command()
def serve(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model path or HuggingFace ID",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Server host",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Server port",
    ),
    use_vllm: bool = typer.Option(
        True,
        "--vllm/--no-vllm",
        help="Use vLLM for fast inference",
    ),
):
    """
    Start an inference server with OpenAI-compatible API.

    Example:
        slmflow serve --model ./my_model --port 8000
    """
    from slmflow.inference import SLMServer

    console.print("\n[bold blue]SLMFlow Inference Server[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Endpoint: http://{host}:{port}")
    console.print()

    server = SLMServer(model, use_vllm=use_vllm)
    server.serve(host=host, port=port)


@app.command()
def export(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model path",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path",
    ),
    format: str = typer.Option(
        "gguf",
        "--format",
        "-f",
        help="Export format (gguf, int4, int8, awq)",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quant",
        "-q",
        help="Quantization type for GGUF",
    ),
):
    """
    Export model to different formats for deployment.

    Example:
        slmflow export --model ./my_model --output ./my_model.gguf --format gguf
    """
    from slmflow.inference.quantization import export_gguf, quantize_model

    console.print("\n[bold blue]SLMFlow Export[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Format: {format}")
    console.print(f"Output: {output}")
    console.print()

    with console.status(f"Exporting to {format}..."):
        if format == "gguf":
            export_gguf(model, output, quantization=quantization)
        else:
            quantize_model(model, output, method=format)

    console.print("\n[bold green]✓ Export complete![/bold green]")
    console.print(f"Model exported to: {output}")


@app.command()
def generate(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model path or HuggingFace ID",
    ),
    prompt: str = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Input prompt",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive chat mode",
    ),
):
    """
    Generate text from a model.

    Example:
        slmflow generate --model ./my_model --prompt "Hello, how are you?"
        slmflow generate --model ./my_model --interactive
    """
    from slmflow.inference import SLMServer

    server = SLMServer(model, use_vllm=False)

    if interactive:
        console.print("\n[bold blue]SLMFlow Interactive Mode[/bold blue]")
        console.print("Type 'exit' or 'quit' to exit.\n")

        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
                if user_input.lower() in ("exit", "quit"):
                    break

                response = server.generate(
                    user_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                console.print(f"[bold blue]Assistant:[/bold blue] {response}\n")

            except KeyboardInterrupt:
                break

        console.print("\nGoodbye!")
    else:
        if prompt is None:
            prompt = typer.prompt("Enter prompt")

        console.print("\n[dim]Generating...[/dim]")
        response = server.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        console.print(f"\n{response}")


@app.command()
def info(
    model: str = typer.Argument(
        ...,
        help="Model path or HuggingFace ID",
    ),
):
    """
    Show information about a model.

    Example:
        slmflow info HuggingFaceTB/SmolLM-135M
    """
    from slmflow.core.models import get_model_info
    from slmflow.inference.quantization import get_model_size

    console.print("\n[bold blue]Model Information[/bold blue]")
    console.print(f"Model: {model}\n")

    # Get info
    info = get_model_info(model)
    size_info = get_model_size(model)

    # Display as table
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    for key, value in info.items():
        if value is not None:
            table.add_row(key, str(value))

    if "size_gb" in size_info:
        table.add_row("size", f"{size_info['size_gb']:.2f} GB")

    console.print(table)


@app.command()
def benchmark(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model path",
    ),
    num_prompts: int = typer.Option(
        10,
        "--prompts",
        "-n",
        help="Number of test prompts",
    ),
    runs: int = typer.Option(
        3,
        "--runs",
        "-r",
        help="Number of benchmark runs",
    ),
):
    """
    Benchmark model inference performance.

    Example:
        slmflow benchmark --model ./my_model --prompts 10
    """
    from slmflow.inference import SLMServer

    console.print("\n[bold blue]SLMFlow Benchmark[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Prompts: {num_prompts}")
    console.print(f"Runs: {runs}\n")

    # Generate test prompts
    prompts = [f"Write a short story about topic {i}." for i in range(num_prompts)]

    server = SLMServer(model)

    with console.status("Running benchmark..."):
        results = server.benchmark(prompts, num_runs=runs)

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Backend", results["backend"])
    table.add_row("Prompts", str(results["prompts"]))
    table.add_row("Avg Latency", f"{results['avg_latency_ms']:.2f} ms")
    table.add_row("Min Latency", f"{results['min_latency_ms']:.2f} ms")
    table.add_row("Max Latency", f"{results['max_latency_ms']:.2f} ms")
    table.add_row("Throughput", f"{results['throughput_tokens_per_sec']:.1f} tokens/s")

    console.print(table)


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version",
    ),
):
    """
    SLMFlow: Democratized Small Language Model Training

    Train, fine-tune, distill, and deploy sub-500M parameter models
    on Colab T4 in 30-40 minutes with enterprise-grade features.
    """
    if version:
        from slmflow import __version__

        console.print(f"slmflow version {__version__}")
        raise typer.Exit()

    if verbose:
        logging.getLogger("slmflow").setLevel(logging.DEBUG)


if __name__ == "__main__":
    app()
