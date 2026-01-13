import logging
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lmfast.core.config import TrainingConfig

logger = logging.getLogger(__name__)


class PreTrainer:
    """
    Trainer for pre-training tiny models (<200M params) from scratch on a single GPU.
    Optimized for Colab T4.
    """

    def __init__(
        self,
        model_type: str = "gpt2",
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_position_embeddings: int = 1024,
        config: TrainingConfig | None = None,
    ):
        self.config = config or TrainingConfig()

        # Define architecture (e.g., config for a TinyLlama or GPT-2 style model)
        # We default to a small configuration
        self.model_config = AutoConfig.from_pretrained(model_type)
        self.model_config.vocab_size = vocab_size
        self.model_config.n_embd = hidden_size
        self.model_config.n_layer = num_layers
        self.model_config.n_head = num_heads
        self.model_config.n_positions = max_position_embeddings

        self.model = AutoModelForCausalLM.from_config(self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)  # Reuse tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, dataset: Any, output_dir: str):
        """Train the model from scratch."""
        logger.info(
            f"Starting pre-training from scratch (Params: {self.model.num_parameters()/1e6:.1f}M)..."
        )

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            logging_steps=10,
            save_steps=100,
            fp16=True,  # Critical for T4
            optim="adamw_torch",
            report_to="none",
        )

        trainer = Trainer(
            model=self.model, args=args, train_dataset=dataset, processing_class=self.tokenizer
        )

        trainer.train()
        trainer.save_model(output_dir)
        logger.info(f"Pre-training complete! Saved to {output_dir}")


def pretrain(
    dataset: Any,
    output_dir: str = "./my_tiny_model",
    hidden_size: int = 384,  # Tiny
    num_layers: int = 6,
    max_steps: int = 1000,
    **kwargs,
) -> PreTrainer:
    """Pre-train a tiny model."""
    config = TrainingConfig(max_steps=max_steps, **kwargs)
    trainer = PreTrainer(hidden_size=hidden_size, num_layers=num_layers, config=config)
    trainer.train(dataset, output_dir)
    return trainer
