"""
SLMTrainer: Unified training interface for Small Language Models.

Provides end-to-end training optimized for Colab T4 (12GB VRAM).
Automatically applies best practices for efficient fine-tuning.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from slmflow.core.config import SLMConfig, TrainingConfig
from slmflow.core.models import (
    load_model,
    prepare_model_for_training,
    save_model,
)
from slmflow.training.data import prepare_dataset
from slmflow.training.optimizations import get_memory_stats, optimize_for_t4

logger = logging.getLogger(__name__)


class SLMTrainer:
    """
    End-to-end trainer for Small Language Models.

    Optimized for Colab T4 (12GB VRAM) with automatic:
    - QLoRA quantization
    - Gradient checkpointing
    - Mixed precision training
    - Unsloth acceleration (if available)

    Example:
        >>> from slmflow import SLMTrainer, SLMConfig, TrainingConfig
        >>>
        >>> # Configure
        >>> model_config = SLMConfig(model_name="HuggingFaceTB/SmolLM-135M")
        >>> train_config = TrainingConfig(max_steps=500)
        >>>
        >>> # Create trainer
        >>> trainer = SLMTrainer(model_config, train_config)
        >>>
        >>> # Train
        >>> trainer.train(dataset)
        >>>
        >>> # Save
        >>> trainer.save("./my_model")
    """

    def __init__(
        self,
        model_config: SLMConfig | None = None,
        training_config: TrainingConfig | None = None,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        use_unsloth: bool = True,
    ):
        """
        Initialize the SLMTrainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            use_unsloth: Use Unsloth for acceleration
        """
        self.model_config = model_config or SLMConfig()
        self.training_config = training_config or TrainingConfig()
        self.use_unsloth = use_unsloth

        self._model = model
        self._tokenizer = tokenizer
        self._trainer = None
        self._is_prepared = False

        # Log configuration
        logger.info("SLMTrainer initialized")
        logger.info(f"  Model: {self.model_config.model_name}")
        logger.info(f"  Max steps: {self.training_config.max_steps}")
        logger.info(f"  Effective batch size: {self.training_config.effective_batch_size}")
        logger.info(
            f"  Est. training time: ~{self.training_config.estimated_training_time_minutes} min"
        )

    @property
    def model(self) -> PreTrainedModel:
        """Get the model, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info("Loading model and tokenizer...")

        # Apply T4 optimizations first
        optimize_for_t4()

        # Load model
        self._model, self._tokenizer = load_model(
            self.model_config,
            for_training=True,
            use_unsloth=self.use_unsloth,
        )

        # Log memory after loading
        mem_stats = get_memory_stats()
        logger.info(f"Memory after model load: {mem_stats}")

    def prepare(self) -> "SLMTrainer":
        """
        Prepare model for training with LoRA adapters.

        Returns:
            Self for chaining
        """
        if self._is_prepared:
            logger.info("Model already prepared for training")
            return self

        # Ensure model is loaded
        _ = self.model

        # Apply LoRA
        self._model = prepare_model_for_training(
            self._model,
            self.training_config,
            use_unsloth=self.use_unsloth,
        )

        self._is_prepared = True

        # Log memory after LoRA
        mem_stats = get_memory_stats()
        logger.info(f"Memory after LoRA: {mem_stats}")

        return self

    def train(
        self,
        dataset: Any,
        *,
        text_field: str = "text",
        output_dir: str | None = None,
        callbacks: list | None = None,
        resume_from_checkpoint: str | bool | None = None,
    ) -> dict[str, float]:
        """
        Train the model on a dataset.

        Args:
            dataset: Training dataset (HuggingFace Dataset or similar)
            text_field: Field containing the text to train on
            output_dir: Output directory (overrides config)
            callbacks: Optional training callbacks
            resume_from_checkpoint: Resume from checkpoint path

        Returns:
            Training metrics

        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
            >>> metrics = trainer.train(dataset)
        """
        # Prepare model if not already done
        if not self._is_prepared:
            self.prepare()

        # Override output dir if provided
        if output_dir is not None:
            self.training_config.output_dir = output_dir

        # Prepare dataset
        logger.info("Preparing dataset...")
        train_dataset = prepare_dataset(
            dataset,
            self.tokenizer,
            text_field=text_field,
            max_seq_length=self.model_config.max_seq_length,
        )

        logger.info(f"Dataset prepared: {len(train_dataset)} samples")

        # Create trainer
        self._trainer = self._create_trainer(
            train_dataset,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting training...")
        logger.info(f"  Output dir: {self.training_config.output_dir}")

        train_result = self._trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # Log final metrics
        metrics = train_result.metrics
        logger.info(f"Training complete! Final metrics: {metrics}")

        # Save final model
        self.save(self.training_config.output_dir)

        return metrics

    def _create_trainer(
        self,
        train_dataset: Any,
        callbacks: list | None = None,
    ):
        """Create the SFT trainer."""
        from trl import SFTConfig, SFTTrainer

        # Create SFT config
        sft_kwargs = self.training_config.to_sft_config_kwargs()
        sft_kwargs["dataset_text_field"] = "text"
        sft_kwargs["max_seq_length"] = self.model_config.max_seq_length
        sft_kwargs["packing"] = True  # Pack sequences for efficiency

        sft_config = SFTConfig(**sft_kwargs)

        # Create trainer
        trainer = SFTTrainer(
            model=self._model,
            args=sft_config,
            train_dataset=train_dataset,
            tokenizer=self._tokenizer,
            callbacks=callbacks,
        )

        return trainer

    def save(
        self,
        output_dir: str | Path,
        *,
        merge_adapters: bool = False,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
    ) -> Path:
        """
        Save the trained model.

        Args:
            output_dir: Directory to save to
            merge_adapters: Merge LoRA adapters into base model
            push_to_hub: Push to HuggingFace Hub
            hub_model_id: Hub model ID

        Returns:
            Path to saved model
        """
        return save_model(
            self._model,
            self._tokenizer,
            output_dir,
            merge_adapters=merge_adapters,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional generation kwargs

        Returns:
            Generated text
        """
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_seq_length,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        training_config: TrainingConfig | None = None,
    ) -> "SLMTrainer":
        """
        Load a trainer from a saved model.

        Args:
            model_path: Path to saved model
            training_config: Training configuration for continued training

        Returns:
            SLMTrainer instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_config = SLMConfig(model_name=model_path)

        return cls(
            model_config=model_config,
            training_config=training_config or TrainingConfig(),
            model=model,
            tokenizer=tokenizer,
        )
