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

from lmfast.core.config import SLMConfig, TrainingConfig
from lmfast.core.models import (
    TokenizerType,
    load_model,
    prepare_model_for_training,
    save_model,
)
from lmfast.training.data import prepare_dataset
from lmfast.training.optimizations import get_memory_stats, optimize_for_t4

logger = logging.getLogger(__name__)


class SLMTrainer:
    _model: PreTrainedModel | None
    _tokenizer: TokenizerType | None
    _trainer: Any | None

    """
    End-to-end trainer for Small Language Models.

    Optimized for Colab T4 (12GB VRAM) with automatic:
    - QLoRA quantization
    - Gradient checkpointing
    - Mixed precision training
    - Unsloth acceleration (if available)

    Example:
        >>> from lmfast import SLMTrainer, SLMConfig, TrainingConfig
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
        assert self._model is not None, "Model failed to load"
        return self._model

    @property
    def tokenizer(self) -> TokenizerType:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self._load_model()
        assert self._tokenizer is not None, "Tokenizer failed to load"
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
        model = self.model
        self._model = prepare_model_for_training(
            model,
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

        if self._trainer is None:
            raise RuntimeError("Trainer not created. Call prepare() first.")

        train_result = self._trainer.train(
            resume_from_checkpoint=resume_from_checkpoint,
        )

        # Log final metrics
        metrics = train_result.metrics
        logger.info(f"Training complete! Final metrics: {metrics}")

        # Save final model
        self.save(self.training_config.output_dir)

        return dict(metrics)

    def _create_trainer(
        self,
        train_dataset: Any,
        callbacks: list | None = None,
    ):
        """Create the SFT trainer."""
        from trl import SFTConfig, SFTTrainer

        # Create SFT config
        sft_kwargs = self.training_config.to_sft_config_kwargs()

        # SFTConfig (TrainingArguments) does not accept these args in recent versions
        # They should be passed to the SFTConfig constructor ONLY if supported, but
        # standard TrainingArguments don't support them.
        # Alternatively, SFTTrainer accepts them.

        # We'll filter them out from sft_kwargs if they were somehow added,
        # but to_sft_config_kwargs mostly adds standard args.
        # The specific fields causing error were manually added below in previous code.

        # We pass these directly to SFTConfig ONLY if they are valid, but the error says they aren't.
        # So we pass them to SFTConfig correctly? usage: args=SFTConfig(...)
        # Wait, if SFTConfig is a dataclass for arguments, maybe it DOESN'T have max_seq_length.
        # In newer TRL, you pass them to SFTConfig.
        # BUT if the user gets an error, we must assume they are using a version where SFTConfig doesn't have it (or we are using it wrong).

        # Let's try passing them to SFTConfig only if we are sure, but here we see failure.
        # Actually, in TRL < 0.8, these were SFTTrainer args. In TRL >= 0.8, they moved to SFTConfig?
        # The user has trl-0.24.0. In 0.24.0, SFTConfig SHOULD have them.
        # Wait, the error `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
        # implies SFTConfig does NOT accept it.
        # Perhaps `max_seq_length` is passed as a positional arg or something? No.
        #
        # Let's verify `trl` version. The user log says `trl-0.24.0`.
        # In `trl` 0.24.0, `SFTConfig` definitely has `max_seq_length`.
        # UNLESS `lmfast` imports `SFTConfig` from somewhere else?
        # `from trl import SFTConfig, SFTTrainer` matches.

        # Maybe I am shadowing something?
        # Ah, maybe I should check if `max_seq_length` is passed to `SFTTrainer` directly instead?
        # A common pattern is: args=training_args, max_seq_length=...
        # Let's try moving them to SFTConfig properties if possible, or just pass to SFTTrainer if the Config fails.

        # The SAFEST bet for compatibility with weird versions or if I'm misremembering:
        # Pass standard TrainingArguments to SFTConfig.
        # Pass SFT-specific args (max_seq_length, packing, dataset_text_field) to SFTTrainer.

        sft_config = SFTConfig(**sft_kwargs)

        # Safely set attributes based on TRL version
        for attr, value in [
            ("dataset_text_field", "text"),
            ("max_seq_length", self.model_config.max_seq_length),
            ("max_length", self.model_config.max_seq_length),
            ("packing", True),
        ]:
            if hasattr(sft_config, attr):
                setattr(sft_config, attr, value)

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
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
            self.model,
            self.tokenizer,
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
        tokenizer = self.tokenizer
        from typing import cast

        inputs = cast(Any, tokenizer)(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.max_seq_length,
        ).to(self.model.device)

        # Generate
        model = self.model
        with torch.no_grad():
            outputs = cast(Any, model).generate(
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

        return str(response)

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
