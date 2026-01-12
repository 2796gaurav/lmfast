"""
Teacher-Student Knowledge Distillation.

Transfer knowledge from larger teacher models to smaller student models.
Optimized for Colab T4 where both models need to fit in memory.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import PreTrainedModel

from slmflow.core.config import DistillationConfig, SLMConfig, TrainingConfig
from slmflow.core.models import load_model, load_tokenizer, save_model
from slmflow.training.optimizations import clear_memory, get_memory_stats

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.

    Combines:
    - Cross-entropy loss with hard labels
    - KL divergence loss with soft labels from teacher
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Softmax temperature for soft labels
            alpha: Balance between CE (1-alpha) and KL (alpha) loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels

        Returns:
            Dictionary with loss values
        """
        # Hard label loss (cross-entropy)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Soft label loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (self.temperature**2)

        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
        }


class DistillationTrainer:
    """
    Knowledge distillation trainer.

    Enables training a small student model to mimic a larger teacher model.
    Optimized for Colab T4 memory constraints.

    Example:
        >>> from slmflow import DistillationTrainer
        >>> from slmflow.core.config import DistillationConfig
        >>>
        >>> # Configure
        >>> config = DistillationConfig(
        ...     teacher_model="Qwen/Qwen2-1.5B",
        ...     temperature=2.0,
        ... )
        >>>
        >>> # Create trainer
        >>> trainer = DistillationTrainer(
        ...     student_model="HuggingFaceTB/SmolLM-135M",
        ...     distillation_config=config,
        ... )
        >>>
        >>> # Distill
        >>> trainer.distill(dataset)
    """

    def __init__(
        self,
        student_model: str | SLMConfig | PreTrainedModel,
        distillation_config: DistillationConfig,
        training_config: TrainingConfig | None = None,
        *,
        tokenizer: Any | None = None,
    ):
        """
        Initialize distillation trainer.

        Args:
            student_model: Student model (name, config, or model instance)
            distillation_config: Distillation configuration
            training_config: Training configuration
            tokenizer: Optional tokenizer
        """
        self.distillation_config = distillation_config
        self.training_config = training_config or TrainingConfig()

        self._student = None
        self._teacher = None
        self._tokenizer = tokenizer
        self._student_config = (
            student_model
            if isinstance(student_model, SLMConfig)
            else SLMConfig(model_name=student_model) if isinstance(student_model, str) else None
        )

        if isinstance(student_model, PreTrainedModel):
            self._student = student_model

        # Loss function
        self.loss_fn = DistillationLoss(
            temperature=distillation_config.temperature,
            alpha=distillation_config.alpha,
        )

        logger.info("DistillationTrainer initialized")
        logger.info(f"  Teacher: {distillation_config.teacher_model}")
        logger.info(f"  Temperature: {distillation_config.temperature}")
        logger.info(f"  Alpha: {distillation_config.alpha}")

    @property
    def student(self) -> PreTrainedModel:
        """Get student model, loading if necessary."""
        if self._student is None:
            self._load_student()
        return self._student

    @property
    def teacher(self) -> PreTrainedModel:
        """Get teacher model, loading if necessary."""
        if self._teacher is None:
            self._load_teacher()
        return self._teacher

    @property
    def tokenizer(self):
        """Get tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.distillation_config.teacher_model)
        return self._tokenizer

    def _load_student(self) -> None:
        """Load student model."""
        logger.info("Loading student model...")
        self._student, self._tokenizer = load_model(
            self._student_config,
            for_training=True,
            use_unsloth=True,
        )
        logger.info(f"Student model loaded: {get_memory_stats()}")

    def _load_teacher(self) -> None:
        """Load teacher model in inference mode."""
        logger.info("Loading teacher model...")

        # Create config for teacher
        teacher_config = SLMConfig(
            model_name=self.distillation_config.teacher_model,
            dtype=self.distillation_config.teacher_dtype,
            load_in_4bit=self.distillation_config.teacher_load_in_4bit,
            use_gradient_checkpointing=False,  # Don't need for inference
        )

        self._teacher, _ = load_model(
            teacher_config,
            for_training=False,
            use_unsloth=False,  # Just for inference
        )

        # Set to eval mode
        self._teacher.eval()

        logger.info(f"Teacher model loaded: {get_memory_stats()}")

    def distill(
        self,
        dataset: Any,
        *,
        output_dir: str | None = None,
        text_field: str = "text",
    ) -> dict[str, float]:
        """
        Run knowledge distillation.

        Args:
            dataset: Training dataset
            output_dir: Output directory
            text_field: Field containing text

        Returns:
            Training metrics
        """
        from tqdm import tqdm

        from slmflow.core.models import prepare_model_for_training
        from slmflow.training.data import prepare_dataset

        output_dir = output_dir or self.training_config.output_dir

        # Load models
        student = self.student
        teacher = self.teacher
        tokenizer = self.tokenizer

        # Prepare student for training
        student = prepare_model_for_training(
            student,
            self.training_config,
            use_unsloth=True,
        )

        # Prepare dataset
        train_dataset = prepare_dataset(
            dataset,
            tokenizer,
            text_field=text_field,
            max_seq_length=self._student_config.max_seq_length,
        )

        logger.info(f"Starting distillation on {len(train_dataset)} samples")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Training loop
        student.train()
        total_loss = 0.0
        total_steps = min(self.training_config.max_steps, len(train_dataset))

        progress_bar = tqdm(range(total_steps), desc="Distilling")

        for step in progress_bar:
            # Get batch
            example = train_dataset[step % len(train_dataset)]

            # Tokenize
            inputs = tokenizer(
                example[text_field],
                return_tensors="pt",
                truncation=True,
                max_length=self._student_config.max_seq_length,
                padding="max_length",
            ).to(student.device)

            # Forward pass - student
            student_outputs = student(**inputs)
            student_logits = student_outputs.logits

            # Forward pass - teacher (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(**inputs.to(teacher.device))
                teacher_logits = teacher_outputs.logits.to(student.device)

            # Compute loss
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            losses = self.loss_fn(student_logits, teacher_logits, labels)
            loss = losses["loss"]

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # Update progress
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{losses['ce_loss'].item():.4f}",
                    "kl": f"{losses['kl_loss'].item():.4f}",
                }
            )

            # Logging
            if (step + 1) % self.training_config.logging_steps == 0:
                logger.info(
                    f"Step {step + 1}: loss={loss.item():.4f}, "
                    f"ce={losses['ce_loss'].item():.4f}, "
                    f"kl={losses['kl_loss'].item():.4f}"
                )

        # Save model
        save_model(student, tokenizer, output_dir)

        avg_loss = total_loss / total_steps
        logger.info(f"Distillation complete! Average loss: {avg_loss:.4f}")

        return {
            "loss": avg_loss,
            "total_steps": total_steps,
        }

    def generate_teacher_labels(
        self,
        dataset: Any,
        text_field: str = "text",
        batch_size: int = 4,
    ) -> Any:
        """
        Pre-generate teacher logits/labels for offline distillation.

        Useful when teacher is too large to keep in memory during training.

        Args:
            dataset: Dataset to generate labels for
            text_field: Field containing text
            batch_size: Batch size for generation

        Returns:
            Dataset with teacher logits added
        """
        teacher = self.teacher
        tokenizer = self.tokenizer

        logger.info("Generating teacher labels...")

        def add_teacher_logits(examples):
            inputs = tokenizer(
                examples[text_field],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            ).to(teacher.device)

            with torch.no_grad():
                outputs = teacher(**inputs)
                # Store top-k logits to save memory
                topk_values, topk_indices = outputs.logits.topk(50, dim=-1)

            return {
                "teacher_logits": topk_values.cpu().numpy().tolist(),
                "teacher_indices": topk_indices.cpu().numpy().tolist(),
            }

        dataset = dataset.map(
            add_teacher_logits,
            batched=True,
            batch_size=batch_size,
            desc="Generating teacher labels",
        )

        # Free teacher memory
        del self._teacher
        self._teacher = None
        clear_memory()

        return dataset
