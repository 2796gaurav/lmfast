"""
Self-Distillation for SLMs.

A model learns from its own outputs to improve performance.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from slmflow.core.config import SLMConfig, TrainingConfig
from slmflow.core.models import load_model, prepare_model_for_training, save_model

logger = logging.getLogger(__name__)


class SelfDistillationTrainer:
    """
    Self-distillation trainer.

    The model learns from its own previous outputs, which can help
    improve consistency and reduce hallucinations.

    Example:
        >>> trainer = SelfDistillationTrainer(
        ...     model_config=SLMConfig(model_name="HuggingFaceTB/SmolLM-135M"),
        ...     num_iterations=3,
        ... )
        >>> trainer.distill(dataset)
    """

    def __init__(
        self,
        model_config: SLMConfig,
        training_config: TrainingConfig | None = None,
        *,
        num_iterations: int = 3,
        temperature: float = 1.5,
    ):
        """
        Initialize self-distillation trainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
            num_iterations: Number of self-distillation iterations
            temperature: Temperature for soft labels
        """
        self.model_config = model_config
        self.training_config = training_config or TrainingConfig()
        self.num_iterations = num_iterations
        self.temperature = temperature

        self._model = None
        self._tokenizer = None

        logger.info("SelfDistillationTrainer initialized")
        logger.info(f"  Model: {model_config.model_name}")
        logger.info(f"  Iterations: {num_iterations}")

    @property
    def model(self):
        """Get model, loading if necessary."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer."""
        return self._tokenizer

    def _load_model(self) -> None:
        """Load model."""
        self._model, self._tokenizer = load_model(
            self.model_config,
            for_training=True,
            use_unsloth=True,
        )

    def distill(
        self,
        dataset: Any,
        *,
        output_dir: str | None = None,
        text_field: str = "text",
    ) -> list[dict[str, float]]:
        """
        Run iterative self-distillation.

        Args:
            dataset: Training dataset
            output_dir: Output directory
            text_field: Text field name

        Returns:
            List of metrics per iteration
        """

        output_dir = output_dir or self.training_config.output_dir
        all_metrics = []

        for iteration in range(self.num_iterations):
            logger.info(f"Self-distillation iteration {iteration + 1}/{self.num_iterations}")

            # Get current model predictions
            current_model = self.model

            # Generate soft labels from current model
            logger.info("Generating soft labels from current model...")
            soft_labels = self._generate_soft_labels(
                dataset,
                text_field=text_field,
            )

            # Prepare for training
            current_model = prepare_model_for_training(
                current_model,
                self.training_config,
            )

            # Train on soft labels
            metrics = self._train_iteration(
                dataset,
                soft_labels,
                text_field=text_field,
            )
            metrics["iteration"] = iteration + 1
            all_metrics.append(metrics)

            logger.info(f"Iteration {iteration + 1} complete: {metrics}")

        # Save final model
        save_model(self._model, self._tokenizer, output_dir)

        return all_metrics

    def _generate_soft_labels(
        self,
        dataset: Any,
        text_field: str,
    ) -> list[torch.Tensor]:
        """Generate soft labels from current model."""
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        soft_labels = []

        with torch.no_grad():
            for example in tqdm(dataset, desc="Generating soft labels"):
                inputs = tokenizer(
                    example[text_field],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.model_config.max_seq_length,
                ).to(model.device)

                outputs = model(**inputs)
                probs = F.softmax(outputs.logits / self.temperature, dim=-1)
                soft_labels.append(probs.cpu())

        model.train()
        return soft_labels

    def _train_iteration(
        self,
        dataset: Any,
        soft_labels: list[torch.Tensor],
        text_field: str,
    ) -> dict[str, float]:
        """Train one iteration of self-distillation."""
        model = self.model
        tokenizer = self.tokenizer

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
        )

        total_loss = 0.0
        steps = min(len(dataset), self.training_config.max_steps)

        for step in tqdm(range(steps), desc="Training"):
            example = dataset[step]
            soft_target = soft_labels[step].to(model.device)

            inputs = tokenizer(
                example[text_field],
                return_tensors="pt",
                truncation=True,
                max_length=self.model_config.max_seq_length,
            ).to(model.device)

            outputs = model(**inputs)
            log_probs = F.log_softmax(outputs.logits / self.temperature, dim=-1)

            # KL divergence loss
            loss = F.kl_div(log_probs, soft_target, reduction="batchmean")
            loss = loss * (self.temperature**2)

            loss.backward()

            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        return {"loss": total_loss / steps}


# Import for tqdm
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable
