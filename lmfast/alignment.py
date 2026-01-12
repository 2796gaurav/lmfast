
import logging
from typing import Any, Optional
from lmfast.core.config import SLMConfig, TrainingConfig
from lmfast.core.models import load_model, load_tokenizer

logger = logging.getLogger(__name__)

class PreferenceTrainer:
    """
    Unified trainer for Preference Optimization (ORPO, DPO, KTO).
    Defaults to ORPO (2025 Standard) for best performance on small models.
    """
    def __init__(
        self,
        model_name: str,
        config: TrainingConfig,
        method: str = "orpo"
    ):
        self.model_name = model_name
        self.config = config
        self.method = method
        self.trainer = None

    def train(self, dataset: Any, output_dir: str):
        """Train alignment."""
        logger.info(f"Starting {self.method.upper()} alignment...")
        
        # Load Model & Tokenizer
        # For ORPO we mainly rely on TRL
        try:
            from trl import ORPOConfig, ORPOTrainer
        except ImportError:
            raise ImportError("Please install trl>=0.8.0")

        # Basic Check
        if self.method.lower() != "orpo":
            raise NotImplementedError("Currently only ORPO (Odds Ratio Preference Optimization) is supported as it is SOTA for SLMs.")

        # Config Setup
        orpo_config = ORPOConfig(
            output_dir=output_dir,
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            optim="paged_adamw_8bit", # Optim for Colab T4
            logging_steps=10
        )
        
        # Load model with optimizations
        # Note: We rely on the internal loading of ORPOTrainer or pass the loaded model
        # Passing model_init or model directly is preferred
        
        self.trainer = ORPOTrainer(
            model=self.model_name, # TRL handles loading
            args=orpo_config,
            train_dataset=dataset,
            tokenizer=load_tokenizer(self.model_name),
        )
        
        self.trainer.train()
        self.trainer.save_model(output_dir)
        logger.info(f"Alignment complete! Model saved to {output_dir}")

def align(
    model: str,
    dataset: Any,
    output_dir: str = "./aligned_model",
    method: str = "orpo",
    **kwargs
) -> PreferenceTrainer:
    """
    Align a model to preferences.
    """
    config = TrainingConfig(**kwargs)
    trainer = PreferenceTrainer(model, config, method)
    trainer.train(dataset, output_dir)
    return trainer
