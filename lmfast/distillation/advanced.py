"""
Advanced Distillation Techniques for LMFast.

Implements SOTA distillation methods from 2024-2025 research:
- TAID: Temporally Adaptive Interpolated Distillation
- GKD: Generalized Knowledge Distillation (on-policy)
- CoT Distillation: Chain-of-Thought transfer
- Agent Distillation: Full behavior transfer

References:
- TAID: "Temporally Adaptive Interpolated Distillation" (ICLR 2025)
- GKD: "On-Policy Distillation of Language Models" (EMNLP 2024)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class AdvancedDistillationConfig:
    """Configuration for advanced distillation methods."""

    # Models
    teacher_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_model: str = "HuggingFaceTB/SmolLM-135M"

    # Method
    method: Literal["kd", "gkd", "taid", "cot", "agent"] = "gkd"

    # Temperature
    temperature: float = 2.0

    # Loss mixing
    alpha: float = 0.5  # Weight for distillation loss vs CE loss

    # TAID-specific
    taid_warmup_steps: int = 100
    taid_total_steps: int = 1000

    # GKD-specific
    gkd_sample_temperature: float = 1.0
    gkd_num_samples: int = 1

    # CoT-specific
    cot_max_reasoning_tokens: int = 256

    # Training
    max_seq_length: int = 512
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 500


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.

    L = (1 - α) * L_CE + α * L_KL

    where:
    - L_CE: Cross-entropy with ground truth
    - L_KL: KL divergence with teacher
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, use_reverse_kl: bool = False):
        """
        Args:
            temperature: Softmax temperature for softer distributions
            alpha: Weight for distillation loss (0 = pure CE, 1 = pure KL)
            use_reverse_kl: Use reverse KL (student || teacher) for mode-seeking
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.use_reverse_kl = use_reverse_kl

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: [batch, seq, vocab]
            teacher_logits: [batch, seq, vocab]
            labels: [batch, seq] ground truth tokens
        """
        # Soften distributions with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        if self.use_reverse_kl:
            # Reverse KL: mode-seeking, better for generation
            kl_loss = F.kl_div(
                F.log_softmax(teacher_logits / self.temperature, dim=-1),
                F.softmax(student_logits / self.temperature, dim=-1),
                reduction="batchmean",
            )
        else:
            # Forward KL: mean-seeking, standard
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")

        # Scale by T^2 as per Hinton et al.
        kl_loss = kl_loss * (self.temperature**2)

        if labels is not None and self.alpha < 1.0:
            # Cross-entropy with ground truth
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        else:
            total_loss = kl_loss

        return total_loss


class TAIDScheduler:
    """
    Temporally Adaptive Interpolated Distillation scheduler.

    Gradually increases distillation weight over training to bridge
    large capacity gaps between teacher and student.

    λ(t) = sigmoid((t - T/2) / τ)

    where T is total steps and τ controls transition sharpness.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_lambda: float = 0.1,
        max_lambda: float = 0.9,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

    def get_lambda(self, step: int) -> float:
        """Get interpolation weight at given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.min_lambda * (step / self.warmup_steps)

        # Sigmoid schedule after warmup
        normalized_step = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        sigmoid_val = 1 / (1 + math.exp(-10 * (normalized_step - 0.5)))

        return self.min_lambda + (self.max_lambda - self.min_lambda) * sigmoid_val


import math


class AdvancedDistillationTrainer:
    """
    Advanced distillation trainer supporting multiple methods.

    Methods:
    - kd: Standard knowledge distillation (KL on logits)
    - gkd: Generalized KD (on-policy, student's own generations)
    - taid: Temporally Adaptive Interpolated Distillation
    - cot: Chain-of-Thought distillation
    - agent: Full agent behavior transfer

    Example:
        >>> trainer = AdvancedDistillationTrainer(
        ...     student_model="HuggingFaceTB/SmolLM-135M",
        ...     teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
        ...     method="taid"
        ... )
        >>> trainer.distill(dataset, output_dir="./distilled")
    """

    def __init__(
        self,
        student_model: str,
        teacher_model: str,
        config: AdvancedDistillationConfig | None = None,
        method: str | None = None,
        device: str = "auto",
    ):
        """
        Initialize trainer.

        Args:
            student_model: Student model path or HF ID
            teacher_model: Teacher model path or HF ID
            config: Distillation configuration
            method: Override config method
            device: Device to use
        """
        self.config = config or AdvancedDistillationConfig(
            student_model=student_model, teacher_model=teacher_model
        )

        if method:
            self.config.method = method

        self.device = self._get_device(device)
        self.student = None
        self.teacher = None
        self.tokenizer = None

        logger.info(f"AdvancedDistillationTrainer initialized with method: {self.config.method}")

    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_models(self):
        """Load teacher and student models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading teacher: {self.config.teacher_model}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model, torch_dtype=torch.float16, device_map="auto"
        )
        self.teacher.eval()

        logger.info(f"Loading student: {self.config.student_model}")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.config.student_model, torch_dtype=torch.float16, device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.student_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def distill(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """
        Run distillation.

        Args:
            dataset: Training dataset
            output_dir: Directory to save distilled model
            **kwargs: Additional training arguments

        Returns:
            Training metrics
        """
        # Load models if not already loaded
        if self.student is None:
            self._load_models()

        # Dispatch to appropriate method
        method_map = {
            "kd": self._distill_standard,
            "gkd": self._distill_gkd,
            "taid": self._distill_taid,
            "cot": self._distill_cot,
            "agent": self._distill_agent,
        }

        distill_fn = method_map.get(self.config.method)
        if distill_fn is None:
            raise ValueError(f"Unknown method: {self.config.method}")

        return distill_fn(dataset, output_dir, **kwargs)

    def _distill_standard(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """Standard knowledge distillation."""
        logger.info("Running standard knowledge distillation...")


        # Create loss function
        loss_fn = DistillationLoss(temperature=self.config.temperature, alpha=self.config.alpha)

        # Optimizer
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.config.learning_rate)

        # Training loop
        self.student.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(dataset, total=self.config.max_steps)):
            if step >= self.config.max_steps:
                break

            # Tokenize
            inputs = self.tokenizer(
                batch["text"] if "text" in batch else batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)

            # Get teacher logits
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
                teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = self.student(**inputs)
            student_logits = student_outputs.logits

            # Compute loss
            labels = inputs.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            loss = loss_fn(student_logits, teacher_logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        avg_loss = total_loss / min(step + 1, self.config.max_steps)
        logger.info(f"Distillation complete. Avg loss: {avg_loss:.4f}")

        return {"loss": avg_loss, "method": "kd"}

    def _distill_taid(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """
        TAID: Temporally Adaptive Interpolated Distillation.

        Gradually increases distillation weight to bridge large capacity gaps.
        """
        logger.info("Running TAID distillation...")

        # Create scheduler
        scheduler = TAIDScheduler(
            warmup_steps=self.config.taid_warmup_steps, total_steps=self.config.taid_total_steps
        )

        # Modified loss with adaptive alpha
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.config.learning_rate)

        self.student.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(dataset, total=self.config.max_steps)):
            if step >= self.config.max_steps:
                break

            # Get adaptive alpha
            alpha = scheduler.get_lambda(step)

            # Create loss with current alpha
            loss_fn = DistillationLoss(temperature=self.config.temperature, alpha=alpha)

            # Tokenize
            inputs = self.tokenizer(
                batch["text"] if "text" in batch else batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)

            # Get logits
            with torch.no_grad():
                teacher_logits = self.teacher(**inputs).logits

            student_logits = self.student(**inputs).logits

            # Compute loss
            labels = inputs.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            loss = loss_fn(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return {"loss": total_loss / self.config.max_steps, "method": "taid"}

    def _distill_gkd(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """
        GKD: Generalized Knowledge Distillation.

        Uses on-policy distillation where student generates,
        then learns from teacher's distribution on its own outputs.
        This addresses exposure bias in autoregressive distillation.
        """
        logger.info("Running GKD (on-policy) distillation...")

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.config.learning_rate)

        self.student.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(dataset, total=self.config.max_steps)):
            if step >= self.config.max_steps:
                break

            # Get prompt
            prompt = batch["prompt"] if "prompt" in batch else batch["text"]

            # Student generates (on-policy)
            prompt_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length // 2,
            ).to(self.device)

            with torch.no_grad():
                student_gen = self.student.generate(
                    **prompt_tokens,
                    max_new_tokens=self.config.max_seq_length // 2,
                    temperature=self.config.gkd_sample_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Get teacher logits on student generation
            with torch.no_grad():
                teacher_logits = self.teacher(student_gen).logits

            # Train student to match teacher on its own generation
            student_outputs = self.student(student_gen)
            student_logits = student_outputs.logits

            # KL loss
            loss_fn = DistillationLoss(
                temperature=self.config.temperature, alpha=1.0  # Pure distillation for GKD
            )

            loss = loss_fn(student_logits, teacher_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return {"loss": total_loss / self.config.max_steps, "method": "gkd"}

    def _distill_cot(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """
        Chain-of-Thought Distillation.

        Teacher generates reasoning traces, student learns to reproduce them.
        """
        logger.info("Running CoT distillation...")

        # First, generate CoT traces from teacher
        cot_data = []

        for batch in tqdm(dataset, desc="Generating CoT traces"):
            prompt = batch["prompt"] if "prompt" in batch else batch["text"]

            # Augment prompt with CoT instruction
            cot_prompt = f"{prompt}\n\nLet's think step by step:"

            inputs = self.tokenizer(
                cot_prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.teacher.generate(
                    **inputs,
                    max_new_tokens=self.config.cot_max_reasoning_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            cot_trace = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cot_data.append({"prompt": prompt, "cot_response": cot_trace})

            if len(cot_data) >= 100:  # Limit for demo
                break

        # Now train student on CoT traces
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.config.learning_rate)

        self.student.train()
        total_loss = 0

        for step, item in enumerate(tqdm(cot_data)):
            if step >= self.config.max_steps:
                break

            # Train on full CoT trace
            full_text = f"{item['prompt']}\n\nLet's think step by step:{item['cot_response']}"

            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)

            labels = inputs.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.student(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return {"loss": total_loss / min(step + 1, self.config.max_steps), "method": "cot"}

    def _distill_agent(self, dataset: Any, output_dir: str, **kwargs) -> dict[str, Any]:
        """
        Agent Distillation.

        Transfers full agent behavior including:
        - Tool selection
        - Reasoning patterns
        - Multi-turn interaction
        """
        logger.info("Running agent distillation...")
        logger.warning("Agent distillation requires tool-augmented dataset. Using CoT fallback.")

        # For now, fall back to CoT distillation
        return self._distill_cot(dataset, output_dir, **kwargs)


def distill_advanced(
    student_model: str,
    teacher_model: str,
    dataset: Any,
    output_dir: str,
    method: str = "gkd",
    **kwargs,
) -> dict[str, Any]:
    """
    One-line advanced distillation.

    Args:
        student_model: Student model path
        teacher_model: Teacher model path
        dataset: Training data
        output_dir: Output directory
        method: Distillation method (kd, gkd, taid, cot, agent)

    Returns:
        Training metrics

    Example:
        >>> from lmfast.distillation import distill_advanced
        >>> distill_advanced(
        ...     student_model="smolLM",
        ...     teacher_model="qwen",
        ...     dataset=my_data,
        ...     output_dir="./distilled",
        ...     method="taid"
        ... )
    """
    config = AdvancedDistillationConfig(
        student_model=student_model,
        teacher_model=teacher_model,
        method=method,
        **{k: v for k, v in kwargs.items() if hasattr(AdvancedDistillationConfig, k)},
    )

    trainer = AdvancedDistillationTrainer(
        student_model=student_model, teacher_model=teacher_model, config=config
    )

    return trainer.distill(dataset, output_dir)
