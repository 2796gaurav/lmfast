"""Distillation module initialization."""

from slmflow.distillation.self_distillation import SelfDistillationTrainer
from slmflow.distillation.teacher_student import DistillationTrainer

__all__ = [
    "DistillationTrainer",
    "SelfDistillationTrainer",
]
