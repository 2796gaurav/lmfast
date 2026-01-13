"""
Distillation module for LMFast.

Provides:
- DistillationTrainer: Standard teacher-student distillation
- SelfDistillationTrainer: Self-distillation with label smoothing
- AdvancedDistillationTrainer: SOTA methods (TAID, GKD, CoT, Agent)
"""

from lmfast.distillation.self_distillation import SelfDistillationTrainer
from lmfast.distillation.teacher_student import DistillationTrainer

# Try to import advanced distillation
try:
    from lmfast.distillation.advanced import (
        AdvancedDistillationConfig,
        AdvancedDistillationTrainer,
        DistillationLoss,
        TAIDScheduler,
        distill_advanced,
    )

    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

__all__ = [
    "DistillationTrainer",
    "SelfDistillationTrainer",
]

if _ADVANCED_AVAILABLE:
    __all__.extend(
        [
            "AdvancedDistillationTrainer",
            "AdvancedDistillationConfig",
            "DistillationLoss",
            "TAIDScheduler",
            "distill_advanced",
        ]
    )
