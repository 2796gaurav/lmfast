"""
Unit tests for core configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestSLMConfig:
    """Tests for SLMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from lmfast.core.config import SLMConfig

        config = SLMConfig()

        assert config.model_name == "HuggingFaceTB/SmolLM-135M"
        assert config.max_seq_length == 2048
        assert config.load_in_4bit is True
        assert config.use_gradient_checkpointing is True

    def test_custom_config(self):
        """Test custom configuration."""
        from lmfast.core.config import SLMConfig

        config = SLMConfig(
            model_name="custom/model",
            max_seq_length=4096,
            load_in_4bit=False,
        )

        assert config.model_name == "custom/model"
        assert config.max_seq_length == 4096
        assert config.load_in_4bit is False

    def test_quantization_validation(self):
        """Test that both 4-bit and 8-bit can't be enabled."""
        from lmfast.core.config import SLMConfig

        with pytest.raises(ValidationError):
            SLMConfig(load_in_4bit=True, load_in_8bit=True)

    def test_is_colab_t4_compatible(self):
        """Test T4 compatibility check."""
        from lmfast.core.config import SLMConfig

        # Should be compatible
        config = SLMConfig(
            load_in_4bit=True,
            use_gradient_checkpointing=True,
            max_seq_length=2048,
        )
        assert config.is_colab_t4_compatible() is True

        # Not compatible (no 4-bit)
        config = SLMConfig(
            load_in_4bit=False,
            use_gradient_checkpointing=True,
        )
        assert config.is_colab_t4_compatible() is False

    def test_model_size_estimate(self):
        """Test model size estimation."""
        from lmfast.core.config import SLMConfig

        config = SLMConfig(model_name="model-135m")
        assert "135M" in config.model_size_estimate

        config = SLMConfig(model_name="model-1b")
        assert "1B" in config.model_size_estimate


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default training configuration."""
        from lmfast.core.config import TrainingConfig

        config = TrainingConfig()

        assert config.max_steps == 500
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.use_lora is True

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        from lmfast.core.config import TrainingConfig

        config = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16

    def test_estimated_training_time(self):
        """Test training time estimation."""
        from lmfast.core.config import TrainingConfig

        config = TrainingConfig(max_steps=500)
        # ~2 seconds per step, so 500 steps ≈ 16-17 minutes
        assert config.estimated_training_time_minutes > 10
        assert config.estimated_training_time_minutes < 30

    def test_sft_config_kwargs(self):
        """Test conversion to SFT config kwargs."""
        from lmfast.core.config import TrainingConfig

        config = TrainingConfig(
            output_dir="./test",
            max_steps=100,
            batch_size=2,
        )

        kwargs = config.to_sft_config_kwargs()

        assert kwargs["output_dir"] == "./test"
        assert kwargs["max_steps"] == 100
        assert kwargs["per_device_train_batch_size"] == 2

    def test_lora_config_kwargs(self):
        """Test conversion to LoRA config kwargs."""
        from lmfast.core.config import TrainingConfig

        config = TrainingConfig(
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )

        kwargs = config.to_lora_config_kwargs()

        assert kwargs["r"] == 32
        assert kwargs["lora_alpha"] == 64
        assert kwargs["lora_dropout"] == 0.1


class TestDistillationConfig:
    """Tests for DistillationConfig."""

    def test_default_values(self):
        """Test default distillation config."""
        from lmfast.core.config import DistillationConfig

        config = DistillationConfig(teacher_model="test/teacher")

        assert config.teacher_model == "test/teacher"
        assert config.temperature == 2.0
        assert config.alpha == 0.5

    def test_temperature_bounds(self):
        """Test temperature validation."""
        from lmfast.core.config import DistillationConfig

        # Valid
        config = DistillationConfig(
            teacher_model="test",
            temperature=5.0,
        )
        assert config.temperature == 5.0

        # Invalid (too low)
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="test", temperature=0.5)

        # Invalid (too high)
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="test", temperature=15.0)


class TestInferenceConfig:
    """Tests for InferenceConfig."""

    def test_generation_kwargs(self):
        """Test conversion to generation kwargs."""
        from lmfast.core.config import InferenceConfig

        config = InferenceConfig(
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
        )

        kwargs = config.to_generation_kwargs()

        assert kwargs["max_new_tokens"] == 128
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.95
        assert kwargs["do_sample"] is True

    def test_greedy_sampling(self):
        """Test that do_sample is False for temperature=0."""
        from lmfast.core.config import InferenceConfig

        config = InferenceConfig(temperature=0.0)
        kwargs = config.to_generation_kwargs()

        assert kwargs["do_sample"] is False
