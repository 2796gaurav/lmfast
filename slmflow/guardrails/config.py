"""
Guardrails configuration.

Configuration for input validation and output filtering.
"""

from pydantic import BaseModel, Field


class GuardrailsConfig(BaseModel):
    """
    Configuration for guardrails and safety features.

    Example:
        >>> config = GuardrailsConfig(
        ...     enable_pii_detection=True,
        ...     enable_toxicity_filter=True,
        ... )
    """

    # PII Detection
    enable_pii_detection: bool = Field(
        default=True, description="Enable PII (Personally Identifiable Information) detection"
    )
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "IP_ADDRESS",
        ],
        description="PII entity types to detect",
    )
    pii_action: str = Field(
        default="redact", description="Action for PII: 'redact', 'mask', or 'block'"
    )

    # Toxicity Filter
    enable_toxicity_filter: bool = Field(
        default=True, description="Enable toxicity detection and filtering"
    )
    toxicity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Toxicity score threshold for filtering"
    )
    toxicity_categories: list[str] = Field(
        default_factory=lambda: [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
        ],
        description="Toxicity categories to check",
    )

    # Prompt Injection Detection
    enable_prompt_injection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            "ignore previous instructions",
            "ignore all previous",
            "disregard your instructions",
            "you are now",
            "act as if",
            "pretend you are",
            "system prompt",
            "your instructions are",
            "forget everything",
        ],
        description="Patterns to detect prompt injection",
    )

    # Content Policy
    blocked_phrases: list[str] = Field(default_factory=list, description="Custom phrases to block")
    blocked_topics: list[str] = Field(default_factory=list, description="Topics to avoid")

    # Length Limits
    max_input_tokens: int = Field(default=4096, ge=1, description="Maximum input tokens")
    max_output_tokens: int = Field(default=2048, ge=1, description="Maximum output tokens")

    # Rate Limiting (for production)
    enable_rate_limiting: bool = Field(default=False, description="Enable rate limiting")
    requests_per_minute: int = Field(default=60, ge=1, description="Maximum requests per minute")

    # Logging
    log_violations: bool = Field(default=True, description="Log guardrail violations")
