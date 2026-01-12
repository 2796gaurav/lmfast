"""
Output filtering and sanitization.

Filters and sanitizes model outputs before returning to users.
"""

import logging
import re

from slmflow.guardrails.config import GuardrailsConfig

logger = logging.getLogger(__name__)


class OutputFilterResult:
    """Result of output filtering."""

    def __init__(
        self,
        is_safe: bool,
        filtered_output: str | None = None,
        violations: list[str] | None = None,
        scores: dict[str, float] | None = None,
    ):
        self.is_safe = is_safe
        self.filtered_output = filtered_output
        self.violations = violations or []
        self.scores = scores or {}

    def __bool__(self) -> bool:
        return self.is_safe


class OutputFilter:
    """
    Filters and sanitizes model outputs.

    Detects and handles:
    - Toxic content
    - PII in outputs
    - Blocked phrases
    - Content policy violations

    Example:
        >>> filter = OutputFilter(config)
        >>> result = filter.filter("Model's response here...")
        >>> if result.is_safe:
        ...     return result.filtered_output
        >>> else:
        ...     return "I cannot provide that response."
    """

    def __init__(self, config: GuardrailsConfig | None = None):
        """
        Initialize output filter.

        Args:
            config: Guardrails configuration
        """
        self.config = config or GuardrailsConfig()
        self._toxicity_model = None
        self._pii_analyzer = None
        self._anonymizer = None

    @property
    def toxicity_model(self):
        """Lazy load toxicity detection model."""
        if self._toxicity_model is None and self.config.enable_toxicity_filter:
            try:
                from detoxify import Detoxify

                self._toxicity_model = Detoxify("original")
                logger.info("Toxicity model initialized")
            except ImportError:
                logger.warning(
                    "detoxify not installed. " "Install with: pip install slmflow[guardrails]"
                )
        return self._toxicity_model

    @property
    def pii_analyzer(self):
        """Lazy load PII analyzer."""
        if self._pii_analyzer is None and self.config.enable_pii_detection:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._pii_analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
                logger.info("PII analyzer and anonymizer initialized")
            except ImportError:
                logger.warning(
                    "presidio not installed. " "Install with: pip install slmflow[guardrails]"
                )
        return self._pii_analyzer

    def filter(
        self,
        text: str,
        *,
        block_on_violation: bool = False,
        fallback_response: str = "I cannot provide that response.",
    ) -> OutputFilterResult:
        """
        Filter and sanitize output text.

        Args:
            text: Output text to filter
            block_on_violation: If True, replace entire output on violation
            fallback_response: Response to use when blocking

        Returns:
            OutputFilterResult with filtering status and sanitized output
        """
        violations = []
        scores = {}
        filtered = text

        # Check length
        estimated_tokens = len(text) / 4
        if estimated_tokens > self.config.max_output_tokens:
            violations.append("output_too_long")
            # Truncate
            max_chars = self.config.max_output_tokens * 4
            filtered = filtered[:max_chars] + "..."

        # Check toxicity
        if self.config.enable_toxicity_filter:
            toxicity_result = self._check_toxicity(text)
            scores.update(toxicity_result["scores"])

            if toxicity_result["is_toxic"]:
                violations.extend(toxicity_result["violations"])
                if block_on_violation:
                    return OutputFilterResult(
                        is_safe=False,
                        filtered_output=fallback_response,
                        violations=violations,
                        scores=scores,
                    )

        # Check blocked phrases
        for phrase in self.config.blocked_phrases:
            if phrase.lower() in text.lower():
                violations.append(f"blocked_phrase: {phrase}")
                # Remove the phrase
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                filtered = pattern.sub("[REDACTED]", filtered)

        # Check and redact PII
        if self.config.enable_pii_detection:
            pii_result = self._check_and_redact_pii(filtered)
            if pii_result["found"]:
                violations.extend([f"pii: {e}" for e in pii_result["entities"]])
                filtered = pii_result["anonymized"]

        # Log violations
        if violations and self.config.log_violations:
            logger.warning(f"Output violations detected: {violations}")

        return OutputFilterResult(
            is_safe=len(violations) == 0,
            filtered_output=filtered,
            violations=violations,
            scores=scores,
        )

    def _check_toxicity(self, text: str) -> dict:
        """Check text for toxic content."""
        if self.toxicity_model is None:
            return {"is_toxic": False, "scores": {}, "violations": []}

        try:
            results = self.toxicity_model.predict(text)

            scores = {k: float(v) for k, v in results.items()}
            violations = []

            for category in self.config.toxicity_categories:
                if category in scores:
                    if scores[category] > self.config.toxicity_threshold:
                        violations.append(f"toxicity:{category}={scores[category]:.2f}")

            return {
                "is_toxic": len(violations) > 0,
                "scores": scores,
                "violations": violations,
            }

        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return {"is_toxic": False, "scores": {}, "violations": []}

    def _check_and_redact_pii(self, text: str) -> dict:
        """Check and redact PII from text."""
        if self.pii_analyzer is None:
            return {"found": False, "entities": [], "anonymized": text}

        try:
            # Analyze
            results = self.pii_analyzer.analyze(
                text=text,
                entities=self.config.pii_entities,
                language="en",
            )

            if not results:
                return {"found": False, "entities": [], "anonymized": text}

            # Anonymize
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
            )

            entities = list({r.entity_type for r in results})
            return {
                "found": True,
                "entities": entities,
                "anonymized": anonymized.text,
            }

        except Exception as e:
            logger.error(f"PII redaction failed: {e}")
            return {"found": False, "entities": [], "anonymized": text}

    def is_safe(self, text: str) -> bool:
        """Quick check if output is safe."""
        result = self.filter(text)
        return result.is_safe


class GuardedModel:
    """
    Wrapper that applies guardrails to a model.

    Example:
        >>> guarded = GuardedModel(model, tokenizer, config)
        >>> response = guarded.generate("Hello!")
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: GuardrailsConfig | None = None,
    ):
        """
        Initialize guarded model.

        Args:
            model: The language model
            tokenizer: The tokenizer
            config: Guardrails configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GuardrailsConfig()

        from slmflow.guardrails.input_validator import InputValidator

        self.input_validator = InputValidator(self.config)
        self.output_filter = OutputFilter(self.config)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Generate text with guardrails applied.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation kwargs

        Returns:
            Filtered response
        """
        import torch

        # Validate input
        input_result = self.input_validator.validate(prompt)
        if not input_result.is_valid:
            return "I cannot process that request."

        sanitized_prompt = input_result.sanitized_input

        # Generate
        inputs = self.tokenizer(
            sanitized_prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Filter output
        output_result = self.output_filter.filter(response)

        return output_result.filtered_output
