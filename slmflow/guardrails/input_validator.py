"""
Input validation and sanitization.

Validates and sanitizes user inputs before processing.
"""

import logging
import re

from slmflow.guardrails.config import GuardrailsConfig

logger = logging.getLogger(__name__)


class InputValidationResult:
    """Result of input validation."""

    def __init__(
        self,
        is_valid: bool,
        sanitized_input: str | None = None,
        violations: list[str] | None = None,
        risk_score: float = 0.0,
    ):
        self.is_valid = is_valid
        self.sanitized_input = sanitized_input
        self.violations = violations or []
        self.risk_score = risk_score

    def __bool__(self) -> bool:
        return self.is_valid


class InputValidator:
    """
    Validates and sanitizes input prompts.

    Detects:
    - Prompt injection attempts
    - PII in inputs
    - Blocked phrases/topics
    - Length violations

    Example:
        >>> validator = InputValidator(config)
        >>> result = validator.validate("Tell me about AI")
        >>> if result.is_valid:
        ...     process(result.sanitized_input)
    """

    def __init__(self, config: GuardrailsConfig | None = None):
        """
        Initialize input validator.

        Args:
            config: Guardrails configuration
        """
        self.config = config or GuardrailsConfig()
        self._pii_analyzer = None

        # Compile injection patterns for efficiency
        self._injection_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.injection_patterns
        ]

    @property
    def pii_analyzer(self):
        """Lazy load PII analyzer."""
        if self._pii_analyzer is None and self.config.enable_pii_detection:
            try:
                from presidio_analyzer import AnalyzerEngine

                self._pii_analyzer = AnalyzerEngine()
                logger.info("PII analyzer initialized")
            except ImportError:
                logger.warning(
                    "presidio-analyzer not installed. "
                    "Install with: pip install slmflow[guardrails]"
                )
        return self._pii_analyzer

    def validate(
        self,
        text: str,
        *,
        strict: bool = False,
    ) -> InputValidationResult:
        """
        Validate and optionally sanitize input text.

        Args:
            text: Input text to validate
            strict: If True, block on any violation. If False, sanitize and continue.

        Returns:
            InputValidationResult with validation status and sanitized input
        """
        violations = []
        risk_score = 0.0
        sanitized = text

        # Check length
        estimated_tokens = len(text) / 4  # Rough estimate
        if estimated_tokens > self.config.max_input_tokens:
            violations.append("input_too_long")
            risk_score += 0.3
            if strict:
                return InputValidationResult(
                    is_valid=False,
                    violations=violations,
                    risk_score=risk_score,
                )
            # Truncate
            max_chars = self.config.max_input_tokens * 4
            sanitized = sanitized[:max_chars]

        # Check prompt injection
        if self.config.enable_prompt_injection:
            injection_result = self._check_prompt_injection(text)
            if injection_result:
                violations.append(f"prompt_injection: {injection_result}")
                risk_score += 0.5
                if strict:
                    return InputValidationResult(
                        is_valid=False,
                        violations=violations,
                        risk_score=risk_score,
                    )

        # Check blocked phrases
        for phrase in self.config.blocked_phrases:
            if phrase.lower() in text.lower():
                violations.append(f"blocked_phrase: {phrase}")
                risk_score += 0.2
                if strict:
                    return InputValidationResult(
                        is_valid=False,
                        violations=violations,
                        risk_score=risk_score,
                    )

        # Check PII
        if self.config.enable_pii_detection:
            pii_result = self._check_pii(text)
            if pii_result["found"]:
                violations.extend([f"pii: {e}" for e in pii_result["entities"]])
                risk_score += 0.3

                if self.config.pii_action == "block" and strict:
                    return InputValidationResult(
                        is_valid=False,
                        violations=violations,
                        risk_score=risk_score,
                    )
                elif self.config.pii_action in ("redact", "mask"):
                    sanitized = self._redact_pii(sanitized, pii_result["results"])

        # Log violations if configured
        if violations and self.config.log_violations:
            logger.warning(f"Input violations detected: {violations}")

        return InputValidationResult(
            is_valid=True,  # Passed validation (possibly after sanitization)
            sanitized_input=sanitized,
            violations=violations,
            risk_score=min(risk_score, 1.0),
        )

    def _check_prompt_injection(self, text: str) -> str | None:
        """Check for prompt injection patterns."""
        text_lower = text.lower()

        for pattern in self._injection_patterns:
            if pattern.search(text_lower):
                return pattern.pattern

        return None

    def _check_pii(self, text: str) -> dict:
        """Check for PII in text."""
        if self.pii_analyzer is None:
            return {"found": False, "entities": [], "results": []}

        try:
            results = self.pii_analyzer.analyze(
                text=text,
                entities=self.config.pii_entities,
                language="en",
            )

            entities = list({r.entity_type for r in results})
            return {
                "found": len(results) > 0,
                "entities": entities,
                "results": results,
            }
        except Exception as e:
            logger.error(f"PII analysis failed: {e}")
            return {"found": False, "entities": [], "results": []}

    def _redact_pii(self, text: str, pii_results: list) -> str:
        """Redact PII from text."""
        if not pii_results:
            return text

        # Sort by position (reverse) to maintain indices
        sorted_results = sorted(pii_results, key=lambda x: x.start, reverse=True)

        result = text
        for pii in sorted_results:
            replacement = f"[{pii.entity_type}]"
            result = result[: pii.start] + replacement + result[pii.end :]

        return result

    def is_safe(self, text: str) -> bool:
        """Quick check if input is safe (no violations)."""
        result = self.validate(text, strict=True)
        return result.is_valid
