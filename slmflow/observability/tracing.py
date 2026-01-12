"""
Tracing and logging for SLM operations.

Provides integration with Langfuse and OpenTelemetry for observability.
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class TraceSpan:
    """A single span in a trace."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        parent_id: str | None = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = f"{trace_id}_{name}_{time.time_ns()}"
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: float | None = None
        self.attributes: dict[str, Any] = {}
        self.events: list[dict] = []

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
        }


class SLMTracer:
    """
    Tracing for SLM training and inference.

    Supports:
    - Local tracing (in-memory)
    - Langfuse integration (if available)
    - OpenTelemetry export

    Example:
        >>> tracer = SLMTracer(project_name="my_slm")
        >>>
        >>> with tracer.trace("inference") as span:
        ...     span.set_attribute("model", "smollm-135m")
        ...     response = model.generate(prompt)
        ...     span.set_attribute("output_tokens", len(response))
    """

    def __init__(
        self,
        project_name: str = "slmflow",
        *,
        use_langfuse: bool = True,
        langfuse_host: str | None = None,
        langfuse_public_key: str | None = None,
        langfuse_secret_key: str | None = None,
    ):
        """
        Initialize tracer.

        Args:
            project_name: Project name for traces
            use_langfuse: Try to use Langfuse for tracing
            langfuse_host: Langfuse server host
            langfuse_public_key: Langfuse public key
            langfuse_secret_key: Langfuse secret key
        """
        self.project_name = project_name
        self.traces: list[dict] = []
        self._langfuse = None
        self._current_trace_id: str | None = None

        if use_langfuse:
            self._init_langfuse(
                langfuse_host,
                langfuse_public_key,
                langfuse_secret_key,
            )

    def _init_langfuse(
        self,
        host: str | None,
        public_key: str | None,
        secret_key: str | None,
    ) -> None:
        """Initialize Langfuse client."""
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                host=host,
                public_key=public_key,
                secret_key=secret_key,
            )
            logger.info("Langfuse tracing initialized")

        except ImportError:
            logger.info(
                "Langfuse not installed. Using local tracing. "
                "Install with: pip install slmflow[observability]"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        metadata: dict | None = None,
    ):
        """
        Create a trace context.

        Args:
            name: Trace name
            metadata: Additional metadata

        Yields:
            TraceSpan for the trace
        """
        trace_id = f"{self.project_name}_{name}_{time.time_ns()}"
        self._current_trace_id = trace_id
        span = TraceSpan(name, trace_id)

        if metadata:
            for key, value in metadata.items():
                span.set_attribute(key, value)

        try:
            yield span
        finally:
            span.end()
            self._record_trace(span)
            self._current_trace_id = None

    @contextmanager
    def span(self, name: str):
        """
        Create a child span in current trace.

        Args:
            name: Span name

        Yields:
            TraceSpan for the span
        """
        if self._current_trace_id is None:
            # No active trace, create one
            with self.trace(name) as span:
                yield span
            return

        span = TraceSpan(name, self._current_trace_id, self._current_trace_id)

        try:
            yield span
        finally:
            span.end()
            self._record_trace(span)

    def _record_trace(self, span: TraceSpan) -> None:
        """Record a completed trace/span."""
        trace_data = span.to_dict()
        self.traces.append(trace_data)

        # Send to Langfuse if available
        if self._langfuse is not None:
            try:
                self._langfuse.trace(
                    id=span.trace_id,
                    name=span.name,
                    metadata=span.attributes,
                )
            except Exception as e:
                logger.debug(f"Failed to send trace to Langfuse: {e}")

    def log_generation(
        self,
        input_text: str,
        output_text: str,
        *,
        model_name: str | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Log a generation event.

        Args:
            input_text: Input prompt
            output_text: Generated output
            model_name: Model used
            latency_ms: Generation latency
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata
        """
        event = {
            "type": "generation",
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "output": output_text,
            "model": model_name,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "metadata": metadata or {},
        }

        self.traces.append(event)

        # Log to Langfuse
        if self._langfuse is not None:
            try:
                trace = self._langfuse.trace(name="generation")
                trace.generation(
                    name="llm",
                    input=input_text,
                    output=output_text,
                    model=model_name,
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug(f"Failed to log generation to Langfuse: {e}")

    def log_training_step(
        self,
        step: int,
        loss: float,
        *,
        learning_rate: float | None = None,
        grad_norm: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log a training step.

        Args:
            step: Training step number
            loss: Loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            metrics: Additional metrics
        """
        event = {
            "type": "training_step",
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "metrics": metrics or {},
        }

        self.traces.append(event)

    def get_traces(
        self,
        trace_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get recorded traces.

        Args:
            trace_type: Filter by type
            limit: Maximum traces to return

        Returns:
            List of trace dictionaries
        """
        traces = self.traces

        if trace_type:
            traces = [t for t in traces if t.get("type") == trace_type]

        return traces[-limit:]

    def flush(self) -> None:
        """Flush any pending traces to backends."""
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as e:
                logger.debug(f"Failed to flush Langfuse: {e}")


def traced(tracer: SLMTracer, name: str | None = None):
    """
    Decorator to trace a function.

    Args:
        tracer: SLMTracer instance
        name: Trace name (defaults to function name)

    Example:
        >>> @traced(tracer)
        ... def my_function(x):
        ...     return x * 2
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            with tracer.trace(trace_name) as span:
                span.set_attribute("function", func.__name__)
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator
