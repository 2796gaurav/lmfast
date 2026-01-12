"""
Unit tests for observability.
"""

import time


class TestSLMTracer:
    """Tests for SLMTracer."""

    def test_trace_context(self):
        """Test trace context manager."""
        from slmflow.observability import SLMTracer

        tracer = SLMTracer(project_name="test", use_langfuse=False)

        with tracer.trace("test_operation") as span:
            span.set_attribute("key", "value")
            time.sleep(0.01)  # Small delay

        traces = tracer.get_traces()
        assert len(traces) == 1
        assert traces[0]["name"] == "test_operation"
        assert traces[0]["attributes"]["key"] == "value"
        assert traces[0]["duration_ms"] >= 10

    def test_log_generation(self):
        """Test logging generation events."""
        from slmflow.observability import SLMTracer

        tracer = SLMTracer(project_name="test", use_langfuse=False)

        tracer.log_generation(
            input_text="Hello",
            output_text="Hi there!",
            model_name="test-model",
            latency_ms=100,
        )

        traces = tracer.get_traces(trace_type="generation")
        assert len(traces) == 1
        assert traces[0]["input"] == "Hello"
        assert traces[0]["output"] == "Hi there!"

    def test_log_training_step(self):
        """Test logging training steps."""
        from slmflow.observability import SLMTracer

        tracer = SLMTracer(project_name="test", use_langfuse=False)

        for step in range(5):
            tracer.log_training_step(
                step=step,
                loss=1.0 - step * 0.1,
                learning_rate=1e-4,
            )

        traces = tracer.get_traces(trace_type="training_step")
        assert len(traces) == 5


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_log_and_get_metrics(self):
        """Test logging and retrieving metrics."""
        from slmflow.observability import MetricsCollector

        collector = MetricsCollector()

        collector.log("loss", 0.5, step=1)
        collector.log("loss", 0.4, step=2)
        collector.log("loss", 0.3, step=3)

        values = collector.get_values("loss")
        assert values == [0.5, 0.4, 0.3]

    def test_batch_logging(self):
        """Test logging multiple metrics at once."""
        from slmflow.observability import MetricsCollector

        collector = MetricsCollector()

        collector.log_batch(
            {
                "loss": 0.5,
                "accuracy": 0.8,
                "lr": 1e-4,
            },
            step=1,
        )

        assert collector.latest("loss") == 0.5
        assert collector.latest("accuracy") == 0.8
        assert collector.latest("lr") == 1e-4

    def test_statistics(self):
        """Test statistical functions."""
        from slmflow.observability import MetricsCollector

        collector = MetricsCollector()

        for i in range(10):
            collector.log("value", float(i))

        assert collector.min("value") == 0.0
        assert collector.max("value") == 9.0
        assert collector.mean("value") == 4.5
        assert collector.latest("value") == 9.0

    def test_summary(self):
        """Test summary statistics."""
        from slmflow.observability import MetricsCollector

        collector = MetricsCollector()

        collector.log("metric", 1.0)
        collector.log("metric", 2.0)
        collector.log("metric", 3.0)

        summary = collector.summary("metric")

        assert summary["count"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["mean"] == 2.0
        assert summary["latest"] == 3.0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_record_usage(self):
        """Test recording token usage."""
        from slmflow.observability.metrics import CostTracker

        tracker = CostTracker()

        tracker.record(input_tokens=100, output_tokens=50)
        tracker.record(input_tokens=200, output_tokens=100)

        summary = tracker.summary()
        assert summary["requests"] == 2
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150
        assert summary["total_tokens"] == 450

    def test_cost_calculation(self):
        """Test cost calculation with pricing."""
        from slmflow.observability.metrics import CostTracker

        tracker = CostTracker(
            cost_per_million_input=1.0,  # $1 per 1M input tokens
            cost_per_million_output=2.0,  # $2 per 1M output tokens
        )

        tracker.record(input_tokens=1_000_000, output_tokens=500_000)

        cost = tracker.total_cost()
        assert cost == 2.0  # $1 for input + $1 for output


class TestTraceSpan:
    """Tests for TraceSpan."""

    def test_span_attributes(self):
        """Test span attribute setting."""
        from slmflow.observability.tracing import TraceSpan

        span = TraceSpan("test", "trace_123")
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42

    def test_span_events(self):
        """Test span event adding."""
        from slmflow.observability.tracing import TraceSpan

        span = TraceSpan("test", "trace_123")
        span.add_event("event1", {"detail": "info"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"

    def test_span_duration(self):
        """Test span duration calculation."""
        from slmflow.observability.tracing import TraceSpan

        span = TraceSpan("test", "trace_123")
        time.sleep(0.01)
        span.end()

        assert span.duration_ms >= 10
        assert span.end_time is not None
