"""
Metrics collection and visualization.

Collects and visualizes training and inference metrics.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric measurement."""

    value: float
    timestamp: float = field(default_factory=time.time)
    step: int | None = None
    metadata: dict = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and manages metrics for training and inference.

    Example:
        >>> collector = MetricsCollector()
        >>>
        >>> for step in range(100):
        ...     loss = train_step()
        ...     collector.log("loss", loss, step=step)
        >>>
        >>> collector.plot("loss")
    """

    def __init__(self, name: str = "slmflow"):
        """
        Initialize metrics collector.

        Args:
            name: Collector name
        """
        self.name = name
        self.metrics: dict[str, list[MetricPoint]] = defaultdict(list)
        self.start_time = time.time()

    def log(
        self,
        metric_name: str,
        value: float,
        *,
        step: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Log a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Training step
            metadata: Additional metadata
        """
        point = MetricPoint(
            value=value,
            step=step,
            metadata=metadata or {},
        )
        self.metrics[metric_name].append(point)

    def log_batch(
        self,
        metrics: dict[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step
        """
        for name, value in metrics.items():
            self.log(name, value, step=step)

    def get(
        self,
        metric_name: str,
        *,
        last_n: int | None = None,
    ) -> list[MetricPoint]:
        """
        Get metric values.

        Args:
            metric_name: Name of the metric
            last_n: Return only last N values

        Returns:
            List of MetricPoint objects
        """
        points = self.metrics.get(metric_name, [])
        if last_n:
            return points[-last_n:]
        return points

    def get_values(self, metric_name: str) -> list[float]:
        """Get just the values for a metric."""
        return [p.value for p in self.get(metric_name)]

    def get_steps(self, metric_name: str) -> list[int]:
        """Get steps for a metric."""
        return [p.step for p in self.get(metric_name) if p.step is not None]

    def latest(self, metric_name: str) -> float | None:
        """Get the latest value for a metric."""
        points = self.metrics.get(metric_name, [])
        return points[-1].value if points else None

    def mean(self, metric_name: str, last_n: int | None = None) -> float | None:
        """
        Get mean of metric values.

        Args:
            metric_name: Metric name
            last_n: Consider only last N values

        Returns:
            Mean value or None if no data
        """
        values = self.get_values(metric_name)
        if last_n:
            values = values[-last_n:]
        return sum(values) / len(values) if values else None

    def min(self, metric_name: str) -> float | None:
        """Get minimum value for a metric."""
        values = self.get_values(metric_name)
        return min(values) if values else None

    def max(self, metric_name: str) -> float | None:
        """Get maximum value for a metric."""
        values = self.get_values(metric_name)
        return max(values) if values else None

    def summary(self, metric_name: str) -> dict[str, Any]:
        """
        Get summary statistics for a metric.

        Returns:
            Dictionary with count, min, max, mean, latest
        """
        values = self.get_values(metric_name)
        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
        }

    def plot(
        self,
        metric_names: str | list[str],
        *,
        title: str | None = None,
        xlabel: str = "Step",
        ylabel: str = "Value",
        save_path: str | None = None,
    ):
        """
        Plot metrics.

        Args:
            metric_names: Metric name(s) to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "matplotlib not installed. " "Install with: pip install slmflow[observability]"
            )
            return

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        plt.figure(figsize=(10, 6))

        for name in metric_names:
            points = self.get(name)
            if not points:
                continue

            steps = [p.step if p.step is not None else i for i, p in enumerate(points)]
            values = [p.value for p in points]

            plt.plot(steps, values, label=name, marker=".")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or f"{self.name} Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def to_dataframe(self, metric_name: str | None = None):
        """
        Convert metrics to a pandas DataFrame.

        Args:
            metric_name: Specific metric (or all if None)

        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed")
            return None

        if metric_name:
            points = self.get(metric_name)
            return pd.DataFrame(
                [
                    {
                        "metric": metric_name,
                        "value": p.value,
                        "step": p.step,
                        "timestamp": p.timestamp,
                        **p.metadata,
                    }
                    for p in points
                ]
            )

        # All metrics
        rows = []
        for name, points in self.metrics.items():
            for p in points:
                rows.append(
                    {
                        "metric": name,
                        "value": p.value,
                        "step": p.step,
                        "timestamp": p.timestamp,
                    }
                )

        return pd.DataFrame(rows)

    def export_json(self, path: str) -> None:
        """
        Export metrics to JSON.

        Args:
            path: Output file path
        """
        import json

        data = {
            "name": self.name,
            "start_time": self.start_time,
            "metrics": {
                name: [
                    {
                        "value": p.value,
                        "step": p.step,
                        "timestamp": p.timestamp,
                        "metadata": p.metadata,
                    }
                    for p in points
                ]
                for name, points in self.metrics.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Metrics exported to {path}")

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()


class CostTracker:
    """
    Track token usage and estimated costs.

    Example:
        >>> tracker = CostTracker()
        >>> tracker.record(input_tokens=100, output_tokens=50)
        >>> print(tracker.total_cost())
    """

    # Default cost per 1M tokens (can be customized)
    DEFAULT_COSTS = {
        "input": 0.0,  # Self-hosted = free
        "output": 0.0,
    }

    def __init__(
        self,
        cost_per_million_input: float = 0.0,
        cost_per_million_output: float = 0.0,
    ):
        """
        Initialize cost tracker.

        Args:
            cost_per_million_input: Cost per 1M input tokens
            cost_per_million_output: Cost per 1M output tokens
        """
        self.cost_per_million_input = cost_per_million_input
        self.cost_per_million_output = cost_per_million_output

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Record token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.requests += 1

    def total_cost(self) -> float:
        """Calculate total cost."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.cost_per_million_input
        output_cost = (self.total_output_tokens / 1_000_000) * self.cost_per_million_output
        return input_cost + output_cost

    def summary(self) -> dict:
        """Get usage summary."""
        return {
            "requests": self.requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self.total_cost(),
        }
