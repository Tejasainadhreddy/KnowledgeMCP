"""Structured JSON logs + CloudWatch metrics (Embedded Metric Format) + X-Ray tracing via AWS Lambda Powertools.

EMF metrics are written as log lines, so there is no PutMetricData API cost.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

SERVICE = "knowledgemcp"
logger = Logger(service=SERVICE)
metrics = Metrics(namespace="KnowledgeMCP", service=SERVICE)
tracer = Tracer(service=SERVICE)


class StepTimer:
    """Records how long each pipeline step takes, in milliseconds."""

    def __init__(self):
        self.steps: dict[str, float] = {}

    @contextmanager
    def step(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] = round((time.perf_counter() - start) * 1000, 2)


def add_metric(name: str, value: float, unit=MetricUnit.Count) -> None:
    try:
        metrics.add_metric(name=name, unit=unit, value=value)
    except Exception:  # metrics must never break a request
        logger.debug("metric dropped", extra={"metric": name})


__all__ = ["logger", "metrics", "tracer", "StepTimer", "add_metric", "MetricUnit"]
