"""Benchmark module for running jigsaw puzzle solver evaluations across multiple models and images."""

from .config import BenchmarkConfig, ModelSpec, RunConfig
from .cache import BenchmarkCache
from .runner import BenchmarkRunner
from .aggregator import ResultsAggregator
from .plots import BenchmarkPlotter

__all__ = [
    "BenchmarkConfig",
    "ModelSpec",
    "RunConfig",
    "BenchmarkCache",
    "BenchmarkRunner",
    "ResultsAggregator",
    "BenchmarkPlotter",
]
