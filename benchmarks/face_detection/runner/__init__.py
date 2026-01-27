# -*- coding: utf-8 -*-
"""
Runner module - Moteur d'exécution du benchmark.
"""

from .engine import run_benchmark, run_benchmark_parallel, run_benchmark_worker
from .strict_engine import run_benchmark_strict, run_all_detectors_strict, clear_gpu_memory

__all__ = [
    "run_benchmark",
    "run_benchmark_parallel",
    "run_benchmark_worker",
    "run_benchmark_strict",
    "run_all_detectors_strict",
    "clear_gpu_memory",
]
