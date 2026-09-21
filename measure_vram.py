from __future__ import annotations

import functools
import gc
import json
import time
from collections.abc import Callable
from typing import Any
from typing import ParamSpec, TypeVar

import torch

P = ParamSpec("P")
T = TypeVar("T")


def cuda_snapshot(device: int | torch.device = 0) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
        }

    device = torch.device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1024 ** 3,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1024 ** 3,
        "free_gb": free_bytes / 1024 ** 3,
        "total_gb": total_bytes / 1024 ** 3,
    }


def make_vram_usage_report(started_at, ended_at, before, after, peak_allocated_gb=None, peak_reserved_gb=None):
    report: dict[str, Any] = {
        "time": {
            "before": started_at,
            "after": ended_at,
            "used": ended_at - started_at,
        },
        "vram": {
            "delta": {
                "allocated_gb": after["allocated_gb"] - before["allocated_gb"],
                "reserved_gb": after["reserved_gb"] - before["reserved_gb"],
                "free_gb": after["free_gb"] - before["free_gb"],
            },
            "peak": {
                "allocated_gb": peak_allocated_gb,
                "reserved_gb": peak_reserved_gb,
            },
        }
    }
    return report


_vram_load_report_key = "_vram_load_report"


def measure_model_load_vram(
        *,
        device: int | torch.device = 0,
        clear_cache_before: bool = True,
        synchronize: bool = True,
        logger: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for measuring GPU VRAM before and after a model loader runs.

    The returned model object will get an attribute:

        model._vram_load_report

    containing before/after/delta/peak memory data.
    """

    def decorator(model_loader: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(model_loader)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            started_at = time.time()
            if torch.cuda.is_available():
                if clear_cache_before:
                    gc.collect()
                    torch.cuda.empty_cache()

                if synchronize:
                    torch.cuda.synchronize(device)

                torch.cuda.reset_peak_memory_stats(device)

            before = cuda_snapshot(device)

            loaded_result = model_loader(*args, **kwargs)

            if torch.cuda.is_available() and synchronize:
                torch.cuda.synchronize(device)

            after = cuda_snapshot(device)

            peak_allocated_gb = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3
                if torch.cuda.is_available()
                else 0.0
            )
            peak_reserved_gb = (
                torch.cuda.max_memory_reserved(device) / 1024 ** 3
                if torch.cuda.is_available()
                else 0.0
            )
            ended_at = time.time()

            report = make_vram_usage_report(
                started_at,
                ended_at,
                before,
                after,
                peak_allocated_gb,
                peak_reserved_gb,
            )
            if logger is not None:
                logger("*" * 20)
                logger(json.dumps(report, indent=2))
                logger("*" * 20)

            try:
                loaded_model = loaded_result[0]
                setattr(loaded_model, _vram_load_report_key, report)
            except Exception as ex:
                print("*" * 20 + " failed to set _vram_load_report to model: " + str(ex))

            return loaded_result

        return wrapper

    return decorator


def get_model_measurement(model):
    if hasattr(model, _vram_load_report_key):
        return getattr(model, _vram_load_report_key)
    else:
        return {}
