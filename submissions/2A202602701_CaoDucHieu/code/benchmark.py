"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
  - ghi rõ GPU, dtype, batch, độ phân giải, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100, sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    # 2. Đo đạc chính thức
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_arr = np.array(times, dtype=np.float64)
    p50 = float(np.percentile(times_arr, 50))
    p95 = float(np.percentile(times_arr, 95))
    p99 = float(np.percentile(times_arr, 99))
    mean = float(np.mean(times_arr))

    return {
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "mean": round(mean, 3),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    is_cuda = torch.cuda.is_available() and "cuda" in device
    dev = torch.device(device if is_cuda else "cpu")

    model = model.to(dev)
    model.eval()

    sync_fn = torch.cuda.synchronize if is_cuda else None
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16" and is_cuda:
        model = model.half()
        dummy_input = dummy_input.half()
        forward_fn = lambda: model(dummy_input)
    elif dtype == "amp" and is_cuda:
        def forward_fn():
            with torch.cuda.amp.autocast():
                return model(dummy_input)
    else:
        # fp32
        forward_fn = lambda: model(dummy_input)

    with torch.inference_mode():
        metrics = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    p50 = metrics["p50"]
    p95 = metrics["p95"]
    p99 = metrics["p99"]
    imgs_per_s = round(batch_size / (p50 / 1000.0), 2) if p50 > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype.upper(),
        "batch": batch_size,
        "img_size": img_size,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "images_per_s": imgs_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    """Đo độ trễ của TTA với k views."""
    single_res = latency_report(model, batch_size=batch_size, img_size=img_size, dtype=dtype,
                                device=device, warmup=warmup, iters=iters)
    
    # Đo thực tế k lượt forward
    is_cuda = torch.cuda.is_available() and "cuda" in device
    dev = torch.device(device if is_cuda else "cpu")
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    sync_fn = torch.cuda.synchronize if is_cuda else None

    def tta_fn():
        for _ in range(k_views):
            model(dummy_input)

    with torch.inference_mode():
        tta_metrics = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)

    return {
        "k_views": k_views,
        "single_p50": single_res["p50"],
        "tta_p50": tta_metrics["p50"],
        "tta_p95": tta_metrics["p95"],
        "relative_cost": round(tta_metrics["p50"] / max(single_res["p50"], 1e-4), 2),
    }
