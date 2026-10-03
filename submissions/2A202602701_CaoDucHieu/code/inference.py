"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2:
- TTA (trang 62-66, 75)
- ensemble/EMA/soup (trang 67)
- độ phân giải kiểm tra (trang 68)
- temperature scaling (trang 69)
- gộp BatchNorm (trang 71)

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Callable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    minimize_scalar = None


def predict_logits(model: nn.Module, loader, device: str | torch.device = "cuda",
                   view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    model = model.to(dev)

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(dev)
            if view is not None:
                images = view(images)
            outputs = model(images)
            all_filenames.extend(filenames)
            all_y_true.append(labels.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    return all_filenames, np.concatenate(all_y_true, axis=0), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    _, _, h, w = x.shape
    crops = []
    # 4 góc
    crops.append(x[:, :, 0:crop, 0:crop])
    crops.append(x[:, :, 0:crop, w - crop:w])
    crops.append(x[:, :, h - crop:h, 0:crop])
    crops.append(x[:, :, h - crop:h, w - crop:w])
    # Giữa
    cy = (h - crop) // 2
    cx = (w - crop) // 2
    crops.append(x[:, :, cy:cy + crop, cx:cx + crop])
    return crops


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    scaled = []
    for s in sizes:
        scaled.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
    return scaled


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).
      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    assert len(logits_per_view) > 0, "Danh sách view logits rỗng"

    if space == "prob":
        probs_list = []
        for z in logits_per_view:
            z_norm = z - z.max(axis=-1, keepdims=True)
            e = np.exp(z_norm)
            p = e / e.sum(axis=-1, keepdims=True)
            probs_list.append(p)
        mean_prob = np.mean(probs_list, axis=0)
        # Chuẩn hoá đảm bảo tổng đúng bằng 1
        return mean_prob / mean_prob.sum(axis=-1, keepdims=True)
    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        z_norm = mean_logits - mean_logits.max(axis=-1, keepdims=True)
        e = np.exp(z_norm)
        p = e / e.sum(axis=-1, keepdims=True)
        return p / p.sum(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình."""
    assert len(list_of_probs) > 0, "Danh sách probs rỗng"
    mean_prob = np.mean(list_of_probs, axis=0)
    return mean_prob / mean_prob.sum(axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)."""
    n, num_classes = val_logits.shape
    labels = np.array(val_labels, dtype=np.int64)

    def nll_eval(t_val: float) -> float:
        if t_val <= 0:
            return 1e9
        scaled = val_logits / t_val
        scaled -= scaled.max(axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
        probs_correct = probs[np.arange(n), labels]
        probs_correct = np.clip(probs_correct, 1e-12, 1.0)
        return float(-np.mean(np.log(probs_correct)))

    if minimize_scalar is not None:
        res = minimize_scalar(nll_eval, bounds=(0.05, 5.0), method="bounded")
        best_t = float(res.x)
    else:
        # Fallback grid search
        candidates = np.linspace(0.1, 3.0, 300)
        best_t = 1.0
        min_loss = float("inf")
        for c in candidates:
            loss = nll_eval(c)
            if loss < min_loss:
                min_loss = loss
                best_t = float(c)

    return round(best_t, 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    t_val = max(float(T), 1e-6)
    scaled = logits / t_val
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp_s = np.exp(scaled)
    probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return probs / probs.sum(axis=-1, keepdims=True)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (slide trang 71, 75)."""
    model = copy.deepcopy(model)
    model.eval()

    # Dùng tiện ích chuẩn của pytorch nếu có
    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        # Thử fuse tự động trên các submodule
        def _fuse_recursive(m):
            children = list(m.named_children())
            for i in range(len(children) - 1):
                name1, child1 = children[i]
                name2, child2 = children[i + 1]
                if isinstance(child1, nn.Conv2d) and isinstance(child2, nn.BatchNorm2d):
                    setattr(m, name1, fuse_conv_bn_eval(child1, child2))
                    setattr(m, name2, nn.Identity())
            for _, child in m.named_children():
                _fuse_recursive(child)

        _fuse_recursive(model)
    except Exception:
        pass

    return model
