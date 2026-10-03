"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
- label smoothing (trang 56)
- focal loss (trang 57)
- Mixup/CutMix (trang 48)

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

from typing import Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls", "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loại loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, f"smoothing phải trong [0, 1), nhận {smoothing}"
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, target)
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).
    Khi gamma = 0 và alpha = None, tương đương hoàn toàn cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_softmax
        log_p = F.log_softmax(logits, dim=-1)
        # gather log_p của đúng target class
        log_pt = log_p.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(dim=-1, index=target)
            loss = at * loss

        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.
    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về mean = 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng: w_c = (1 - beta) / (1 - beta ** n_c),
      chuẩn hoá tổng trọng số về số lớp (K=9)
    """
    counts_arr = np.array(counts, dtype=np.float64)
    k = len(counts_arr)

    if beta <= 0.0:
        inv = 1.0 / np.maximum(counts_arr, 1.0)
        weights = inv / inv.mean()
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * k

    return torch.tensor(weights, dtype=torch.float32)


def rand_bbox(size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
    """Cắt bounding box ngẫu nhiên cho CutMix."""
    w = size[3]
    h = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
        # Điều chỉnh lại lam theo diện tích thực tế
        box_area = (bbx2 - bbx1) * (bby2 - bby1)
        total_area = x.size(2) * x.size(3)
        lam_adj = 1.0 - (box_area / float(total_area))
        return x_mixed, (y_a, y_b, lam_adj)
    else:
        return x, (y_a, y_b, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
