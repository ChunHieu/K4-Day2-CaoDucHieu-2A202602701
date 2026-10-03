"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

MỘT hàm `run(cfg)` dùng chung cho mọi cấu hình: đổi thí nghiệm bằng cách đổi `Config`.
Chỉ số chọn checkpoint (macro-F1 val) tính bằng `eval.compute_metrics`.
"""
from __future__ import annotations

import os
import sys
import json
import random
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Import các module nội bộ
from dataset import load_split, check_split, build_transforms, make_loader, NUM_CLASSES
from model import build_model, param_groups, freeze_backbone
from losses import build_criterion, class_weights, mix_batch, mixed_loss

# Thêm đường dẫn tới eval.py ở repo gốc nếu cần
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số."""
    groups = param_groups(model, lr_backbone=cfg.lr_backbone, lr_head=cfg.lr_head,
                          weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi Cosine Annealing về 0."""
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def evaluate_loader(model: nn.Module, loader, device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy đánh giá và gom filenames, y_true, probs."""
    model.eval()
    all_filenames = []
    all_y = []
    all_probs = []

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_filenames.extend(filenames)
            all_y.extend(labels.numpy())
            all_probs.append(probs)

    y_true = np.array(all_y, dtype=np.int64)
    probs_arr = np.concatenate(all_probs, axis=0)
    # Chuẩn hoá đảm bảo tổng đúng bằng 1
    probs_arr = probs_arr / probs_arr.sum(axis=-1, keepdims=True)
    return all_filenames, y_true, probs_arr


def run(cfg: Config) -> Dict[str, Any]:
    """Hàm huấn luyện dùng chung cho mọi thí nghiệm."""
    set_seed(cfg.seed)
    out_path = run_dir(cfg)
    out_path.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu & split
    train_df, val_df, test_df = load_split(cfg.labels_dir, fold=cfg.fold)
    _ = check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size,
                               train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size,
                             train=False, num_workers=cfg.num_workers)
    test_loader = make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size,
                              train=False, num_workers=cfg.num_workers)

    # 2. Xây dựng model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, pretrained=True, num_classes=NUM_CLASSES,
                        drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)

    # 3. Xây dựng loss
    weights = None
    if cfg.loss in ("ce_weighted", "focal") and cfg.class_weight_beta is not None:
        counts = [train_df["Label"].value_counts().get(c, 0) for c in range(NUM_CLASSES)]
        weights = class_weights(counts, beta=cfg.class_weight_beta).to(device)

    criterion = build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weights,
        alpha=weights
    )

    # 4. Tối ưu hoá
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None
    scaler = GradScaler(enabled=cfg.amp and torch.cuda.is_available())

    best_macro_f1 = -1.0
    best_epoch = 0
    best_weights = None
    history = []

    # 5. Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.init == "frozen":
            # Giữ BN ở eval mode nếu backbone đóng băng
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        train_loss = 0.0
        n_batches = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=cfg.amp and torch.cuda.is_available()):
                if cfg.mix in ("mixup", "cutmix"):
                    images, targets = mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
                    outputs = model(images)
                    loss = mixed_loss(criterion, outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation cuối epoch
        if ema is not None:
            ema.apply_shadow(model)

        val_fn, val_y, val_probs = evaluate_loader(model, val_loader, device)
        val_metrics = ev.compute_metrics(val_y, val_probs)
        val_macro_f1 = val_metrics["macro_f1"]
        val_top1 = val_metrics["top1"]

        if ema is not None:
            ema.restore(model)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_macro_f1": round(val_macro_f1, 4),
        })

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    # 6. Đánh giá mô hình tốt nhất
    if best_weights is not None:
        model.load_state_dict(best_weights)

    val_fn, val_y, val_probs = evaluate_loader(model, val_loader, device)
    ev.save_predictions(pred_path(cfg, "val"), val_fn, val_y, val_probs)

    if cfg.save_test_predictions:
        test_fn, test_y, test_probs = evaluate_loader(model, test_loader, device)
        ev.save_predictions(pred_path(cfg, "test"), test_fn, test_y, test_probs)

    # Lưu history và config
    pd.DataFrame(history).to_csv(out_path / "history.csv", index=False)
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "best_val_top1": val_top1,
    }
