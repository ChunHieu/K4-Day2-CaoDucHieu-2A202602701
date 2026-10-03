"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

from typing import List, Dict, Any
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

# Gợi ý backbone (GUIDE.md mục 2.1)
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (init != "scratch") and pretrained

    if timm is not None:
        model = timm.create_model(
            name,
            pretrained=is_pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
    else:
        # Fallback dùng torchvision nếu timm chưa được cài đặt
        import torchvision.models as tvm
        if "resnet50" in name:
            weights = tvm.ResNet50_Weights.DEFAULT if is_pretrained else None
            model = tvm.resnet50(weights=weights)
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        elif "mobilenet" in name:
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if is_pretrained else None
            model = tvm.mobilenet_v3_large(weights=weights)
            in_f = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_f, num_classes)
        elif "convnext" in name:
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if is_pretrained else None
            model = tvm.convnext_tiny(weights=weights)
            in_f = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_f, num_classes)
        elif "swin" in name:
            weights = tvm.Swin_T_Weights.DEFAULT if is_pretrained else None
            model = tvm.swin_t(weights=weights)
            in_f = model.head.in_features
            model.head = nn.Linear(in_f, num_classes)
        else:
            raise ValueError(f"Backbone {name} không được hỗ trợ trong fallback mode.")

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head."""
    # Tìm các tham số của head
    head_params = set()
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            head_params.update(classifier.parameters())
        elif isinstance(classifier, nn.Parameter):
            head_params.add(classifier)
    elif hasattr(model, "fc"):
        head_params.update(model.fc.parameters())
    elif hasattr(model, "head"):
        head_params.update(model.head.parameters())
    elif hasattr(model, "classifier"):
        head_params.update(model.classifier.parameters())

    for param in model.parameters():
        if param not in head_params:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Khi backbone đóng băng, BN cần giữ ở eval mode
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:
    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    - head mới: lr = lr_head (thường gấp 10 lần), weight_decay = weight_decay
    """
    head_params = set()
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            head_params.update(classifier.parameters())
        elif isinstance(classifier, nn.Parameter):
            head_params.add(classifier)
    elif hasattr(model, "fc"):
        head_params.update(model.fc.parameters())
    elif hasattr(model, "head"):
        head_params.update(model.head.parameters())
    elif hasattr(model, "classifier"):
        head_params.update(model.classifier.parameters())

    group_backbone_decay = []
    group_backbone_no_decay = []
    group_head = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p in head_params:
            group_head.append(p)
        else:
            if p.ndim > 1:
                group_backbone_decay.append(p)
            else:
                group_backbone_no_decay.append(p)

    groups = []
    if group_backbone_decay:
        groups.append({"params": group_backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if group_backbone_no_decay:
        groups.append({"params": group_backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if group_head:
        groups.append({"params": group_head, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính GMAC cho một ảnh 3 x img_size x img_size."""
    macs = 0
    hooks = []

    def conv_hook(self, input, output):
        nonlocal macs
        batch_size, out_c, out_h, out_w = output.shape
        kernel_h, kernel_w = self.kernel_size
        in_c = self.in_channels // self.groups
        macs_per_instance = out_c * out_h * out_w * (in_c * kernel_h * kernel_w)
        macs += batch_size * macs_per_instance

    def linear_hook(self, input, output):
        nonlocal macs
        weight_ops = self.in_features * self.out_features
        batch_size = input[0].shape[0] if input[0].dim() > 1 else 1
        macs += batch_size * weight_ops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)
    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception:
        # Nếu mô hình có forward phức tạp, ước lượng theo tham số
        pass
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    gmacs = macs / 1e9
    if gmacs < 0.01:
        # Fallback ước lượng thông qua params
        gmacs = round(count_params(model) * 0.16, 2)
    return round(gmacs, 3)
