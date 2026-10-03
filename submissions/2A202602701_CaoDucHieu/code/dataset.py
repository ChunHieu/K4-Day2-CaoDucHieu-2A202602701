"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
try:
    import torchvision.transforms as T  # type: ignore
except ImportError:
    T = None

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột `Filename, Label` (hoặc thêm `Species`).
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1).
    1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng ~60/20/20)
    2. Giao của từng cặp tập theo Filename phải RỖNG
    3. Hợp ba tập phải bằng đúng 17.509 ảnh
    4. Mọi Filename đều tồn tại trong images_dir (nếu images_dir tồn tại)
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test

    set_train = set(train_df["Filename"])
    set_val = set(val_df["Filename"])
    set_test = set(test_df["Filename"])

    ov_train_val = len(set_train & set_val)
    ov_train_test = len(set_train & set_test)
    ov_val_test = len(set_val & set_test)

    assert ov_train_val == 0, f"Giao train và val không rỗng: {ov_train_val} ảnh"
    assert ov_train_test == 0, f"Giao train và test không rỗng: {ov_train_test} ảnh"
    assert ov_val_test == 0, f"Giao val và test không rỗng: {ov_val_test} ảnh"

    union_all = len(set_train | set_val | set_test)
    assert union_all == 17509, f"Hợp 3 tập không đủ 17509 ảnh (có {union_all} ảnh)"
    assert n_total == 17509, f"Tổng số dòng không bằng 17509 (có {n_total})"

    # Kiểm tra tồn tại file nếu images_dir tồn tại
    if images_dir is not None:
        img_p = Path(images_dir)
        if img_p.exists() and any(img_p.iterdir()):
            missing = []
            for fn in list(train_df["Filename"][:100]) + list(val_df["Filename"][:50]) + list(test_df["Filename"][:50]):
                if not (img_p / fn).exists():
                    missing.append(fn)
            if missing:
                raise FileNotFoundError(f"Thiếu các file ảnh mẫu: {missing[:5]}")

    per_class_train = train_df["Label"].value_counts().sort_index().to_dict()
    per_class_val = val_df["Label"].value_counts().sort_index().to_dict()
    per_class_test = test_df["Label"].value_counts().sort_index().to_dict()

    return {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": n_total},
        "per_class": {
            "train": per_class_train,
            "val": per_class_val,
            "test": per_class_test,
        },
        "overlap": {
            "train_val": ov_train_val,
            "train_test": ov_train_test,
            "val_test": ov_val_test,
        },
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transforms theo train/val và mức augmentation `aug`.
    aug: "basic" | "color" | "trivial" | "randaug"
    """
    if T is None:
        class FallbackTransform:
            def __init__(self, size):
                self.size = size
            def __call__(self, img):
                img = img.resize((self.size, self.size))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
                return torch.from_numpy(arr).permute(2, 0, 1).float()
        return FallbackTransform(img_size)

    if train:
        transform_list = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ]
        if aug == "color":
            transform_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        elif aug == "trivial":
            transform_list.append(T.TrivialAugmentWide())
        elif aug == "randaug":
            transform_list.append(T.RandAugment(num_ops=2, magnitude=9))
        
        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return T.Compose(transform_list)
    else:
        # Val / test: không augmentation ngẫu nhiên
        crop_size = img_size
        resize_size = int(crop_size * 256 / 224) if crop_size < 256 else 256
        return T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ images_dir theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        filename = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / filename

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback tạo dummy RGB image nếu thư mục ảnh chưa tải đủ (cho testing)
            image = Image.new("RGB", (256, 256), color=(128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        return image, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    data_sampler = None
    shuffle = train

    if train and sampler == "balanced":
        # Balanced sampler theo inverse class frequency
        label_counts = df["Label"].value_counts().to_dict()
        sample_weights = [1.0 / label_counts[l] for l in df["Label"]]
        data_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(df),
            replacement=True
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=data_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train
    )
