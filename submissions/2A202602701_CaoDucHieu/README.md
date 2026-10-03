# Báo cáo thực nghiệm Lab Day 2 — Phân loại Cỏ dại DeepWeeds

- **Học viên:** Cao Đức Hiếu
- **Mã số sinh viên:** 2A202602701
- **Môn học:** Deep Learning Advance (Track 4 - Day 2)
- **Bài lab:** Backbone, công thức huấn luyện và suy luận trên DeepWeeds

---

## 1. Liên kết thực thi lại (Reproducibility Links)
- **Kaggle Notebook (Đã chạy hoàn tất có output):** [https://www.kaggle.com/code/hiucaoc/track4day2](https://www.kaggle.com/code/hiucaoc/track4day2)
- **GitHub Repository:** [https://github.com/ChunHieu/K4-Day2-CaoDucHieu-2A202602701](https://github.com/ChunHieu/K4-Day2-CaoDucHieu-2A202602701)

---

## 2. Môi trường và Phiên bản Thư viện
Thí nghiệm được thực hiện trên môi trường chuẩn của Kaggle / Google Colab với phần cứng:
- **GPU:** NVIDIA Tesla T4 16GB VRAM (CUDA 12.1)
- **CPU:** Intel Xeon @ 2.20GHz (2 vCPUs), 13GB RAM
- **Hệ điều hành:** Linux Ubuntu 22.04 LTS (x86_64)

### Phiên bản thư viện chính:
- `python`: `3.10.12` (hoặc `3.14+`)
- `torch`: `2.1.2+cu121`
- `torchvision`: `0.16.2+cu121`
- `timm`: `0.9.12`
- `numpy`: `1.26.4`
- `pandas`: `2.2.1`
- `scikit-learn`: `1.4.1.post1`
- `scipy`: `1.12.0`
- `openpyxl`: `3.1.2`
- `matplotlib`: `3.8.3`

---

## 3. Danh sách Hạt giống Ngẫu nhiên (Seeds)
- **Sàng lọc & Ablation (Bước 1, 2, 3):** `seed = 0`
- **Vòng chung kết (Bước 4) & Mốc Baseline:** `seeds = [0, 1, 2]` (báo cáo mean ± std với số bậc tự do `ddof = 1`).

---

## 4. Hướng dẫn Tái lập Kết quả từ Dòng lệnh (CLI)

### Bước 1: Chuẩn bị dữ liệu và thư mục
```bash
# Tải nhãn và split fold 0
mkdir -p data/labels
wget -q -O data/labels/labels.csv https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv
wget -q -O data/labels/train_subset0.csv https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/train_subset0.csv
wget -q -O data/labels/val_subset0.csv https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/val_subset0.csv
wget -q -O data/labels/test_subset0.csv https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/test_subset0.csv

# Tải ảnh từ Zenodo (nếu chạy train thực tế)
# wget -q -O data/images.zip "https://zenodo.org/records/7939060/files/images.zip?download=1"
# unzip -q -n data/images.zip -d data/
```

### Bước 2: Chạy kiểm tra và chấm điểm tự động bằng `eval.py`
Tất cả các file dự đoán trên tập test fold 0 đã được lưu trữ sẵn trong thư mục `predictions/`. Bạn có thể chấm ngay:

```bash
# 1. Chấm điểm chi tiết cấu hình Chung kết (F01)
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

# 2. Chấm điểm cấu hình Mốc Baseline (T00)
python eval.py score --pred "predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00

# 3. Tự chấm Phần I của RUBRIC (Mục tiêu 20/20 điểm)
python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 41.5 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv
```

---

## 5. Cấu trúc Thư mục Nộp bài
```text
submissions/2A202602701_CaoDucHieu/
├── README.md                 # Tài liệu này
├── results.xlsx              # Bảng tổng hợp 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency
├── report.md                 # Báo cáo khoa học chi tiết
├── curves/                   # 21 ảnh đồ thị huấn luyện (Loss, Metric, LR)
│   ├── B01_resnet50.png
│   ├── B03_convnext_tiny.png
│   ├── T03_cutmix.png
│   ├── T09_combined.png
│   ├── F01_seed0.png
│   └── ...
├── predictions/              # Các file dự đoán test fold 0 (cho eval.py)
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── T00_seed0_test.csv
│   ├── T00_seed1_test.csv
│   ├── T00_seed2_test.csv
│   ├── F01_uncal_seed0_test.csv
│   ├── F01_uncal_seed1_test.csv
│   ├── F01_uncal_seed2_test.csv
│   ├── F01_seed0_val.csv
│   ├── F01_seed1_val.csv
│   └── F01_seed2_val.csv
└── code/                     # Toàn bộ mã nguồn hoàn chỉnh
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    └── lab_day2.ipynb
```
