# UIT Research - NFST (Anomaly Detection)

## 📁 Cấu trúc thư mục

```
📦 UIT_Research_NFST
├── 📁 notebooks/                    # Tất cả Jupyter notebooks
│   ├── 📁 experiments/              # Thử nghiệm các model chính
│   │   ├── OC-NSFT_old_noise_Gau.ipynb
│   │   ├── OC-NSFT_old_noise_Kmean_threshold.ipynb
│   │   └── OC-NSFT_old_outlier.ipynb
│   ├── 📁 baselines/                # Chạy baseline models để so sánh
│   │   ├── run_baseline-noise+.ipynb
│   │   └── run_baseline-outliers.ipynb
│   ├── 📁 analysis/                 # Phân tích kết quả & tạo báo cáo
│   │   ├── ChartGenerate.ipynb
│   │   ├── createConfusionMatrix.ipynb
│   │   ├── GenerateReport.ipynb
│   │   └── GenerateReport-outlier.ipynb
│   └── 📁 data_processing/          # Tiền xử lý dữ liệu
│       └── scaleData2_200k.ipynb
│
├── 📁 baseline_model/               # Code wrapper cho các baseline models
│   ├── dasvdd_wrapper.py
│   ├── dif_wrapper.py
│   ├── neutralad_wrapper.py
│   └── 📁 algorithms/               # Thuật toán baseline
│
├── 📁 DASVDD/                       # DASVDD model (submodule/reference)
│
├── 📁 Results/                      # Kết quả thí nghiệm (CSV files)
│   ├── DASVDD.csv, DASVDD_noise.csv
│   ├── DIF_noise.csv
│   ├── NeuTraLAD.csv, NeuTraLAD_noise.csv
│   ├── SUOD.csv, SUOD_noise.csv
│   ├── 📁 final/                    # Kết quả cuối cùng
│   └── 📁 OURMODEL/                 # Kết quả model của chúng ta
│
├── 📁 Structure_Result/             # Kết quả cấu trúc dữ liệu
│
├── 📁 outputs/                      # Hình ảnh, biểu đồ xuất ra
│   └── confusion_matrices.png
│
├── 📁 Trash/                        # File cũ/không dùng (có thể xóa)
│
├── .gitignore
└── README.md
```

## 🔬 Mô tả các thành phần

### Notebooks
- **experiments/**: Chứa các notebook thử nghiệm model OC-NSFT với các phương pháp xử lý noise khác nhau (Gaussian, K-means threshold, outlier detection)
- **baselines/**: Chạy các baseline models (DASVDD, DIF, NeuTraLAD, SUOD) để so sánh hiệu năng
- **analysis/**: Tạo biểu đồ, confusion matrix và báo cáo kết quả
- **data_processing/**: Tiền xử lý và scale dữ liệu

### Baseline Models
Các wrapper để chạy baseline models:
- DASVDD (Deep Anomaly Detection with Self-supervised Learning)
- DIF (Deep Isolation Forest)
- NeuTraLAD (Neural Transformation Learning for Anomaly Detection)
- SUOD (Scalable Unsupervised Outlier Detection)

### Datasets
Các dataset IoT được sử dụng:
- BoTIoT
- CICIoT2023
- N_BaIoT
- ToNIoT

---

## 📝 Changelog - Dọn dẹp workspace

**Ngày: 03/12/2024**

### Đã thực hiện:
1. ✅ Tạo cấu trúc thư mục mới cho notebooks:
   - `notebooks/experiments/` - Thử nghiệm model
   - `notebooks/baselines/` - Baseline models
   - `notebooks/analysis/` - Phân tích & báo cáo
   - `notebooks/data_processing/` - Xử lý dữ liệu

2. ✅ Di chuyển các file notebook vào đúng thư mục:
   - 3 notebooks thử nghiệm → `experiments/`
   - 2 notebooks baseline → `baselines/`
   - 4 notebooks phân tích → `analysis/`
   - 1 notebook xử lý dữ liệu → `data_processing/`

3. ✅ Tạo thư mục `outputs/` cho hình ảnh xuất ra
   - Di chuyển `confusion_matrices.png` vào đây

4. ✅ Đổi tên `Trassh/` → `Trash/` (sửa lỗi chính tả)

5. ✅ Cập nhật `.gitignore`:
   - Thêm `.ipynb_checkpoints/` (bỏ qua Jupyter checkpoints)
   - Thêm `__pycache__/` (bỏ qua Python cache)
   - Thêm `Trash/` (bỏ qua thư mục rác)

6. ✅ Tạo file `README.md` mô tả cấu trúc project

### Gợi ý tiếp theo:
- [ ] Xóa thư mục `Trash/` nếu không cần các file cũ
- [ ] Xóa thư mục `.ipynb_checkpoints/` ở root
- [ ] Commit changes lên git
