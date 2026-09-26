# UIT Research: LOC-NFST & Federated LUNAR (Edge IoT Intrusion Detection)

**Phòng thí nghiệm Hệ thống Nhúng & An ninh Thông tin (IEC Lab)**  
**Trường Đại học Công nghệ Thông tin, ĐHQG-HCM**  
**Repository Branch**: `feature/federated-lunar-novel`

---

## 📌 Tổng quan Dự án (Project Overview)

Repository này chứa toàn bộ mã nguồn, thực nghiệm và văn bản khoa học cho hai nhánh nghiên cứu cốt lõi về Phát hiện Dị biệt & Xâm nhập Mạng IoT (IoT NIDS) phân tán:
1. **LOC-NFST (Local Null Space Feature Transformation)**: Phương pháp hình học phổ dạng giải tích đóng (closed-form spectral null-space) tối ưu hóa suy diễn tại thiết bị biên IoT.
2. **Federated LUNAR (Fed-LUNAR)**: Kiến trúc One-Class Graph dựa trên xếp hạng khoảng cách $k$-NN phân tán, kết hợp kỹ thuật sinh mẫu nhiễu đa không gian (MSSP), phác thảo mật độ đa tạp (FSDS), thanh lọc mẫu âm xâm lấn (CMNP) và căn chỉnh gradient trực giao (DROGA).

---

## 📁 Cấu trúc Thư mục Khoa học (Directory Hierarchy)

```
📦 UIT_Research_NFST
├── 📁 docs/                             # Toàn bộ tài liệu, báo cáo khoa học & đồ án tốt nghiệp
│   ├── 📁 reports/                      # Báo cáo chuyên sâu theo từng đề tài
│   │   ├── 📁 fed_lunar/                # Báo cáo chuyên đề Federated LUNAR
│   │   │   ├── BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md
│   │   │   ├── WALKTHROUGH_FEDERATED_LUNAR.md
│   │   │   └── REPORT_DEVICE_TYPE_AWARE_INTRUSION_DETECTION.md
│   │   └── 📁 loc_nfst/                 # Báo cáo chuyên đề LOC-NFST & Giới hạn giải tích
│   │       ├── FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md
│   │       ├── RESEARCH_DYNAMIC_K_LOC_NFST.md
│   │       ├── GIAI_DAP_PHAN_BIEN_CHUYEN_SAU_LOC_NFST.md
│   │       ├── REPORT_NATURE_OF_DIVERSE_ANOMALIES_AND_CLUSTERING.md
│   │       ├── GIAI_MA_IEEE_VA_CASE_STUDY_TARGET_2013.md
│   │       └── Deployment_Diagram_Proposal.md
│   ├── 📁 thesis/                       # Đề cương & Hồ sơ Khóa luận Tốt nghiệp
│   │   ├── DE_CUONG_KHOA_LUAN_TOT_NGHIEP_LOC_NFST.md
│   │   └── FORM_DANG_KY_DE_CUONG_KLTN.md
│   └── 📁 testing/                      # Tài liệu quy chuẩn kiểm thử & đặc tả kỹ thuật
│       ├── TEST_INFRA.md
│       ├── TEST_READY.md
│       └── ORIGINAL_REQUEST.md
│
├── 📁 paper_latex/                      # Gói bài báo khoa học chuẩn A* (IEEE S&P / ACM CCS format)
│   ├── main.tex                         # File chính bài báo
│   ├── main.pdf                         # Bản PDF bài báo đã biên dịch (16 trang)
│   ├── references.bib                   # Danh mục tài liệu tham khảo chuẩn (38 trích dẫn có DOI)
│   ├── sec_intro.tex                    # Giới thiệu & Đóng góp
│   ├── sec_threat_model.tex             # Mô hình đe dọa & Research Gap
│   ├── sec_formulation.tex              # Định nghĩa bài toán & Cấu trúc toán học
│   ├── sec_methodology.tex              # Phương pháp: MSSP, FSDS, CMNP, DROGA
│   ├── sec_proofs.tex                   # Chứng minh định lý (Theorems 1 & 2)
│   ├── sec_experiments.tex              # Bảng thực nghiệm & Phân tích Ablation
│   ├── sec_related.tex                  # Tổng quan văn hiến & Ma trận so sánh
│   └── sec_conclusion.tex               # Kết luận & Hướng phát triển
│
├── 📁 notebooks/                        # Jupyter Notebooks thực nghiệm & phân tích
│   ├── 📁 experiments/                  # Thử nghiệm các mô hình chính
│   │   ├── 📁 fed_loc_nfst/             # Triển khai Federated LOC-NFST
│   │   ├── OC-NSFT_old_noise_Gau.ipynb
│   │   ├── OC-NSFT_old_noise_Kmean_threshold.ipynb
│   │   └── OC-NSFT_old_outlier.ipynb
│   ├── 📁 baselines/                    # Huấn luyện & đánh giá các mô hình baseline
│   │   ├── run_baseline-noise+.ipynb
│   │   └── run_baseline-outliers.ipynb
│   ├── 📁 analysis/                     # Trực quan hóa & Phân tích độ phức tạp RAM/độ trễ
│   └── 📁 data_processing/              # Tiền xử lý, lọc nhãn One-Class & Chuẩn hóa
│
├── 📁 baseline_model/                   # Wrappers cho các mô hình so sánh (DASVDD, DIF, NeuTraL AD, SUOD)
├── 📁 outputs/                          # Kết quả benchmark, sensitivity sweep & đồ thị
│   └── 📁 lunar_results/                # File CSV kết quả thực nghiệm từ server RTX 5090
│
├── 📁 tests/                            # Bộ 179 test tự động bảo chứng chất lượng (E2E & Unit tests)
│   ├── 📁 e2e/                          # Kiểm thử tính năng đầu cuối (Tier 1 -> Tier 4)
│   ├── test_baselines.py                # Kiểm thử các mô hình baseline
│   ├── test_benchmark_harness.py        # Kiểm thử bộ đo benchmark tự động
│   ├── test_cmnp_purging.py             # Kiểm thử cơ chế thanh lọc mẫu âm CMNP
│   ├── test_droga_alignment.py          # Kiểm thử thuật toán căn chỉnh gradient DROGA
│   └── test_m2_math_verification.py     # Kiểm thử kiểm chứng định lý toán học
│
├── AGENTS.md                            # Quy tắc bất biến của workspace (Workspace Rules)
└── README.md                            # Mục lục & Hướng dẫn sử dụng
```

---

## 📚 Mục lục Tài liệu Chi tiết (Master Documentation Index)

### 1. Nghiên cứu Federated LUNAR (`docs/reports/fed_lunar/`)
- [**Báo cáo Khoa học Federated LUNAR Chi tiết**](docs/reports/fed_lunar/BAO_CAO_KHOA_HOC_FEDERATED_LUNAR_CHI_TIET.md): Báo cáo toàn diện phân tích hiện tượng Distance-Ranking Inversion khi gặp OOD flood và hiện tượng triệt tiêu gradient xuyên đa tạp trong FL; giải pháp MSSP, FSDS, CMNP, DROGA cùng kết quả trên GPU RTX 5090 qua 4 bộ dữ liệu (`BoTIoT`, `EdgeIIoTset`, `CICIoT2023`, `N_BaIoT`).
- [**Walkthrough Kỹ thuật Triển khai**](docs/reports/fed_lunar/WALKTHROUGH_FEDERATED_LUNAR.md): Hướng dẫn chi tiết từng module mã nguồn, kiến trúc pipeline và các giao thức kiểm thử.
- [**Nghiên cứu Nhận biết Loại Thiết bị trong IDS**](docs/reports/fed_lunar/REPORT_DEVICE_TYPE_AWARE_INTRUSION_DETECTION.md): Phân tích lý luận khoa học và khảo sát văn hiến chuyên sâu về sự cần thiết của đặc trưng nhận biết loại thiết bị (Device-Type-Aware Feature / Device Profiling) khi khoảng cách chuẩn hóa của các thiết bị IoT khác nhau có hình thái đa tạp hoàn toàn khác nhau.

### 2. Nghiên cứu LOC-NFST & Giới hạn Giải tích (`docs/reports/loc_nfst/`)
- [**Báo cáo Toàn diện Federated Edge LOC-NFST**](docs/reports/loc_nfst/FEDERATED_EDGE_LOC_NFST_COMPREHENSIVE_REPORT.md): Khảo sát và đánh giá thuật toán LOC-NFST phân tán trên thiết bị biên.
- [**Nghiên cứu Kỹ thuật Dynamic-k trong LOC-NFST**](docs/reports/loc_nfst/RESEARCH_DYNAMIC_K_LOC_NFST.md): Cơ chế tự thích ứng tham số lân cận $k$ theo mật độ cục bộ.
- [**Giải đáp & Phản biện Chuyên sâu LOC-NFST**](docs/reports/loc_nfst/GIAI_DAP_PHAN_BIEN_CHUYEN_SAU_LOC_NFST.md): Luận giải các câu hỏi phản biện học thuật về tính ổn định số học và ma trận suy biến.
- [**Bản chất Dị biệt Đa dạng & Phân cụm**](docs/reports/loc_nfst/REPORT_NATURE_OF_DIVERSE_ANOMALIES_AND_CLUSTERING.md): Phân tích đặc tính cụm dị biệt trong mạng IoT.
- [**Giải mã Chuẩn IEEE & Case Study Target 2013**](docs/reports/loc_nfst/GIAI_MA_IEEE_VA_CASE_STUDY_TARGET_2013.md): Phân tích ca tấn công chuỗi cung ứng Target 2013 dưới lăng kính an ninh IoT.
- [**Đề xuất Kiến trúc Triển khai (Deployment Diagram)**](docs/reports/loc_nfst/Deployment_Diagram_Proposal.md): Sơ đồ kiến trúc phần cứng và luồng dữ liệu tại gateway biên.

### 3. Hồ sơ Khóa luận Tốt nghiệp (`docs/thesis/`)
- [**Đề cương Chi tiết Khóa luận Tốt nghiệp**](docs/thesis/DE_CUONG_KHOA_LUAN_TOT_NGHIEP_LOC_NFST.md): Đề cương hoàn chỉnh phục vụ bảo vệ KLTN ngành An toàn Thông tin / Kỹ thuật Máy tính.
- [**Đơn Đăng ký Đề tài KLTN**](docs/thesis/FORM_DANG_KY_DE_CUONG_KLTN.md): Mẫu đăng ký đề cương chính thức theo quy chuẩn đào tạo UIT.

### 4. Quy chuẩn Kiểm thử & Hạ tầng (`docs/testing/`)
- [**Hạ tầng Kiểm thử (Test Infrastructure)**](docs/testing/TEST_INFRA.md): Mô tả cấu hình máy chủ, môi trường CUDA, thư viện và test harness.
- [**Báo cáo Trạng thái Sẵn sàng Kiểm thử**](docs/testing/TEST_READY.md): Checklist nghiệm thu tính sẵn sàng của các bộ test.
- [**Yêu cầu Nghiên cứu Gốc**](docs/testing/ORIGINAL_REQUEST.md): Ghi chép chi tiết yêu cầu gốc và các tiêu chí đo lường.

---

## ⚡ Hướng dẫn Chạy Kiểm thử (Quick Start & Testing)

Để chạy toàn bộ bộ 179 ca kiểm thử tự động xác thực tính toàn vẹn:

```powershell
# Chạy toàn bộ test suite
pytest tests/ -v

# Chạy riêng nhóm kiểm thử toán học & định lý
pytest tests/test_m2_math_verification.py -v

# Chạy riêng nhóm kiểm thử thuật toán DROGA & CMNP
pytest tests/test_droga_alignment.py tests/test_cmnp_purging.py -v
```

---

## 🖥️ Cấu hình Môi trường Thực nghiệm

- **Máy chủ Tính toán**: `postmaster.iec`
- **Bộ vi xử lý (CPU)**: Intel Core i9-13900K (32 vCPUs)
- **Bộ nhớ RAM**: 62 GB DDR5
- **Bộ tăng tốc đồ họa (GPU)**: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **Môi trường phần mềm**: Ubuntu Linux, Python 3.11, PyTorch 2.11 (CUDA 13.0)
