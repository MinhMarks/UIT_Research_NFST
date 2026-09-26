# BẢN ĐĂNG KÝ ĐỀ CƯƠNG KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC

**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM**  
**KHOA KỸ THUẬT MÁY TÍNH / AN TOÀN THÔNG TIN**  
**PHÒNG THÍ NGHIỆM HỆ THỐNG THÔNG TIN VÀ ĐIỀU KHIỂN NHÚNG (IEC LAB)**

---

### THÔNG TIN CHUNG

* **Tên đề tài tiếng Việt:**  
  Nghiên cứu và phát triển hệ thống phát hiện dị biệt mạng IoT thích ứng trên thiết bị biên dựa trên biến đổi không gian rỗng phân cụm động và đồng thiết kế phần cứng - phần mềm.

* **Tên đề tài tiếng Anh:**  
  Adaptive Edge-Native IoT Anomaly Detection via Dynamic-Cardinality Local Null Foley-Sammon Transformation and Hardware-Software Co-Design.

* **Cán bộ hướng dẫn:**  
  TS. [Họ và tên Cán bộ Hướng dẫn] (IEC Lab, Trường ĐH Công nghệ Thông tin – ĐHQG-HCM)

* **Ngôn ngữ thực hiện:**  
  Tiếng Việt (Kèm báo cáo tóm tắt và bài báo khoa học bằng Tiếng Anh chuẩn IEEE)

* **Thời gian thực hiện:**  
  Từ ngày 07/09/2026 đến ngày 26/12/2026 (16 tuần)

* **Sinh viên thực hiện:**  
  1. Sinh viên 1: [Họ và tên Sinh viên 1] – MSSV: [MSSV 1]  
  2. Sinh viên 2: [Họ và tên Sinh viên 2] – MSSV: [MSSV 2]  

* **Hệ đào tạo:**  
  Chính quy / Trí tuệ nhân tạo / Lớp Kỹ sư Tài năng

---

### NỘI DUNG ĐỀ TÀI

#### 1. Tổng quan đề tài (Literature Review & Current State of the Art)
* **Bối cảnh thực tế:**  
  Mạng lưới Vạn vật kết nối (IoT) đang phát triển bùng nổ trong các đô thị thông minh, nhà máy công nghiệp (IIoT) và y tế. Tuy nhiên, các thiết bị IoT thường có năng lực tính toán và bộ nhớ rất hạn chế, dễ bị xâm nhập và biến thành mạng máy tính ma (Botnet) để thực hiện các cuộc tấn công tinh vi. Trong khi đó, các cuộc tấn công Zero-day thế hệ mới xuất hiện liên tục, khiến các hệ thống phát hiện xâm nhập (NIDS) dựa trên phân loại có giám sát (Closed-set Supervised Learning) bị lỗi thời và sụt giảm độ chính xác lên tới 70% khi đối mặt với mã độc mới. Do đó, bài toán **Phát hiện Dị biệt Một lớp (One-Class Novelty Detection - OCND)**—chỉ học từ dữ liệu lưu lượng bình thường và cảnh báo mọi hành vi sai lệch—là hướng đi tất yếu.
* **Khảo sát các đề tài, sản phẩm liên quan và Thực trạng hạn chế:**
  1. *Các mô hình One-Class truyền thống và Học sâu (Deep Learning):*  
     - Các mô hình kinh điển như One-Class SVM (OCSVM), Isolation Forest (iForest), DeepSVDD (ICML 2018), AutoEncoder thường giả định dữ liệu bình thường là một phân phối đơn khối (Unimodal Gaussian). Trên thực tế mạng IoT, lưu lượng đến từ hàng chục thiết bị không đồng nhất (camera truyền video dung lượng lớn liên tục, cảm biến nhiệt độ 30 giây gửi 60 bytes, khóa thông minh bắt tay TLS rời rạc). Dữ liệu bình thường trong không gian $\mathbb{R}^d$ phân rã thành **nhiều đa tạp con rời rạc, phi lồi (Disconnected Non-Convex Manifolds)**. Các mô hình đơn khối tạo ra một siêu bao lồi (Convex Hull) khổng lồ, biến vùng chân không giữa các thiết bị thành lỗ hổng cho mã độc ẩn nấp, dẫn đến việc bỏ lọt các cuộc tấn công cục bộ (**Local Anomalies**) với độ chính xác tụt giảm sâu.
     - Các mô hình học sâu phức tạp (Deep Learning NIDS) lại đòi hỏi tài nguyên tính toán GPU đắt đỏ, tiêu tốn nhiều watt điện năng và độ trễ suy luận lớn, hoàn toàn bất khả thi để triển khai trực tiếp trên các Gateway biên IoT (Edge Gateways).
  2. *Phương pháp Biến đổi không gian rỗng (Null Foley-Sammon Transform - NFST):*  
     - NFST nổi tiếng với khả năng triệt tiêu phương sai nội lớp về 0 tuyệt đối ($S_w \rightarrow \mathbf{0}$) và tối đa hóa độ phân tách liên lớp ($S_b > 0$). Tuy nhiên, trong lịch sử, NFST **chỉ áp dụng cho bài toán phân loại đa lớp có giám sát ($C \ge 2$)**. Khi đưa vào bài toán One-Class ($C=1$), ma trận tán xạ liên lớp triệt tiêu $S_b \equiv \mathbf{0}$ và $S_w \equiv S_t$, dẫn đến không gian nghiệm rỗng bị sụp đổ hoàn toàn ($\text{dim}(\text{Null}) = 0$). Các nghiên cứu One-Class gần đây đề xuất phân cụm giả (Pseudo-classes) nhưng lại **chốt cứng số cụm $K$ tĩnh ngoại tuyến (Static $K$)**. Khi mạng IoT thực tế xảy ra hiện tượng trôi dạt khái niệm (Concept Drift), $K$ tĩnh gây ra hai thái cực nguy hiểm: nếu $K$ quá nhỏ gây sụp đổ chiều rỗng ($L=0$), nếu $K$ quá lớn gây quá phân cụm (over-clustering) làm suy thoái số học ma trận hiệp phương sai.
  3. *Các hệ thống NIDS trên thiết bị biên hiện hữu:*  
     - Các công trình tiêu biểu như *Kitsune* (NDSS 2018) hay *Pigasus* (ACM SIGCOMM 2020) khi chạy trên chip nhúng ARM thường xuyên gặp hiện tượng giật cục do cấp phát bộ nhớ động (Heap Allocation Jitter), gây tràn bộ nhớ (Out-Of-Memory - OOM Crash) khi mạng bị quá tải. Đồng thời, cơ chế khóa luồng truyền thống (`std::mutex`) trong quá trình cập nhật mô hình gây tắc nghẽn luồng dữ liệu (Data Plane) và làm rớt gói tin mạng (Packet Drop).

---

#### 2. Mục tiêu của đề tài (Research Objectives & Proposed Breakthroughs)
Khóa luận hướng tới giải quyết triệt để các hạn chế trên thông qua **2 Đột phá về Thuật toán** và **1 Đột phá về Phần cứng Vi kiến trúc Biên**:

* **Mục tiêu 1 (Thuật toán 1 - Khung giải thuật LOC-NFST & Federated Learning):**  
  Xây dựng khung giải thuật Local One-Class NFST (LOC-NFST). Bằng cách phân hoạch dữ liệu bình thường thành $K$ lớp giả thông qua K-Means, mô hình giải quyết dứt điểm tính suy biến giải tích của One-Class NFST, nén toàn bộ phương sai nội cụm của các thiết bị về $0$ để tạo thành các điểm kỳ dị thu hút (*Point Attractors*) trong không gian rỗng $\mathbb{R}^L$. Cơ chế này giúp phóng đại độ lệch vi mô của các cuộc tấn công ngụy trang tinh vi (**Local Anomalies**), đạt AUC-ROC $> 99\%$ (vượt trội các mô hình Deep Learning). Đồng thời, mở rộng sang môi trường Học liên kết (**FL-LOC-NFST One-Shot, $T=1$**) với cơ chế hiệu chỉnh tán xạ (*Scatter-Shift Correction*), đảm bảo tính riêng tư dữ liệu và nén băng thông truyền thông xuống mức cực thấp ($\approx 9.4\text{ KB/client}$).
* **Mục tiêu 2 (Thuật toán 2 - Cơ chế Phân cụm Động Thích ứng ADYN-LOC-NFST):**  
  Phát triển thuật toán phân cụm động đầu tiên cho dòng dữ liệu Null-Space Learning. Cho phép số lượng cụm $K(t)$ co giãn tự nhiên theo sự xuất hiện/biến mất của thiết bị IoT thông qua máy trạng thái 5 pha: Cập nhật Welford vi mô $\rightarrow$ Tách cụm (Split) $\rightarrow$ Hợp nhất cụm (Merge) $\rightarrow$ Triệt tiêu cụm chết (Death qua phân rã hàm mũ $2^{-\Delta t / T_{half}}$) $\rightarrow$ Sinh cụm mới (Birth qua Phao cách ly `QuarantineBuffer` chống tấn công đầu độc *Boiling Frog*).  
  *Đột phá giải tích:* Chứng minh tính bảo toàn ma trận tán xạ toàn phần $\Delta S_t = \mathbf{0}$ và rút gọn quá trình cập nhật không gian rỗng thành các toán tử đóng **Rank-1 Downdate/Update** với nhân tử chuẩn hóa kích thước mẫu $\frac{1}{N_{total}}$. Giảm độ phức tạp từ $O(Nd^2)$ xuống $O(r^2)$, hoàn thành cập nhật trong $< 1.5\text{ ms}$.
* **Mục tiêu 3 (Phần cứng 3 - Đồng thiết kế Phần cứng - Phần mềm trên Thiết bị Biên):**  
  Hiện thực hóa hệ thống trên vi kiến trúc chip nhúng ARM Cortex-A72 (Raspberry Pi 4 Model B):
  - *Vi kiến trúc bộ nhớ Zero-Heap:* Sử dụng mảng tĩnh `alignas(64) struct ClusterSlot` với $K_{max}=128$ căn chỉnh đường nhớ L1/L2 Cache ($26\text{ KB}$), loại bỏ hoàn toàn việc gọi `malloc/new` tại runtime, chống phân mảnh RAM và triệt tiêu 100% nguy cơ crash OOM.
  - *Cơ chế hoán đổi con trỏ phi khóa RCU (Read-Copy-Update):* Phân tách Data Plane (suy luận gói tin tốc độ đường truyền Line-rate) và Control Plane (cập nhật thích ứng nền). Quá trình cập nhật mô hình mới diễn ra qua lệnh tráo con trỏ nguyên tử `atomic_exchange` trong đúng **$8\text{ nano-giây}$**, đảm bảo **$0\%$ Packet Drop**.
  - *Tăng tốc phần cứng SIMD ARM NEON & Tiết kiệm năng lượng E-Detector:* Vector hóa 128-bit NEON cho phép nhân ma trận chiếu và kích hoạt chế độ ngủ đông (Sleep mode) khi lưu lượng mạng ổn định, tiết kiệm tới $78\%$ điện năng tiêu thụ.

---

#### 3. Phương pháp thực hiện (Methodology & Workflow)
Đề tài áp dụng phương pháp tiếp cận từ mô hình hóa giải tích toán học, mô phỏng đối chứng đa bộ dữ liệu, đến triển khai thực nghiệm phần cứng nhúng:

```
[Luồng gói tin mạng IoT] ──> [Thu thập & Tiền xử lý Luồng] ──> [Feature Extraction d chiều]
                                                                        │
        ┌───────────────────────────────────────────────────────────────┘
        ▼
[Data Plane: Suy luận phi khóa RCU trên Edge Gateway]
  │  1. Vector hóa SIMD ARM NEON: Chiếu z = W^T * x
  │  2. Tính khoảng cách tới K(t) Centroids trong L2 Cache
  │  3. So sánh ngưỡng phân phối Chi: d_min > theta_anomaly(t)?
  ├───> [BÌNH THƯỜNG] ──> Chuyển tiếp gói tin + Welford Stats
  └───> [BẤT THƯỜNG]  ──> Đẩy vào QuarantineBuffer / Kích hoạt Báo động IDS
                                │
        ┌───────────────────────┘
        ▼
[Control Plane: Máy trạng thái ADYN-LOC-NFST thích ứng ngầm]
  │  - Kiểm định Bimodality Index -> K <- K + 1 (Rank-1 Downdate A = A - uu^T)
  │  - Kiểm định Khoảng cách tâm  -> K <- K - 1 (Rank-1 Update A = A + ww^T)
  │  - Quarantine Buffer đạt mật độ -> Cluster Birth (K <- K + 1)
  │  - Cập nhật ma trận chiếu ngầm shadow_W
  └───> [RCU Atomic Pointer Swap 8ns] ──> Cập nhật tức thời sang Data Plane!
```

1. **Giai đoạn 1: Toán học giải tích và Thuật toán cốt lõi (Phòng thí nghiệm):**
   - Xây dựng công thức toán học giải tích cho bài toán Rank-1 Downdate/Update có nhân tử $\frac{1}{N_{total}}$.
   - Lập trình kiểm chứng 6 unit tests bằng Python (PyTest, NumPy, SciPy) đảm bảo tính đúng đắn toán học tuyệt đối.
2. **Giai đoạn 2: Thực nghiệm diện rộng trên Máy chủ GPU (Server `postmaster.iec`):**
   - Chạy kiểm chứng toàn diện trên 6 bộ dữ liệu IoT thực tế: `CICIoT2023`, `ToN-IoT`, `BoTIoT`, `N_BaIoT`, `EdgeIIoTset`, `IoTID20` với 5 bộ chuẩn hóa (`MinMaxScaler`, `StandardScaler`, `RobustScaler`, `QuantileTransformer`, `Normalizer`).
   - Đánh giá khả năng chống chọi trước các loại dị biệt (Global, Clustered, Local) và khả năng chịu lỗi trước tạp chất huấn luyện ($\rho \in \{1\%, 3\%, 5\%\}$).
3. **Giai đoạn 3: Hiện thực hóa Phần cứng - Phần mềm trên Thiết bị Biên:**
   - Viết toàn bộ module suy luận và quản lý bộ nhớ vi cụm bằng ngôn ngữ C++20 tối ưu vi kiến trúc ARM.
   - Biên dịch bằng GCC với cờ tối ưu `-O3 -march=armv8-a+simd` trên hệ điều hành Linux nhúng (Ubuntu Server 22.04 LTS cho Raspberry Pi 4).
   - Đo đạc hiệu năng phần cứng thực tế bằng công cụ Linux `perf` và thiết bị đo công suất phần cứng chuyên dụng.

---

#### 4. Các nội dung chính và Giới hạn của đề tài (Scope, Evaluation & Demo System)

* **Nội dung chính:**
  1. *Nội dung 1:* Nghiên cứu lý thuyết không gian nghiệm rỗng (Null Space Learning) và hoàn thiện giải thuật LOC-NFST, chứng minh vai trò kiến tạo không gian rỗng của việc phân cụm giả.
  2. *Nội dung 2:* Xây dựng toán học giải tích và giải thuật thích ứng dòng dữ liệu ADYN-LOC-NFST với toán tử Rank-1 đóng.
  3. *Nội dung 3:* Phát triển giao thức Học liên kết One-Shot (FL-LOC-NFST) với cơ chế Scatter-Shift Correction.
  4. *Nội dung 4:* Thiết kế vi kiến trúc phần cứng nhúng: Static Slot-Array L2-Cache Aligned và cơ chế đồng bộ phi khóa RCU Atomic Pointer Swap.
  5. *Nội dung 5:* Thực nghiệm đối chuẩn đối kháng trên 6 bộ dữ liệu IoT chuẩn quốc tế và phân tích chuyên sâu các chế độ dị biệt.
  6. *Nội dung 6:* Xây dựng mô hình thực nghiệm Demo thời gian thực (Hardware Testbed) trên phần cứng Raspberry Pi 4.
* **Giới hạn của đề tài (Delimitations):**
  - Đề tài tập trung xử lý luồng dữ liệu đặc trưng bảng (Tabular Flow-based Features) được trích xuất từ tiêu đề gói tin (Header) và thống kê phiên mạng (Flow statistics), không can thiệp giải mã nội dung gói tin (Payload Deep Packet Inspection) nhằm đảm bảo tính riêng tư người dùng và thông lượng mili-giây.
  - Phần cứng thử nghiệm thực tế tập trung vào dòng vi xử lý ARMv8 64-bit (ARM Cortex-A72 đại diện cho thế hệ Edge Gateway phổ biến nhất hiện nay).
* **Phương pháp dự kiến đánh giá hệ thống & Kịch bản DEMO thực tế:**
  - *Chỉ số đánh giá độ chính xác:* AUC-ROC, AUC-PR, F1-Score, Matthews Correlation Coefficient (MCC), Tỷ lệ báo động giả (False Alarm Rate - FAR).
  - *Chỉ số đánh giá phần cứng (Hardware Metrics):* L1/L2 Cache Miss Rate, Instructions Per Cycle (IPC), độ trễ suy luận P99 (P99 Latency), thông lượng gói tin (Packets Per Second - PPS), mức sử dụng bộ nhớ RAM tĩnh (MB), và công suất tiêu thụ điện (mW).
  - *Hệ thống DEMO thực tế:*  
    Thiết lập mô hình mạng IoT thu nhỏ tại IEC Lab gồm: 01 IP Camera giám sát, 02 Cảm biến ESP32 truyền MQTT, và 01 Laptop tấn công (chạy Kali Linux phát các cuộc tấn công DoS, PortScan, và ngụy trang Camera).  
    Gateway trung tâm là **Raspberry Pi 4 Model B** chạy hệ thống LOC-NFST nhúng C++:
    - Màn hình Dashboard trực quan hiển thị số cụm $K(t)$ co giãn tự nhiên trong thời gian thực khi bật/tắt thiết bị.
    - Cảnh báo tức thời ($< 10\text{ ms}$) khi laptop đóng vai trò kẻ tấn công phát tán mã độc, chứng minh khả năng bắt trúng Local Anomalies mà không gây rớt bất kỳ gói tin bình thường nào của camera.

---

### KẾ HOẠCH THỰC HIỆN VÀ PHÂN CÔNG CÔNG VIỆC

Thời gian thực hiện: **16 tuần (Từ 07/09/2026 đến 26/12/2026)**.  
Tiến độ được phân bổ chi tiết cho 2 sinh viên như sau:

| Tuần | Nội dung công việc chi tiết | Phân công: Sinh viên 1 (Thuật toán & AI) | Phân công: Sinh viên 2 (Hệ thống & Phần cứng) | Sản phẩm đầu ra (Deliverables) |
| :---: | :--- | :--- | :--- | :--- |
| **T1 – T2** | - Thu thập tài liệu, khảo sát SOTA.<br>- Thiết lập môi trường server `postmaster.iec` và kit nhúng Raspberry Pi 4. | - Khảo cứu toán học NFST, KNFST, bài toán suy biến One-Class ($K=1$).<br>- Chuẩn bị dữ liệu 6 benchmark datasets. | - Thiết lập môi trường Linux nhúng trên Raspberry Pi 4.<br>- Cài đặt công cụ đo Linux `perf`, thư viện giám sát phần cứng. | Đề cương chi tiết KLTN, môi trường thực nghiệm hoàn chỉnh. |
| **T3 – T5** | - Phát triển thuật toán LOC-NFST và ADYN-LOC-NFST cốt lõi.<br>- Thiết kế cấu trúc dữ liệu vi cụm phần cứng. | - Chứng minh toán học Rank-1 Downdate/Update có chuẩn hóa $\frac{1}{N_{total}}$.<br>- Viết mã nguồn Python `adyn_core.py` và 6 unit tests. | - Thiết kế cấu trúc C++20 `ClusterSlot` căn chỉnh 64-byte L2 Cache.<br>- Lập trình module Welford đa biến tối ưu SIMD ARM NEON. | Module Python cốt lõi, 6/6 Unit tests Passed, module C++ vi cụm. |
| **T6 – T8** | - Thực nghiệm quy mô lớn trên server GPU.<br>- Lập trình cơ chế RCU Lock-Free Pointer Swap. | - Chạy benchmark 30 cặp (6 datasets $\times$ 5 scalers) so sánh Static-K vs ADYN.<br>- Phân tích dữ liệu kết quả, giải mã hiện tượng cứu sụp đổ. | - Lập trình cơ chế RCU Atomic Swap giữa Data Plane và Control Plane.<br>- Kiểm thử tải truyền dữ liệu mô phỏng, đo tỷ lệ Packet Drop. | File kết quả CSV benchmark 30 cặp, module C++ RCU phi khóa chạy ổn định. |
| **T9 – T11** | - Mở rộng Federated Learning (FL-LOC-NFST).<br>- Tích hợp hệ thống nhúng hoàn chỉnh trên Raspberry Pi 4. | - Thiết kế và lập trình giao thức FL One-shot ($T=1$) với Scatter-Shift Correction.<br>- Thử nghiệm đa client không đồng nhất $K_m(t)$. | - Tích hợp toàn bộ mã nguồn C++ lên Raspberry Pi 4.<br>- Tối ưu hóa tập lệnh NEON, đo đạc Cache Miss và IPC bằng Linux `perf`. | Module Federated Learning, bản build C++ chạy trực tiếp trên Pi 4. |
| **T12 – T13** | - Xây dựng Demo Testbed thực tế tại Lab.<br>- Đánh giá thực nghiệm Diverse Anomalies (Global, Clustered, Local). | - Thực nghiệm đối kháng chi tiết trên 3 chế độ dị biệt ADBench.<br>- Đánh giá khả năng chống nhiễm độc tập mẫu ($\rho \in \{1\%, 3\%, 5\%\}$). | - Dựng mô hình mạng thực tế: Camera + ESP32 + Laptop Kali Linux tấn công.<br>- Hoàn thiện Dashboard hiển thị $K(t)$ và cảnh báo IDS thời gian thực. | Hệ thống DEMO phần cứng hoàn chỉnh chạy ổn định tại phòng Lab. |
| **T14 – T15** | - Viết bản thảo Khóa luận tốt nghiệp (Thesis Draft).<br>- Soạn thảo bài báo khoa học quốc tế (Paper Draft). | - Viết Chương 1, 2, 3, 4 (Phần dẫn luận, thuật toán, toán học giải tích).<br>- Soạn thảo các mục lý thuyết và kết quả thực nghiệm bài báo IEEE. | - Viết Chương 5, 6 (Phần cứng vi kiến trúc, đo đạc thực tế, demo).<br>- Vẽ các biểu đồ kiến trúc hệ thống, chụp ảnh hệ thống thực nghiệm. | Bản thảo toàn văn Khóa luận (100+ trang), Bản thảo bài báo chuẩn IEEE. |
| **T16** | - Hoàn thiện quyển Khóa luận, nghiệm thu tại Lab.<br>- Bảo vệ trước Hội đồng Khoa. | - Rà soát toàn bộ công thức toán học, chỉnh sửa theo góp ý GVHD.<br>- Soạn slide báo cáo phần Thuật toán & Đóng góp học thuật. | - Đóng gói mã nguồn GitHub sạch, chuẩn bị thiết bị Demo trước Hội đồng.<br>- Soạn slide báo cáo phần Phần cứng & Hệ thống thực nghiệm. | **Quyển Khóa luận hoàn chỉnh, bảo vệ thành công đạt Xuất sắc.** |

---

### CAM KẾT CHẤT LƯỢNG VÀ KẾT QUẢ MONG ĐỢI

1. **Về mặt đào tạo và học thuật:** Hoàn thành xuất sắc khóa luận tốt nghiệp với đầy đủ chứng minh toán học giải tích chặt chẽ, được hội đồng đánh giá cao về tính kết hợp liên ngành giữa Trí tuệ nhân tạo (AI/ML) và Hệ thống nhúng (Edge Hardware Systems).
2. **Về mặt công bố khoa học:** Tối thiểu **01 bài báo khoa học** hoàn thiện gửi đăng tại tạp chí uy tín thuộc danh mục ISI/Scopus **Q1 / Core A\*** (ưu tiên *IEEE Transactions on Information Forensics and Security* hoặc *IEEE Internet of Things Journal*).
3. **Về mặt sản phẩm chuyển giao:** Kho mã nguồn mở chuẩn mực trên GitHub kèm hệ thống Demo phần cứng Edge Gateway thực tế có khả năng ứng dụng trực tiếp cho các dự án an ninh mạng IoT tại Việt Nam.

---

*TP. Hồ Chí Minh, ngày ..... tháng ..... năm 2026*

**XÁC NHẬN CỦA CÁN BỘ HƯỚNG DẪN**  
*(Ký và ghi rõ họ tên)*

<br><br><br>

**SINH VIÊN THỰC HIỆN 1** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **SINH VIÊN THỰC HIỆN 2**  
*(Ký và ghi rõ họ tên)* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Ký và ghi rõ họ tên)*
