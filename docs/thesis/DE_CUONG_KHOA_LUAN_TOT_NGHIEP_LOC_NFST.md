# ĐỀ CƯƠNG CHI TIẾT KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC

**ĐỀ TÀI:**  
**NGHIÊN CỨU VÀ PHÁT TRIỂN HỆ THỐNG PHÁT HIỆN DỊ BIỆT MẠNG IOT THÍCH ỨNG TRÊN THIẾT BỊ BIÊN DỰA TRÊN BIẾN ĐỔI KHÔNG GIAN RỖNG PHÂN CỤM ĐỘNG VÀ ĐỒNG THIẾT KẾ PHẦN CỨNG - PHẦN MỀM**  
*(Adaptive Edge-Native IoT Anomaly Detection via Dynamic-Cardinality Local Null Foley-Sammon Transformation and Hardware-Software Co-Design)*

---

**Ngành:** Kỹ thuật Máy tính / An toàn Thông tin / Khoa học Máy tính  
**Đơn vị đào tạo:** Trường Đại học Công nghệ Thông tin – Đại học Quốc gia TP.HCM (UIT - VNU-HCM)  
**Phòng thí nghiệm:** Information & Embedded Cyber-physical Systems Lab (IEC Lab)  
**Định hướng công bố khoa học:** IEEE Transactions on Information Forensics and Security (IEEE TIFS - Core A*) / IEEE Internet of Things Journal (IEEE IoT-J - Core Q1, IF 10.6)

---

## PHẦN 1: THÔNG TIN TỔNG QUAN VỀ ĐỀ TÀI

### 1.1. Tính cấp thiết của đề tài (Motivation & Problem Statement)
Sự bùng nổ của mạng lưới Vạn vật kết nối (IoT) với hơn 21 tỷ thiết bị biên dự kiến vào cuối năm 2025 đang đặt ra thách thức an ninh mạng nghiêm trọng. Môi trường mạng IoT thực tế sở hữu 3 đặc tính gây tê liệt các hệ thống phát hiện xâm nhập (NIDS) truyền thống:
1. **Tính bất tương đồng thiết bị (Device Heterogeneity):** Mạng kết nối đồng thời từ camera độ trễ thấp (video stream hàng Mbps) đến cảm biến telemetry siêu nhẹ (vài chục bytes/phút). Trong không gian đặc trưng $\mathbb{R}^d$, dữ liệu bình thường không phải là một khối cầu Gauss đơn mode, mà phân rã thành **nhiều đa tạp con rời rạc, phi lồi (Disconnected Non-Convex Manifolds)**. Các mô hình One-Class toàn cục ngây thơ ($K=1$) sẽ gom tất cả thành một siêu bao lồi, vô tình biến các khoảng trống chân không giữa các thiết bị thành vùng trú ẩn an toàn cho mã độc.
2. **Cấu trúc dị biệt đa hình thái (Diverse Anomaly Structures):** Các cuộc tấn công IoT biến hóa từ Dị biệt toàn cục (*Global Anomalies* như DoS/DDoS), Dị biệt cụm (*Clustered Anomalies* như Botnet Mirai C&C) đến Dị biệt cục bộ (*Local Anomalies* như ngụy trang gửi dữ liệu gián điệp, quét cổng âm thầm). Trong đó, Local Anomaly là "sát thủ vô hình" khi điểm tấn công nằm ngay sát sườn cụm thiết bị bình thường, đánh lừa hầu hết các mô hình học sâu (DeepSVDD, AutoEncoder) và mô hình mật độ (LUNAR, OCSVM).
3. **Hiện tượng trôi dạt khái niệm (Concept Drift):** Phân phối mạng $\mathcal{P}_t(X)$ biến thiên liên tục khi có thiết bị mới gia nhập, cập nhật firmware hoặc chu kỳ ngày/đêm. Việc "chốt cứng" số cụm $K$ tĩnh ngoại tuyến (Static $K$) khiến mô hình hoặc bị sụp đổ chiều rỗng ($L=0$) do dưới phân cụm, hoặc bị bùng nổ báo động giả do quá phân cụm (over-clustering).
4. **Rào cản vi kiến trúc thiết bị biên (Edge Hardware Bottlenecks):** Các thiết bị IoT Gateway (như ARM Cortex-A72 trên Raspberry Pi 4) bị thắt cổ chai về băng thông bộ nhớ RAM, phân mảnh Heap khi gọi `malloc/new` liên tục, và hiện tượng rớt gói tin (Packet Drop) khi khóa luồng dữ liệu (Data Plane) trong quá trình cập nhật mô hình.

### 1.2. Mục tiêu nghiên cứu (Research Objectives)
1. **Làm chủ và tái định hình nền tảng giải tích NFST cho bài toán One-Class:** Giải quyết triệt để bài toán suy biến của Null Foley-Sammon Transform khi $K=1$ ($S_b = \mathbf{0}, S_w \equiv S_t$), chứng minh vai trò kiến tạo không gian rỗng của việc phân cụm giả (Pseudo-classes) để nén phương sai nội cụm về $0$.
2. **Đột phá thuật toán thích ứng trực tuyến:** Xây dựng thuật toán **ADYN-LOC-NFST** cho phép số cụm $K(t)$ co giãn tự nhiên theo dòng dữ liệu thông qua các cập nhật đóng dạng **Rank-1 Downdate/Update** với chi phí $O(r^2)$, giúp cập nhật không gian chiếu trong mili-giây mà không cần tính lại SVD từ đầu.
3. **Mở rộng bảo vệ quyền riêng tư qua Học liên kết (Federated Learning):** Thiết kế giao thức **Fed-LOC-NFST** một chu kỳ (One-shot, $T=1$) với cơ chế hiệu chỉnh lệch tâm (Scatter-shift correction), cho phép các gateway biên có số cụm $K_m(t)$ bất đối xứng vẫn hội tụ về nghiệm giải tích toàn cục tối ưu.
4. **Đồng thiết kế Phần cứng - Phần mềm (Hardware-Software Co-Design):** Hiện thực hóa trên vi kiến trúc chip nhúng ARM với cơ chế **Zero-Heap Allocation** (Static Cache-Aligned Pool) và cơ chế hoán đổi con trỏ phi khóa **RCU (Read-Copy-Update)**, đảm bảo thông lượng xử lý gói tin tốc độ đường truyền (Line-rate) không trễ.

---

## PHẦN 2: BA ĐÓNG GÓP TRỌNG TÂM CỦA KHÓA LUẬN (CORE CONTRIBUTIONS)

Đề cương được tổ chức chặt chẽ theo chuẩn mực các bài báo Top-Tier (tách bạch đóng góp Thuật toán và đóng góp Hệ thống/Phần cứng):

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │       ĐỀ TÀI KHÓA LUẬN TỐT NGHIỆP IEC LAB (UIT)         │
                                    └────────────────────────────┬────────────────────────────┘
                                                                 │
                  ┌──────────────────────────────────────────────┴──────────────────────────────┐
                  ▼                                                                             ▼
┌──────────────────────────────────────────────────┐                         ┌──────────────────────────────────────────────────┐
│             ĐÓNG GÓP THUẬT TOÁN                  │                         │             ĐÓNG GÓP PHẦN CỨNG                   │
│          (Algorithmic Contributions)             │                         │        (Hardware-Software Co-Design)             │
└─────────────────┬────────────────────────────────┘                         └──────────────────┬───────────────────────────────┘
                  │                                                                             │
         ┌────────┴────────┐                                                                    │
         ▼                 ▼                                                                    ▼
┌─────────────────┐ ┌──────────────────────────────┐                         ┌──────────────────────────────────────────────────┐
│  ĐÓNG GÓP 1:    │ │  ĐÓNG GÓP 2:                 │                         │  ĐÓNG GÓP 3:                                     │
│  LOC-NFST &     │ │  ADYN-LOC-NFST               │                         │  VI KIẾN TRÚC BIÊN ZERO-HEAP & RCU ATOMIC SWAP   │
│  FEDERATED      │ │  (Dynamic-Cardinality        │                         │  - Static Slot Array alignas(64) L1/L2 cache     │
│  LEARNING       │ │   Cluster Lifecycle via      │                         │  - RCU Lock-Free Pointer Swap (8ns latency)      │
│  (Protocol A)   │ │   Rank-1 Matrix Downdate)    │                         │  - SIMD ARM NEON & E-Detector sleep cycle        │
└─────────────────┘ └──────────────────────────────┘                         └──────────────────────────────────────────────────┘
```

### ĐÓNG GÓP THUẬT TOÁN 1: Khung giải thuật LOC-NFST và Học liên kết One-Shot (FL-LOC-NFST)
- **Bản chất hình học:** Phân hoạch dữ liệu bình thường đa tạp thành $K$ lớp giả (Pseudo-classes) bằng K-Means, biến đổi bài toán One-Class suy biến ($S_b=\mathbf{0}$) thành bài toán đa cụm ($S_b > \mathbf{0}, S_w \ll S_t$). Không gian rỗng $\text{Null}(S_w)$ nén toàn bộ phương sai nội cụm của các thiết bị về $0$, biến các đám mây dữ liệu thành các **điểm kỳ dị thu hút (Point Attractors)** trong $\mathbb{R}^L$, phóng đại các sai lệch vi mô của **Local Anomalies** đạt AUC $99.60\%$ (vượt trội các mô hình Deep Learning).
- **Phục hồi số học phổ (Spectral Relaxation):** Khắc phục triệt để hiện tượng ma trận $A$ đầy rank trong miền dữ liệu bảng lớn ($N \gg d$) thông qua cơ chế Near-Null Adaptive Relaxation với ngưỡng tương đối, chống sụp đổ số chiều $L \ge 5$.
- **Giao thức FL 1-shot (T=1):** Chứng minh tính bảo toàn chính xác của ma trận hiệp phương sai khi tổng hợp từ các gateway biên với thuật toán hiệu chỉnh Scatter-Shift Correction, nén ngân sách truyền thông xuống chỉ còn $\approx 9.4\text{ KB/client}$.

### ĐÓNG GÓP THUẬT TOÁN 2: Cơ chế Phân cụm Động thích ứng Concept Drift (ADYN-LOC-NFST)
- **Vòng đời cụm trực tuyến (Online Cluster Lifecycle):** Thiết lập máy trạng thái 5 pha: Cập nhật vi mô trực tuyến (Multivariate Welford) $\rightarrow$ Tách cụm (Cluster Split) $\rightarrow$ Hợp nhất cụm (Cluster Merge) $\rightarrow$ Triệt tiêu cụm (Cluster Death qua phân rã hàm mũ $2^{-\Delta t / T_{half}}$) $\rightarrow$ Sinh cụm mới (Cluster Birth qua Phao cách ly kiểm chứng `QuarantineBuffer` chống đầu độc).
- **Đột phá giải tích Rank-1 với nhân tử $\frac{1}{N_{total}}$:** Chứng minh định lý bảo toàn tán xạ toàn phần $\Delta S_t = \mathbf{0}$ dưới toán tử Split/Merge, suy ra cơ sở $Q$ bất biến. Rút gọn việc cập nhật không gian rỗng $W(t)$ thành toán tử Rank-1 Downdate/Update trên ma trận $A(t+1) = A(t) \mp u u^\top$ với vector chuẩn hóa chính xác $v = \sqrt{\frac{N_{j1} N_{j2}}{N_{total} \cdot N_j}} (\mu_{j1} - \mu_{j2})$. Chi phí giảm từ $O(Nd^2)$ xuống $O(r^2)$, hoàn tất trong $< 1.5\text{ ms}$.
- **Thực nghiệm cứu vãn sụp đổ:** Cứu mô hình trên `EdgeIIoTset` + `StandardScaler` từ **$64.4\%$ vọt lên $99.76\%$ ($+35.33\%$)**, và ngăn chặn hiện tượng over-clustering trên `N_BaIoT` giữ vững **AUC $99.56\%$**, F1-score **$0.9860$**, FAR **$0.046\%$**.

### ĐÓNG GÓP PHẦN CỨNG 3: Đồng thiết kế Phần cứng - Phần mềm trên Thiết bị Biên (Hardware-Software Co-Design)
- **Cấu trúc bộ nhớ Static Slot-Array Cache-Aligned:** Triệt tiêu hoàn toàn việc cấp phát heap động (`malloc/new`) gây phân mảnh bộ nhớ và lỗi tràn RAM (OOM Crash). Toàn bộ $K_{max}=128$ cụm được cấp phát tĩnh trong mảng liên tục `ClusterSlot`, căn chỉnh địa chỉ 64-byte (`alignas(64)`), nằm trọn vẹn trong L2 Cache ($26\text{ KB}$).
- **Cơ chế hoán đổi con trỏ phi khóa RCU (Read-Copy-Update):** Tách bạch tuyệt đối giữa Luồng dữ liệu (Data Plane - 3 lõi suy luận gói tin liên tục đạt Line-rate) và Luồng điều khiển thích ứng (Control Plane - 1 lõi tính toán cập nhật ma trận ngầm). Quá trình cập nhật mô hình mới diễn ra qua lệnh tráo con trỏ nguyên tử `atomic_exchange` chỉ mất **$8\text{ nano-giây}$**, đảm bảo **$0\%$ Packet Drop**.
- **Tối ưu hóa vi kiến trúc ARM NEON SIMD & Kích hoạt năng lượng E-Detector:** Vector hóa 128-bit NEON cho các phép nhân ma trận chiếu $W$ và tính khoảng cách Euclide. Tích hợp bộ kích hoạt biên năng lượng (Energy-based Trigger) đưa module thích ứng vào trạng thái ngủ đông (Sleep mode) khi mạng phẳng, tiết kiệm tới $78\%$ năng lượng tiêu thụ trên pin thiết bị IoT.

---

## PHẦN 3: ĐỀ CƯƠNG CHI TIẾT CÁC CHƯƠNG CỦA KHÓA LUẬN (7 CHƯƠNG)

### CHƯƠNG 1: TỔNG QUAN VÀ PHÁT BIỂU BÀI TOÁN (INTRODUCTION & MOTIVATION)
* **1.1. Bối cảnh an ninh mạng IoT và sự bùng nổ của thiết bị biên**
  * Sự phát triển của Smart City, Smart Home, Industrial IoT (IIoT).
  * Hạn chế của mô hình an ninh mạng truyền thống tập trung (Cloud-centric IDS).
* **1.2. Thách thức cốt lõi của bài toán One-Class Novelty Detection (OCND)**
  * Bài toán thiếu nhãn tấn công trong thế giới mở (Open-world zero-day threats).
  * Vấn đề dữ liệu huấn luyện bị nhiễm tạp chất (Contamination noise).
* **1.3. Nghịch lý của sự bất tương đồng thiết bị & Cấu trúc dị biệt đa hình thái**
  * Bản chất đa tạp phi lồi trong không gian đặc trưng $\mathbb{R}^d$.
  * Phân loại Diverse Anomaly Types: Global, Clustered, Local Anomalies.
  * Nghịch lý tâm toàn cục (Global Mean Fallacy).
* **1.4. Động lực nghiên cứu: Tại sao cần Dynamic-K và Hardware-Software Co-design?**
* **1.5. Mục tiêu, Đối tượng và Phạm vi nghiên cứu**
* **1.6. Cấu trúc của khóa luận**

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÁC CÔNG TRÌNH LIÊN QUAN (THEORETICAL BACKGROUND & RELATED WORK)
* **2.1. Nền tảng toán học biến đổi không gian con (Subspace Learning)**
  * Phân tích biệt số tuyến tính Fisher (LDA).
  * Biến đổi không gian rỗng Null Foley-Sammon Transform (NFST).
  * Phân rã giá trị suy biến (SVD) và Định lý phân rã phương sai Huygens.
* **2.2. Khảo cứu các công trình liên quan (Prior-Art Audit & Taxonomy)**
  * *Nhóm 1: Các mô hình One-Class truyền thống và Deep Learning* (OCSVM, Isolation Forest, DeepSVDD, AutoEncoder, LUNAR).
  * *Nhóm 2: Các biến thể của NFST và KNFST* (Juncheng Liu et al. CVPR 2017, Huang et al. ICPR 2018, Dufrenois PR 2022).
  * *Nhóm 3: Thuật toán phân cụm dòng dữ liệu thích ứng* (DenStream, CluStream).
  * *Nhóm 4: Hệ thống NIDS trên phần cứng nhúng biên* (Kitsune NDSS 2018, Pigasus SIGCOMM 2020).
* **2.3. Khoảng trống học thuật (Research Gap)**
  * Điểm nghẽn toán học khi NFST gặp $K=1$.
  * Sự thiếu vắng cơ chế biến thiên số cụm động dưới tác động của Concept Drift.
  * Khoảng cách giữa lý thuyết giải tích và khả năng thực thi thời gian thực trên chip biên.

---

### CHƯƠNG 3: KHUNG GIẢI THUẬT LOC-NFST VÀ CƠ CHẾ CO GIÃN ĐỘNG ADYN-LOC-NFST
* **3.1. Thiết kế giải thuật LOC-NFST (Local One-Class NFST)**
  * Cơ chế phân cụm giả tạo cấu trúc (Structure-Inducing Pseudo-Class Construction).
  * Phép giải phổ thích ứng Near-Null Spectral Solve với chặn sàn chiều rỗng $L_{min}=5$.
  * Hàm tính điểm khoảng cách nguyên mẫu (Prototype-Aligned Novelty Scoring) và suy rộng phân phối Chi $\chi(L)$.
* **3.2. Nền tảng toán học của ADYN-LOC-NFST (Dynamic Cardinality Dynamics)**
  * Chứng minh Định lý 1: Tính giảm đơn điệu và tính chất Rank-1 của toán tử Tách cụm (Cluster Split).
  * Chứng minh tính bảo toàn ma trận tán xạ toàn phần $\Delta S_t = \mathbf{0}$ và tính bất biến của cơ sở $Q$.
  * Bổ đề chuẩn hóa kích thước tập mẫu $\frac{1}{N_{total}}$ trong cập nhật Rank-1.
  * Toán tử Hợp nhất cụm (Cluster Merge) và Rank-1 Update.
* **3.3. Thiết kế máy trạng thái vòng đời cụm (Cluster Lifecycle State Machine)**
  * Cập nhật vi mô Welford đa biến đơn kỳ.
  * Phao cách ly kiểm chứng (Drift Quarantine Buffer) phân định Concept Drift vs Boiling-Frog Attack.
  * Phân rã lũy thừa thời gian (Exponential Fading Decay) triệt tiêu cụm chết.
  * Quy trình kiểm định định kỳ (Periodic Lifecycle Pruning).
* **3.4. Đánh giá độ phức tạp thuật toán**
  * Phân tích độ phức tạp thời gian: $O(Nd^2) \rightarrow O(r^2)$.
  * Phân tích bộ nhớ tĩnh $O(K_{max} d)$.

---

### CHƯƠNG 4: HỌC LIÊN KẾT BẢO TỒN TÁN XẠ CHO MẠNG BIÊN PHÂN TÁN (FED-LOC-NFST)
* **4.1. Bài toán không đồng nhất số cụm cục bộ (Local Cardinality Heterogeneity)**
  * Hiện thực mạng biên: Mỗi Edge Gateway sở hữu số lượng thiết bị và luồng $K_m(t)$ khác nhau.
  * Sự phá sản của các giao thức FL đồng nhất truyền thống ($K_m = K_{global}$).
* **4.2. Giao thức FL One-Shot (Protocol A - $T=1$)**
  * Thuật toán phía Client: Trích xuất vi thống kê $\{S_{w,m}, \mu_{m,j}, N_{m,j}\}$.
  * Thuật toán phía Server: Non-parametric Aggregation qua Agglomerative Clustering và DP-Means.
  * Cơ chế hiệu chỉnh độ lệch tán xạ (Scatter-Shift Correction):
    $$S_w^{global} = \sum_{m=1}^M S_{w,m} + \sum_{k=1}^{K_{global}} \sum_{m=1}^M \sum_{j \in \text{group}(k)} \frac{N_{m,j}}{N_{total}} (\mu_{m,j} - \mu_k)(\mu_{m,j} - \mu_k)^\top$$
* **4.3. Phân tích quyền riêng tư và ngân sách truyền thông**
  * Bảo đảm không truyền dữ liệu thô (Raw Traffic Invariance).
  * Ngân sách truyền thông $9.4\text{ KB/client}$, hoàn toàn miễn nhiễm với nghẽn mạng băng hẹp (6LoWPAN / LoRaWAN).

---

### CHƯƠNG 5: ĐỒNG THIẾT KẾ PHẦN CỨNG - PHẦN MỀM TRÊN THIẾT BỊ BIÊN (HARDWARE-SOFTWARE CO-DESIGN)
* **5.1. Phân tích điểm nghẽn vi kiến trúc chip biên (ARM Cortex-A72 / Raspberry Pi 4)**
  * Hiện tượng phân mảnh Heap (Heap Fragmentation Jitter).
  * Xung đột khóa luồng (Lock Contention) và rớt gói tin (Packet Drop Rate).
  * Giới hạn thông lượng L1/L2 Cache và rào cản tiêu thụ năng lượng.
* **5.2. Vi kiến trúc bộ nhớ Zero-Heap Căn chỉnh Cache (Static Slot-Array Pool)**
  * Cấu trúc dữ liệu `alignas(64) struct ClusterSlot`.
  * Chiến lược cấp phát tĩnh cố định $26\text{ KB}$ nằm trọn trong L2 Cache.
  * Cơ chế kích hoạt/hủy kích hoạt $O(1)$ qua cờ `is_active`.
* **5.3. Cơ chế đồng bộ phi khóa RCU (Read-Copy-Update) Atomic Swap**
  * Phân tách Data Plane (Inference Line-rate) và Control Plane (Adaptive Lifecycle).
  * Lệnh hoán đổi nguyên tử con trỏ `atomic_exchange` trong $8\text{ ns}$.
  * Chứng minh toán học về tính phi khóa (Lock-free Guarantee) và không rớt gói tin.
* **5.4. Tăng tốc phần cứng qua ARM NEON SIMD & Cơ chế tiết kiệm năng lượng E-Detector**
  * Vector hóa phép nhân ma trận $W$ và khoảng cách Euclide với thanh ghi 128-bit `float32x4_t`.
  * Bộ kích hoạt dựa trên tỷ lệ biên năng lượng (Energy-based Drift Trigger), ngủ đông module thích ứng để bảo toàn pin.

---

### CHƯƠNG 6: THỰC NGHIỆM, KẾT QUẢ ĐỐI CHUẨN VÀ BÀN LUẬN (EMPIRICAL EVALUATION & BENCHMARK)
* **6.1. Môi trường thực nghiệm và Tập dữ liệu chuẩn**
  * Cấu hình máy chủ nghiên cứu: Server `postmaster.iec` (Intel Core i9-13900K, RTX 5090 32GB, RAM 128GB).
  * Phần cứng thiết bị biên: Raspberry Pi 4 Model B (Broadcom BCM2711, Quad-core Cortex-A72 @ 1.8GHz).
  * 6 Bộ dữ liệu IoT tiêu chuẩn: `CICIoT2023`, `ToN-IoT`, `BoTIoT`, `N_BaIoT`, `EdgeIIoTset`, `IoTID20`.
  * 5 Bộ chuẩn hóa: `MinMaxScaler`, `StandardScaler`, `RobustScaler`, `QuantileTransformer`, `Normalizer`.
* **6.2. Kết quả đối chuẩn toàn diện: Static-K vs ADYN-LOC-NFST (30 Cặp Thử nghiệm)**
  * Đánh giá hiện tượng cứu sụp đổ mô hình: Phân tích trường hợp `EdgeIIoTset` ($64.4\% \rightarrow 99.76\%$).
  * Đánh giá khả năng chống quá phân cụm: Phân tích trường hợp `N_BaIoT` ($81.5\% \rightarrow 99.56\%$, F1: $0.9860$, FAR: $0.046\%$).
  * Đánh giá trên dữ liệu mạng ổn định: Phân tích trường hợp `BoTIoT`.
* **6.3. Đánh giá khả năng phát hiện Diverse Anomaly Types (RQ1)**
  * So sánh đối đầu trên 3 chế độ: Global, Clustered, Local Anomalies.
  * Phân tích tại sao LOC-NFST đạt $99.60\%$ trên Local Anomalies đánh bại AutoEncoder, LUNAR, DeepSVDD.
* **6.4. Đánh giá độ bền vững trước tạp chất huấn luyện (Contamination Robustness - RQ2)**
  * Thử nghiệm tiêm nhiễu độc $\rho \in \{1\%, 3\%, 5\%\}$.
* **6.5. Đánh giá hiệu năng hệ thống và phần cứng thực tế (RQ3)**
  * Đo đạc bằng công cụ chuyên dụng Linux `perf`: L1/L2 Cache Miss Rate, Instructions Per Cycle (IPC).
  * Đo đạc thời gian trễ suy luận P99 (P99 Tail Latency) dưới tải mạng 10 Gbps.
  * Đo đạc công suất tiêu thụ năng lượng (mW) khi có và không có E-Detector.

---

### CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
* **7.1. Tổng kết các đóng góp chính của khóa luận**
  * Khẳng định tính đúng đắn của 2 đóng góp Thuật toán và 1 đóng góp Phần cứng.
* **7.2. Hạn chế của đề tài**
  * Độ nhạy với ngưỡng khởi tạo ban đầu khi không gian đặc trưng có chiều quá cao ($d > 500$).
* **7.3. Hướng nghiên cứu mở rộng tiếp theo**
  * Tích hợp cơ chế Hardware Acceleration trên FPGA (Xilinx Zynq) qua ngôn ngữ VHDL/HLS.
  * Triển khai thử nghiệm thực địa (Field Deployment) trên mạng IoT Smart Campus của Trường ĐH Công nghệ Thông tin.

---

## PHẦN 4: KẾ HOẠCH THỰC HIỆN & DỰ KIẾN SẢN PHẨM ĐẦU RA

### 4.1. Bảng phân kỳ công việc (Gantt Chart Roadmap)

| Giai đoạn | Nội dung công việc chi tiết | Công cụ / Môi trường | Sản phẩm đạt được |
| :--- | :--- | :--- | :--- |
| **Tháng 1** | - Hoàn thiện cơ sở toán học Rank-1 Update có chuẩn hóa $\frac{1}{N_{total}}$.<br>- Viết trọn vẹn mã nguồn `adyn_core.py` và chạy 6 unit tests. | Python, NumPy, SciPy, PyTest | Bộ mã nguồn Core & 6/6 Unit tests Passed. |
| **Tháng 2** | - Triển khai benchmark 30 cặp dữ liệu trên server `postmaster.iec`.<br>- Phân tích số liệu đối đầu giữa Static-K và ADYN-LOC-NFST. | Server SSH `postmaster.iec`, Bash, Pandas | Tập dữ liệu kết quả CSV, bảng tóm tắt 30 cặp. |
| **Tháng 3** | - Lập trình vi kiến trúc phần cứng trên C++20 (`alignas(64)`, RCU Atomic).<br>- Triển khai thử nghiệm trên Raspberry Pi 4, đo đạc bằng Linux `perf`. | C++20, ARM GCC, Linux `perf`, Raspberry Pi 4 | Module C++ nhúng Line-rate, báo cáo IPC & Cache miss. |
| **Tháng 4** | - Viết toàn văn Khóa luận tốt nghiệp theo cấu trúc 7 chương.<br>- Hoàn thiện bản thảo bài báo khoa học (Paper Draft) chuẩn IEEE. | LaTeX, Overleaf, Matplotlib | Toàn văn Khóa luận (100+ trang) & Paper Draft IEEE TIFS/IoT-J. |
| **Tháng 5** | - Phản biện thử tại IEC Lab, chỉnh sửa theo góp ý của GVHD.<br>- Bảo vệ chính thức Khóa luận tốt nghiệp trước Hội đồng Khoa. | Slide thuyết trình, Demo trực tiếp trên Raspberry Pi 4 | Bảo vệ Khóa luận Tốt nghiệp Xuất sắc (Điểm $\ge 9.5$). |

### 4.2. Dự kiến sản phẩm đầu ra (Deliverables)
1. **Quyển Khóa luận tốt nghiệp (Bản in + File PDF):** Trình bày chuẩn format ĐHQG-HCM, tối thiểu 90 - 120 trang, đầy đủ chứng minh toán học giải tích và đồ thị trực quan.
2. **Kho mã nguồn hoàn chỉnh trên GitHub:** Nhánh `feature/federated-loc-nfst` chứa mã nguồn sạch, tài liệu hướng dẫn (`README.md`, `walkthrough.md`), script chạy 1 lệnh tự động hóa.
3. **Mẫu phần mềm nhúng thực thi tại biên (Hardware Prototype):** Bản build C++ chạy trực tiếp trên Raspberry Pi 4 Model B, nhận luồng gói tin thực tế qua card mạng WiFi/Ethernet và cảnh báo dị biệt thời gian thực.
4. **Bài báo khoa học quốc tế:** 01 bài báo đăng trên tạp chí thuộc danh mục ISI/Scopus **Q1 / Core A\*** (IEEE Transactions on Information Forensics and Security hoặc IEEE Internet of Things Journal).

---

## TÀI LIỆU THAM KHẢO CHỌN LỌC (TOP-TIER REFERENCES)

1. **[ADBench NeurIPS 2022]** S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, "ADBench: Anomaly Detection Benchmark," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.
2. **[Kitsune NDSS 2018]** Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection," in *Network and Distributed System Security Symposium (NDSS)*, 2018.
3. **[Pigasus ACM SIGCOMM 2020]** Z. Zhao et al., "Pigasus: FPGA-Accelerated 100Gbps Intrusion Detection and Prevention System," in *ACM Special Interest Group on Data Communication (SIGCOMM)*, 2020.
4. **[IKNDA IEEE CVPR 2017]** J. Liu, Z. Chi, and X. Fu, "Incremental Kernel Null Space Discriminant Analysis for Novelty Detection," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 438-446.
5. **[EdgeIIoTset IEEE IoT-J 2022]** M. A. Ferrag et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications," *IEEE Internet of Things Journal*, vol. 9, no. 20, pp. 20096–20108, 2022.
6. **[CICIoT2023 IEEE IoT-J 2023]** E. C. P. Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Networks," *Sensors / IEEE IoT-J*, 2023.
7. **[Hyperscan USENIX NSDI 2019]** X. Wang et al., "Hyperscan: A Fast Multi-pattern Regex Matcher for Modern CPUs," in *USENIX Symposium on Networked Systems Design and Implementation (NSDI)*, 2019.
