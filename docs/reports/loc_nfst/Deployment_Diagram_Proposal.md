# A* Conference Level Deployment Diagram Proposal: LOC-NFST

Để đạt được tiêu chuẩn của các hội nghị A* (CCS, USENIX Security, NDSS), một Deployment Diagram không thể chỉ là luồng mũi tên "A $\rightarrow$ B $\rightarrow$ C". Biểu đồ cần phải tỏa ra **sức nặng toán học (Mathematical Grounding)** lồng ghép chặt chẽ vào **kiến trúc hệ thống (System Architecture)**. 

Dưới đây là một bản thiết kế chi tiết để bạn có thể vẽ lại bằng draw.io, Visio hoặc TikZ. Khung hình (Canvas) nên được chia làm 3 cột (hoặc 3 tầng) chính để tạo chiều sâu:

---

## Tầng 1 (Bên trái): Operational Environment \& Data Preparation (Môi trường \& Tiền xử lý)
*Nơi phát sinh dữ liệu, bắt gói tin và xử lý quy chuẩn.*

1. **IoT End-devices (Nodes):**
   * **Hình ảnh:** Icon các thiết bị Smart Home (N-BaIoT), Industrial Sensors (ToN-IoT).
   * **Bố cục:** Rải rác, thể hiện sự hỗn mang các nền tảng (heterogeneous data sources).
   * **Ghi chú Toán học:** Ký hiệu luồng dữ liệu thô $Raw \ Traffic \ Streams$.
2. **Feature Extraction (Tại Edge Gateway):**
   * **Hình ảnh:** Một server gờ nổi thu thập packet mạng.
   * **Logic Nội bộ:** Hộp (Box) mỏng ghi "Flow Aggregation \& Encoding" (biến gói tin mạng thành số).
3. **Flow-level Spatial Alignment \& Normalization (Chuẩn hóa không gian Luồng):**
   * **Đừng viết sơ sài là "Data Processing", hãy biến nó thành một khối chuyên sâu.** 
   * **Hình ảnh:** Một Cylinder (Trụ dữ liệu) đi qua một phễu lọc 3 tầng với các ký hiệu thống kê.
   * **Nội dung bên trong (3 Layers):**
     * *Layer 1: Heavy-tailed Clipping \& Imputation.* (Mạng IoT thường có các luồng Bytes/Packet đâm thẳng lên Vô cực do rác mạng hoặc DDoS. Dùng hàm cắt gọt chặn trên/dưới để ngăn tràn số liệu). Ký hiệu: $X \leftarrow \max(\min(X, \tau_{max}), -\tau_{max})$.
     * *Layer 2: Categorical/Protocol Embedding.* (Biểu diễn các đại lượng rời rạc như TCP/UDP, Port thành vector không gian).
     * *Layer 3: Variance Stabilization \& Scaling.* (Đưa toàn bộ mọi Feature về chung một tham chiếu để khoảng cách Euclidean sau này không bị thiên vị). Ký hiệu thống kê: $X \leftarrow (X - \mu) \odot \Sigma^{-1}$.
   * **Sự liên kết Toán học:** Bước này biến đổi mớ hỗn độn (Chaos) từ mạng thô thành một **Dense Feature Matrix** cấu trúc chặt chẽ ($X \in \mathbb{R}^{d}$) thỏa mãn điều kiện tĩnh (stationary) để các phép chiếu Không gian (SVD/NFST) ở Tầng 2 không bị sụp đổ vì ma trận suy biến.

---

## Tầng 2 (Ở giữa): Offline Training / Cloud Phase (Nền tảng Toán học)
*Mô tả cách thuật toán học được không gian biểu diễn chỉ từ dữ liệu sạch.*

Tầng này chứa một khối hình hộp chữ nhật lớn gọi là **LOC-NFST Core Engine**. Bên trong hộp này, chia làm 2 phase toán học:

1. **Surrogate Structure Formulation (Tạo cấu trúc giả):**
   * **Hình ảnh:** Một cụm điểm dữ liệu lộn xộn màu xanh (Benign Data). Mũi tên dẫn đến K-Means $c$-clusters $\Rightarrow$ Các điểm được gom thành từng cụm nhỏ.
   * **Ghi chú Toán học:** 
     $$ \mathcal{D}_{train} \rightarrow \bigcup_{j=1}^{c} C_j $$
     Tính Toán Ma Trận Scatter: $S_w = \Sigma(x - \mu_j)(x - \mu_j)^T$ và $S_b = \Sigma n_j(\mu_j - \mu)(\mu_j - \mu)^T$.

2. **Null-Space Projection (Chiếu không gian SVD):**
   * **Hình ảnh:** Ma trận Dữ liệu $\mathbf{X}$ đi qua phép SVD Truncation, loại bỏ bộ lọc $\epsilon$, tạo ra không gian chiếu mới nhỏ gọn hơn (Diminished planes).
   * **Ghi chú Toán học (Rất quan trọng để ăn điểm A*):**
     $$ X_c = U_r \Sigma_r V_r^T \quad \xrightarrow{\text{Filter } \lambda \le \epsilon} \quad \text{Basis } B_\epsilon $$
     Kết xuất ra Ma trận chiếu $W_{opt} \in \mathbb{R}^{d \times L}$ (L là bậc rank tối ưu, $L \ll d$). 

3. **Empirical Threshold Calibration (Định chuẩn Ngưỡng Thực nghiệm):**
   * **Trọng tâm (Rất quan trọng):** Bước tính Threshold này bắt buộc phải nằm bên trong luồng Offline Training Cloud, vì nó được estimate từ biểu đồ phân phối của tập Train (Benign Training Scores $s_i$), hoàn toàn độc lập với luồng test!
   * **Hình ảnh:** Vẽ một dải phân phối phân cực (Density curve/Histogram) của các điểm $s_i$. Kẻ một vạch dọc mỏng (vạch $\tau$) chia cắt vùng đuôi phải (Right-tail).
   * **Ghi chú Toán học:** 
     $$ \tau = Q_{1-\rho}\big(\{s_i\}_{i=1}^{n}\big) $$
   *(Tổng kết Tầng 2: Hệ thống Offline Cloud sau khi hoàn thành sẽ "đóng gói" 3 bộ tham số cấu hình: $W_{opt}$ (Ma trận lọc), $p_j$ (Các tâm cụm đại diện), và $\tau$ (Ngưỡng ranh giới) truyền tải xuống cho Thiết bị Edge ở Tầng 3 bằng nét gạch đứt `dashed-line`).*

---

## Tầng 3 (Bên phải): Online Inference / Edge Deployment (Suy luận mức thiết bị)
*Cách thức xử lý Real-time cực nhẹ trên Edge.*

Khối này nhận đầu vào $X_{test}$ từ Tầng 1 và nhận Ma trận $W_{opt}$ từ Tầng 2.

1. **Lightweight Embedding:**
   * **Hình ảnh:** Phép nhân vector tuyến tính.
   * **Thông tin Toán học:** $z_{test} = W_{opt}^T x_{test}$ ($O(dL)$ Complexity - Biểu thị độ phức tạp siêu thấp để khoe tính Lightweight).
2. **Prototype Scoring:**
   * **Hình ảnh:** Một không gian 2D với các cụm tròn (Pseudo-classes) có tâm là các chấm đậm (Centroids $p_j = W_{opt}^T \mu_j$). Điểm Test màu đỏ/xanh (Anomalous/Normal) rơi vào không gian này.
   * **Thông tin Toán học:** Phương trình tính điểm ranh giới ngây thơ nhưng hiệu quả:
     $$ \text{Score}(x) = \min_j \|z_{test} - p_j\|_2 $$
3. **Decision Layer (So sánh Real-time cực nhẹ):**
   * **Hình ảnh:** Một khối Logic nhị phân `IF...ELSE`, có vạch kẻ Ngưỡng **$\tau$** đã được nhúng (embedded) sẵn từ server đưa xuống. Điểm $Test \ Score$ chạy vào đập vào vạch này giống như một cái cổng chặn dòng nước.
   * **Ký hiệu Logic học thuật:**
     $$ f(x_{test}) = \begin{cases} 1 \ (\text{Intrusion}), & \text{if } Score(x_{test}) > \tau \\ 0 \ (\text{Normal}), & \text{if } Score(x_{test}) \le \tau \end{cases} $$
   * Mũi tên 2 ngã rẽ lối ra: **"Normal Flow"** (Traffic được pass) và **"Anomaly Alert Triggered"** (Còi báo động/DDoS alert).

---

## Mẹo thiết kế màu sắc (Design Aesthetics)
1. **Phân cực Không gian:** Hãy dùng màu Xanh Xám (Blue-Grey) cho Tầng Offline Training (thể hiện Compute to, Server) và màu Xanh Lá Sáng/Cam cho khối Online Inference (thể hiện tốc độ, Real-time trên Edge).
2. **Đường nét (Lines):** Dùng nét đứt (dashed line) để mô tả quá trình truyền tham số học được (truyền $W_{opt}, p_j, \tau$ từ Training Server gởi xuống Edge), và dùng nét liền mạch dày (solid bold line) cho đường đi của Data traffic $X_{test}$.
3. **Typography:** Font chữ cho phần Toán học MỚI NHẤT thống nhất phải dùng bộ font chuẩn (như Computer Modern trong LaTeX). 

## Cách thuyết minh trong bài (Text Alignment)
Dưới Diagram, kèm dòng Caption:
> **Figure X:** Systematic architecture and mathematical workflow of LOC-NFST. The framework decouples the computationally intensive surrogate structure induction and SVD truncation (Offline Phase) from the extremely lightweight distance-based inference (Online Edge Phase), enforcing a hyperparameter-robust decision boundary via threshold $\tau$.
