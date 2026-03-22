# Giải Thích Vấn Đề AUC = 0.0 Trên Tập Dữ Liệu BoTIoT và Giải Pháp Xử Lý

Tài liệu này giải thích bản chất toán học của lỗi khiến mô hình One-Class NFST (OC-NFST) trả về **AUCROC = 0.0** và **Accuracy = 100%** (nếu tập test 99.9% anomaly) hoặc **50%** (nếu tập test balance) khi chạy trên dataset BoTIoT.

---

## 1. Hiện Tượng Chẩn Đoán (The Symptoms)

Khi huấn luyện OC-NFST trên tập dữ liệu BoTIoT (DDoS / IoT Botnet), bạn nhận được các chỉ số sau:
*   **AUCROC = 0.0**: Điều này có nghĩa là Khả năng xếp hạng (Ranking) của mô hình bị **đảo ngược hoàn hảo**. Tức là: Điểm bất thường (Anomaly Score / Distance) của các mẫu Anomaly lại **THẤP HƠN** điểm bất thường của các mẫu Normal. Dữ liệu Anomaly bị mô hình hiểu lầm là "rất Normal", trong khi dữ liệu Normal lại bị chấm là "bất thường".
*   **Accuracy = 100%**: Mặc dù xếp ngược, nhưng do tập Test có thể chứa 99.9% là mẫu Anomaly, hàm `Youden's J` của ROC dồn ngưỡng (threshold) đến mức mô hình dự đoán **tất cả các mẫu test đều là Anomaly**. Khi dự đoán mù tất cả là Anomaly trên một tập dữ liệu gần như 100% Anomaly, độ chính xác (Accuracy) giả tạo sẽ đạt 100%. Nếu bạn dùng tập test cân bằng (balance 50-50), Accuracy sẽ giảm xuống đúng 50%.

## 2. Nguyên Nhân Kỹ Thuật (Mathematical Blind Spot)

Hãy đi sâu vào quá trình tạo **Không Gian Null Space (NFST)**:

1. **Phép chiếu Q (Principal Subspace)**
   Thuật toán bắt đầu bằng cách tìm các vector riêng lớn nhất ($Q$) của ma trận hiệp phương sai ($S_t$) trên tập huấn luyện (Training Set). Trong bài toán One-Class, tập huấn luyện **chỉ chứa dữ liệu Normal**. 
   $Q$ chính là không gian chứa toàn bộ các sự biến thiên (phương sai) tự nhiên của dữ liệu Normal.

2. **Đặc Thù Của BoTIoT Data**
   Dữ liệu BoTIoT chứa các dạng tấn công mang giá trị cực kỳ lớn (ví dụ: `rate`, `pkts`, `bytes` tăng vọt gấp 10,000 lần so với mức thông thường). 
   Tuy nhiên, trong tập huấn luyện (chỉ Normal), phương sai của các cột này gần như bằng `0` hoặc không đáng kể so với các cột khác. Do đó, ma trận $Q$ **sẽ tự động loại bỏ (zero-out)** các chiều dữ liệu/hướng đi này vì chúng có eigenvalue (giá trị riêng) bằng 0 trên tập Normal.

3. **Cái Bẫy Null Space**
   NFST sau đó tìm không gian rỗng $W$ (Null Space) nằm TRONG không gian con $Q$.
   Khi một mẫu **Anomaly** khổng lồ xuất hiện lúc test, nó bay vào mô hình. Do độ lớn khổng lồ của nó nằm ở những chiều dữ liệu bị Normal bỏ qua (các chiều trực giao với $Q$), khi nhân ma trận:
   $$ X_{anom\_projected} = X_{anom} \times Q \times B $$
   Toàn bộ giá trị lớn bất thường của nó **bị triệt tiêu hoàn toàn thành 0** (do nhân với ma trận Q không chứa hướng đó).
   Mẫu Anomaly khủng khiếp bỗng dưng trở thành một điểm hoàn hảo ngay tại gốc tọa độ của Null Space (Khoảng cách = 0.0).

4. **Nghịch Lý Kết Quả**
   * Khoảng cách Null Space của Anomaly = `0.0` (Bị triệt tiêu).
   * Khoảng cách Null Space của Normal test data = `0.001` (Do có chút nhiễu tự nhiên trong không gian $Q$).
   
   👉 Vô hình trung: Mẫu Normal lại có điểm bất thường **CAO HƠN** mẫu Anomaly khổng lồ. Từ đó AUC rớt thẳng xuống 0.0.

---

## 3. Giải Pháp Khắc Phục (The Solution)

Để giải quyết điểm mù (Blind Spot) chết người này của OC-NFST (khi chỉ huấn luyện trên Normal class), ta BẮT BUỘC phải đo đạc những thứ bị $Q$ vứt bỏ.

Chúng ta thêm **Reconstruction Error (Khoảng Cách Trực Giao - Orthogonal Distance)** vào tổng điểm Anomaly Score.

Công thức gốc (chỉ tính khoảng cách bên trong Null space):
$$ Score_{cũ} = || (X - center) \times W ||^2 $$

Công thức hoàn chỉnh mới được cập nhật trong file `OC_NFST_memory_optimized.py`:
1. **Tìm phần bị Q loại bỏ (Khoảng cách trực giao)**: 
   Ta dự đoán $X$ vào không gian $Q$, sau đó khôi phục lại (Reconstruct) xem bị mất mát bao nhiêu.
   $$ X_{recon} = (X - center) \times Q \times Q^T $$
   $$ Lỗi\_Trực\_Giao = || (X - center) - X_{recon} ||^2 $$

2. **Tổng hợp Anomaly Score Mới**:
   $$ Score_{mới} = \sqrt{ || (X - center) \times W ||^2 + \alpha \times Lỗi\_Trực\_Giao } $$

### Kết Quả Của Giải Pháp:
Với công thức mới:
* Mẫu Normal sẽ có cả 2 khoảng cách (Null và Orthogonal) cực kỳ nhỏ.
* Mẫu Anomaly của BoTIoT sẽ có khoảng cách Orthogonal **KẾT XÙ** (vì toàn bộ giá trị cực khủng 10,000 của nó nằm bên ngoài không gian Normal $Q$ và tạo thành Lỗi Trực Giao khổng lồ).
* Điểm bất thường của Anomaly giờ sẽ lớn hơn Normal rất rất nhiều.
* Khả năng xếp hạng được khôi phục chính xác $\rightarrow$ AUCROC sẽ từ 0.0 tăng vọt.

Đó là lý do chúng ta buộc phải đổi hàm `compute_scores` trong code để kết hợp cả 2 hình thái khoảng cách này!
