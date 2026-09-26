# BẢN CHẤT CỦA SỰ KHÁC BIỆT THIẾT BỊ IOT, DIVERSE ANOMALY TYPES VÀ TẠI SAO PHẢI PHÂN CỤM TRƯỚC KHI CHIẾU KHÔNG GIAN RỖNG (LOC-NFST)

**Tác giả:** Nghiên cứu sinh / Tác giả công trình LOC-NFST  
**Đơn vị:** IEC Lab, Trường Đại học Công nghệ Thông tin (ĐHQG-HCM)  
**Mục đích:** Giải thích tường minh từ bản chất vật lý mạng đến nền tảng giải tích đại số tuyến tính; làm rõ triết lý cốt lõi của phương pháp Local One-Class Null Foley-Sammon Transform (LOC-NFST).

---

## MỞ ĐẦU: CÂU HỎI TỰ VẤN CỐT LÕI

Khi viết báo khoa học, chúng ta thường viết những câu mang tính quy ước như:
> *"Vì dữ liệu IoT đến từ nhiều thiết bị không đồng nhất (camera, cảm biến, khóa thông minh...) với nhiều giao thức khác nhau, và mạng phải đối mặt với diverse anomaly types, nên chúng tôi phân cụm dữ liệu bình thường thành $K$ pseudo-classes trước khi áp dụng NFST."*

Tuy nhiên, nếu một Reviewer khó tính của IEEE TIFS hoặc ACM CCS hỏi:
1. **"Các thiết bị khác nhau thì sao chứ?"** Bản chất vật lý của lưu lượng mạng biểu diễn lên không gian đặc trưng toán học $\mathbb{R}^d$ tạo ra hình thái hình học gì?
2. **"Nếu không phân cụm mà để nguyên dữ liệu bình thường làm 1 khối ($K=1$) thì chuyện gì xảy ra với toán học của NFST?"**
3. **"Diverse anomaly types thực chất là gì về mặt topo hình học?"** Tại sao phương pháp chiếu không gian rỗng (Null Space Projection) lại giải quyết được triệt để bài toán này trong khi các mô hình nổi tiếng như OCSVM, DeepSVDD, AutoEncoder hay LUNAR lại gặp khó khăn?

Báo cáo này bóc tách cặn kẽ 4 tầng bản chất: **Vật lý mạng $\rightarrow$ Hình học không gian đặc trưng $\rightarrow$ Đại số tuyến tính của NFST $\rightarrow$ Cơ chế phát hiện dị biệt.**

---

## 1. TẦNG 1: "CÁC THIẾT BỊ KHÁC NHAU THÌ SAO CHỨ?" — BẢN CHẤT VẬT LÝ VÀ HÌNH THÁI HÌNH HỌC

### 1.1. Bản chất Vật lý mạng: Sự bất tương đồng về cấu trúc luồng (Flow Profile Heterogeneity)

Một mạng IoT thực tế (Smart Home, Smart Factory, Smart City) không bao giờ có một phân phối lưu lượng đồng nhất (Homogeneous Distribution). Hãy nhìn vào 4 thiết bị điển hình:

| Thiết bị | Giao thức truyền | Kích thước gói tin (Packet Length) | Tần suất gửi (Inter-Arrival Time) | Hướng truyền & Cờ TCP/UDP |
| :--- | :--- | :--- | :--- | :--- |
| **IP Camera** | UDP / RTSP | Lớn (1000 – 1500 bytes), xấp xỉ MTU | Liên tục, băng thông cực cao (Mbps), độ trễ nhỏ | Luồng 1 chiều áp đảo (Outbound video stream) |
| **Cảm biến nhiệt độ (DHT22 / MQTT)** | TCP / MQTT | Nhỏ (50 – 120 bytes) | Rời rạc, chu kỳ cố định (vd: 30 giây/lần) | Trao đổi song công ngắn (Publish + ACK) |
| **Khóa cửa thông minh (Smart Lock)** | TCP / TLS | Trung bình (200 – 500 bytes) | Bất thường theo sự kiện (Event-driven khi mở cửa) | Bắt tay mã hóa ngắn, đa số thời gian ở trạng thái Idle |
| **Hệ thống HVAC Công nghiệp** | Modbus / CoAP | Nhỏ, cố định từng trường thanh ghi (Register polling) | Tuần hoàn mili-giây cực kỳ nghiêm ngặt | Client-Server Polling cố định |

### 1.2. Biểu diễn toán học trên không gian đặc trưng $\mathbb{R}^d$: Đa tạp phi lồi, phân mảnh (Disconnected Non-Convex Manifolds)

Mỗi mẫu lưu lượng mạng được trích xuất thành vector $d$ chiều $x \in \mathbb{R}^d$ gồm các đặc trưng:
$$x = \big[ \text{Packet Length Mean, Std, Flow Duration, Byte Rate, Packet Rate, TCP Flags Ratio, ...} \big]^\top$$

Khi các thiết bị khác nhau hoạt động, trên không gian $\mathbb{R}^d$:
- Dữ liệu camera tạo thành một "đám mây" (cluster) nằm ở góc: **Packet Size cao, Rate cao**.
- Dữ liệu cảm biến tạo thành một đám mây nằm ở góc hoàn toàn đối lập: **Packet Size cực nhỏ, Duration cực ngắn, Idle time cực cao**.
- Dữ liệu khóa thông minh tạo thành một đám mây rời rạc khác.

$$\Longrightarrow \text{Tập hợp toàn bộ lưu lượng bình thường } \mathcal{D}_{normal} \text{ KHÔNG PHẢI là một khối cầu Gauss đơn mode (Unimodal Gaussian).}$$
Nó là một **tập hợp của nhiều đa tạp con rời rạc, phi lồi (Disconnected Sub-manifolds)** lơ lửng trong không gian $d$ chiều:
$$\mathcal{M}_{normal} = \bigcup_{k=1}^K \mathcal{M}_k, \quad \text{với } \mathcal{M}_i \cap \mathcal{M}_j \approx \emptyset \quad (i \ne j)$$

### 1.3. Nghịch lý của Tâm toàn cục (The Global Mean Fallacy)

Nếu ta coi tất cả các thiết bị này chung một lớp bình thường duy nhất ($K=1$) và tính kỳ vọng toàn cục:
$$\mu_{global} = \frac{1}{N} \sum_{i=1}^N x_i$$
- $\mu_{global}$ là điểm nằm ở **chính giữa khoảng trống chân không (Empty Space)** giữa đám mây của Camera và đám mây của Cảm biến!
- Trong thực tế mạng, **hoàn toàn KHÔNG CÓ bất kỳ gói tin bình thường nào có đặc trưng nằm gần $\mu_{global}$ cả!**
- Nếu một thuật toán học ranh giới đơn khối (như One-Class SVM hay DeepSVDD bán kính cầu $R$), nó buộc phải tạo ra một hình cầu hoặc một siêu bao lồi (Convex Hull) khổng lồ bao quanh tất cả các cụm này.
- **Hậu quả chí tử:** Toàn bộ không gian rỗng nằm giữa các cụm thiết bị bị gộp nhầm thành "vùng an toàn bình thường". Kẻ tấn công chỉ cần ẩn nấp vào vùng chân không này là hoàn toàn tàng hình!

---

## 2. TẦNG 2: BẢN CHẤT TOÁN HỌC CỦA VIỆC PHÂN CỤM TRONG ONE-CLASS NFST

Đây là câu hỏi quan trọng nhất về mặt giải tích đại số: **Tại sao bắt buộc phải chia thành $K$ cụm giả (Pseudo-classes) thì NFST mới hoạt động được?**

### 2.1. Sự phá sản của NFST cổ điển khi $K=1$ (The Degeneracy of One-Class NFST)

Mục tiêu của biến đổi Null Foley-Sammon (NFST) là tìm các hướng chiếu $w$ sao cho:
1. **Phương sai nội lớp bị triệt tiêu hoàn toàn về 0:** $w^\top S_w w = 0$.
2. **Phương sai liên lớp đạt giá trị dương cực đại:** $w^\top S_b w > 0$.

Bây giờ, giả sử chúng ta **KHÔNG phân cụm** ($K=1$), chỉ có một lớp bình thường duy nhất. Hãy xem ma trận tán xạ liên lớp $S_b$ và toàn phần $S_t$:
- Tán xạ liên lớp: Vì chỉ có $K=1$ lớp, tâm của lớp đó chính là tâm toàn cục $\mu_1 = \mu_{global}$.
  $$S_b = \sum_{k=1}^1 \frac{N_k}{N} (\mu_k - \mu_{global})(\mu_k - \mu_{global})^\top = \mathbf{0}_{d \times d}$$
  **Ma trận tán xạ liên lớp triệt tiêu bằng 0 tuyệt đối!**
- Theo định lý phân rã phương sai toàn phần Huygens:
  $$S_t = S_w + S_b = S_w + \mathbf{0} \equiv S_w$$
  **Ma trận tán xạ nội lớp đồng nhất với ma trận tán xạ toàn phần!**
- Bây giờ xét không gian con chính $Q$ trích từ SVD của $S_t$ ($Q^\top S_t Q = \Lambda_t$). Ma trận rút gọn $A$ của NFST trở thành:
  $$A = Q^\top S_w Q = Q^\top S_t Q = \Lambda_t = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_r)$$
- Vì dữ liệu bảng thực tế có phương sai trên mọi chiều cơ sở ($S_t$ có đầy đủ rank $r$), tất cả các trị riêng $\lambda_i > 0$.
- **Không gian nghiệm rỗng của $A$:**
  $$\text{Null}(A) = \{ v \in \mathbb{R}^r \mid A v = \mathbf{0} \} = \emptyset \quad (\text{hoặc } \{ \mathbf{0} \})$$
  $$\text{dim}(\text{Null}(A)) = 0$$

> [!CAUTION]
> **ĐỊNH LÝ SỤP ĐỔ (Degeneracy Theorem):**  
> Nếu không phân hoạch lớp bình thường thành ít nhất $K \ge 2$ lớp giả, **NFST không thể tồn tại về mặt toán học**. Ma trận $S_b$ bằng 0 và không gian nghiệm rỗng $\text{Null}(A)$ có số chiều bằng 0. Mô hình bị liệt hoàn toàn!

### 2.2. Phân cụm giả (Pseudo-Classes): Cứu tinh toán học kiến tạo không gian nghiệm rỗng

Khi ta dùng K-Means để chia tập bình thường thành $K$ cụm giả $\{C_1, C_2, \dots, C_K\}$ với các tâm $\mu_k$ khác biệt:
1. **Kiến tạo $S_b$ dương xác định trên không gian tâm cụm:**
   $$S_b = \sum_{k=1}^K \frac{N_k}{N} (\mu_k - \mu_{global})(\mu_k - \mu_{global})^\top \ne \mathbf{0}$$
   Rank của $S_b$ đạt tới $\min(K-1, d) > 0$.
2. **Thu nhỏ ma trận tán xạ nội cụm $S_w$:**
   Thay vì đo khoảng cách từ điểm $x$ đến tâm toàn cục $\mu_{global}$ xa xôi, $S_w$ bây giờ chỉ đo độ lệch cục bộ từ $x$ đến tâm cụm gần nhất $\mu_k$:
   $$S_w = \sum_{k=1}^K \frac{1}{N} \sum_{x \in C_k} (x - \mu_k)(x - \mu_k)^\top \ll S_t$$
3. **Mở ra không gian nghiệm rỗng (Null Space Opening):**
   Vì $S_w \lneq S_t$, ma trận rút gọn $A = Q^\top S_w Q$ xuất hiện các hướng mà phương sai nội cụm bị triệt tiêu ($A v \approx \mathbf{0}$) trong khi độ phân tách liên cụm vẫn được bảo toàn ($Q^\top S_b Q > 0$).
   Ta thu được ma trận chiếu $W \in \mathbb{R}^{d \times L}$ với $L \ge 1$ chiều nghiệm rỗng thực sự!

### 2.3. Ý nghĩa hình học: Co cụm thành các "Điểm kỳ dị" (Point Attractors)

Trong không gian gốc $\mathbb{R}^d$, mỗi cụm thiết bị là một đám mây có thể tích và phương sai.  
Nhưng khi chiếu qua ma trận không gian rỗng $W$ ($z = W^\top x$):
- Vì $W$ nằm trong $\text{Null}(S_w)$, **mọi phương sai nội cụm dọc theo các hướng chiếu này đều bị nén về 0!**
- Toàn bộ đám mây dữ liệu phức tạp của Camera trong $\mathbb{R}^d$ bị **co sụp (collapse) thành đúng một điểm duy nhất** $c_{camera} = W^\top \mu_{camera}$ trong không gian $\mathbb{R}^L$.
- Tương tự, toàn bộ dữ liệu Cảm biến co sụp thành điểm $c_{sensor} = W^\top \mu_{sensor}$.
- Không gian bình thường trong $\mathbb{R}^L$ bây giờ không còn là một khối mây bầy hầy, mà trở thành **$K$ điểm neo (Point Attractors)** siêu gọn!

```
Không gian gốc R^d:                               Không gian Null R^L:
┌───────────────────────────────┐                 ┌───────────────────────────────┐
│     ( Đám mây Camera )        │                 │                               │
│           • • •               │   Chiếu qua W   │      • c_camera               │
│          • • • •              │ ──────────────> │                               │
│                               │  (S_w nén về 0) │                               │
│     ( Đám mây Sensor )        │                 │                               │
│           * * *               │                 │               * c_sensor      │
│          * * * *              │                 │                               │
└───────────────────────────────┘                 └───────────────────────────────┘
```

---

## 3. TẦNG 3: BẢN CHẤT CỦA "DIVERSE ANOMALY TYPES" & TẠI SAO LOC-NFST GIẢI QUYẾT TRIỆT ĐỂ

Trong các bài toán IDS và chuẩn benchmark quốc tế như **ADBench (NeurIPS 2022)**, dị biệt không xuất hiện theo một kiểu duy nhất mà chia thành 3 cấu trúc hình thái:

```
          GLOBAL ANOMALY                        CLUSTER ANOMALY                         LOCAL ANOMALY
┌────────────────────────────────┐    ┌────────────────────────────────┐    ┌────────────────────────────────┐
│             • (Normal)         │    │             • (Normal)         │    │             • •                │
│            • • •               │    │            • • •               │    │           • • • •              │
│                                │    │                                │    │          •  ▲ (Local Anomaly!) │
│                                │    │       ▲ ▲ ▲ (Botnet C&C)       │    │           • • •                │
│                                │    │        ▲ ▲ ▲ (New cluster)     │    │                                │
│                                │    │                                │    │                                │
│    ▲ (DDoS Flooding)           │    │                                │    │             * * *              │
│    (Rất xa mọi cụm)            │    │                                │    │            * * * * (Sensor)    │
└────────────────────────────────┘    └────────────────────────────────┘    └────────────────────────────────┘
```

### 3.1. Phân tích 3 loại dị biệt:

#### 1. Dị biệt toàn cục (Global Anomalies)
- **Hành vi mạng:** Các cuộc tấn công brute-force cường độ cao, DoS/DDoS SYN Flood, quét cổng diện rộng (Aggressive Port Scan).
- **Vị trí hình học:** Giá trị các đặc trưng (Packet rate, Flow byte) vượt ngưỡng cực đại, văng ra cực kỳ xa khỏi toàn bộ các cụm bình thường.
- **Mức độ khó:** **Dễ nhất.** Hầu hết mọi mô hình (kNN, OCSVM, Isolation Forest) đều bắt được loại này vì khoảng cách Euclidean tới mọi điểm bình thường đều lớn.

#### 2. Dị biệt cụm (Cluster Anomalies)
- **Hành vi mạng:** Tấn công có tổ chức, tự động hóa từ phần mềm độc hại (Botnet Mirai C&C communication, định kỳ gửi beacon, tấn công dò mật khẩu chậm phân tán - Slow Distributed Brute-force).
- **Vị trí hình học:** Bản thân các gói tin tấn công không quá bất thường nếu xét đơn lẻ, nhưng chúng tụ lại thành một **cụm dị biệt mới** nằm ở vùng trống giữa các cụm bình thường.
- **Mức độ khó:** **Trung bình.** Các mô hình mật độ (Density-based) dễ bị đánh lừa vì tưởng đây là một cụm bình thường mới có mật độ cao.

#### 3. Dị biệt cục bộ (Local Anomalies) — "Kẻ sát thủ vô hình"
- **Hành vi mạng:** 
  - Tấn công ngụy trang (Stealthy / Mimicry Attacks): Kẻ tấn công giả mạo làm IP Camera để gửi dữ liệu gián điệp ra ngoài (Data Exfiltration), hoặc giả mạo gói tin ARP/DNS của Gateway.
  - Các cuộc tấn công khai thác lỗ hổng nhắm đích (Targeted Exploits, Buffer Overflow) chỉ làm biến dạng 1-2 trường nhỏ trong gói tin trong khi kích thước và tần suất vẫn giống hệt thiết bị bình thường.
- **Vị trí hình học:** 
  - Điểm dị biệt nằm **ngay sát vách hoặc chui vào bên trong vùng lân cận của cụm Camera** (nằm sâu trong phân phối chung của mạng).
  - Khoảng cách từ nó đến tâm Camera $\mu_{camera}$ thậm chí còn **nhỏ hơn** khoảng cách từ Camera đến Cảm biến!
- **Mức độ khó:** **Cực kỳ khó!** 
  - Nếu dùng mô hình toàn cục (Global baseline): Điểm dị biệt này lập tức bị gán nhãn là "Normal" vì nó nằm trong không gian bao bọc của mạng.
  - Các mô hình như OCSVM, DeepSVDD, AutoEncoder, LUNAR thường bị tụt giảm hiệu năng nghiêm trọng ở chế độ Local Regime (như số liệu trong bài báo: các baseline giảm sâu, chỉ đạt $95\% - 97\%$).

### 3.2. Tại sao LOC-NFST lại là "Kính hiển vi" phát hiện Local Anomalies với AUC 99.60%?

Hãy nhìn vào cơ chế giải tích của LOC-NFST:
1. Giả sử điểm bình thường $x_{norm}$ thuộc cụm $k$. Khoảng cách chiếu của nó trong không gian Null:
   $$d(x_{norm}) = \| (x_{norm} - \mu_k) W \|_2$$
   Vì $W$ là cơ sở trực giao triệt tiêu phương sai nội cụm ($W^\top S_{w, k} W \approx \mathbf{0}$), mọi biến thiên tự nhiên của thiết bị camera theo các hướng này bị triệt tiêu hoàn toàn:
   $$d(x_{norm}) \approx 0$$
2. Bây giờ, xét một điểm tấn công cục bộ $x_{local\_attack}$ ngụy trang trong cụm camera:
   - Trong không gian gốc $\mathbb{R}^d$, kẻ tấn công cố gắng làm cho $x_{local\_attack} \approx \mu_k$ ở các chiều dễ thấy (kích thước gói tin, cổng dịch vụ).
   - Tuy nhiên, để thực hiện hành vi tấn công, vector sai khác $(x_{local\_attack} - \mu_k)$ **bắt buộc phải có thành phần nằm dọc theo các hướng tương quan vi mô** (vốn là hướng bất biến của thiết bị bình thường).
   - Khi chiếu qua $W$, vì các hướng bình thường đã bị nén phẳng về 0, **sai lệch vi mô của cuộc tấn công không còn bị che mờ bởi phương sai tự nhiên của thiết bị nữa!**
   - Sai lệch này bị **phóng đại (magnified)** trong không gian $\mathbb{R}^L$:
     $$d(x_{local\_attack}) = \| (x_{local\_attack} - \mu_k) W \|_2 \gg 0$$
3. Điểm tấn công bị văng ra khỏi điểm co $c_k$, làm điểm dị biệt bùng nổ lên sát ngưỡng $1.0$!

> [!TIP]
> **Bản chất triết lý:**  
> LOC-NFST phân cụm để **"bóc tách tiếng ồn riêng của từng thiết bị"** ($S_w$), sau đó dùng toán tử Null Space để **"triệt tiêu hoàn toàn tiếng ồn đó"**. Khi tiếng ồn bình thường của thiết bị bị tắt đi, dù cuộc tấn công có tinh vi hay cục bộ đến đâu, nó cũng trở nên cực kỳ chói lòa trong không gian rỗng!

---

## 4. TẦNG 4: MỐI QUAN HỆ BIỆN CHỨNG GIỮA HETEROGENEOUS DEVICES, DYNAMIC K VÀ KẾT QUẢ THỰC NGHIỆM TRÊN SERVER

Từ những phân tích ở Tầng 1, 2 và 3, ta thấy rõ mắt xích kết nối trực tiếp đến ý tưởng nghiên cứu **ADYN-LOC-NFST** mà chúng ta vừa thực nghiệm thành công trên server `postmaster.iec`:

```
┌────────────────────────────────┐
│   HETEROGENEOUS IOT DEVICES    │  Các thiết bị khác nhau tạo ra K_thực tế cụm đa tạp rời rạc.
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    NGHỊCH LÝ CỦA K CỐ ĐỊNH     │  K_static cố định ngoại tuyến (Grid Search) sẽ:
│        (STATIC K)              │   - K quá nhỏ: Dồn nhiều thiết bị vào 1 cụm -> Phình Sw -> Sụp đổ L=0.
└───────────────┬────────────────┘   - K quá lớn: Băm nhỏ 1 thiết bị -> Overfitting, vỡ ma trận hiệp phương sai.
                │
                ▼
┌────────────────────────────────┐
│       ADYN-LOC-NFST            │  Cho phép K(t) co giãn tự nhiên:
│     (DYNAMIC CARDINALITY)      │   - Thiết bị mới vào mạng -> Cluster Birth.
└───────────────┬────────────────┘   - Lưu lượng phân tách -> Cluster Split (Rank-1 downdate).
                │                    - Thiết bị tắt -> Cluster Death.
                ▼
┌────────────────────────────────┐
│ BẢO VỆ TUYỆT ĐỐI KHÔNG GIAN    │  Triệt tiêu Sw chính xác ở mọi thời điểm,
│       NGHIỆM RỖNG (NULL)       │  bảo đảm độ nhạy tối đa với DIVERSE ANOMALY TYPES!
└────────────────────────────────┘
```

### 4.1. Giải mã kết quả thực nghiệm thực tế trên server `postmaster.iec`:

1. **Tại sao trên `EdgeIIoTset` + `StandardScaler`, Static-K bị kẹt ở $64\%$ còn ADYN nhảy vọt lên $99.76\%$ ($+35.33\%$)?**
   - `EdgeIIoTset` chứa hơn 10 loại thiết bị IoT công nghiệp khác nhau (cảm biến áp suất, nhiệt độ, van công nghiệp, cánh tay robot...).
   - Với Static-K, dù chọn $K=20, 50$ hay $100$, việc cố định cứng nhắc số cụm trong khi dùng `StandardScaler` (vốn nhạy cảm với các đuôi phân phối dài) làm cho các cụm bị gán lệch tâm. Ma trận $S_w$ bị phình to, không gian rỗng bị co ép xuống mức sàn $L=5$ khiến mô hình mất khả năng phân biệt.
   - **ADYN-LOC-NFST** cho phép các cụm tự động tách ($20 \rightarrow 26$) đúng theo ranh giới tự nhiên của các thiết bị công nghiệp thông qua các cập nhật Rank-1. Kết quả là ma trận $A$ được nới rộng chính xác, đưa AUC lên mức hoàn hảo **$99.76\%$**!

2. **Tại sao trên `N_BaIoT`, khi Static-K tăng từ $20 \rightarrow 100$ thì AUC bị "rơi tự do" ($95\% \rightarrow 81\%$), còn ADYN giữ vững $99.56\%$?**
   - `N_BaIoT` thu thập từ 9 thiết bị IoT thương mại cụ thể (chuông cửa thông minh, camera an ninh...). Bản chất vật lý của mạng này chỉ có khoảng $\approx 20 - 30$ chế độ hoạt động bình thường.
   - Khi cố ép Static $K=100$, một hành vi bình thường của chiếc chuông cửa bị băm vụn thành 5 cụm nhỏ li ti. Mỗi cụm chỉ có vài mẫu ($N_k < d$), dẫn đến ma trận hiệp phương sai bị suy thoái số học (Rank-deficiency), gây ra báo động giả hàng loạt (Overfitting).
   - **ADYN-LOC-NFST** chỉ tăng nhẹ $K$ từ $20 \rightarrow 27$ rồi dừng lại vì kiểm định phân tách (Split Criterion) nhận diện các cụm đã đạt trạng thái đơn mode ổn định. Nhờ đó, ADYN đạt **AUC $99.56\%$**, F1-Score **$0.9860$**, và tỷ lệ báo động giả FAR chỉ **$0.046\%$**!

---

## 5. BẢNG TỔNG HỢP SO SÁNH: GÓC NHÌN TOÁN HỌC & HỆ THỐNG

| Tiêu chí | Tiếp cận Ngây thơ (Naive / $K=1$) | Tiếp cận LOC-NFST Tĩnh (Static $K$) | Tiếp cận ADYN-LOC-NFST (Động $K(t)$) |
| :--- | :--- | :--- | :--- |
| **Mô hình hóa Thiết bị IoT** | Ép tất cả thiết bị vào 1 phân phối Gauss đơn mode. | Chia thành $K$ lớp giả cố định bằng K-Means ngoại tuyến. | Co giãn động theo số lượng và hành vi thực tế của thiết bị ($K(t)$). |
| **Tâm biểu diễn ($\mu$)** | Rơi vào vùng chân không (nghịch lý $\mu_{global}$). | Có $K$ tâm cục bộ đại diện cho từng thiết bị. | Các tâm thích ứng theo thời gian thực qua Welford online. |
| **Ma trận $S_b$** | $S_b = \mathbf{0}$ (Sụp đổ hoàn toàn). | $S_b > \mathbf{0}$, rank cố định $\le K-1$. | $S_b(t) > \mathbf{0}$, biến thiên Rank-1 bảo toàn tính giải tích. |
| **Không gian rỗng $\text{Null}(A)$** | Bị triệt tiêu ($\text{dim} = 0$). | Tồn tại, nhưng có nguy cơ sụp đổ nếu chọn sai $K$. | Luôn được tối ưu hóa số chiều $L(t)$ nhờ Rank-1 downdate/update. |
| **Bắt Global Anomaly** | Tốt. | Rất tốt. | Hoàn hảo ($> 99.9\%$). |
| **Bắt Local Anomaly** | Thất bại (bị nuốt vào bụng lớp bình thường). | Rất tốt ($99.6\%$), vượt trội AutoEncoder / LUNAR. | Hoàn hảo và tự động thích ứng khi mạng xuất hiện luồng mới. |
| **Khả năng chống Overfitting** | Không overfit nhưng underfit cực nặng. | Dễ overfit nếu chọn $K$ quá lớn ($N\_BaIoT$ sụt về $81\%$). | Tự động cân bằng, triệt tiêu overfit ($N\_BaIoT$ đạt $99.56\%$). |

---

## 6. KHUNG DIỄN NGÔN HỌC THUẬT (DÀNH CHO KHÓA LUẬN VÀ BÀI BÁO Q1/A*)

Dưới đây là đoạn văn mẫu chuẩn phong cách Academic English (IEEE Transactions style) để bạn đưa vào Section Methodology / Motivation của bài báo:

> *"In real-world IoT infrastructures, network traffic does not emanate from a single monolithic source, but rather from a federation of heterogeneous edge entities (e.g., streaming IP surveillance, intermittent telemetry sensors, and event-driven actuators). In the extracted feature space $\mathbb{R}^d$, this physical heterogeneity manifests as a disconnected, non-convex union of sub-manifolds rather than a unimodal Gaussian distribution. Under a conventional global one-class formulation ($K=1$), the between-class scatter matrix identically vanishes ($S_b \equiv \mathbf{0}_{d \times d}$), causing the within-class scatter to coincide with the total scatter ($S_w \equiv S_t$) and resulting in the complete collapse of the exact null space ($\text{dim}(\text{Null}(A)) = 0$).*  
> 
> *To circumvent this theoretical singularity, LOC-NFST establishes a structure-inducing pseudo-class partitioning. Geometrically, this partitioning decomposes the heterogeneous traffic into compact local modes, enabling NFST to project intra-cluster variances to zero ($W^\top S_{w, k} W \to \mathbf{0}$). In the transformed subspace $\mathbb{R}^L$, the complex normal manifolds collapse into discrete point attractors. Consequently, while global and clustered anomalies remain naturally separable, local anomalies—which subtly deviate from specific device profiles and are traditionally obscured by global variance—are sharply exposed as prominent metric deviations, achieving unprecedented fidelity in one-class intrusion detection."*

---

## KẾT LUẬN

1. **"Các thiết bị khác nhau thì sao chứ?"** $\rightarrow$ Chúng tạo ra các đám mây đa tạp rời rạc trong $\mathbb{R}^d$. Tâm toàn cục rơi vào vùng chân không. Nếu không phân cụm, ranh giới quyết định sẽ bị thủng lỗ chỗ, tạo điều kiện cho mã độc ẩn nấp.
2. **"Tại sao phải phân cụm?"** $\rightarrow$ Để $S_b \ne \mathbf{0}$ và $S_w < S_t$. Đây là **điều kiện tiên quyết bắt buộc về mặt giải tích** để không gian nghiệm rỗng $\text{Null}(S_w)$ có thể tồn tại.
3. **"Diverse anomaly types bản chất là gì?"** $\rightarrow$ Là sự khác biệt về khoảng cách hình học đối với các cụm: Global (xa tít), Cluster (tụ thành cụm lạ ở khoảng trống), Local (nằm sát sườn cụm bình thường). LOC-NFST phân cụm để nén phẳng tiếng ồn riêng của từng thiết bị về 0, biến không gian rỗng thành một chiếc **kính hiển vi phóng đại các sai lệch vi mô của Local Anomalies**, giúp mô hình đạt hiệu năng vượt bậc so với mọi phương pháp hiện hành.
