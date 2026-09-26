# GIẢI MÃ TOÀN DIỆN: TỔ CHỨC IEEE VÀ PHÁP Y VỤ TẤN CÔNG LỊCH SỬ TARGET (2013) QUA HỆ THỐNG HVAC

**Đơn vị:** Information & Embedded Cyber-physical Systems Lab (IEC Lab) – Trường ĐH Công nghệ Thông tin (ĐHQG-HCM)  
**Mục tiêu tài liệu:** Cung cấp tri thức chuẩn mực, chính thống và phân tích pháp y chi tiết (Forensic Analysis) cho 2 chủ đề:  
1. **IEEE là gì:** Tại sao tổ chức này vừa đặt ra các tiêu chuẩn phần cứng (Ethernet 802.3, Wi-Fi 802.11) vừa là nhà xuất bản học thuật Top-1 thế giới?  
2. **Hồ sơ pháp y vụ tấn công Target (2013):** Từng bước kỹ thuật (Step-by-step Cyber Kill Chain) giải thích cách hacker lợi dụng một nhà thầu điều hòa/nhiệt độ (HVAC) để đánh cắp 110 triệu thẻ tín dụng, và bài học sống còn cho nghiên cứu an ninh mạng IoT / LOC-NFST.

---

## PHẦN 1: GIẢI MÃ TOÀN DIỆN VỀ IEEE (INSTITUTE OF ELECTRICAL AND ELECTRONICS ENGINEERS)

### 1.1. IEEE là gì? Lịch sử hình thành và Vị thế toàn cầu
- **Tên viết tắt:** **IEEE** (phát âm là *"Eye-triple-E"* / *Ai-tríp-pồ-i*).
- **Tên đầy đủ:** **Institute of Electrical and Electronics Engineers** (*Viện Kỹ sư Điện và Điện tử*).
- **Quy mô:** Là tổ chức nghề nghiệp kỹ thuật lớn nhất thế giới, thành lập tại Hoa Kỳ với hơn **430,000 hội viên tại hơn 160 quốc gia**.
- **Lịch sử ra đời:**
  - Năm **1884**, Viện Kỹ sư Điện Hoa Kỳ (**AIEE** - American Institute of Electrical Engineers) được thành lập bởi các nhà khoa học vĩ đại: **Thomas Edison** (nhà phát minh bóng đèn, điện một chiều) và **Alexander Graham Bell** (nhà phát minh điện thoại).
  - Năm **1912**, Viện Kỹ sư Vô tuyến điện (**IRE** - Institute of Radio Engineers) ra đời khi ngành vô tuyến và điện tử phát triển.
  - Ngày **01/01/1963**, AIEE và IRE chính thức sáp nhập lại thành **IEEE**.

---

### 1.2. "Hai bộ mặt" quyền lực của IEEE: Tiêu chuẩn hóa Công nghiệp & Nhà xuất bản Học thuật

Nhiều người thường thắc mắc: *"Tại sao thấy IEEE vừa quy định chuẩn Wi-Fi, Ethernet, lại vừa thấy các giáo sư xuất bản bài báo khoa học (Paper) trên IEEE?"*  
Lý do là vì IEEE vận hành qua hai nhánh trụ cột độc lập nhưng hỗ trợ nhau:

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                          IEEE                           │
                                    │    (Institute of Electrical & Electronics Engineers)    │
                                    └────────────────────────────┬────────────────────────────┘
                                                                 │
                  ┌──────────────────────────────────────────────┴──────────────────────────────┐
                  ▼                                                                             ▼
┌──────────────────────────────────────────────────┐                         ┌──────────────────────────────────────────────────┐
│                   NHÁNH 1:                       │                         │                   NHÁNH 2:                       │
│           HIỆP HỘI TIÊU CHUẨN HÓA                │                         │          XUẤT BẢN KHOA HỌC & HỘI NGHỊ            │
│         (IEEE Standards Association - SA)        │                         │          (IEEE Societies & Publications)         │
├──────────────────────────────────────────────────┤                         ├──────────────────────────────────────────────────┤
│ • Định nghĩa quy chuẩn phần cứng thế giới        │                         │ • Xuất bản 30% tài liệu kỹ thuật điện-CNTT toàn cầu│
│ • IEEE 802.3: Chuẩn mạng có dây Ethernet         │                         │ • Thư viện số IEEE Xplore Digital Library        │
│ • IEEE 802.11: Chuẩn mạng không dây Wi-Fi       │                         │ • Tạp chí danh giá: IEEE TIFS, IEEE IoT-J...     │
│ • IEEE 802.15.4: Chuẩn mạng IoT (Zigbee/6LoWPAN) │                         │ • Hội nghị Top-tier: IEEE S&P (Oakland), INFOCOM │
│ • IEEE 754: Chuẩn biểu diễn số thực Float32/64   │                         │ • Nơi công bố các nghiên cứu như LOC-NFST        │
└──────────────────────────────────────────────────┘                         └──────────────────────────────────────────────────┘
```

#### Nhánh 1: Hiệp hội Tiêu chuẩn hóa IEEE (IEEE-SA - Standards Association)
Đây là cơ quan tối cao định hình cách thế giới công nghệ kết nối với nhau:
- **IEEE 802.3 (Ethernet):** Quy định chính xác từ cấu trúc giắc cắm RJ45, cáp mạng xoắn đôi, tín hiệu điện áp, cấu trúc khung dữ liệu (Frame) và ngưỡng **MTU 1500 bytes**. Mọi máy tính, router, switch trên hành tinh muốn cắm dây mạng nói chuyện được với nhau đều bắt buộc phải tuân theo chuẩn này.
- **IEEE 802.11 (Wi-Fi):** Chuẩn mạng không dây phổ thông (từ 802.11a/b/g/n/ac đến Wi-Fi 6 là 802.11ax, Wi-Fi 7 là 802.11be).
- **IEEE 802.15.4:** Chuẩn truyền thông vô tuyến năng lượng cực thấp cho IoT (nền tảng của Zigbee, WirelessHART, 6LoWPAN).
- **IEEE 754:** Chuẩn định dạng số thực dấu phẩy động (Floating-Point Arithmetic) quy định cấu trúc 32-bit (`float32`) và 64-bit (`float64`) mà mọi chip xử lý Intel, AMD, ARM (kể cả Raspberry Pi) đều sử dụng trong các phép tính ma trận của thuật toán NFST.

#### Nhánh 2: Nhà xuất bản Học thuật & Tổ chức Hội nghị Khoa học
IEEE là nguồn cung cấp hơn **30% tổng số tài liệu học thuật toàn cầu** trong lĩnh vực Kỹ thuật điện, Điện tử, Khoa học Máy tính và Viễn thông:
- Thư viện số **IEEE Xplore Digital Library** lưu trữ hơn 5 triệu công trình nghiên cứu.
- Các tạp chí hàng đầu thế giới (Transactions/Journals) thuộc danh mục ISI/Scopus Q1:
  - *IEEE Transactions on Information Forensics and Security (TIFS)*: Tạp chí bảo mật uy tín hàng đầu (Core A*).
  - *IEEE Internet of Things Journal (IoT-J)*: Tạp chí chuyên sâu IoT số 1 thế giới (Impact Factor ~10.6).
  - *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*: Tạp chí AI/ML danh giá nhất lịch sử.
- Các hội nghị học thuật đỉnh cao (Tier-1 / Flagship Conferences):
  - *IEEE S&P (Oakland)*: Hội nghị an ninh thông tin hàng đầu cùng với ACM CCS, USENIX Security, NDSS.
  - *IEEE INFOCOM*: Hội nghị mạng máy tính danh giá.
  - *IEEE CVPR*: Hội nghị thị giác máy tính và trí tuệ nhân tạo lớn nhất thế giới.

### 1.3. Mối liên hệ mật thiết với Đề tài Khóa luận của chúng ta
- Nghiên cứu của chúng ta chạy trên các gói tin mạng tuân theo chuẩn **IEEE 802.3** và **IEEE 802.11**.
- Các đặc trưng luồng (Flow Features) của các bộ dữ liệu như CICIoT2023 hay EdgeIIoTset (như Header Length, Inter-arrival Time) đều bắt nguồn từ quy định đóng gói của các chuẩn IEEE này.
- Mục tiêu đầu ra của Khóa luận tại IEC Lab là hoàn thiện bài báo khoa học chất lượng cao để xuất bản trực tiếp lên hệ thống tạp chí của **IEEE** (*IEEE TIFS* hoặc *IEEE IoT-J*).

---

## PHẦN 2: HỒ SƠ PHÁP Y CHI TIẾT VỤ TẤN CÔNG TARGET (2013) QUA HỆ THỐNG HVAC

Vụ tấn công vào tập đoàn bán lẻ **Target Corporation** vào tháng 11 - 12/2013 là một trong những thảm họa an ninh mạng lớn nhất lịch sử thương mại thế giới. Nó trở thành **case study kinh điển bắt buộc phải giảng dạy** tại mọi viện an toàn thông tin để chứng minh rằng: **Trong mạng IoT, không có thiết bị nào là "quá tầm thường để bị hack".**

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CHỖI TẤN CÔNG (CYBER KILL CHAIN) VỤ TARGET (2013)                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  [1. PHISHING EMAIL]                 [2. VENDOR PORTAL BREACH]           [3. PIVOT & LATERAL MOVEMENT]
  Hacker gửi email mã độc             Đánh cắp tài khoản của              Xâm nhập cổng nội bộ Target,
  Citadel Trojan tới Fazio            Fazio, đăng nhập vào cổng           phát hiện mạng không phân vùng,
  Mechanical (nhà thầu HVAC)          Vendor Portal của Target            leo thang sang Domain Controller
          │                                   │                                         │
          ▼                                   ▼                                         ▼
┌──────────────────┐                ┌──────────────────┐                      ┌──────────────────┐
│ Fazio Mechanical │                │ Target Supplier  │                      │ Target Corporate │
│ (Nhà thầu HVAC)  │ ─────────────> │  Vendor Portal   │ ───────────────────> │ Active Directory │
└──────────────────┘                └──────────────────┘                      └──────────────────┘
                                                                                        │
                                                                                        ▼
  [6. DATA EXFILTRATION]              [5. RAM SCRAPING TRÊN POS]          [4. MALWARE DEPLOYMENT]
  Tuồn 11 triệu thẻ tín dụng          BlackPOS quét RAM tại máy quẹt      Cài đặt mã độc BlackPOS lên
  về máy chủ tại Nga & Brazil         thẻ trước khi dữ liệu kịp mã hóa    1,800 máy POS tại các cửa hàng
          ▲                                   ▲                                         │
          │                                   │                                         │
┌──────────────────┐                ┌──────────────────┐                                │
│ External C&C     │ <───────────── │ Staging Server   │ <──────────────────────────────┘
│ Server (Russia)  │                │  (Nội bộ Target) │
└──────────────────┘                └──────────────────┘
```

### 2.1. Nạn nhân và Hậu quả Thiệt hại
- **Nạn nhân:** Target Corporation (Chuỗi bán lẻ lớn thứ 2 tại Mỹ thời điểm đó với hơn 1,800 siêu thị).
- **Quy mô thiệt hại:**
  - Bị đánh cắp thông tin thẻ tín dụng/ghi nợ của **40 triệu khách hàng** (gồm số thẻ, ngày hết hạn, mã CVV bí mật, tên chủ thẻ).
  - Bị lộ dữ liệu cá nhân của **70 triệu khách hàng khác** (họ tên, địa chỉ nhà, email, số điện thoại).
  - Tổng cộng ảnh hưởng tới **110 triệu người** (chiếm 1/3 dân số nước Mỹ lúc bấy giờ).
  - Target phải chi trả hơn **$292 triệu USD** cho chi phí khắc phục, tiền bồi thường pháp lý và tiền phạt từ các ngân hàng/tổ chức phát hành thẻ.
  - Tổng giám đốc điều hành (CEO Gregg Steinhafel) và Giám đốc thông tin (CIO Beth Jacob) đều phải **từ chức**.

---

### 2.2. Pháp y Chi tiết Chuỗi Tấn công (Detailed Cyber Kill Chain)

#### Bước 1: Trinh sát và Tấn công Nhà thầu phụ HVAC (Fazio Mechanical Services)
- Target là một tập đoàn khổng lồ với hệ thống an ninh mạng đa tầng, tường lửa kiên cố và các phần mềm bảo mật đắt giá (bao gồm hệ thống FireEye trị giá $1.6 triệu USD). Hacker biết rằng tấn công trực diện vào máy chủ của Target là rất khó.
- **Kẻ hở chí mạng:** Để quản lý nhiệt độ và hệ thống làm lạnh bảo quản thực phẩm tại hàng ngàn siêu thị, Target ký hợp đồng với một công ty cơ điện lạnh địa phương nhỏ tên là **Fazio Mechanical Services** (có trụ sở tại Sharpsburg, Pennsylvania).
- Target cấp cho nhân viên của Fazio một tài khoản truy cập từ xa vào cổng thông tin nhà cung cấp (**Target Supplier Vendor Portal**) để nộp hóa đơn thanh toán và theo dõi thông số điều hòa/nhiệt độ từ xa.
- Tháng 9/2013, hacker gửi một email lừa đảo (**Spear-Phishing Email**) có đính kèm phần mềm độc hại **Citadel Trojan** (một biến thể nguy hiểm của Zeus Banking Trojan) cho nhân viên của Fazio Mechanical.
- Một nhân viên của Fazio đã mở tệp đính kèm. Trojan Citadel âm thầm ghi lại thao tác bàn phím (Keylogger) và **đánh cắp toàn bộ tên đăng nhập cùng mật khẩu truy cập vào mạng Target của nhà thầu này**.

#### Bước 2: Đột nhập vào Cổng thông tin Target (Initial Access)
- Ngày 15/11/2013, hacker sử dụng tài khoản đánh cắp của Fazio Mechanical để đăng nhập từ xa vào Vendor Portal của Target.
- **Lỗ hổng chết người của Target:**
  - Target **KHÔNG áp dụng xác thực hai yếu tố (2FA / MFA)** cho cổng kết nối nhà thầu phụ (chỉ dùng mật khẩu thông thường).
  - Nghiêm trọng hơn: Target mắc sai lầm kiến trúc cơ bản là **Mạng phẳng (Flat Network / No Network Segmentation)**. Cổng kết nối quản lý hóa đơn/nhiệt độ của nhà thầu phụ lại được cắm chung một dải mạng nội bộ với hệ thống văn phòng và hệ thống bán lẻ!

#### Bước 3: Tấn công leo thang và Chiếm quyền điều khiển (Lateral Movement)
- Từ cổng Vendor Portal, hacker khai thác các lỗ hổng chưa vá trên máy chủ Windows của Target để chiếm quyền quản trị miền (**Domain Administrator** trên hệ thống Active Directory).
- Khi đã có quyền kiểm soát nội bộ, hacker quét toàn bộ hạ tầng mạng của Target để tìm kiếm mục tiêu béo bở nhất: **Các máy quẹt thẻ thanh toán tại quầy thu ngân (POS - Point of Sale Terminals)**.

#### Bước 4: Vũ khí bí mật — Mã độc BlackPOS (Kaptoxa Memory Scraper)
- Target áp dụng tiêu chuẩn bảo mật dữ liệu thẻ quốc tế (PCI-DSS): Mọi dữ liệu thẻ truyền trên dây mạng hay lưu trong ổ cứng đều được **mã hóa đầu cuối (End-to-End Encryption)**. Nếu hacker nghe lén trên dây mạng, dữ liệu thu được chỉ là các khối mã hóa vô nghĩa.
- Để hóa giải điều này, hacker đã sử dụng một vũ khí đặc chế mang tên **BlackPOS** (tên nội bộ là *Kaptoxa*, được viết bởi một hacker trẻ người Nga):
  - **Kỹ thuật RAM Scraping (Vét sạch bộ nhớ RAM):** Khi một khách hàng quẹt thẻ qua máy POS, thông tin từ dải băng từ (Magnetic Stripe) gồm Track 1 và Track 2 (chứa số thẻ, họ tên, ngày hết hạn, mã xác thực) bắt buộc phải được nạp vào bộ nhớ RAM của máy POS trong một vài mili-giây ở dạng **văn bản thô (Plaintext)** trước khi phần mềm thanh toán kịp mã hóa nó để gửi đi.
  - BlackPOS chạy ngầm trên máy POS, liên tục rà quét không gian bộ nhớ RAM của tiến trình thanh toán. Cứ mỗi khi khách hàng quẹt thẻ, mã độc lập tức "chộp" lấy dữ liệu thô trong RAM trước khi nó bị mã hóa!

#### Bước 5: Cài cắm hàng loạt và Thu hoạch dữ liệu (Harvesting)
- Hacker lợi dụng hệ thống cập nhật phần mềm tự động của Target để đẩy mã độc BlackPOS xuống **hơn 1,800 siêu thị Target trên khắp nước Mỹ**, lây nhiễm thành công trên **hàng chục ngàn máy quẹt thẻ POS**.
- Cuộc thu hoạch diễn ra đúng vào dịp cao điểm mua sắm lớn nhất trong năm: **Lễ Tạ Ơn (Thanksgiving) và Thứ Sáu Đen Tối (Black Friday)** từ ngày 27/11 đến 15/12/2013. Hàng chục triệu người quẹt thẻ mua hàng đều bị mã độc lưu lại thông tin.

#### Bước 6: Tập kết nội bộ và Tuồn dữ liệu ra ngoài (Data Exfiltration)
- Các máy POS không có kết nối trực tiếp ra Internet (bị chặn bởi tường lửa).
- Hacker thiết lập các máy chủ nội bộ bị chiếm quyền trong mạng Target làm **Trạm trung chuyển (Staging Servers)**. Các máy POS định kỳ gửi dữ liệu thẻ đã đánh cắp về Staging Server thông qua giao thức nội bộ NetBIOS / SMB.
- Từ Staging Server, hacker dùng giao thức truyền file mã hóa và WebDAV để tuồn các tệp dữ liệu nén ra các máy chủ điều khiển (C&C Server) đặt tại **Nga, Brazil và Đông Âu**. Sau đó, các tệp dữ liệu thẻ này được đem bán đấu giá trên các chợ đen ngầm (Dark Web).

---

### 2.3. Bí ẩn: Tại sao Hệ thống Bảo mật triệu đô của Target bị "Liệt"?
Một chi tiết gây chấn động trong cuộc điều tra pháp y của Thượng viện Mỹ sau đó:
- Target đã chi **$1.6 triệu USD** trang bị giải pháp bảo mật phát hiện mã độc tiên tiến nhất thế giới thời điểm đó là **FireEye**.
- Vào ngày 30/11/2013, khi hacker bắt đầu cài đặt BlackPOS, **hệ thống FireEye đã phát hiện chính xác mã độc và liên tục gửi cảnh báo khẩn cấp (Security Alerts)** về trung tâm giám sát an ninh (SOC) của Target tại Bangalore (Ấn Độ) và Minneapolis (Mỹ).
- **Vì sao không ai ngăn chặn?**
  - Đội ngũ SOC của Target mỗi ngày nhận được hàng ngàn cảnh báo từ đủ loại phần mềm khác nhau (hiện tượng **Bội thực cảnh báo - Alert Fatigue**).
  - Do tỷ lệ báo động giả (False Positive) trong mạng quá cao, các chuyên viên an ninh Target cho rằng đây chỉ là một cảnh báo giả hoặc một tiến trình bảo trì bình thường của hệ thống, nên đã **bỏ qua hoặc tắt tiếng cảnh báo**!
  - Mãi đến khi Bộ Tư pháp Mỹ và Mật vụ Hoa Kỳ (US Secret Service) nhận được báo cáo từ các ngân hàng về việc hàng loạt thẻ bị đánh cắp đều từng quẹt tại Target, Target mới bàng hoàng phát hiện mình đã bị thủng lưới hoàn toàn trong gần 1 tháng.

---

### 2.4. Bài học Sống còn cho An ninh mạng IoT và Giá trị Cốt lõi của LOC-NFST

Vụ tấn công Target 2013 đã thay đổi vĩnh viễn tư duy an ninh mạng thế giới và trực tiếp minh chứng cho tính cấp thiết của đề tài nghiên cứu của chúng ta:

| Bài học từ Vụ Target (2013) | Vấn đề Thực trạng | Giải pháp Đột phá của Khóa luận LOC-NFST |
| :--- | :--- | :--- |
| **1. Không có thiết bị nào là "vô hại"** | Kẻ tấn công luôn tìm thiết bị yếu nhất (như hệ thống điều hòa HVAC, cảm biến nhiệt độ) để đột nhập và làm bàn đạp (Pivot). | LOC-NFST được thiết kế như một **NIDS đặt trực tiếp tại Gateway biên (Edge Gateway)**, giám sát toàn bộ các luồng lưu lượng của từng cảm biến/thiết bị ngay tại lớp mạng cục bộ, ngăn chặn ngay hành vi bất thường từ trứng nước. |
| **2. Tấn công ngụy trang (Local Anomalies)** | Hacker dùng tài khoản nhà thầu hợp lệ, truyền dữ liệu qua giao thức bình thường, ngụy trang hoàn hảo để tránh bị phát hiện bởi các hệ thống dựa trên chữ ký (Signature-based). | LOC-NFST áp dụng phép chiếu không gian rỗng $W \in \text{Null}(S_w)$, triệt tiêu tiếng ồn bình thường của thiết bị, biến sai lệch vi mô của cuộc tấn công ngụy trang thành **tín hiệu chói lòa trong $\mathbb{R}^L$**, đạt **AUC $99.6\%$** trên các cuộc tấn công Local Anomalies. |
| **3. Thảm họa của Bội thực cảnh báo (Alert Fatigue)** | Hệ thống FireEye của Target sinh ra quá nhiều cảnh báo giả khiến đội ngũ an ninh phớt lờ cảnh báo thật. | LOC-NFST với cơ chế co giãn động **ADYN-LOC-NFST** đạt tỷ lệ báo động giả (FAR) siêu thấp: **$0.046\%$** (chỉ 4 lần báo sai trên 10,000 gói tin trên dataset N-BaIoT), giải quyết triệt để vấn đề Alert Fatigue cho các trung tâm giám sát. |
| **4. Rào cản phần cứng và độ trễ tại biên** | Kiểm tra sâu gói tin (DPI) tốn quá nhiều CPU, làm chậm mạng nên thường bị tắt đi tại các cổng chi nhánh. | Khóa luận đồng thiết kế phần cứng: **Zero-Heap Static Cache Pool** và **RCU Lock-free Swap (8ns)** trên chip ARM Cortex-A72 (Raspberry Pi 4), suy luận tốc độ đường truyền (Line-rate) không trễ, không rớt gói tin. |

---

## TỔNG KẾT TƯ DUY KHOA HỌC

1. **Về IEEE:** Không phải là một công ty hay nhà xuất bản thương mại đơn thuần, IEEE là tổ chức bảo trợ khoa học kỹ thuật lớn nhất thế giới, nắm giữ cả quyền lực **định hình chuẩn mực công nghệ toàn cầu (IEEE-SA)** lẫn vị thế **trọng tài học thuật tối cao (IEEE Journals/Conferences)**.
2. **Về vụ hack Target 2013:** Là bằng chứng thép cho thấy ranh giới mạng IoT/CPS (điều hòa, nhiệt độ, camera) và mạng máy tính truyền thống đã hòa làm một. Bất kỳ một cảm biến hay thiết bị IoT nào bị bỏ quên đều có thể trở thành "cổng hậu" đánh sập toàn bộ doanh nghiệp hàng tỷ USD. Khóa luận **LOC-NFST và ADYN-LOC-NFST** chính là lời giải khoa học giải quyết trực diện bài toán bảo vệ này ngay tại lớp biên.
