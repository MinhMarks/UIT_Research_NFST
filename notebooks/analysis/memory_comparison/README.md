# Scientific Memory Plots Generator

Đây là thư mục chứa đoạn mã Python tự động sinh các biểu đồ tiêu chuẩn dùng cho bài **Scientific Paper (Báo cáo nghiên cứu)** nhằm so sánh giữa các Baseline Models và thuật toán để xuất nạp của bạn.

## Các loại biểu đồ được hỗ trợ sinh tự động:
1. **Bar_Memory_Footprint.png**: Biểu đồ dạng cột (Bar plot) so sánh bộ nhớ `Peak RAM Train (MB)` giữa các thuật toán.
2. **Scatter_Efficiency_Tradeoff.png**: Biểu đồ phân tán (Scatter Plot) với Trục X là Bộ nhớ (RAM) và Trục Y là Hiệu năng (AUCPR). Góc cung cấp góc nhìn **"Trade-off" (Đánh đổi Hệ sinh thái vs Cấu hình)**. Thuật toán của bạn lý tưởng nhất sẽ nằm ở góc Trái - Trên cùng (RAM ít nhất, AUCPR cao nhất).
3. **Bar_Time_Complexity.png**: Biểu đồ so sánh thời gian training và test (Thang đo Logarit) của model.

## Cách sử dụng

**Bước 1:** Sau khi chạy xong quá trình Tune, hãy copy file `Tuned_Baseline_Results_All.csv` (hoặc `Best_Baseline_Results_Per_Model.csv`) vào thư mục này và đổi tên thành `baseline_results.csv`.

**Bước 2:** Chạy script đánh giá Model do bạn đề xuất (`OC-NSFT...`), xuất kết quả ra file định dạng `.csv`. Hãy lưu ý cấu trúc file của bạn phải có các cột sau để chương trình vẽ biểu đồ hiểu được:
- `Model` (hoặc `Method`): Tên thuật toán (VD: "OC-NSFT")
- `AUCPR`
- `Peak RAM Train (MB)`
- `Time Train`

Copy file đó vào thư mục này và đổi tên thành `model_results.csv`.

**Bước 3:** Mở Terminal và truy cập vào đường dẫn thư mục này, tiến hành chạy lệnh:

```bash
python scientific_memory_plots.py --baseline baseline_results.csv --model model_results.csv --output plots
```

Hình ảnh vẽ ra với DPI=300 cực nét sẽ được lưu ở trong thư mục `plots/`. Bạn chỉ cần attach vào file LaTeX hoặc Word của paper!
