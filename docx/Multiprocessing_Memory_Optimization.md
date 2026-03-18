# Tối ưu hóa Hiệu suất và Bộ nhớ (Multiprocessing & Memory Optimization) trong OC-NFST

Tài liệu này giải thích chi tiết về kỹ thuật đã được áp dụng vào file `OC_NFST_memory_optimized.py` để giúp tận dụng tối đa lượng RAM dư thừa (55GB+) nhằm tăng tốc độ chạy Experiment lên nhiều lần.

## 1. Vấn đề ban đầu (The Bottleneck)
Trước đây, file script chạy thử nghiệm (experiment) đối với mô hình quét qua 100 giá trị `n_clusters` (từ 1 đến 301) theo cách **tuần tự (sequential)**.
```python
# Code cũ chạy tuần tự:
for n_clusters in n_clusters_list:
    # 1. Chạy KMeans
    # 2. Chạy SVD / Tính toán Không gian Null (NPD)
    # 3. Chấm điểm rào cản FAISS
    # 4. Lưu kết quả
```
Với vòng lặp `for` thông thường, Python chỉ sử dụng duy nhất **1 nhân CPU (Core)** tại một thời điểm. Dù Server có 64GB RAM và hàng chục nhân CPU, nó vẫn nằm chơi xơi nước trong khi 1 nhân duy nhất "gánh team" rùa bò qua 100 vòng lặp.

## 2. Giải pháp: Lập trình Đa tiến trình (Multiprocessing) với `joblib`

Thay vì cố gắng tiết kiệm từng MB RAM, chúng ta đưa ra chiến lược: **"Đánh đổi RAM lấy Tốc độ"**.

Chúng ta phân chia 100 giá trị `n_clusters` thành những lô công việc độc lập và giao cho toàn bộ số nhân CPU có sẵn trên Server xử lý **cùng một lúc** thông qua thư viện `joblib`.

### Chi tiết cách áp dụng:

1. **Đóng gói vòng lặp thành một hàm độc lập (Worker function)**:
   Toàn bộ logic đo đạc bên trong vòng lặp được gom lại thành hàm `_process_one_cluster(n_clusters)`.
   Hàm này tự thân nó chứa biến đo lường bộ nhớ (tracemalloc) và có tính "thuần khiết" (không phụ thuộc vào số `n_clusters` kết quả trước đó).

2. **Kích hoạt Multiprocessing pool (`Parallel` và `delayed`)**:
```python
from joblib import Parallel, delayed

# n_jobs=-1 nghĩa là: Giao việc cho TẤT CẢ các CPU cores hiện có.
parallel_outputs = Parallel(n_jobs=-1, verbose=10)(
    delayed(_process_one_cluster)(nc) for nc in n_clusters_list
)
```

### Tại sao nó lại cực kỳ ngốn RAM?
Khi CPU số 1 đang tính toán không gian đa chiều (NPD) cho `n_clusters = 50`, thì CPU số 2 cũng đang phân bổ bộ nhớ để tính toán cho `n_clusters = 53`. 
Hệ điều hành buộc phải nhân bản (copy) ma trận `X_train` và `X_test` vào từng phân vùng bộ nhớ cô lập cho các tiến trình (Processes) này. 

Nếu có 16 nhân CPU cùng chạy song song, lượng RAM bùng nổ tạm thời sẽ tăng lên gấp xấp xỉ 16 lần so với bình thường. Tuy nhiên, vì máy tính của bạn có tới **55GB RAM trống**, việc tiêu tốn 10GB hay 20GB RAM bù lại giúp **tiết kiệm 16 lần thời gian** là một sự đánh đổi cực kỳ khôn ngoan!

## 3. Tổng kết Lợi ích Đạt được

1. **Thời gian chạy giảm phễu (Drastic speedup)**: Rút ngắn thời gian thí nghiệm từ vài tiếng đồng hồ xuống chỉ còn vài phút (hoặc mười mấy phút cho dataset khủng như CICIoT2023).
2. **Khai thác phần cứng 100%**: Đẩy CPU usage lên mức tối đa (`n_jobs=-1`), không phí phạm bất kỳ linh kiện nào trong máy mướn.
3. **Thong dong cho tương lai**: Kể cả khi có thêm bước thử nghiệm noise_ratios (1%, 3%, 5%, 10%), Multiprocessing vẫn đảm bảo quét sạch lưới hyperparameter với tốc độ ánh sáng mà không làm nhà nghiên cứu phải chờ đợi qua đêm.
