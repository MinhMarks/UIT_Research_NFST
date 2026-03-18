# Hướng dẫn Xử lý File ZIP Dữ liệu Khổng lồ (Memory-Safe ZIP Streaming)

## 1. Vấn đề gặp phải (The Problem)
Các bộ dữ liệu IoT như N_BaIoT, CICIoT2023, BoTIoT, ToNIoT có kích thước vô cùng lớn. Một file áp suất nén `.zip` có thể nặng từ 1.7GB đến vài chục GB, nếu bung nén (extract) hoàn toàn ra thư mục sẽ tiêu tốn hàng chục thậm chí hàng trăm GB ổ cứng (Disk space). Điều này khiến các Server hoặc máy trạm có dung lượng ổ cứng hạn hẹp bị báo lỗi `Disk out of space` và sập tiến trình, hoặc mất cực kì nhiều thời gian để chuyển dữ liệu qua lại.

## 2. Giải pháp: Đọc luồng trực tiếp (Streaming) không giải nén
Thay vì dùng `zipfile.extractall()` ghi toàn bộ dữ liệu ra ổ cứng, hệ thống đã được tối ưu hóa để **đọc luồng (stream) trực tiếp từng dòng dữ liệu từ bên trong file nén `.zip`** rồi đưa thẳng vào RAM để xử lý dạng chunk-by-chunk bằng `pandas`.

Nhờ kỹ thuật này, dữ liệu chiếm dụng thêm trên ổ cứng sẽ **luôn là 0 MB** (ngoại trừ dung lượng của bản thân file `.zip` ban đầu mà bạn đã tải). 

## 3. Các công cụ và thư viện được sử dụng
- **`zipfile` (Python Standard Library)**: Dùng để phân tích cấu trúc file `.zip` và mở luồng (file pointer ảo) để luân chuyển nội dung ở dạng nhị phân, mà không hề giải phóng chúng ra file vật lý.
- **`pandas`**: Kết hợp với tham số `chunksize` để đọc từng mớ nhỏ dữ liệu từ cái file pointer ảo kia, giúp không bị tràn RAM (Out-of-Memory / OOM).

## 4. Chi tiết triển khai (Implementation Detail)

Toàn bộ logic thao tác cực mạnh tay này được áp dụng trong hàm `__load_raw_default()` của tất cả các class dataset (Ex: `N_BaIoT.py`, `CICIoT2023.py`...).

### Bước 1: Mở kiến trúc Zip mà không giải nén
Sử dụng context manager để đọc cấu trúc Zip.
```python
import zipfile

zip_path = "N_BaIoT.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    # Lấy danh sách tên các file .csv nằm bên trong lõi file zip (vẫn đang bị nén chặt)
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
```

### Bước 2: Truyền Pointer ảo vào Pandas Streaming
Hàm `z.open(file_name)` sẽ đóng vai trò như một object file ảo, chuyển đổi liền mạch từ luồng nén thành luồng dữ liệu đọc được thay vì tốn băng thông ổ cứng. Ta truyền nó vào mồm thuật toán `pd.read_csv`.
```python
    for file_name in csv_files:
        with z.open(file_name) as f:
            # Đọc từng cụm (chunk) với kích thước chỉ 10,000 dòng vào RAM
            for chunk in pd.read_csv(f, chunksize=10000, low_memory=False):
                # ...
                # Trích xuất nhãn (labels), gạn lọc (sample) ngẫu nhiên mảnh chunk này 
                # (để đạt số lượng `limit_cnt` nhanh nhất mà không phải load toàn bộ)
                # ...
```

### Bước 3: Nối các mảnh Chunk đã qua chắt lọc
Sau khi chắt lọc được dữ liệu cần thiết (giảm dung lượng xuống cực nhỏ) từ từng chunk, ta chỉ việc gộp chúng lại với hàm `CustomMerger`.
```python
                list_ss.append(sub_set) # Các mảnh vài chục/vài trăm dòng đã lọc cực nhỏ
                base_self.__label_cnt[x] += sub_set.shape[0]
        # Sau chu trình, ghép tất cả DataFrame nhỏ lại
        df_ans = CustomMerger().fit_transform([df_ans] + list_ss)
```

## 5. Xử lý Quái chiêu từ Google Drive: Zip giả mạo
Do các Dataset này được chia sẻ qua Google Drive ở chế độ công khai với link có chứa Authenticate Token (`at=...`). Token này **có thời hạn sống hữu hạn**. 
Khi hết thời hạn Token mà chạy Script Download trong code, tài khoản Google Drive sẽ không trả về luồng tải của file Zip GBs nữa mà trả về tập tin báo lỗi HTML (kích thước vài chục KB). Thấy có file tạo ra (dù là HTML đội lốt), code vẫn tưởng đã hoàn thành nên đi tiếp, nhưng nếu cố đem `.zip` ra phân tích bằng thư viện `zipfile` sẽ báo lỗi: `File is not a zip file`.

Vì thế code download có một cơ chế vệ sĩ túc trực để tự động tóm cổ tàn dư HTML và xóa sổ, rồi yêu cầu user tự tải thủ công:
```python
import zipfile
if not zipfile.is_zipfile(zip_file):
    print("================ Zip file not valid (possibly expired link)!!!=================")
    os.remove(zip_file)
    print("ERROR: Google Drive link expired. Please manually download.")
    sys.exit(1)
```

## 6. Lợi ích thu được
1. **Disk-safe**: Không tiêu tốn dung lượng lưu trữ trung gian/ổ cứng hệ thống.
2. **RAM-safe**: Bằng cách kết hợp Chunking (`chunksize=10000`), bộ nhớ RAM dù yếu tới mấy vẫn có thể cày nát các bộ dữ liệu khổng lồ bằng cách nhai từng miếng bánh thay vì nhét nguyên ổ vào họng.
3. **Robust**: Kháng được hiện tượng tải nhầm file rác cực oái oăm từ Google Drive ảnh hưởng tới lần chạy sau mà không ai ngờ tới.
