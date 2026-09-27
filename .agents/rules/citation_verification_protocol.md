---
trigger: model_decision
description: Invariant requiring all cited academic literature to be verified real publications, with an independent subagent content-alignment check against the paper's full text.
---

# Quy trình Bắt buộc: Xác thực Nguồn trích dẫn & Kiểm định Nội dung Bài báo (Citation Verification & Content-Alignment Protocol)

## 1. Nguyên tắc Cốt lõi (Invariants)

1. **Xác thực Danh tính Bài báo (Authenticity Check)**:
   - Mọi tài liệu tham khảo phải có đầy đủ: Danh sách tác giả thực tế, Tên bài báo chính xác, Năm xuất bản, Tên hội nghị/tạp chí uy tín, DOI hợp lệ hoặc đường dẫn URL chính thức (arXiv / IEEE Xplore / ACM DL / Nature / v.v.).
   - Tuyệt đối cấm tạo ra các bài báo "tổng hợp" từ nhiều tên gọi hoặc gán ghép tên nhà nghiên cứu nổi tiếng vào một chủ đề họ chưa từng công bố.

2. **Kiểm định Sự Khớp nối Nội dung (Content-Alignment Check)**:
   - **Không dừng lại ở Abstract**: Abstract thường mang tính khái quát cao; việc trích dẫn các đặc tính kỹ thuật, thuật toán, công thức toán hoặc kết quả đối chuẩn đòi hỏi phải kiểm tra trong phần Methodology và Experiments của bài báo.
   - **Đúng Challenge - Đúng Ngữ cảnh**: Tránh hiện tượng bài báo giải quyết bài toán A nhưng lại trích dẫn như thể họ giải quyết bài toán B, hoặc trích dẫn một bài báo như một baseline thất bại trong khi thiết lập thực nghiệm của họ hoàn toàn khác biệt.

## 2. Quy trình Thực thi bằng Subagent (Subagent Verification Workflow)

Khi thực hiện các nhiệm vụ nghiên cứu chuyên sâu, khảo sát văn hiến (Literature Survey) hoặc viết bài báo khoa học:

1. **Bước 1 (Soạn thảo & Thu thập Nguồn)**:
   - Agent/Subagent nghiên cứu tìm kiếm và tổng hợp các công trình khoa học có DOI/URL rõ ràng.
2. **Bước 2 (Kích hoạt Auditor Subagent)**:
   - Sử dụng một Subagent đóng vai trò Kiểm định viên Độc lập (Literature Auditor / Fact-Checker) để:
     - Đọc trực tiếp tài liệu nguồn (thông qua `read_url_content`, `view_file` trên bản cache HTML/PDF, hoặc API học thuật).
     - Đối chiếu từng câu khẳng định: *"Tác giả X trong bài báo Y chứng minh/đo lường được Z"* $\to$ Tìm chính xác vị trí trong văn bản nguồn (Section, Theorem, Table, Page).
3. **Bước 3 (Báo cáo Phản biện & Hiệu chỉnh)**:
   - Auditor Subagent trả về danh sách xác nhận (Passed) hoặc cảnh báo sai lệch (Discrepancy Warning). Chỉ khi mọi trích dẫn đã được chứng thực thì tài liệu mới được xem là hoàn tất.
