---
trigger: model_decision
description: Invariant requiring all generated research reports, scientific analyses, and walkthrough documents to quote the originating user prompt verbatim at the top.
---

# Quy tắc Bắt buộc: Trích dẫn Nguyên văn Prompt ở Đầu Mọi Báo cáo

Mỗi khi tạo mới hoặc cập nhật lớn một tài liệu báo cáo nghiên cứu (`.md`), báo cáo đối chuẩn (benchmark report), bài phân tích khoa học hoặc walkthrough kỹ thuật:

1. **Vị trí bắt buộc**:
   - Ngay dưới tiêu đề chính (`# Title` / `## Subtitle`) và phần thông tin meta (Đơn vị, Ngày tháng, Git branch), TRƯỚC mục lục hoặc nội dung chính.

2. **Định dạng chuẩn mực**:
   Sử dụng blockquote GitHub markdown với tiêu đề rõ ràng:

   ```markdown
   > ### 📋 Yêu cầu Nghiên cứu Gốc (Originating Research Prompt)
   > 
   > *"Trích dẫn nguyên văn toàn bộ prompt/câu lệnh của người dùng tại đây..."*
   ```

   Nếu tài liệu tổng hợp từ nhiều prompt liên hoàn, trích dẫn chuỗi prompt chính kèm theo ngữ cảnh kích hoạt.

3. **Tính bất biến (Invariant)**:
   - Tuyệt đối không tự ý rút gọn hoặc diễn giải lại sai lệch ý của người dùng trong phần trích dẫn này.
   - Luôn giữ nguyên văn để đảm bảo tính tái lập (reproducibility) và minh bạch học thuật.
