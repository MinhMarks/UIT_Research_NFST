# Workspace Rules & Invariants for UIT_Research_NFST

## Academic Reporting Standards
1. **Originating Prompt Header Invariant**:
   Whenever authoring or significantly updating any research report (`BAO_CAO_*.md`), benchmark assessment, or technical walkthrough (`WALKTHROUGH_*.md`), the document MUST start with a blockquote immediately under the header quoting the exact user prompt(s) verbatim:
   ```markdown
   > ### 📋 Yêu cầu Nghiên cứu Gốc (Originating Research Prompt)
   > 
   > *"Trích dẫn nguyên văn prompt của người dùng tại đây..."*
   ```
2. **Empirical Integrity**:
   - Never invent or fabricate benchmarks, metrics, or citations. All numbers must come from executed experiments or verified published papers.
   - All comparisons must compare like-with-like (same challenge addressed, same dataset protocol).
3. **Grounded Citations & Content-Alignment Audit Invariant**:
   - **Zero-Hallucination Citations**: Mọi paper được trích dẫn trong báo cáo, bài báo khoa học, hoặc khảo sát văn hiến BẮT BUỘC phải là công trình khoa học có thật, đã xuất bản hoặc xuất hiện trên các kho lưu trữ uy tín (arXiv, IEEE Xplore, ACM DL, Springer, Elsevier, USENIX, NeurIPS, ICML, AAAI). Tuyệt đối không trích dẫn nguồn không tồn tại.
   - **Full-Text Content Alignment**: Không suy diễn nội dung bài báo từ tiêu đề hoặc abstract. Luận điểm đưa ra phải phản ánh đúng kết quả, định lý, giới hạn hoặc cơ chế thực tế mà các tác giả trong bài báo đã chứng minh hoặc tuyên bố.
   - **Subagent Verification Protocol**: Khi xây dựng báo cáo khoa học hoặc khảo sát văn hiến chuyên sâu, Agent chính PHẢI kích hoạt hoặc yêu cầu một Subagent thẩm định độc lập (Auditor / Fact-Checker) thực hiện:
     1. Tra cứu và đọc toàn văn (full-text / arXiv source / official PDF / HTML).
     2. Đối chiếu từng claim / số liệu / công thức trích dẫn với nội dung thực tế của paper.
     3. Báo cáo cụ thể nếu phát hiện sai lệch (mismatch) hoặc suy diễn quá đà trước khi hoàn tất tài liệu.

