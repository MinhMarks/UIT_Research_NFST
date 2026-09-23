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
