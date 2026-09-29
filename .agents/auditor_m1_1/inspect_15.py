import json

with open(r"d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\doi_audit_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, d in enumerate(data[:15]):
    status = d['status']
    key = d['key']
    bib_t = d['bib_title']
    real_t = d['real_title']
    doi = d['doi']
    bib_a = d['bib_author']
    real_a = d['real_authors']
    print(f"[{i+1}] {key} ({status})")
    print(f"    DOI:       {doi}")
    print(f"    Bib Title: {bib_t}")
    print(f"    Real Title:{real_t}")
    print(f"    Bib Auth:  {bib_a}")
    print(f"    Real Auth: {real_a}")
    print()
