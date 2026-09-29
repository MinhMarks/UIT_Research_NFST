import json

with open(r"d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\doi_audit_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    if d['status'] == 'RESOLVED':
        print(f"KEY: {d['key']}")
        print(f"  BIB:  {d['bib_title']}")
        print(f"  REAL: {d['real_title']}")
        print(f"  BIB_AUTH:  {d['bib_author'][:40] if d['bib_author'] else ''}")
        print(f"  REAL_AUTH: {d['real_authors'][:40] if d['real_authors'] else ''}")
        print("-" * 50)
