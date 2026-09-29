import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(".agents/explorer_remediation_m1_1/audit_50_entries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
for d in data:
    if not d['match']:
        print(f"Key: {d['key']}")
        print(f"  Status:        {d['status']}")
        print(f"  Claimed DOI:   {d['doi']}")
        print(f"  Claimed Title: {d['bib_title']}")
        print(f"  Real Title:    {d['real_title']}")
        print(f"  Real Author:   {d['real_author']}")
        print("-" * 50)
