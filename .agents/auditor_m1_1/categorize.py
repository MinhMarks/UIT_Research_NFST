import json
import re

with open(r"d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\doi_audit_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

genuine = []
fake_doi = []
spoofed_doi = []

for d in data:
    status = d['status']
    key = d['key']
    bib_t = d['bib_title'] or ''
    real_t = d['real_title'] or ''
    doi = d['doi']
    bib_a = d['bib_author'] or ''
    real_a = d['real_authors'] or ''
    
    if status != 'RESOLVED':
        fake_doi.append(d)
        continue
        
    b_words = set(re.findall(r'\w+', bib_t.lower()))
    r_words = set(re.findall(r'\w+', real_t.lower()))
    # Remove common stop words
    stop = {'the', 'a', 'an', 'and', 'for', 'of', 'in', 'on', 'with', 'to', 'using', 'from', 'across', 'via'}
    b_sig = b_words - stop
    r_sig = r_words - stop
    overlap = len(b_sig & r_sig)
    
    # Also check first author match
    first_bib_author = bib_a.split()[0].replace(',', '').lower() if bib_a else ''
    author_match = first_bib_author in real_a.lower() if first_bib_author else False
    
    if overlap >= 2 or author_match:
        genuine.append(d)
    else:
        spoofed_doi.append(d)

print(f"Total entries: {len(data)}")
print(f"GENUINE: {len(genuine)}")
print(f"FAKE DOIs (HTTP 404): {len(fake_doi)}")
print(f"SPOOFED DOIs (Mismatched Papers): {len(spoofed_doi)}")

print("\n--- FAKE DOIs (404) ---")
for d in fake_doi:
    print(f"Key: {d['key']} | DOI: {d['doi']} | Claimed Title: {d['bib_title']}")

print("\n--- SPOOFED DOIs (Paper Mismatch) ---")
for d in spoofed_doi:
    print(f"Key: {d['key']} | DOI: {d['doi']}")
    print(f"  Claimed: {d['bib_title']} (by {d['bib_author']})")
    print(f"  Actual:  {d['real_title']} (by {d['real_authors']})")
