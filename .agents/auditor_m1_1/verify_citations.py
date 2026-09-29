import re
import urllib.request
import json
import time

BIB_FILE = r"d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\paper_latex\references.bib"

with open(BIB_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Parse bibtex entries
raw_entries = re.split(r'\n@', text)
entries = []
for raw in raw_entries:
    if not raw.strip():
        continue
    match = re.match(r'(\w+)\s*\{\s*([^,]+),', raw)
    if not match:
        continue
    entry_type = match.group(1)
    key = match.group(2).strip()
    doi_m = re.search(r'doi\s*=\s*\{([^}]+)\}', raw)
    title_m = re.search(r'title\s*=\s*\{([^}]+)\}', raw)
    author_m = re.search(r'author\s*=\s*\{([^}]+)\}', raw)
    year_m = re.search(r'year\s*=\s*\{([^}]+)\}', raw)
    
    doi = doi_m.group(1).strip() if doi_m else None
    title = title_m.group(1).strip().replace('{', '').replace('}', '') if title_m else None
    author = author_m.group(1).strip() if author_m else None
    year = year_m.group(1).strip() if year_m else None
    
    entries.append({
        'type': entry_type,
        'key': key,
        'doi': doi,
        'title': title,
        'author': author,
        'year': year
    })

print(f"Total parsed entries: {len(entries)}")

audit_results = []

for e in entries:
    doi = e['doi']
    key = e['key']
    bib_title = e['title']
    bib_author = e['author']
    
    # Try resolving via doi.org with json accept header
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "ForensicAuditor/1.0 (mailto:audit@uit.edu.vn)"
        }
    )
    
    status = "UNKNOWN"
    real_title = ""
    real_authors = ""
    real_container = ""
    error_msg = ""
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            real_title = data.get('title', '')
            authors_list = data.get('author', [])
            real_authors = ", ".join([a.get('family', '') for a in authors_list if isinstance(a, dict)])
            real_container = data.get('container-title', '')
            status = "RESOLVED"
    except urllib.error.HTTPError as err:
        status = f"HTTP_{err.code}"
        error_msg = str(err)
    except Exception as err:
        status = f"ERROR_{type(err).__name__}"
        error_msg = str(err)
        
    audit_results.append({
        'key': key,
        'doi': doi,
        'bib_title': bib_title,
        'real_title': real_title,
        'bib_author': bib_author,
        'real_authors': real_authors,
        'status': status,
        'error_msg': error_msg
    })
    time.sleep(0.1)

# Write full audit log to file
with open(r"d:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\.agents\auditor_m1_1\doi_audit_results.json", "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

# Print summary
resolved_count = sum(1 for r in audit_results if r['status'] == 'RESOLVED')
failed_count = sum(1 for r in audit_results if r['status'] != 'RESOLVED')

print(f"\n--- AUDIT SUMMARY ---")
print(f"Total checked: {len(audit_results)}")
print(f"Resolved via doi.org: {resolved_count}")
print(f"Failed to resolve: {failed_count}")

print("\n--- UNRESOLVED DOIs (404/ERRORS) ---")
for r in audit_results:
    if r['status'] != 'RESOLVED':
        print(f"[{r['status']}] Key: {r['key']} | DOI: {r['doi']} | Bib Title: {r['bib_title']}")

print("\n--- RESOLVED DOIs TITLE CHECK ---")
mismatches = []
for r in audit_results:
    if r['status'] == 'RESOLVED':
        # Simple similarity check
        b_clean = re.sub(r'[^a-z0-9]', '', (r['bib_title'] or '').lower())
        r_clean = re.sub(r'[^a-z0-9]', '', (r['real_title'] or '').lower())
        if b_clean not in r_clean and r_clean not in b_clean:
            # Check overlap of first few words
            b_words = set(re.findall(r'\w+', (r['bib_title'] or '').lower()))
            r_words = set(re.findall(r'\w+', (r['real_title'] or '').lower()))
            overlap = len(b_words & r_words)
            if overlap < 3:
                mismatches.append(r)
                print(f"[MISMATCH] Key: {r['key']}")
                print(f"  Bib Title:  {r['bib_title']}")
                print(f"  Real Title: {r['real_title']}")
                print(f"  DOI:        {r['doi']}")
                print(f"  Real Authors: {r['real_authors']}")

print(f"\nTotal title mismatches (Spoofed DOIs): {len(mismatches)}")
