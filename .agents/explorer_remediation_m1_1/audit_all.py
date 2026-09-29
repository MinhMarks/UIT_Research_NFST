import re
import urllib.request
import urllib.parse
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

headers = {'User-Agent': 'AcademicAuditor/1.0 (mailto:auditor@uit.edu.vn)'}

with open("paper_latex/references.bib", "r", encoding="utf-8") as f:
    content = f.read()

entries = re.split(r'\n@', content)
if entries[0].startswith('@'):
    entries[0] = entries[0][1:]
else:
    entries = entries[1:]

results = []

for e in entries:
    lines = e.strip().splitlines()
    header = lines[0]
    entry_type, rest = header.split('{', 1)
    key = rest.split(',')[0].strip()
    doi_m = re.search(r'doi\s*=\s*\{([^}]+)\}', e)
    title_m = re.search(r'title\s*=\s*\{([^}]+)\}', e)
    author_m = re.search(r'author\s*=\s*\{([^}]+)\}', e)
    
    doi = doi_m.group(1).strip() if doi_m else ''
    title = title_m.group(1).strip() if title_m else ''
    author = author_m.group(1).strip() if author_m else ''
    
    item = {
        'key': key,
        'doi': doi,
        'bib_title': title,
        'bib_author': author,
        'status': 'UNKNOWN',
        'real_title': '',
        'real_author': '',
        'match': False
    }
    
    if not doi:
        item['status'] = 'NO_DOI'
        results.append(item)
        continue
        
    # Check CrossRef or DataCite
    try:
        if doi.startswith('10.48550/'):
            url = f"https://api.datacite.org/dois/{doi}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                attrs = data.get('data', {}).get('attributes', {})
                titles = attrs.get('titles', [])
                real_title = titles[0].get('title', '') if titles else ''
                creators = attrs.get('creators', [])
                real_author = ", ".join([c.get('name', '') for c in creators[:3]])
                item['real_title'] = real_title
                item['real_author'] = real_author
                item['status'] = 'RESOLVED'
        else:
            url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                msg = data.get('message', {})
                titles = msg.get('title', [])
                real_title = titles[0] if titles else ''
                authors = msg.get('author', [])
                real_author = ", ".join([a.get('family', '') for a in authors[:3]])
                item['real_title'] = real_title
                item['real_author'] = real_author
                item['status'] = 'RESOLVED'
                
        # Simple match check
        clean_bib_t = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
        clean_real_t = re.sub(r'[^a-zA-Z0-9]', '', real_title.lower())
        
        # Check first 20 chars of title
        if clean_bib_t[:15] in clean_real_t or clean_real_t[:15] in clean_bib_t:
            item['match'] = True
        else:
            item['match'] = False
            
    except urllib.error.HTTPError as e:
        item['status'] = f"HTTP_{e.code}"
    except Exception as e:
        item['status'] = f"ERROR: {str(e)[:30]}"
        
    results.append(item)
    time.sleep(0.1)

with open(".agents/explorer_remediation_m1_1/audit_50_entries.json", "w", encoding="utf-8") as out:
    json.dump(results, out, indent=2, ensure_ascii=False)

resolved_match = [r for r in results if r['match']]
resolved_mismatch = [r for r in results if r['status'] == 'RESOLVED' and not r['match']]
http_errors = [r for r in results if r['status'].startswith('HTTP_')]
others = [r for r in results if r not in resolved_match and r not in resolved_mismatch and r not in http_errors]

print(f"Total entries: {len(results)}")
print(f"VERIFIED (Resolved & Matched): {len(resolved_match)}")
print(f"MISMATCHED (Resolved to different paper): {len(resolved_mismatch)}")
print(f"HTTP ERRORS (404/etc): {len(http_errors)}")
print(f"OTHERS/ERRORS: {len(others)}")
