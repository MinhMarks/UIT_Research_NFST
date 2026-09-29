import re

with open(r'd:\UIT\Research\IEC2023\LOC-NFST\UIT_Research_NFST\references_master.bib', 'r', encoding='utf-8') as f:
    content = f.read()

# Match entries
entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),', content)
print(f'Total bib entries found: {len(entries)}')

# Split into blocks
blocks = re.split(r'\n(?=@)', content)
parsed = []
for block in blocks:
    if not block.strip():
        continue
    m_key = re.match(r'@(\w+)\s*\{\s*([^,]+),', block)
    if not m_key:
        continue
    etype, key = m_key.groups()
    title_m = re.search(r'title\s*=\s*[\"{](.*?)[\"}],', block, re.DOTALL | re.IGNORECASE)
    venue_m = re.search(r'(?:journal|booktitle)\s*=\s*[\"{](.*?)[\"}],', block, re.DOTALL | re.IGNORECASE)
    year_m = re.search(r'year\s*=\s*[\"{]?(\d{4})[\"}],?', block, re.IGNORECASE)
    doi_m = re.search(r'doi\s*=\s*[\"{](.*?)[\"}],', block, re.IGNORECASE)
    
    title = title_m.group(1).replace('\n', ' ').strip() if title_m else 'NO_TITLE'
    venue = venue_m.group(1).replace('\n', ' ').strip() if venue_m else 'NO_VENUE'
    year = year_m.group(1) if year_m else 'NO_YEAR'
    doi = doi_m.group(1).strip() if doi_m else None
    parsed.append({'type': etype, 'key': key.strip(), 'title': title, 'venue': venue, 'year': year, 'doi': doi})

has_doi = [p for p in parsed if p['doi']]
print(f'Entries with explicit DOI: {len(has_doi)} / {len(parsed)}')
print('\n--- All parsed entries ---')
for i, p in enumerate(parsed, 1):
    doi_str = f"DOI: {p['doi']}" if p['doi'] else "NO_DOI"
    print(f"{i:2d}. [{p['key']}] ({p['year']}) [{p['venue']}] | {doi_str}")
    print(f"    Title: {p['title']}")
