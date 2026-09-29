import re
import json

with open("paper_latex/references.bib", "r", encoding="utf-8") as f:
    content = f.read()

entries = re.split(r'\n@', content)
if entries[0].startswith('@'):
    entries[0] = entries[0][1:]
else:
    entries = entries[1:]

parsed = []
for e in entries:
    lines = e.strip().splitlines()
    header = lines[0]
    entry_type, rest = header.split('{', 1)
    key = rest.split(',')[0].strip()
    doi_m = re.search(r'doi\s*=\s*\{([^}]+)\}', e)
    title_m = re.search(r'title\s*=\s*\{([^}]+)\}', e)
    author_m = re.search(r'author\s*=\s*\{([^}]+)\}', e)
    year_m = re.search(r'year\s*=\s*\{([^}]+)\}', e)
    journal_m = re.search(r'journal\s*=\s*\{([^}]+)\}', e)
    book_m = re.search(r'booktitle\s*=\s*\{([^}]+)\}', e)
    venue = (journal_m.group(1) if journal_m else (book_m.group(1) if book_m else ''))
    parsed.append({
        'key': key,
        'type': entry_type.strip(),
        'doi': doi_m.group(1) if doi_m else '',
        'title': title_m.group(1) if title_m else '',
        'author': author_m.group(1) if author_m else '',
        'year': year_m.group(1) if year_m else '',
        'venue': venue
    })

print(f"Total parsed: {len(parsed)}")

target_keys = [
    'nguyen2024locnfst', 'aaai2025fedclgn', 'shen2021ares', 'ngo2019fence',
    'sun2021flpa', 'segurola2024unsupervised', 'sarhan2023evaluating',
    'wang2022fedod', 'ferrag2022edgeiiotset', 'roesch1999snort',
    'ruff2018deep', 'qiu2021neural', 'bergman2020classification',
    'jin2021anemone', 'sakurada2014anomaly', 'bodesheim2013kernel',
    'shen2022connective', 'yuan2021federated', 'rey2022federated',
    'neto2023botiot', 'xiang2026federated', 'prabowo2026contrastive'
]

for p in parsed:
    if p['key'] in target_keys:
        print(f"KEY: {p['key']}")
        print(f"  DOI:    {p['doi']}")
        print(f"  Title:  {p['title']}")
        print(f"  Author: {p['author'][:60]}...")
        print(f"  Venue:  {p['venue']}")
        print(f"  Year:   {p['year']}")
        print("-" * 50)
