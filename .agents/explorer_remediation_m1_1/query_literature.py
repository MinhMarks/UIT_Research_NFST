import urllib.request
import urllib.parse
import json
import time

headers = {'User-Agent': 'AcademicAuditor/1.0 (mailto:auditor@uit.edu.vn)'}

queries = [
    {"key": "ruff2018deep", "title": "Deep One-Class Classification", "author": "Ruff"},
    {"key": "qiu2021neural", "title": "Neural Transformation Learning for Deep Anomaly Detection Beyond Images", "author": "Qiu"},
    {"key": "bergman2020classification", "title": "Classification-Based Anomaly Detection for General Data", "author": "Bergman"},
    {"key": "jin2021anemone", "title": "ANEMONE: Multi-scale Contrastive Learning for Graph Anomaly Detection", "author": "Jin"},
    {"key": "sakurada2014anomaly", "title": "Anomaly Detection Using Autoencoders with Extreme Value Theory", "author": "Sakurada"},
    {"key": "bodesheim2013kernel", "title": "Kernel Null Space Methods for Novelty Detection", "author": "Bodesheim"},
    {"key": "shen2022connective", "title": "Connective Gradient Descent for Heterogeneous Federated Learning", "author": "Shen"},
    {"key": "yuan2021federated", "title": "Federated Graph Learning with Local Differential Privacy", "author": "Yuan"},
    {"key": "rey2022federated", "title": "Federated Learning for Intrusion Detection in the Internet of Things: A Review", "author": "Rey"},
    {"key": "neto2023botiot", "title": "A Systematic Assessment of the BoT-IoT Dataset for Network Intrusion Detection", "author": "Neto"},
    {"key": "xiang2026federated", "title": "Federated Isolation Forest for Network Intrusion Detection in Edge Computing", "author": "Xiang"},
    {"key": "ngo2019fence", "title": "Fence GAN: Towards Better Anomaly Detection", "author": "Ngo"},
    {"key": "segurola2024unsupervised", "title": "Unsupervised Network Intrusion Detection in IoT", "author": "Segurola"},
    {"key": "sarhan2023evaluating", "title": "Evaluating Machine Learning Network Intrusion Detection Systems in Zero-Day Attack Scenarios", "author": "Sarhan"},
    {"key": "wang2022fedod", "title": "FedOD: Federated Outlier Detection", "author": "Wang"},
    {"key": "prabowo2026contrastive", "title": "Multi-Scale Graph Contrastive Representation Learning", "author": "Al-Sabri"},
    {"key": "roesch1999snort", "title": "Snort - Lightweight Intrusion Detection for Networks", "author": "Roesch"},
]

def search_crossref(title, author):
    params = urllib.parse.urlencode({'query.title': title, 'query.author': author, 'rows': 3})
    url = f"https://api.crossref.org/works?{params}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data['message']['items']
            results = []
            for it in items:
                results.append({
                    'doi': it.get('DOI', ''),
                    'title': it.get('title', [''])[0] if it.get('title') else '',
                    'author': [a.get('family', '') for a in it.get('author', [])],
                    'container': it.get('container-title', [''])[0] if it.get('container-title') else '',
                    'year': it.get('issued', {}).get('date-parts', [[None]])[0][0],
                    'score': it.get('score', 0)
                })
            return results
    except Exception as e:
        return [{'error': str(e)}]

def search_openalex(title):
    params = urllib.parse.urlencode({'search': title, 'per-page': 3})
    url = f"https://api.openalex.org/works?{params}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = []
            for it in data.get('results', []):
                authors = [a.get('author', {}).get('display_name', '') for a in it.get('authorships', [])]
                results.append({
                    'doi': it.get('doi', ''),
                    'title': it.get('display_name', ''),
                    'authors': authors[:4],
                    'venue': it.get('primary_location', {}).get('source', {}).get('display_name', '') if it.get('primary_location', {}).get('source') else '',
                    'year': it.get('publication_year', '')
                })
            return results
    except Exception as e:
        return [{'error': str(e)}]

for q in queries:
    print(f"\n=================== SEARCHING FOR: {q['key']} ===================")
    print(f"Title: {q['title']} | Author: {q['author']}")
    cr_res = search_crossref(q['title'], q['author'])
    print("--- CrossRef Top Results ---")
    for r in cr_res:
        if 'error' in r:
            print("  Error:", r['error'])
        else:
            print(f"  DOI: {r['doi']} | Title: {r['title'][:70]} | Year: {r['year']} | Authors: {r['author'][:3]}")
    
    oa_res = search_openalex(q['title'])
    print("--- OpenAlex Top Results ---")
    for r in oa_res:
        if 'error' in r:
            print("  Error:", r['error'])
        else:
            print(f"  DOI: {r['doi']} | Title: {r['title'][:70]} | Year: {r['year']} | Authors: {r['authors'][:3]}")
    time.sleep(0.5)
