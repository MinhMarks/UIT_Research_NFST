import urllib.request
import urllib.parse
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

headers = {'User-Agent': 'AcademicAuditor/1.0 (mailto:auditor@uit.edu.vn)'}

items_to_search = [
    # Category B
    ("ruff2018deep", "Deep One-Class Classification", "Ruff"),
    ("qiu2021neural", "Neural Transformation Learning for Deep Anomaly Detection Beyond Images", "Qiu"),
    ("bergman2020classification", "Classification-Based Anomaly Detection for General Data", "Bergman"),
    ("jin2021anemone", "ANEMONE Multi-scale Contrastive Learning for Graph Anomaly Detection", "Jin"),
    ("sakurada2014anomaly", "Anomaly Detection Using Autoencoders", "Sakurada"),
    ("bodesheim2013kernel", "Kernel Null Space Methods for Novelty Detection", "Bodesheim"),
    ("shen2022connective", "Connective Gradient Descent for Heterogeneous Federated Learning", "Shen"),
    ("yuan2021federated", "Federated Graph Learning with Local Differential Privacy", "Yuan"),
    ("rey2022federated", "Federated Learning for Intrusion Detection in the Internet of Things", "Rey"),
    ("neto2023botiot", "BoT-IoT Dataset for Network Intrusion Detection", "Neto"),
    ("xiang2026federated", "Federated Isolation Forest for Network Intrusion Detection", "Xiang"),
    ("prabowo2026contrastive", "Multi-Scale Graph Contrastive Representation Learning For Network Intrusion Detection", "Al-Sabri"),
    # Category A
    ("ngo2019fence", "Fence GAN Towards Better Anomaly Detection", "Ngo"),
    ("segurola2024unsupervised", "Unsupervised Network Intrusion Detection in IoT", "Segurola"),
    ("sarhan2023evaluating", "Evaluating Machine Learning Network Intrusion Detection Systems", "Sarhan"),
    ("wang2022fedod", "Federated Outlier Detection", "Wang"),
    ("roesch1999snort", "Snort Lightweight Intrusion Detection for Networks", "Roesch"),
    ("ferrag2022edgeiiotset", "Edge-IIoTset A New Comprehensive Realistic Cyber Security Dataset", "Ferrag"),
    ("sun2021flpa", "Data Poisoning Attacks on Federated Machine Learning in IoT", "Sun"),
]

def search(key, title, author):
    print(f"\n=================== KEY: {key} ===================")
    print(f"Query: title='{title}', author='{author}'")
    
    # OpenAlex
    try:
        oa_url = f"https://api.openalex.org/works?{urllib.parse.urlencode({'search': title, 'per-page': 3})}"
        req = urllib.request.Request(oa_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("--- OpenAlex ---")
            for it in data.get('results', []):
                authors = [a.get('author', {}).get('display_name', '') for a in it.get('authorships', [])]
                doi = it.get('doi', '')
                t = it.get('display_name', '')
                year = it.get('publication_year', '')
                venue = it.get('primary_location', {}).get('source', {}).get('display_name', '') if it.get('primary_location', {}).get('source') else ''
                print(f"  DOI: {doi}")
                print(f"  Title: {t}")
                print(f"  Year: {year} | Venue: {venue}")
                print(f"  Authors: {', '.join(authors[:4])}")
                print()
    except Exception as e:
        print("  OpenAlex error:", e)

    # CrossRef
    try:
        cr_url = f"https://api.crossref.org/works?{urllib.parse.urlencode({'query.title': title, 'query.author': author, 'rows': 2})}"
        req = urllib.request.Request(cr_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("--- CrossRef ---")
            for it in data['message']['items']:
                doi = it.get('DOI', '')
                t = it.get('title', [''])[0] if it.get('title') else ''
                year = it.get('issued', {}).get('date-parts', [[None]])[0][0]
                container = it.get('container-title', [''])[0] if it.get('container-title') else ''
                authors = [a.get('family', '') for a in it.get('author', [])]
                print(f"  DOI: {doi}")
                print(f"  Title: {t}")
                print(f"  Year: {year} | Venue: {container}")
                print(f"  Authors: {', '.join(authors[:4])}")
                print()
    except Exception as e:
        print("  CrossRef error:", e)
    
    time.sleep(0.3)

for key, title, author in items_to_search:
    search(key, title, author)
