import requests
import json

endpoints = [
    "https://metax.fairdata.fi/v3/datasets/9d13ef28-2ca7-44b0-9950-225359afac65",
    "https://metax.fairdata.fi/v2/datasets/9d13ef28-2ca7-44b0-9950-225359afac65",
    "https://metax.fairdata.fi/rest/v2/datasets/9d13ef28-2ca7-44b0-9950-225359afac65",
    "https://metax.fairdata.fi/rest/datasets/9d13ef28-2ca7-44b0-9950-225359afac65",
]

for url in endpoints:
    try:
        response = requests.get(url)
        print(f"URL: {url} -> Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            with open(f"scratch/metax_{url.split('/')[-2] if '/' in url else 'out'}.json", "w") as f:
                json.dump(data, f, indent=2)
            print("  Data saved!")
    except Exception as e:
        print(f"URL: {url} -> Error: {e}")
