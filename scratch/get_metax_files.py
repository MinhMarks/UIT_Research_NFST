import requests
import json

endpoints = [
    "https://metax.fairdata.fi/v3/datasets/9d13ef28-2ca7-44b0-9950-225359afac65/files",
    "https://metax.fairdata.fi/rest/v2/datasets/9d13ef28-2ca7-44b0-9950-225359afac65/files",
    "https://metax.fairdata.fi/rest/v2/datasets/9d13ef28-2ca7-44b0-9950-225359afac65/directories",
]

for url in endpoints:
    try:
        response = requests.get(url)
        print(f"URL: {url} -> Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            name = url.replace("https://metax.fairdata.fi/", "").replace("/", "_")
            with open(f"scratch/{name}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved to scratch/{name}.json")
    except Exception as e:
        print(f"URL: {url} -> Error: {e}")
