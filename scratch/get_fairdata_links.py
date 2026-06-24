import requests
import json

url = "https://etsin.fairdata.fi/api/v1/datasets/9d13ef28-2ca7-44b0-9950-225359afac65"
response = requests.get(url)
print(f"V1 Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    # Save to file
    with open("scratch/fairdata_v1.json", "w") as f:
        json.dump(data, f, indent=2)
    print("V1 data saved!")

url2 = "https://etsin.fairdata.fi/api/v2/datasets/9d13ef28-2ca7-44b0-9950-225359afac65"
response2 = requests.get(url2)
print(f"V2 Status: {response2.status_code}")
if response2.status_code == 200:
    data2 = response2.json()
    with open("scratch/fairdata_v2.json", "w") as f:
        json.dump(data2, f, indent=2)
    print("V2 data saved!")
