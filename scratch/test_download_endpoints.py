import requests

urls = [
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65",
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/single?file=Combined.zip",
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/single?file=/Combined.zip",
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/single?file_id=639515c6e1664373518560f71497050",
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/download",
    "https://download.fairdata.fi/v2/dataset/9d13ef28-2ca7-44b0-9950-225359afac65/single?cr_id=9d13ef28-2ca7-44b0-9950-225359afac65&file=/Combined.zip",
]

for url in urls:
    try:
        response = requests.head(url, allow_redirects=True)
        print(f"HEAD {url} -> Status: {response.status_code}")
    except Exception as e:
        print(f"Error {url}: {e}")
