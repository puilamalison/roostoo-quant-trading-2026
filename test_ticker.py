import time
import requests

BASE_URL = "https://mock-api.roostoo.com"

timestamp = int(time.time() * 1000)

params = {
    "timestamp": timestamp,
    "pair": "BTC/USD"
}

response = requests.get(
    f"{BASE_URL}/v3/ticker",
    params=params,
    timeout=10
)

print("HTTP status:", response.status_code)
print("Response:", response.json())
