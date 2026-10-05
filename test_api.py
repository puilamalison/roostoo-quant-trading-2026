import requests

BASE_URL = "https://mock-api.roostoo.com"

url = f"{BASE_URL}/v3/serverTime"

response = requests.get(url, timeout=10)

print("HTTP status:", response.status_code)
print("Response:", response.json())

