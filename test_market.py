import requests

BASE_URL = "https://mock-api.roostoo.com"

url = f"{BASE_URL}/v3/exchangeInfo"

response = requests.get(url, timeout=10)

print("HTTP status:", response.status_code)
print(response.json())
