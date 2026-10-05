import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROOSTOO_API_KEY")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY")

BASE_URL = "https://mock-api.roostoo.com"

# Current timestamp in milliseconds
timestamp = int(time.time() * 1000)

# Parameters that will be signed
params = {
    "timestamp": timestamp
}

# Convert parameters into the string:
# timestamp=123456789...
query_string = "&".join(
    f"{key}={value}"
    for key, value in params.items()
)

# Create HMAC-SHA256 signature
signature = hmac.new(
    SECRET_KEY.encode("utf-8"),
    query_string.encode("utf-8"),
    hashlib.sha256
).hexdigest()

headers = {
    "RST-API-KEY": API_KEY,
    "MSG-SIGNATURE": signature
}

response = requests.get(
    f"{BASE_URL}/v3/balance",
    params=params,
    headers=headers,
    timeout=10
)

print("HTTP status:", response.status_code)
print("Response:", response.json())
