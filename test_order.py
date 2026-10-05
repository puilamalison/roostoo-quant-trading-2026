import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# Load API credentials from .env
load_dotenv()

API_KEY = os.getenv("ROOSTOO_API_KEY")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY")

BASE_URL = "https://mock-api.roostoo.com"

# Safety check: make sure credentials were loaded
if not API_KEY or not SECRET_KEY:
    raise ValueError("API key or secret key not found in .env")

# --------------------------------------------------
# 1. Define our test order
# --------------------------------------------------

params = {
    "pair": "BTC/USD",
    "side": "BUY",
    "type": "MARKET",
    "quantity": "0.00012",
    "timestamp": int(time.time() * 1000)
}

# --------------------------------------------------
# 2. Create the string that Roostoo expects us to sign
# --------------------------------------------------

# Sort alphabetically by parameter name
query_string = "&".join(
    f"{key}={params[key]}"
    for key in sorted(params)
)

print("Signing:", query_string)

# --------------------------------------------------
# 3. Generate HMAC-SHA256 signature
# --------------------------------------------------

signature = hmac.new(
    SECRET_KEY.encode("utf-8"),
    query_string.encode("utf-8"),
    hashlib.sha256
).hexdigest()

# --------------------------------------------------
# 4. Create HTTP headers
# --------------------------------------------------

headers = {
    "RST-API-KEY": API_KEY,
    "MSG-SIGNATURE": signature,
    "Content-Type": "application/x-www-form-urlencoded"
}

# --------------------------------------------------
# 5. Send order to Roostoo
# --------------------------------------------------

response = requests.post(
    f"{BASE_URL}/v3/place_order",
    data=params,
    headers=headers,
    timeout=10
)

# --------------------------------------------------
# 6. Show result
# --------------------------------------------------

print("HTTP status:", response.status_code)

try:
    print("Response:", response.json())
except ValueError:
    print("Raw response:", response.text)
