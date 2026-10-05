import os
import time
import hmac
import hashlib
import requests

from dotenv import load_dotenv


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

BASE_URL = "https://mock-api.roostoo.com"

API_KEY = os.getenv("ROOSTOO_API_KEY")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY")


# ==================================================
# ROOSTOO CLIENT
# ==================================================

class RoostooClient:

    def __init__(
        self,
        api_key=None,
        secret_key=None,
        base_url=BASE_URL,
        timeout=10
    ):
        """
        Create a Roostoo API client.

        Parameters
        ----------
        api_key : str or None
            Roostoo API key.

        secret_key : str or None
            Roostoo API secret.

        base_url : str
            Roostoo API base URL.

        timeout : int
            Maximum seconds to wait for an HTTP response.
        """

        self.api_key = (
            api_key
            or
            API_KEY
        )

        self.secret_key = (
            secret_key
            or
            SECRET_KEY
        )

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "ROOSTOO_API_KEY was not found."
            )

        if not self.secret_key:
            raise ValueError(
                "ROOSTOO_SECRET_KEY was not found."
            )

    # ==================================================
    # INTERNAL HELPERS
    # ==================================================

    def _timestamp(self):
        """
        Return the current Unix timestamp
        in milliseconds.

        Returns
        -------
        int
        """

        return int(
            time.time() * 1000
        )

    def _build_query_string(
        self,
        params
    ):
        """
        Sort parameters alphabetically and convert
        them to Roostoo's signing format.

        Example output:

        pair=BTC/USD&quantity=0.01&side=BUY&timestamp=...

        Parameters
        ----------
        params : dict

        Returns
        -------
        str
        """

        return "&".join(
            f"{key}={params[key]}"
            for key in sorted(params)
        )

    def _sign(
        self,
        params
    ):
        """
        Create HMAC-SHA256 signature.

        Parameters
        ----------
        params : dict

        Returns
        -------
        str
            Hexadecimal HMAC signature.
        """

        query_string = (
            self._build_query_string(
                params
            )
        )

        signature = hmac.new(
            self.secret_key.encode(
                "utf-8"
            ),
            query_string.encode(
                "utf-8"
            ),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _signed_headers(
        self,
        params
    ):
        """
        Create authentication headers.

        Parameters
        ----------
        params : dict

        Returns
        -------
        dict
        """

        signature = (
            self._sign(params)
        )

        return {
            "RST-API-KEY":
                self.api_key,

            "MSG-SIGNATURE":
                signature
        }

    def _handle_response(
        self,
        response
    ):
        """
        Validate HTTP response and return JSON.

        Parameters
        ----------
        response : requests.Response

        Returns
        -------
        dict
        """

        try:

            response.raise_for_status()

        except requests.exceptions.HTTPError:

            print(
                "HTTP error:",
                response.status_code
            )

            print(
                "Response:",
                response.text
            )

            raise

        try:

            data = response.json()

        except ValueError:

            raise RuntimeError(
                "Roostoo returned invalid JSON: "
                + response.text
            )

        return data

    # ==================================================
    # PUBLIC ENDPOINTS
    # ==================================================

    def get_server_time(self):
        """
        Get Roostoo server time.

        Returns
        -------
        dict
        """

        response = requests.get(
            f"{self.base_url}/v3/serverTime",
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    def get_exchange_info(self):
        """
        Get trading-pair rules and precision.

        Returns
        -------
        dict
        """

        response = requests.get(
            f"{self.base_url}/v3/exchangeInfo",
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    def get_ticker(
        self,
        pair=None
    ):
        """
        Get market ticker.

        Parameters
        ----------
        pair : str or None
            Example: "BTC/USD"

        Returns
        -------
        dict
        """

        params = {
            "timestamp":
                self._timestamp()
        }

        if pair is not None:

            params["pair"] = pair

        response = requests.get(
            f"{self.base_url}/v3/ticker",
            params=params,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    # ==================================================
    # SIGNED GET ENDPOINTS
    # ==================================================

    def get_balance(self):
        """
        Get account balances.

        Returns
        -------
        dict
        """

        params = {
            "timestamp":
                self._timestamp()
        }

        headers = (
            self._signed_headers(
                params
            )
        )

        response = requests.get(
            f"{self.base_url}/v3/balance",
            params=params,
            headers=headers,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    def get_pending_count(self):
        """
        Get number of pending orders.

        Returns
        -------
        dict
        """

        params = {
            "timestamp":
                self._timestamp()
        }

        headers = (
            self._signed_headers(
                params
            )
        )

        response = requests.get(
            f"{self.base_url}/v3/pending_count",
            params=params,
            headers=headers,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    # ==================================================
    # PLACE ORDER
    # ==================================================

    def place_order(
        self,
        pair,
        side,
        quantity,
        order_type="MARKET",
        price=None
    ):
        """
        Place a Roostoo order.

        Parameters
        ----------
        pair : str
            Example: "BTC/USD"

        side : str
            "BUY" or "SELL"

        quantity : float or str
            Number of units to trade.

        order_type : str
            "MARKET" or "LIMIT"

        price : float, str or None
            Required for LIMIT orders.

        Returns
        -------
        dict
        """

        side = side.upper()

        order_type = (
            order_type.upper()
        )

        if side not in {
            "BUY",
            "SELL"
        }:

            raise ValueError(
                "side must be BUY or SELL"
            )

        if order_type not in {
            "MARKET",
            "LIMIT"
        }:

            raise ValueError(
                "order_type must be "
                "MARKET or LIMIT"
            )

        if (
            order_type == "LIMIT"
            and
            price is None
        ):

            raise ValueError(
                "LIMIT orders require price"
            )

        params = {
            "pair":
                pair,

            "side":
                side,

            "type":
                order_type,

            "quantity":
                str(quantity),

            "timestamp":
                self._timestamp()
        }

        if order_type == "LIMIT":

            params["price"] = (
                str(price)
            )

        headers = (
            self._signed_headers(
                params
            )
        )

        headers[
            "Content-Type"
        ] = (
            "application/"
            "x-www-form-urlencoded"
        )

        # Send exactly the same sorted
        # parameter string that we signed.
        body = (
            self._build_query_string(
                params
            )
        )

        response = requests.post(
            f"{self.base_url}/v3/place_order",
            data=body,
            headers=headers,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    # ==================================================
    # QUERY ORDERS
    # ==================================================

    def query_order(
        self,
        order_id=None,
        pair=None,
        pending_only=None,
        limit=None,
        offset=None
    ):
        """
        Query previous or pending orders.

        Parameters
        ----------
        order_id : int, str or None

        pair : str or None

        pending_only : bool or None

        limit : int or None

        offset : int or None

        Returns
        -------
        dict
        """

        params = {}

        # Roostoo requires order_id to be
        # used by itself if provided.
        if order_id is not None:

            params["order_id"] = (
                str(order_id)
            )

        else:

            if pair is not None:

                params["pair"] = pair

            if pending_only is not None:

                params[
                    "pending_only"
                ] = (
                    "TRUE"
                    if pending_only
                    else
                    "FALSE"
                )

            if limit is not None:

                params["limit"] = (
                    str(limit)
                )

            if offset is not None:

                params["offset"] = (
                    str(offset)
                )

        params["timestamp"] = (
            self._timestamp()
        )

        headers = (
            self._signed_headers(
                params
            )
        )

        headers[
            "Content-Type"
        ] = (
            "application/"
            "x-www-form-urlencoded"
        )

        body = (
            self._build_query_string(
                params
            )
        )

        response = requests.post(
            f"{self.base_url}/v3/query_order",
            data=body,
            headers=headers,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )

    # ==================================================
    # CANCEL ORDER
    # ==================================================

    def cancel_order(
        self,
        order_id=None,
        pair=None
    ):
        """
        Cancel pending orders.

        Parameters
        ----------
        order_id : int, str or None

        pair : str or None

        Returns
        -------
        dict

        Rules
        -----
        order_id only:
            cancel one specific order

        pair only:
            cancel pending orders for pair

        neither:
            cancel all pending orders
        """

        if (
            order_id is not None
            and
            pair is not None
        ):

            raise ValueError(
                "Use either order_id or pair, "
                "not both."
            )

        params = {}

        if order_id is not None:

            params["order_id"] = (
                str(order_id)
            )

        elif pair is not None:

            params["pair"] = pair

        params["timestamp"] = (
            self._timestamp()
        )

        headers = (
            self._signed_headers(
                params
            )
        )

        headers[
            "Content-Type"
        ] = (
            "application/"
            "x-www-form-urlencoded"
        )

        body = (
            self._build_query_string(
                params
            )
        )

        response = requests.post(
            f"{self.base_url}/v3/cancel_order",
            data=body,
            headers=headers,
            timeout=self.timeout
        )

        return self._handle_response(
            response
        )


# ==================================================
# SAFE TEST
# ==================================================

if __name__ == "__main__":

    client = RoostooClient()

    print()
    print("=" * 50)
    print("ROOSTOO CLIENT TEST")
    print("=" * 50)

    print()
    print("1. SERVER TIME")
    print("-" * 50)

    print(
        client.get_server_time()
    )

    print()
    print("2. BTC TICKER")
    print("-" * 50)

    ticker = (
        client.get_ticker(
            "BTC/USD"
        )
    )

    print(ticker)

    print()
    print("3. ACCOUNT BALANCE")
    print("-" * 50)

    balance = (
        client.get_balance()
    )

    print(balance)

    print()
    print("4. PENDING ORDERS")
    print("-" * 50)

    pending = (
        client.get_pending_count()
    )

    print(pending)

    print()
    print("=" * 50)
    print(
        "TEST COMPLETE - "
        "NO ORDER WAS PLACED"
    )
    print("=" * 50)
