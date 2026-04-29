"""GMOコイン APIクライアント"""
import hashlib
import hmac
import time
import requests
from typing import Optional


class GMOClient:
    BASE_PUBLIC = "https://api.coin.z.com/public"
    BASE_PRIVATE = "https://api.coin.z.com/private"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = timestamp + method + path + body
        return hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    def _private_headers(self, method: str, path: str, body: str = "") -> dict:
        timestamp = str(int(time.time() * 1000))
        return {
            "API-KEY": self.api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": self._sign(timestamp, method, path, body),
            "Content-Type": "application/json",
        }

    # --- Public API ---

    def get_ticker(self, symbol: str = "BTC") -> dict:
        r = requests.get(f"{self.BASE_PUBLIC}/v1/ticker?symbol={symbol}", timeout=10)
        r.raise_for_status()
        return r.json()

    def get_klines(self, symbol: str = "BTC", interval: str = "1hour", date: str = "") -> dict:
        """ローソク足データ取得"""
        if not date:
            date = time.strftime("%Y%m%d")
        url = f"{self.BASE_PUBLIC}/v1/klines?symbol={symbol}&interval={interval}&date={date}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_orderbooks(self, symbol: str = "BTC") -> dict:
        r = requests.get(f"{self.BASE_PUBLIC}/v1/orderbooks?symbol={symbol}", timeout=10)
        r.raise_for_status()
        return r.json()

    # --- Private API ---

    def get_account_margin(self) -> dict:
        path = "/v1/account/margin"
        headers = self._private_headers("GET", path)
        r = requests.get(f"{self.BASE_PRIVATE}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_positions(self, symbol: str = "BTC_JPY") -> dict:
        path = f"/v1/openPositions?symbol={symbol}"
        headers = self._private_headers("GET", path)
        r = requests.get(f"{self.BASE_PRIVATE}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def place_order(self, symbol: str, side: str, size: str, order_type: str = "MARKET") -> dict:
        """成行注文"""
        path = "/v1/order"
        import json
        body = json.dumps({
            "symbol": symbol,
            "side": side,          # BUY or SELL
            "executionType": order_type,
            "size": size,
        })
        headers = self._private_headers("POST", path, body)
        r = requests.post(f"{self.BASE_PRIVATE}{path}", headers=headers, data=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def close_position(self, symbol: str, position_id: str, side: str, size: str) -> dict:
        """ポジション決済"""
        path = "/v1/closeOrder"
        import json
        body = json.dumps({
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "settlePosition": [{"positionId": position_id, "size": size}],
        })
        headers = self._private_headers("POST", path, body)
        r = requests.post(f"{self.BASE_PRIVATE}{path}", headers=headers, data=body, timeout=10)
        r.raise_for_status()
        return r.json()
