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
        """証拠金残高取得（信用取引用）"""
        path = "/v1/account/margin"
        headers = self._private_headers("GET", path)
        r = requests.get(f"{self.BASE_PRIVATE}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_account_assets(self) -> dict:
        """現物資産残高取得"""
        path = "/v1/account/assets"
        headers = self._private_headers("GET", path)
        r = requests.get(f"{self.BASE_PRIVATE}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_jpy_balance(self) -> float:
        """JPY現物残高を取得"""
        resp = self.get_account_assets()
        for item in resp.get("data", []):
            if item.get("symbol") == "JPY":
                return float(item.get("available", 0))
        return 0.0

    def get_positions(self, symbol: str = "BTC_JPY") -> dict:
        """現物保有残高確認（BTCの保有量を確認）"""
        path = "/v1/account/assets"
        headers = self._private_headers("GET", path)
        r = requests.get(f"{self.BASE_PRIVATE}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        resp = r.json()
        # 現物はopenPositionsではなくassetsで確認
        # BTCの保有量が0より大きければポジションあり
        spot_symbol = symbol.replace("_JPY", "")
        for item in resp.get("data", []):
            if item.get("symbol") == spot_symbol:
                amount = float(item.get("amount", 0))
                if amount > 0:
                    # ポジションあり形式に変換して返す
                    return {"data": {"list": [{"symbol": symbol, "size": str(amount), "positionId": "spot"}]}}
        return {"data": {"list": []}}

    def place_order(self, symbol: str, side: str, size: str, order_type: str = "MARKET") -> dict:
        """現物成行注文"""
        path = "/v1/order"
        import json
        # 現物取引はシンボルから_JPYを除いた形式（例: BTC_JPY → BTC）
        spot_symbol = symbol.replace("_JPY", "")
        body = json.dumps({
            "symbol": spot_symbol,
            "side": side,
            "executionType": order_type,
            "size": size,
        })
        headers = self._private_headers("POST", path, body)
        r = requests.post(f"{self.BASE_PRIVATE}{path}", headers=headers, data=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def close_position(self, symbol: str, position_id: str, side: str, size: str) -> dict:
        """現物売り注文（ポジション決済）"""
        path = "/v1/order"
        import json
        spot_symbol = symbol.replace("_JPY", "")
        body = json.dumps({
            "symbol": spot_symbol,
            "side": side,
            "executionType": "MARKET",
            "size": size,
        })
        headers = self._private_headers("POST", path, body)
        r = requests.post(f"{self.BASE_PRIVATE}{path}", headers=headers, data=body, timeout=10)
        r.raise_for_status()
        return r.json()
