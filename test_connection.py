"""接続テスト：残高・現在価格の取得のみ（注文なし）"""
import hashlib
import hmac
import json
import os
import time
from datetime import datetime

import requests

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")
BASE_PUB   = "https://api.coin.z.com/public"
BASE_PRI   = "https://api.coin.z.com/private"


def sign(method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    msg = ts + method + path + body
    sig = hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return {"API-KEY": API_KEY, "API-TIMESTAMP": ts, "API-SIGN": sig}


def get_price() -> float:
    r = requests.get(f"{BASE_PUB}/v1/ticker?symbol=BTC", timeout=10)
    data = r.json()
    print("ticker response:", json.dumps(data, ensure_ascii=False)[:300])
    items = data.get("data", [])
    if items:
        return float(items[0].get("last", 0))
    return 0.0


def get_balance() -> float:
    if not API_KEY:
        print("APIキー未設定 → 残高取得スキップ")
        return 0.0
    path = "/v1/account/assets"
    headers = sign("GET", path)
    r = requests.get(f"{BASE_PRI}{path}", headers=headers, timeout=10)
    data = r.json()
    print("assets response:", json.dumps(data, ensure_ascii=False)[:300])
    for item in data.get("data", []):
        if item.get("symbol") == "JPY":
            return float(item.get("available", 0))
    return 0.0


def main():
    print(f"[{datetime.now()}] 接続テスト開始")

    price   = get_price()
    balance = get_balance()

    print(f"BTC現在価格: ¥{price:,.0f}")
    print(f"JPY残高: ¥{balance:,.0f}")

    # results.json に書き出し（ダッシュボード表示用）
    os.makedirs("data", exist_ok=True)
    path = "data/results.json"
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)

    results.append({
        "timestamp": datetime.now().isoformat(),
        "price":      price,
        "action":     "HOLD",
        "confidence": 0.5,
        "executed":   False,
        "balance_jpy": balance,
        "reason":     "test_run",
    })
    results = results[-500:]

    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, default=str)

    print("results.json 書き出し完了")


if __name__ == "__main__":
    main()
