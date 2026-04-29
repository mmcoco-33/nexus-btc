"""価格データ取得・整形"""
import time
import pandas as pd
from datetime import datetime, timedelta
from src.api.gmo_client import GMOClient


class DataFetcher:
    def __init__(self, client: GMOClient):
        self.client = client

    def fetch_ohlcv(self, symbol: str = "BTC", interval: str = "1hour", days: int = 30) -> pd.DataFrame:
        """過去N日分のOHLCVデータを取得"""
        frames = []
        for i in range(days, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            try:
                resp = self.client.get_klines(symbol=symbol, interval=interval, date=date)
                if resp.get("status") == 0 and resp.get("data"):
                    frames.append(self._parse(resp["data"]))
                time.sleep(0.3)  # レート制限対策
            except Exception:
                continue

        if not frames:
            raise ValueError("OHLCVデータの取得に失敗しました")

        df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

    def _parse(self, data: list) -> pd.DataFrame:
        rows = []
        for d in data:
            rows.append({
                "timestamp": pd.to_datetime(d[0], unit="ms"),
                "open":  float(d[1]),
                "high":  float(d[2]),
                "low":   float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
            })
        return pd.DataFrame(rows)
