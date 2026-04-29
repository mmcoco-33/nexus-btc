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
                print(f"[fetcher] date={date} status={resp.get('status')} keys={list(resp.get('data', {}).keys()) if isinstance(resp.get('data'), dict) else type(resp.get('data'))}")
                if resp.get("status") == 0:
                    raw = resp.get("data")
                    if isinstance(raw, dict):
                        raw = raw.get("list", [])
                    if raw:
                        frames.append(self._parse(raw))
                time.sleep(0.3)
            except Exception as e:
                print(f"[fetcher] date={date} error={e}")
                continue

        if not frames:
            raise ValueError("OHLCVデータの取得に失敗しました")

        df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_multi_timeframe(self, symbol: str = "BTC", days_1h: int = 60, days_4h: int = 120) -> tuple:
        """1時間足と4時間足を両方取得して返す"""
        df_1h = self.fetch_ohlcv(symbol=symbol, interval="1hour", days=days_1h)
        df_4h = self.fetch_ohlcv(symbol=symbol, interval="4hour", days=days_4h)
        return df_1h, df_4h

    def _parse(self, data: list) -> pd.DataFrame:
        rows = []
        for d in data:
            rows.append({
                "timestamp": pd.to_datetime(int(d[0]), unit="ms"),
                "open":   float(d[1]),
                "high":   float(d[2]),
                "low":    float(d[3]),
                "close":  float(d[4]),
                "volume": float(d[5]),
            })
        return pd.DataFrame(rows)
