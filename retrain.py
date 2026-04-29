"""毎日0時に実行：モデル再学習スクリプト"""
import os
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")


def main():
    print(f"[{datetime.now()}] 日次再学習開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    print("データ取得中（60日分）...")
    df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=60)
    df    = add_features(df_1h)
    print(f"データ件数: {len(df)}件")

    print("モデル学習中...")
    model = NexusEnsemble()
    model.train(df)

    print(f"[{datetime.now()}] 再学習完了 → models/ に保存")


if __name__ == "__main__":
    main()
