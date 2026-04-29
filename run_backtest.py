"""バックテスト実行スクリプト"""
import json
import os
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble
from src.backtest.engine import run_backtest, print_report

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")


def main():
    print(f"[{datetime.now()}] バックテスト開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    # 過去90日分取得（学習60日 + 検証30日）
    print("データ取得中...")
    try:
        df_1h, df_4h = fetcher.fetch_multi_timeframe(symbol="BTC", days_1h=90, days_4h=180)
        df = add_features(df_1h, df_4h=df_4h)
    except Exception as e:
        print(f"4時間足取得失敗、1時間足のみで続行: {e}")
        df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=90)
        df = add_features(df_1h)

    print(f"データ件数: {len(df)}件")

    # 学習データ（前60日）/ 検証データ（後30日）に分割
    split = int(len(df) * 0.67)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    print(f"学習: {len(train_df)}件 / 検証: {len(test_df)}件")

    # モデル学習
    print("モデル学習中...")
    model = NexusEnsemble()
    model.train(train_df)

    # 検証データで予測確率を生成
    print("バックテスト実行中...")
    probas = []
    for i in range(len(test_df)):
        # 直近データを使って予測（学習データ + 検証データの先頭iまで）
        ctx = pd.concat([train_df, test_df.iloc[:i+1]]) if i > 0 else train_df
        try:
            p = model.predict_proba(ctx)
        except Exception:
            p = 0.5
        probas.append(p)

    import pandas as pd
    test_df = test_df.copy()
    test_df["proba"] = probas

    # バックテスト実行
    result = run_backtest(
        test_df,
        proba_col="proba",
        buy_threshold=0.60,
        sell_threshold=0.40,
        stop_loss=0.03,
        take_profit=0.06,
        initial_capital=100_000,
    )

    print_report(result)

    # 結果をJSONに保存
    os.makedirs("data", exist_ok=True)
    save = {k: v for k, v in result.items() if k != "equity"}
    save["timestamp"] = datetime.now().isoformat()
    save["train_size"] = len(train_df)
    save["test_size"]  = len(test_df)

    with open("data/backtest_result.json", "w") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)

    # results.json にもサマリーを追記（ダッシュボード表示用）
    path = "results.json"
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)

    results.append({
        "timestamp": datetime.now().isoformat(),
        "price": test_df["close"].iloc[-1],
        "action": "BACKTEST",
        "confidence": 0.0,
        "executed": False,
        "reason": f"BT: {result['total_return_pct']:+.1f}% vs BH: {result['buy_hold_pct']:+.1f}%",
    })
    with open(path, "w") as f:
        json.dump(results[-500:], f, ensure_ascii=False, default=str)

    print("完了")


if __name__ == "__main__":
    import pandas as pd
    main()
