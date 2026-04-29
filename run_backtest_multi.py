"""複数閾値パターンのバックテスト比較"""
import json
import os
import pandas as pd
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble
from src.backtest.engine import run_backtest, print_report

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")

# テストする閾値パターン（buy_threshold, sell_threshold）
PATTERNS = [
    {"name": "超積極(30%)",  "buy": 0.30, "sell": 0.70},
    {"name": "積極(40%)",    "buy": 0.40, "sell": 0.60},
    {"name": "中間(45%)",    "buy": 0.45, "sell": 0.55},
    {"name": "標準(50%)",    "buy": 0.50, "sell": 0.50},
    {"name": "保守(55%)",    "buy": 0.55, "sell": 0.45},
    {"name": "現行(60%)",    "buy": 0.60, "sell": 0.40},
]


def main():
    print(f"[{datetime.now()}] マルチバックテスト開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    # 学習データを増やす（180日）
    print("データ取得中（180日分）...")
    df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df = add_features(df_1h)
    print(f"データ件数: {len(df)}件")

    # 学習70% / 検証30%
    split = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    print(f"学習: {len(train_df)}件 / 検証: {len(test_df)}件")

    # モデル学習（1回だけ）
    print("モデル学習中...")
    model = NexusEnsemble()
    model.train(train_df)

    # 検証データで予測確率を一括生成
    print("予測確率を生成中...")
    probas = []
    for i in range(len(test_df)):
        ctx = pd.concat([train_df, test_df.iloc[:i+1]]) if i > 0 else train_df
        try:
            p = model.predict_proba(ctx)
        except Exception:
            p = 0.5
        probas.append(p)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_df)} 完了")

    test_df = test_df.copy()
    test_df["proba"] = probas

    # 各閾値パターンでバックテスト
    results = []
    print("\n" + "=" * 60)
    print("閾値別バックテスト結果比較")
    print("=" * 60)
    print(f"{'パターン':<14} {'総リターン':>10} {'BH':>8} {'アルファ':>10} {'シャープ':>8} {'最大DD':>8} {'勝率':>7} {'取引数':>6}")
    print("-" * 60)

    for p in PATTERNS:
        r = run_backtest(
            test_df,
            proba_col="proba",
            buy_threshold=p["buy"],
            sell_threshold=p["sell"],
            stop_loss=0.03,
            take_profit=0.06,
            initial_capital=100_000,
        )
        results.append({**p, **{k: v for k, v in r.items() if k != "equity"}})
        print(
            f"{p['name']:<14} "
            f"{r['total_return_pct']:>+9.2f}% "
            f"{r['buy_hold_pct']:>+7.2f}% "
            f"{r['alpha_pct']:>+9.2f}% "
            f"{r['sharpe']:>8.3f} "
            f"{r['max_drawdown_pct']:>7.2f}% "
            f"{r['win_rate_pct']:>6.1f}% "
            f"{r['total_trades']:>6}"
        )

    print("=" * 60)

    # 最良パターンを表示
    best = max(results, key=lambda x: x["total_return_pct"])
    print(f"\n最良パターン: {best['name']} (総リターン {best['total_return_pct']:+.2f}%)")

    # 結果をJSONに保存
    os.makedirs("data", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "buy_hold_pct": results[0]["buy_hold_pct"],
        "patterns": results,
        "best_pattern": best["name"],
    }
    with open("data/backtest_multi.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print("\ndata/backtest_multi.json に保存しました")


if __name__ == "__main__":
    main()
