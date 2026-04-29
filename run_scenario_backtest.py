"""
シナリオ別バックテスト
- 初期資産: 10万円
- 学習期間とバックテスト期間は完全に分離
- 複数の時期でテストして再現性を確認
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")

INITIAL_CAPITAL = 100_000  # 10万円
STOP_LOSS       = 0.025    # 損切り2.5%
TAKE_PROFIT     = 0.05     # 利確5%
BUY_THRESHOLD   = 0.40     # 買いシグナル閾値

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "bb_width", "bb_pct", "atr_pct",
    "vwap_dist", "vol_ratio",
    "ret_1", "ret_3", "ret_6", "ret_24",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vol_regime",
]


class TrendAIModel:
    def __init__(self):
        self.xgb    = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.65,
            min_child_weight=8, gamma=0.3,
            reg_alpha=0.5, reg_lambda=1.5,
            use_label_encoder=False, eval_metric="auc", random_state=42,
        )
        self.scaler = StandardScaler()

    def train(self, df: pd.DataFrame):
        y = (df["close"].shift(-1) > df["close"]).astype(int).values
        X = df[FEATURE_COLS].values
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        X_s = self.scaler.fit_transform(X)
        pos, neg = y.sum(), len(y) - y.sum()
        self.xgb.set_params(scale_pos_weight=neg / pos if pos > 0 else 1)
        self.xgb.fit(X_s, y, verbose=False)

    def predict(self, df: pd.DataFrame) -> float:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        X_s  = self.scaler.transform(df[cols].values[-1:])
        return self.xgb.predict_proba(X_s)[0, 1]


def backtest(test_df: pd.DataFrame, model: TrendAIModel) -> dict:
    capital   = INITIAL_CAPITAL
    position  = 0.0
    entry_px  = 0.0
    entry_ts  = None
    trades    = []
    equity    = [capital]

    for i, row in test_df.iterrows():
        price = row["close"]
        ts    = row["timestamp"]

        # 損切り・利確
        if position > 0:
            if price <= entry_px * (1 - STOP_LOSS):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"ts": str(ts), "type": "SL", "pnl": pnl,
                                "pnl_pct": (price/entry_px-1)*100, "entry": entry_px, "exit": price})
                position = 0.0
            elif price >= entry_px * (1 + TAKE_PROFIT):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"ts": str(ts), "type": "TP", "pnl": pnl,
                                "pnl_pct": (price/entry_px-1)*100, "entry": entry_px, "exit": price})
                position = 0.0

        # シグナル
        ctx  = test_df.loc[:i]
        prob = model.predict(ctx)

        uptrend  = row.get("ema_cross_21_50", 0) == 1
        above200 = row.get("price_vs_ema200", 0) == 1
        adx_ok   = row.get("adx", 0) > 20

        if uptrend and above200 and adx_ok and prob >= BUY_THRESHOLD and position == 0 and capital > 0:
            position = capital / price
            entry_px = price
            entry_ts = ts
            capital  = 0.0

        elif (not uptrend) and position > 0 and prob < 0.45:
            pnl = position * (price - entry_px)
            capital += position * price
            trades.append({"ts": str(ts), "type": "SIG", "pnl": pnl,
                            "pnl_pct": (price/entry_px-1)*100, "entry": entry_px, "exit": price})
            position = 0.0

        equity.append(capital + position * price)

    # 最終決済
    if position > 0:
        fp  = test_df["close"].iloc[-1]
        pnl = position * (fp - entry_px)
        capital += position * fp
        trades.append({"ts": str(test_df["timestamp"].iloc[-1]), "type": "END", "pnl": pnl,
                        "pnl_pct": (fp/entry_px-1)*100, "entry": entry_px, "exit": fp})

    eq  = pd.Series(equity)
    ret = eq.pct_change().dropna()
    bh  = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    sharpe    = (ret.mean() / ret.std()) * np.sqrt(24 * 365) if ret.std() > 0 else 0
    max_dd    = ((eq - eq.cummax()) / eq.cummax()).min()
    wins      = [t for t in trades if t["pnl"] > 0]
    loses     = [t for t in trades if t["pnl"] <= 0]
    pf        = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in loses)) if loses else float("inf")

    return {
        "total_return_pct":  round(total_ret * 100, 2),
        "profit_jpy":        round(capital - INITIAL_CAPITAL, 0),
        "final_capital_jpy": round(capital, 0),
        "buy_hold_pct":      round(bh * 100, 2),
        "alpha_pct":         round((total_ret - bh) * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "profit_factor":     round(pf, 2),
        "total_trades":      len(trades),
        "win_rate_pct":      round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_pct":       round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0,
        "avg_lose_pct":      round(np.mean([t["pnl_pct"] for t in loses]), 2) if loses else 0,
        "trades":            trades,
    }


def print_scenario(name: str, train_period: str, test_period: str, r: dict):
    win = "✅" if r["total_return_pct"] > 0 else "❌"
    beat = "🏆" if r["alpha_pct"] > 0 else "  "
    print(f"\n{'─'*60}")
    print(f"シナリオ: {name}")
    print(f"  学習期間: {train_period}  →  検証期間: {test_period}")
    print(f"{'─'*60}")
    print(f"  {win} 総リターン:     {r['total_return_pct']:>+7.2f}%   (¥{r['profit_jpy']:>+8,.0f})")
    print(f"  {beat} Buy&Hold:       {r['buy_hold_pct']:>+7.2f}%")
    print(f"     アルファ:       {r['alpha_pct']:>+7.2f}%")
    print(f"     最終資産:       ¥{r['final_capital_jpy']:>10,.0f}")
    print(f"  📊 シャープ:       {r['sharpe']:>7.3f}")
    print(f"  📉 最大DD:         {r['max_drawdown_pct']:>7.2f}%")
    print(f"  💹 PF:             {r['profit_factor']:>7.2f}")
    print(f"  🔄 取引回数:       {r['total_trades']:>7}回  勝率: {r['win_rate_pct']:.1f}%")
    print(f"     平均利益:       {r['avg_win_pct']:>+7.2f}%  平均損失: {r['avg_lose_pct']:>+7.2f}%")
    if r["trades"]:
        print(f"  📋 トレード詳細:")
        for i, t in enumerate(r["trades"], 1):
            mark = "✅" if t["pnl"] > 0 else "❌"
            print(f"     {i:>2}. {t['ts'][:16]}  {t['type']:<4}  {t['pnl_pct']:>+6.2f}%  ¥{t['pnl']:>+8,.0f}")


def main():
    print(f"[{datetime.now()}] シナリオ別バックテスト開始")
    print(f"初期資産: ¥{INITIAL_CAPITAL:,}  損切り: {STOP_LOSS*100}%  利確: {TAKE_PROFIT*100}%")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    # 180日分取得（学習+検証を完全分離するため多めに取得）
    print("\nデータ取得中（180日分）...")
    df_raw = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df     = add_features(df_raw)
    print(f"総データ: {len(df)}件  期間: {df['timestamp'].iloc[0].strftime('%Y/%m/%d')} 〜 {df['timestamp'].iloc[-1].strftime('%Y/%m/%d')}")

    total = len(df)

    # シナリオ定義（学習:検証 = 完全分離）
    scenarios = [
        {"name": "シナリオ1: 前半学習→後半検証",
         "train": (0,    int(total*0.60)),
         "test":  (int(total*0.60), total)},
        {"name": "シナリオ2: 序盤学習→中盤検証",
         "train": (0,    int(total*0.40)),
         "test":  (int(total*0.40), int(total*0.65))},
        {"name": "シナリオ3: 中盤学習→終盤検証",
         "train": (int(total*0.35), int(total*0.70)),
         "test":  (int(total*0.70), total)},
        {"name": "シナリオ4: 長期学習→直近検証",
         "train": (0,    int(total*0.75)),
         "test":  (int(total*0.75), total)},
        {"name": "シナリオ5: 短期学習→短期検証",
         "train": (int(total*0.50), int(total*0.70)),
         "test":  (int(total*0.70), int(total*0.85))},
    ]

    all_results = []

    for sc in scenarios:
        train_df = df.iloc[sc["train"][0]:sc["train"][1]].copy()
        test_df  = df.iloc[sc["test"][0]:sc["test"][1]].copy().reset_index(drop=True)

        train_start = train_df["timestamp"].iloc[0].strftime("%Y/%m/%d")
        train_end   = train_df["timestamp"].iloc[-1].strftime("%Y/%m/%d")
        test_start  = test_df["timestamp"].iloc[0].strftime("%Y/%m/%d")
        test_end    = test_df["timestamp"].iloc[-1].strftime("%Y/%m/%d")

        print(f"\n{sc['name']} 学習中... ({len(train_df)}件)")
        model = TrendAIModel()
        model.train(train_df)

        result = backtest(test_df, model)
        print_scenario(sc["name"],
                       f"{train_start}〜{train_end}({len(train_df)}本)",
                       f"{test_start}〜{test_end}({len(test_df)}本)",
                       result)
        all_results.append({"scenario": sc["name"], **{k: v for k, v in result.items() if k != "trades"}})

    # サマリー
    print(f"\n{'='*60}")
    print("全シナリオ サマリー（初期資産 ¥100,000）")
    print(f"{'='*60}")
    print(f"{'シナリオ':<28} {'リターン':>8} {'利益(円)':>10} {'BH':>8} {'勝率':>6} {'取引':>5}")
    print(f"{'─'*60}")
    for r in all_results:
        win = "✅" if r["total_return_pct"] > 0 else "❌"
        print(f"{win} {r['scenario'][:26]:<26} {r['total_return_pct']:>+7.2f}% "
              f"¥{r['profit_jpy']:>+8,.0f} {r['buy_hold_pct']:>+7.2f}% "
              f"{r['win_rate_pct']:>5.1f}% {r['total_trades']:>4}回")

    avg_ret    = np.mean([r["total_return_pct"] for r in all_results])
    avg_profit = np.mean([r["profit_jpy"] for r in all_results])
    win_count  = sum(1 for r in all_results if r["total_return_pct"] > 0)
    print(f"{'─'*60}")
    print(f"平均リターン: {avg_ret:>+.2f}%  平均利益: ¥{avg_profit:>+,.0f}  勝ちシナリオ: {win_count}/{len(all_results)}")
    print(f"{'='*60}")

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/scenario_results.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "initial_capital": INITIAL_CAPITAL,
                   "scenarios": all_results}, f, ensure_ascii=False, indent=2, default=str)
    print("\ndata/scenario_results.json に保存しました")


if __name__ == "__main__":
    main()
