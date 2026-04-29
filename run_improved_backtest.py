"""
改善版バックテスト
改善点:
1. トレーリングストップ（利益を伸ばす）
2. ATRベースの損切り（市場ボラに合わせる）
3. エントリー条件を緩和（機会を増やす）
4. 複利運用（利益を次の取引に再投資）
5. 複数バージョンを比較
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
INITIAL    = 100_000

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
        self.xgb = XGBClassifier(
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

    def predict(self, row_features: np.ndarray) -> float:
        X_s = self.scaler.transform(row_features.reshape(1, -1))
        return self.xgb.predict_proba(X_s)[0, 1]


# ============================================================
# バージョン1: ベースライン（現行）
# ============================================================
def backtest_v1(test_df, model, stop_loss=0.025, take_profit=0.05, threshold=0.40):
    """固定損切り・固定利確"""
    capital = INITIAL
    pos = entry = 0.0
    trades = []
    equity = [capital]

    for _, row in test_df.iterrows():
        price = row["close"]
        feat  = np.array([row.get(c, 0) for c in FEATURE_COLS])
        prob  = model.predict(feat)

        if pos > 0:
            if price <= entry * (1 - stop_loss):
                capital += pos * price; trades.append(("SL", price, entry, pos)); pos = 0.0
            elif price >= entry * (1 + take_profit):
                capital += pos * price; trades.append(("TP", price, entry, pos)); pos = 0.0

        up = row.get("ema_cross_21_50", 0) == 1
        a200 = row.get("price_vs_ema200", 0) == 1
        adx = row.get("adx", 0) > 20

        if up and a200 and adx and prob >= threshold and pos == 0 and capital > 0:
            pos = capital / price; entry = price; capital = 0.0
        elif (not up) and pos > 0 and prob < 0.45:
            capital += pos * price; trades.append(("SIG", price, entry, pos)); pos = 0.0

        equity.append(capital + pos * price)

    if pos > 0:
        fp = test_df["close"].iloc[-1]
        capital += pos * fp; trades.append(("END", fp, entry, pos))

    return _calc(equity, trades, test_df)


# ============================================================
# バージョン2: トレーリングストップ + ATR損切り
# ============================================================
def backtest_v2(test_df, model, atr_mult=2.0, trail_pct=0.03, threshold=0.40):
    """ATRベース損切り + トレーリングストップ"""
    capital = INITIAL
    pos = entry = peak = 0.0
    trades = []
    equity = [capital]

    for _, row in test_df.iterrows():
        price = row["close"]
        atr   = row.get("atr", price * 0.01)
        feat  = np.array([row.get(c, 0) for c in FEATURE_COLS])
        prob  = model.predict(feat)

        if pos > 0:
            peak = max(peak, price)
            sl_atr   = entry - atr * atr_mult          # ATRベース損切り
            sl_trail = peak * (1 - trail_pct)           # トレーリングストップ
            sl_price = max(sl_atr, sl_trail)

            if price <= sl_price:
                reason = "TRAIL" if price <= peak * (1 - trail_pct) else "ATR_SL"
                capital += pos * price; trades.append((reason, price, entry, pos)); pos = 0.0; peak = 0.0

        up   = row.get("ema_cross_21_50", 0) == 1
        a200 = row.get("price_vs_ema200", 0) == 1
        adx  = row.get("adx", 0) > 15  # 条件を少し緩和

        if up and a200 and adx and prob >= threshold and pos == 0 and capital > 0:
            pos = capital / price; entry = price; peak = price; capital = 0.0
        elif (not up) and pos > 0 and prob < 0.42:
            capital += pos * price; trades.append(("SIG", price, entry, pos)); pos = 0.0; peak = 0.0

        equity.append(capital + pos * price)

    if pos > 0:
        fp = test_df["close"].iloc[-1]
        capital += pos * fp; trades.append(("END", fp, entry, pos))

    return _calc(equity, trades, test_df)


# ============================================================
# バージョン3: 複利 + トレーリング + 緩和条件
# ============================================================
def backtest_v3(test_df, model, atr_mult=1.8, trail_pct=0.025, threshold=0.38):
    """複利運用 + より積極的なエントリー"""
    capital = INITIAL
    pos = entry = peak = 0.0
    trades = []
    equity = [capital]

    for _, row in test_df.iterrows():
        price = row["close"]
        atr   = row.get("atr", price * 0.01)
        feat  = np.array([row.get(c, 0) for c in FEATURE_COLS])
        prob  = model.predict(feat)

        if pos > 0:
            peak = max(peak, price)
            sl = max(entry - atr * atr_mult, peak * (1 - trail_pct))
            if price <= sl:
                capital += pos * price; trades.append(("SL", price, entry, pos)); pos = 0.0; peak = 0.0

        # 条件をさらに緩和: EMA9>EMA21 でも可
        up_fast  = row.get("ema_cross_9_21", 0) == 1
        up_slow  = row.get("ema_cross_21_50", 0) == 1
        a200     = row.get("price_vs_ema200", 0) == 1
        adx      = row.get("adx", 0) > 15
        rsi_ok   = row.get("rsi_14", 50) < 70  # 過買いでない

        entry_ok = (up_slow or up_fast) and a200 and adx and rsi_ok

        if entry_ok and prob >= threshold and pos == 0 and capital > 0:
            pos = capital / price; entry = price; peak = price; capital = 0.0
        elif pos > 0 and prob < 0.40 and not up_slow:
            capital += pos * price; trades.append(("SIG", price, entry, pos)); pos = 0.0; peak = 0.0

        equity.append(capital + pos * price)

    if pos > 0:
        fp = test_df["close"].iloc[-1]
        capital += pos * fp; trades.append(("END", fp, entry, pos))

    return _calc(equity, trades, test_df)


# ============================================================
# バージョン4: 押し目買い特化（RSI+BB）
# ============================================================
def backtest_v4(test_df, model, trail_pct=0.03, threshold=0.42):
    """押し目（RSI低い・BB下限付近）でのみエントリー"""
    capital = INITIAL
    pos = entry = peak = 0.0
    trades = []
    equity = [capital]

    for _, row in test_df.iterrows():
        price = row["close"]
        atr   = row.get("atr", price * 0.01)
        feat  = np.array([row.get(c, 0) for c in FEATURE_COLS])
        prob  = model.predict(feat)

        if pos > 0:
            peak = max(peak, price)
            sl = max(entry - atr * 2.5, peak * (1 - trail_pct))
            if price <= sl:
                capital += pos * price; trades.append(("SL", price, entry, pos)); pos = 0.0; peak = 0.0

        up   = row.get("ema_cross_21_50", 0) == 1
        a200 = row.get("price_vs_ema200", 0) == 1
        rsi  = row.get("rsi_14", 50)
        bb   = row.get("bb_pct", 0.5)
        adx  = row.get("adx", 0)

        # 押し目条件: 上昇トレンド中のRSI低下 or BB下限付近
        dip = (rsi < 45) or (bb < 0.35)
        entry_ok = up and a200 and adx > 15 and dip

        if entry_ok and prob >= threshold and pos == 0 and capital > 0:
            pos = capital / price; entry = price; peak = price; capital = 0.0
        elif pos > 0 and prob < 0.40 and not up:
            capital += pos * price; trades.append(("SIG", price, entry, pos)); pos = 0.0; peak = 0.0

        equity.append(capital + pos * price)

    if pos > 0:
        fp = test_df["close"].iloc[-1]
        capital += pos * fp; trades.append(("END", fp, entry, pos))

    return _calc(equity, trades, test_df)


def _calc(equity, trades, df):
    capital = equity[-1]
    eq = pd.Series(equity)
    ret = eq.pct_change().dropna()
    bh  = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    total_ret = (capital - INITIAL) / INITIAL
    sharpe = (ret.mean() / ret.std()) * np.sqrt(24 * 365) if ret.std() > 0 else 0
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
    wins  = [(p, e, s) for t, p, e, s in trades if p > e]
    loses = [(p, e, s) for t, p, e, s in trades if p <= e]
    pf = (sum((p-e)*s for p,e,s in wins) / abs(sum((p-e)*s for p,e,s in loses))) if loses else float("inf")
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win  = np.mean([(p/e-1)*100 for p,e,s in wins]) if wins else 0
    avg_lose = np.mean([(p/e-1)*100 for p,e,s in loses]) if loses else 0
    return {
        "total_return_pct":  round(total_ret * 100, 2),
        "profit_jpy":        round(capital - INITIAL, 0),
        "final_capital":     round(capital, 0),
        "buy_hold_pct":      round(bh * 100, 2),
        "alpha_pct":         round((total_ret - bh) * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "profit_factor":     round(pf, 2),
        "total_trades":      len(trades),
        "win_rate_pct":      round(win_rate, 1),
        "avg_win_pct":       round(avg_win, 2),
        "avg_lose_pct":      round(avg_lose, 2),
    }


def run_all_scenarios(df, versions):
    total = len(df)
    scenarios = [
        ("前半→後半",  (0, int(total*0.60)), (int(total*0.60), total)),
        ("中盤→終盤",  (int(total*0.35), int(total*0.70)), (int(total*0.70), total)),
        ("長期→直近",  (0, int(total*0.75)), (int(total*0.75), total)),
    ]

    all_results = {v["name"]: [] for v in versions}

    for sc_name, train_range, test_range in scenarios:
        train_df = df.iloc[train_range[0]:train_range[1]].copy()
        test_df  = df.iloc[test_range[0]:test_range[1]].copy().reset_index(drop=True)

        model = TrendAIModel()
        model.train(train_df)

        for v in versions:
            r = v["fn"](test_df, model)
            all_results[v["name"]].append({
                "scenario": sc_name,
                "train": f"{train_df['timestamp'].iloc[0].strftime('%m/%d')}〜{train_df['timestamp'].iloc[-1].strftime('%m/%d')}",
                "test":  f"{test_df['timestamp'].iloc[0].strftime('%m/%d')}〜{test_df['timestamp'].iloc[-1].strftime('%m/%d')}",
                **r
            })

    return all_results


def main():
    print(f"[{datetime.now()}] 改善版バックテスト開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)
    print("データ取得中（180日分）...")
    df_raw = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df     = add_features(df_raw)
    print(f"総データ: {len(df)}件")

    versions = [
        {"name": "V1: ベースライン",          "fn": backtest_v1},
        {"name": "V2: トレーリング+ATR",       "fn": backtest_v2},
        {"name": "V3: 複利+緩和条件",          "fn": backtest_v3},
        {"name": "V4: 押し目買い特化",         "fn": backtest_v4},
    ]

    print("全バージョン × 3シナリオ実行中...")
    results = run_all_scenarios(df, versions)

    # 結果表示
    print(f"\n{'='*75}")
    print("改善版バックテスト 比較結果（初期資産 ¥100,000）")
    print(f"{'='*75}")
    print(f"{'バージョン':<22} {'シナリオ':<10} {'リターン':>8} {'利益(円)':>10} {'BH':>8} {'シャープ':>8} {'PF':>5} {'勝率':>6} {'取引':>4}")
    print(f"{'─'*75}")

    summary = {}
    for vname, sc_results in results.items():
        avg_ret    = np.mean([r["total_return_pct"] for r in sc_results])
        avg_profit = np.mean([r["profit_jpy"] for r in sc_results])
        summary[vname] = {"avg_ret": avg_ret, "avg_profit": avg_profit, "results": sc_results}

        for r in sc_results:
            win = "✅" if r["total_return_pct"] > 0 else "❌"
            print(f"{win} {vname:<20} {r['scenario']:<10} "
                  f"{r['total_return_pct']:>+7.2f}% ¥{r['profit_jpy']:>+8,.0f} "
                  f"{r['buy_hold_pct']:>+7.2f}% {r['sharpe']:>8.3f} "
                  f"{r['profit_factor']:>5.2f} {r['win_rate_pct']:>5.1f}% {r['total_trades']:>4}回")
        print(f"   {'平均':>22} {'':>10} {avg_ret:>+7.2f}% ¥{avg_profit:>+8,.0f}")
        print(f"{'─'*75}")

    # 最良バージョン
    best = max(summary.items(), key=lambda x: x[1]["avg_ret"])
    print(f"\n🏆 最良バージョン: {best[0]}")
    print(f"   平均リターン: {best[1]['avg_ret']:+.2f}%  平均利益: ¥{best[1]['avg_profit']:+,.0f}")

    # 詳細トレード（最良バージョンの最良シナリオ）
    best_sc = max(best[1]["results"], key=lambda x: x["total_return_pct"])
    print(f"\n最良シナリオ詳細: {best_sc['scenario']} ({best_sc['train']} → {best_sc['test']})")
    print(f"  リターン: {best_sc['total_return_pct']:+.2f}%  利益: ¥{best_sc['profit_jpy']:+,.0f}")
    print(f"  シャープ: {best_sc['sharpe']:.3f}  最大DD: {best_sc['max_drawdown_pct']:.2f}%")
    print(f"  勝率: {best_sc['win_rate_pct']:.1f}%  PF: {best_sc['profit_factor']:.2f}")
    print(f"  平均利益: {best_sc['avg_win_pct']:+.2f}%  平均損失: {best_sc['avg_lose_pct']:+.2f}%")

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/improved_results.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "best_version": best[0],
                   "summary": {k: {"avg_ret": v["avg_ret"], "avg_profit": v["avg_profit"]}
                                for k, v in summary.items()}},
                  f, ensure_ascii=False, indent=2, default=str)
    print("\ndata/improved_results.json に保存しました")


if __name__ == "__main__":
    main()
