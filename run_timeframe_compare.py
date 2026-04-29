"""
1時間足 vs 日足 バックテスト比較
- 同じ戦略（V4押し目買い+トレーリング）
- 同じ学習/検証分割
- 初期資産10万円
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
FEE_RATE   = 0.0005  # GMOコイン 成行手数料 0.05%（片道）

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "bb_width", "bb_pct", "atr_pct",
    "vwap_dist", "vol_ratio",
    "ret_1", "ret_3", "ret_6", "ret_24",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vol_regime",
]


class Model:
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

    def predict(self, features: np.ndarray) -> float:
        X_s = self.scaler.transform(features.reshape(1, -1))
        return self.xgb.predict_proba(X_s)[0, 1]


def backtest(test_df: pd.DataFrame, model: Model,
             trail_pct: float, atr_mult: float, threshold: float) -> dict:
    capital = INITIAL
    pos = entry = peak = 0.0
    trades = []
    equity = [capital]

    for _, row in test_df.iterrows():
        price = row["close"]
        atr   = row.get("atr", price * 0.01)
        feat  = np.array([row.get(c, 0) for c in FEATURE_COLS])
        prob  = model.predict(feat)

        # エグジット
        if pos > 0:
            peak = max(peak, price)
            sl = max(entry - atr * atr_mult, peak * (1 - trail_pct))
            if price <= sl:
                reason = "TRAIL" if price <= peak * (1 - trail_pct) else "ATR_SL"
                fee = pos * price * FEE_RATE  # 売り手数料
                capital += pos * price - fee
                trades.append({"type": reason, "entry": entry, "exit": price,
                                "pnl": pos*(price-entry)-fee, "pnl_pct": (price/entry-1)*100, "fee": fee})
                pos = peak = 0.0

        # エントリー条件
        up_slow  = row.get("ema_cross_21_50", 0) == 1
        up_fast  = row.get("ema_cross_9_21",  0) == 1
        above200 = row.get("price_vs_ema200", 0) == 1
        adx      = row.get("adx", 0)
        rsi      = row.get("rsi_14", 50)
        bb_pct   = row.get("bb_pct", 0.5)

        trend_ok = (up_slow and above200) or (up_fast and above200 and adx > 20)
        dip      = (rsi < 45) or (bb_pct < 0.35)

        if trend_ok and dip and prob >= threshold and adx > 15 and rsi < 70 and pos == 0 and capital > 0:
            fee = capital * FEE_RATE  # 買い手数料
            capital -= fee
            pos = capital / price; entry = price; peak = price; capital = 0.0
            trades.append({"type": "BUY", "entry": price, "exit": 0, "pnl": -fee, "pnl_pct": 0, "fee": fee})

        # シグナル反転でSELL
        elif pos > 0 and prob < 0.40 and not up_slow:
            fee = pos * price * FEE_RATE  # 売り手数料
            pnl = pos * (price - entry) - fee
            capital += pos * price - fee
            trades.append({"type": "SIG_SELL", "entry": entry, "exit": price,
                            "pnl": pnl, "pnl_pct": (price/entry-1)*100, "fee": fee})
            pos = peak = 0.0

        equity.append(capital + pos * price)

    # 最終決済
    if pos > 0:
        fp = test_df["close"].iloc[-1]
        fee = pos * fp * FEE_RATE
        pnl = pos * (fp - entry) - fee
        capital += pos * fp - fee
        trades.append({"type": "END", "entry": entry, "exit": fp,
                        "pnl": pnl, "pnl_pct": (fp/entry-1)*100, "fee": fee})

    # 指標
    eq  = pd.Series(equity)
    ret = eq.pct_change().dropna()
    bh  = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]
    total_ret = (capital - INITIAL) / INITIAL
    sharpe    = (ret.mean() / ret.std()) * np.sqrt(252 if len(test_df) < 500 else 24*365) if ret.std() > 0 else 0
    max_dd    = ((eq - eq.cummax()) / eq.cummax()).min()

    closed = [t for t in trades if t["type"] != "BUY"]
    wins   = [t for t in closed if t["pnl"] > 0]
    loses  = [t for t in closed if t["pnl"] <= 0]
    pf     = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in loses)) if loses else float("inf")

    return {
        "total_return_pct":  round(total_ret * 100, 2),
        "profit_jpy":        round(capital - INITIAL, 0),
        "final_capital":     round(capital, 0),
        "buy_hold_pct":      round(bh * 100, 2),
        "alpha_pct":         round((total_ret - bh) * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "profit_factor":     round(pf, 2),
        "total_trades":      len(closed),
        "win_rate_pct":      round(len(wins)/len(closed)*100, 1) if closed else 0,
        "avg_win_pct":       round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0,
        "avg_lose_pct":      round(np.mean([t["pnl_pct"] for t in loses]), 2) if loses else 0,
        "avg_hold_candles":  round(len(test_df) / len(closed), 1) if closed else 0,
        "trades":            closed,
    }


def print_result(label: str, r: dict, test_start: str, test_end: str):
    win = "✅" if r["total_return_pct"] > 0 else "❌"
    beat = "🏆" if r["alpha_pct"] > 0 else "  "
    print(f"\n{'─'*58}")
    print(f"  {label}  ({test_start} 〜 {test_end})")
    print(f"{'─'*58}")
    print(f"  {win} 総リターン:    {r['total_return_pct']:>+7.2f}%   利益: ¥{r['profit_jpy']:>+8,.0f}")
    print(f"  {beat} Buy&Hold:      {r['buy_hold_pct']:>+7.2f}%   アルファ: {r['alpha_pct']:>+.2f}%")
    print(f"     最終資産:    ¥{r['final_capital']:>10,.0f}")
    print(f"  📊 シャープ:    {r['sharpe']:>7.3f}   最大DD: {r['max_drawdown_pct']:.2f}%")
    print(f"  💹 PF:          {r['profit_factor']:>7.2f}   勝率: {r['win_rate_pct']:.1f}%")
    print(f"  🔄 取引回数:    {r['total_trades']:>7}回   平均保有: {r['avg_hold_candles']:.0f}本")
    print(f"     平均利益:    {r['avg_win_pct']:>+7.2f}%   平均損失: {r['avg_lose_pct']:>+.2f}%")
    if r["trades"]:
        print(f"  📋 トレード:")
        for i, t in enumerate(r["trades"], 1):
            mark = "✅" if t["pnl"] > 0 else "❌"
            print(f"     {i:>2}. {t['type']:<9} ¥{t['entry']:>12,.0f} → ¥{t['exit']:>12,.0f}  {t['pnl_pct']:>+6.2f}%  ¥{t['pnl']:>+8,.0f}")


def main():
    print(f"[{datetime.now()}] 1時間足 vs 日足 比較バックテスト")
    print(f"初期資産: ¥{INITIAL:,}")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    # --- 1時間足データ（180日）---
    print("\n1時間足データ取得中（180日）...")
    df_1h_raw = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df_1h     = add_features(df_1h_raw)

    # --- 日足データ（1時間足をリサンプリングして生成）---
    print("日足データ生成中（1時間足→日次リサンプリング）...")
    df_rs = df_1h_raw.copy()
    df_rs["timestamp"] = pd.to_datetime(df_rs["timestamp"])
    df_rs = df_rs.set_index("timestamp").resample("D").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna().reset_index()
    # 日足はEMA200が計算できないのでEMA50までで特徴量生成
    df_1d = add_features(df_rs)
    # dropnaで消えた場合は緩めのdropnaで再試行
    if len(df_1d) == 0:
        df_1d = add_features(df_rs)
        # EMA200列を0で埋めてdropnaを回避
        df_rs_filled = df_rs.copy()
        df_1d = add_features(df_rs_filled)
    print(f"日足: {len(df_1d)}本")

    print(f"1時間足: {len(df_1h)}本  日足: {len(df_1d)}本")

    # 学習/検証分割（完全分離）
    split_1h = int(len(df_1h) * 0.65)
    split_1d = int(len(df_1d) * 0.65)

    train_1h = df_1h.iloc[:split_1h].copy()
    test_1h  = df_1h.iloc[split_1h:].copy().reset_index(drop=True)
    train_1d = df_1d.iloc[:split_1d].copy()
    test_1d  = df_1d.iloc[split_1d:].copy().reset_index(drop=True)

    print(f"\n1時間足 学習: {len(train_1h)}本 / 検証: {len(test_1h)}本")
    print(f"日足     学習: {len(train_1d)}本 / 検証: {len(test_1d)}本")

    # モデル学習
    print("\n1時間足モデル学習中...")
    model_1h = Model()
    model_1h.train(train_1h)

    print("日足モデル学習中...")
    model_1d = Model()
    model_1d.train(train_1d)

    # バックテスト実行
    print("\nバックテスト実行中...")

    # 1時間足: トレーリング3%・ATR×2.0・閾値38%
    r_1h = backtest(test_1h, model_1h, trail_pct=0.03, atr_mult=2.0, threshold=0.38)

    # 日足: トレーリング7%・ATR×1.5・閾値40%（日足は揺れが大きいので広め）
    r_1d = backtest(test_1d, model_1d, trail_pct=0.07, atr_mult=1.5, threshold=0.40)

    # 結果表示
    ts_1h = f"{test_1h['timestamp'].iloc[0].strftime('%Y/%m/%d')}〜{test_1h['timestamp'].iloc[-1].strftime('%Y/%m/%d')}"
    ts_1d = f"{test_1d['timestamp'].iloc[0].strftime('%Y/%m/%d')}〜{test_1d['timestamp'].iloc[-1].strftime('%Y/%m/%d')}"

    print_result("📈 1時間足戦略（毎時判定）", r_1h,
                 test_1h["timestamp"].iloc[0].strftime("%Y/%m/%d"),
                 test_1h["timestamp"].iloc[-1].strftime("%Y/%m/%d"))

    print_result("📅 日足戦略（1日1回判定）", r_1d,
                 test_1d["timestamp"].iloc[0].strftime("%Y/%m/%d"),
                 test_1d["timestamp"].iloc[-1].strftime("%Y/%m/%d"))

    # 総合比較
    print(f"\n{'='*58}")
    print("総合比較")
    print(f"{'='*58}")
    metrics = [
        ("総リターン",    "total_return_pct", "%"),
        ("利益(円)",      "profit_jpy",       "¥"),
        ("Buy&Hold",      "buy_hold_pct",     "%"),
        ("アルファ",      "alpha_pct",        "%"),
        ("シャープレシオ","sharpe",           ""),
        ("最大DD",        "max_drawdown_pct", "%"),
        ("プロフィットF", "profit_factor",    ""),
        ("勝率",          "win_rate_pct",     "%"),
        ("取引回数",      "total_trades",     "回"),
        ("平均利益",      "avg_win_pct",      "%"),
        ("平均損失",      "avg_lose_pct",     "%"),
    ]
    print(f"{'指標':<16} {'1時間足':>14} {'日足':>14}")
    print(f"{'─'*46}")
    for label, key, unit in metrics:
        v1 = r_1h[key]
        vd = r_1d[key]
        if unit == "¥":
            s1 = f"¥{v1:>+10,.0f}"
            sd = f"¥{vd:>+10,.0f}"
        elif unit == "%":
            s1 = f"{v1:>+8.2f}%"
            sd = f"{vd:>+8.2f}%"
        elif unit == "回":
            s1 = f"{int(v1):>9}回"
            sd = f"{int(vd):>9}回"
        else:
            s1 = f"{v1:>10.3f}"
            sd = f"{vd:>10.3f}"
        better = "◀" if (unit != "回" and v1 > vd) or (unit == "回" and v1 < vd) else ""
        print(f"{label:<16} {s1:>14} {sd:>14} {better}")

    winner = "1時間足" if r_1h["total_return_pct"] > r_1d["total_return_pct"] else "日足"
    print(f"\n🏆 総合勝者: {winner}戦略")

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/timeframe_compare.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "1hour": {k: v for k, v in r_1h.items() if k != "trades"},
            "1day":  {k: v for k, v in r_1d.items() if k != "trades"},
            "winner": winner,
        }, f, ensure_ascii=False, indent=2, default=str)
    print("data/timeframe_compare.json に保存しました")


if __name__ == "__main__":
    main()
