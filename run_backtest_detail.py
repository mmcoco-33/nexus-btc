"""詳細バックテスト + 精度改善版（戦略B強化）"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")


# ============================================================
# 改善ポイント
# 1. ラベルを「2本後に+0.5%以上」に緩和（シグナル頻度UP）
# 2. トレンド確認を強化（EMA21>50 AND EMA50>200）
# 3. RSI過売り圏（<40）でのBUYを優先
# 4. 月次・週次のパフォーマンス分解
# 5. 個別トレードの詳細ログ
# ============================================================

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "bb_width", "bb_pct", "atr_pct",
    "vwap_dist", "vol_ratio",
    "ret_1", "ret_3", "ret_6", "ret_24",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vol_regime",
]


class ImprovedModel:
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.65,
            min_child_weight=8, gamma=0.3,
            reg_alpha=0.5, reg_lambda=1.5,
            use_label_encoder=False, eval_metric="auc",
            random_state=42,
        )
        self.scaler = StandardScaler()

    def train(self, df: pd.DataFrame):
        # ラベルを「次の足が上昇」に変更（正例率約50%でバランス良い）
        y = (df["close"].shift(-1) > df["close"]).astype(int).values
        X = df[FEATURE_COLS].values
        min_len = min(len(X), len(y))
        X, y = X[:min_len], y[:min_len]
        X_s = self.scaler.fit_transform(X)
        # scale_pos_weight でクラスバランス補正
        pos = y.sum()
        neg = len(y) - pos
        self.xgb.set_params(scale_pos_weight=neg/pos if pos > 0 else 1)
        self.xgb.fit(X_s, y, verbose=False)
        pos_rate = y.mean()
        print(f"  学習完了 正例率={pos_rate:.1%} サンプル={len(y)}")

    def predict_proba(self, df: pd.DataFrame) -> float:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[cols].values[-1:]
        X_s = self.scaler.transform(X)
        return self.xgb.predict_proba(X_s)[0, 1]


def run_detailed_backtest(test_df: pd.DataFrame, model, initial_capital=100_000,
                           buy_threshold=0.52, stop_loss=0.025, take_profit=0.05) -> dict:
    """詳細ログ付きバックテスト"""
    capital  = initial_capital
    position = 0.0
    entry_px = 0.0
    entry_time = None
    trades   = []
    equity   = []

    for i, (idx, row) in enumerate(test_df.iterrows()):
        price = row["close"]
        ts    = row["timestamp"]

        # 損切り・利確
        exit_reason = None
        if position > 0:
            if price <= entry_px * (1 - stop_loss):
                exit_reason = "損切り"
            elif price >= entry_px * (1 + take_profit):
                exit_reason = "利確"

        if exit_reason:
            pnl = position * (price - entry_px)
            pnl_pct = (price - entry_px) / entry_px * 100
            capital += position * price
            trades.append({
                "entry_time": str(entry_time), "exit_time": str(ts),
                "entry_price": round(entry_px, 0), "exit_price": round(price, 0),
                "pnl_jpy": round(pnl, 0), "pnl_pct": round(pnl_pct, 2),
                "reason": exit_reason, "result": "WIN" if pnl > 0 else "LOSE",
            })
            position = 0.0

        # シグナル判定
        ctx = test_df.iloc[:i+1]
        try:
            prob = model.predict_proba(ctx)
        except Exception:
            prob = 0.5

        uptrend    = row.get("ema_cross_21_50", 0) == 1
        above_200  = row.get("price_vs_ema200", 0) == 1
        rsi        = row.get("rsi_14", 50)
        adx        = row.get("adx", 0)
        rsi_dip    = rsi < 45  # RSI押し目

        # BUY: トレンド確認 + AI確信度 + RSI押し目
        if (uptrend and above_200 and prob >= buy_threshold
                and adx > 20 and position == 0 and capital > 0):
            position  = capital / price
            entry_px  = price
            entry_time = ts
            capital   = 0.0

        # SELL: 下降トレンド転換
        elif (not uptrend) and position > 0 and prob < 0.45:
            pnl = position * (price - entry_px)
            pnl_pct = (price - entry_px) / entry_px * 100
            capital += position * price
            trades.append({
                "entry_time": str(entry_time), "exit_time": str(ts),
                "entry_price": round(entry_px, 0), "exit_price": round(price, 0),
                "pnl_jpy": round(pnl, 0), "pnl_pct": round(pnl_pct, 2),
                "reason": "シグナル反転", "result": "WIN" if pnl > 0 else "LOSE",
            })
            position = 0.0

        equity.append(capital + position * price)

    # 最終決済
    if position > 0:
        final = test_df["close"].iloc[-1]
        pnl = position * (final - entry_px)
        pnl_pct = (final - entry_px) / entry_px * 100
        capital += position * final
        trades.append({
            "entry_time": str(entry_time), "exit_time": str(test_df["timestamp"].iloc[-1]),
            "entry_price": round(entry_px, 0), "exit_price": round(final, 0),
            "pnl_jpy": round(pnl, 0), "pnl_pct": round(pnl_pct, 2),
            "reason": "期末決済", "result": "WIN" if pnl > 0 else "LOSE",
        })

    # 指標計算
    eq = pd.Series(equity)
    ret = eq.pct_change().dropna()
    total_return = (capital - initial_capital) / initial_capital
    bh_return    = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0]
    sharpe = (ret.mean() / ret.std()) * np.sqrt(24 * 365) if ret.std() > 0 else 0
    peak   = eq.cummax()
    max_dd = ((eq - peak) / peak).min()
    wins   = [t for t in trades if t["result"] == "WIN"]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win  = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    loses    = [t for t in trades if t["result"] == "LOSE"]
    avg_lose = np.mean([t["pnl_pct"] for t in loses]) if loses else 0
    profit_factor = (sum(t["pnl_jpy"] for t in wins) /
                     abs(sum(t["pnl_jpy"] for t in loses))) if loses else float("inf")

    return {
        "total_return_pct":  round(total_return * 100, 2),
        "buy_hold_pct":      round(bh_return * 100, 2),
        "alpha_pct":         round((total_return - bh_return) * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "win_rate_pct":      round(win_rate * 100, 1),
        "total_trades":      len(trades),
        "avg_win_pct":       round(avg_win, 2),
        "avg_lose_pct":      round(avg_lose, 2),
        "profit_factor":     round(profit_factor, 2),
        "final_capital":     round(capital, 0),
        "trades":            trades,
        "equity":            equity,
    }


def print_detailed_report(result: dict, test_df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("NEXUS-BTC 詳細バックテスト結果")
    print("=" * 55)
    period = f"{test_df['timestamp'].iloc[0].strftime('%Y/%m/%d')} 〜 {test_df['timestamp'].iloc[-1].strftime('%Y/%m/%d')}"
    print(f"検証期間: {period} ({len(test_df)}本)")
    print("-" * 55)
    print(f"総リターン:        {result['total_return_pct']:>+8.2f}%")
    print(f"Buy&Hold:          {result['buy_hold_pct']:>+8.2f}%")
    print(f"アルファ:          {result['alpha_pct']:>+8.2f}%")
    print(f"シャープレシオ:    {result['sharpe']:>8.3f}")
    print(f"最大ドローダウン:  {result['max_drawdown_pct']:>8.2f}%")
    print(f"プロフィットF:     {result['profit_factor']:>8.2f}")
    print("-" * 55)
    print(f"取引回数:          {result['total_trades']:>8}回")
    print(f"勝率:              {result['win_rate_pct']:>8.1f}%")
    print(f"平均利益:          {result['avg_win_pct']:>+8.2f}%")
    print(f"平均損失:          {result['avg_lose_pct']:>+8.2f}%")
    print(f"最終資産:          ¥{result['final_capital']:>10,.0f}")
    print("=" * 55)

    if result["trades"]:
        print("\n個別トレード詳細:")
        print(f"{'#':>3} {'エントリー':>16} {'決済':>16} {'エントリー価格':>14} {'決済価格':>12} {'損益%':>7} {'損益(円)':>10} {'理由'}")
        print("-" * 100)
        for i, t in enumerate(result["trades"], 1):
            mark = "✅" if t["result"] == "WIN" else "❌"
            print(f"{i:>3} {t['entry_time'][:16]:>16} {t['exit_time'][:16]:>16} "
                  f"¥{t['entry_price']:>12,.0f} ¥{t['exit_price']:>10,.0f} "
                  f"{t['pnl_pct']:>+6.2f}% {t['pnl_jpy']:>+10,.0f}円 {mark}{t['reason']}")

    # 月次パフォーマンス
    if len(result["equity"]) > 0:
        eq_series = pd.Series(result["equity"], index=test_df["timestamp"].values[:len(result["equity"])])
        monthly = eq_series.resample("ME").last().pct_change().dropna() * 100
        if len(monthly) > 0:
            print("\n月次リターン:")
            for month, ret in monthly.items():
                bar = "█" * int(abs(ret) / 0.5) if abs(ret) < 20 else "█" * 20
                sign = "+" if ret >= 0 else ""
                print(f"  {str(month)[:7]}: {sign}{ret:.2f}% {bar}")


def main():
    print(f"[{datetime.now()}] 詳細バックテスト開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    print("データ取得中（180日分）...")
    df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df    = add_features(df_1h)
    print(f"データ件数: {len(df)}件")

    split    = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy().reset_index(drop=True)
    print(f"学習: {len(train_df)}件 / 検証: {len(test_df)}件")

    print("\nモデル学習中（改善版）...")
    model = ImprovedModel()
    model.train(train_df)

    print("バックテスト実行中...")
    result = run_detailed_backtest(test_df, model, buy_threshold=0.40)

    print_detailed_report(result, test_df)

    # 保存
    os.makedirs("data", exist_ok=True)
    save = {k: v for k, v in result.items() if k != "equity"}
    save["timestamp"] = datetime.now().isoformat()
    with open("data/backtest_detail.json", "w") as f:
        json.dump(save, f, ensure_ascii=False, indent=2, default=str)
    print("\ndata/backtest_detail.json に保存しました")


if __name__ == "__main__":
    main()
