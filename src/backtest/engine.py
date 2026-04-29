"""③ バックテストエンジン"""
import pandas as pd
import numpy as np


def run_backtest(
    df: pd.DataFrame,
    proba_col: str = "proba",
    buy_threshold: float = 0.60,
    sell_threshold: float = 0.40,
    stop_loss: float = 0.03,
    take_profit: float = 0.06,
    initial_capital: float = 100_000,
) -> dict:
    """
    シグナル列 proba_col を使ってバックテストを実行。
    Buy&Hold との比較結果を返す。
    """
    capital   = initial_capital
    position  = 0.0   # 保有BTC量
    entry_px  = 0.0
    trades    = []
    equity    = []

    for i, row in df.iterrows():
        price = row["close"]
        prob  = row[proba_col]

        # 損切り・利確チェック
        if position > 0:
            if price <= entry_px * (1 - stop_loss):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"type": "SELL_SL", "price": price, "pnl": pnl})
                position = 0.0
            elif price >= entry_px * (1 + take_profit):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"type": "SELL_TP", "price": price, "pnl": pnl})
                position = 0.0

        # シグナル判定
        if prob >= buy_threshold and position == 0 and capital > 0:
            position  = capital / price
            entry_px  = price
            capital   = 0.0
            trades.append({"type": "BUY", "price": price, "pnl": 0})

        elif prob <= sell_threshold and position > 0:
            pnl = position * (price - entry_px)
            capital += position * price
            trades.append({"type": "SELL_SIG", "price": price, "pnl": pnl})
            position = 0.0

        equity.append(capital + position * price)

    # 最終決済
    if position > 0:
        final_price = df["close"].iloc[-1]
        capital += position * final_price

    # 指標計算
    equity_series = pd.Series(equity)
    returns       = equity_series.pct_change().dropna()
    total_return  = (capital - initial_capital) / initial_capital

    # Buy&Hold
    bh_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    # シャープレシオ（年率換算、1時間足想定）
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0

    # 最大ドローダウン
    peak = equity_series.cummax()
    dd   = (equity_series - peak) / peak
    max_dd = dd.min()

    # 勝率
    sell_trades = [t for t in trades if t["type"].startswith("SELL")]
    win_trades  = [t for t in sell_trades if t["pnl"] > 0]
    win_rate    = len(win_trades) / len(sell_trades) if sell_trades else 0

    return {
        "total_return_pct":  round(total_return * 100, 2),
        "buy_hold_pct":      round(bh_return * 100, 2),
        "alpha_pct":         round((total_return - bh_return) * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "win_rate_pct":      round(win_rate * 100, 2),
        "total_trades":      len(sell_trades),
        "final_capital":     round(capital, 0),
        "equity":            equity,
    }


def print_report(result: dict):
    print("=" * 40)
    print("NEXUS-BTC バックテスト結果")
    print("=" * 40)
    print(f"総リターン:      {result['total_return_pct']:+.2f}%")
    print(f"Buy&Hold:        {result['buy_hold_pct']:+.2f}%")
    print(f"アルファ:        {result['alpha_pct']:+.2f}%")
    print(f"シャープレシオ:  {result['sharpe']:.3f}")
    print(f"最大DD:          {result['max_drawdown_pct']:.2f}%")
    print(f"勝率:            {result['win_rate_pct']:.1f}%")
    print(f"取引回数:        {result['total_trades']}")
    print(f"最終資産:        ¥{result['final_capital']:,.0f}")
    print("=" * 40)
