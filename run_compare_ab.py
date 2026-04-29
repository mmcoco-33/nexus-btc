"""
A: モデル見直し（特徴量重要度で絞り込み + 過学習対策）
B: トレンドフォロー戦略（AIシグナル + EMAトレンド確認）
の比較バックテスト
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble
from src.backtest.engine import run_backtest

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")


# ============================================================
# 戦略A: モデル見直し
# - 特徴量重要度上位20個に絞る
# - XGBoostのみ（LSTMなし）でシンプルに
# - 過学習対策: max_depth=3, min_child_weight=10
# ============================================================
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

FEATURE_COLS_ALL = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "rsi_21", "stoch_k", "stoch_d", "cci", "williams_r", "roc",
    "bb_width", "bb_pct", "atr", "atr_pct",
    "obv", "vwap_dist", "vol_ratio",
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "vol_regime",
]


class ModelA:
    """シンプルXGBoost + 特徴量選択"""
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, gamma=0.5,
            reg_alpha=1.0, reg_lambda=2.0,
            use_label_encoder=False, eval_metric="auc",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.top_features = []

    def train(self, df: pd.DataFrame):
        X = df[FEATURE_COLS_ALL].values
        y = df["target"].values
        X_s = self.scaler.fit_transform(X)
        self.xgb.fit(X_s, y, verbose=False)

        # 重要度上位20特徴量を選択
        imp = self.xgb.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        self.top_features = [FEATURE_COLS_ALL[i] for i in top_idx]
        print(f"  [A] 選択特徴量: {self.top_features[:5]}... (top20)")

        # 選択した特徴量で再学習
        X2 = df[self.top_features].values
        X2_s = self.scaler.fit_transform(X2)
        self.xgb.fit(X2_s, y, verbose=False)

    def predict_proba(self, df: pd.DataFrame) -> float:
        cols = [c for c in self.top_features if c in df.columns]
        X = df[cols].values
        X_s = self.scaler.transform(X)
        return self.xgb.predict_proba(X_s[-1:])[0, 1]


# ============================================================
# 戦略B: トレンドフォロー（AIシグナル + EMAトレンド確認）
# - EMA21 > EMA50 のときだけBUYシグナルを有効化
# - AIの確信度は30%以上で十分（トレンドが主役）
# - 下降トレンドでは積極的にSELL
# ============================================================

class StrategyB:
    """トレンドフォロー戦略"""
    def __init__(self, ai_model):
        self.model = ai_model

    def get_signal(self, df: pd.DataFrame) -> tuple[str, float]:
        """(action, confidence) を返す"""
        prob = self.model.predict_proba(df)
        row = df.iloc[-1]

        uptrend   = row.get("ema_cross_21_50", 0) == 1  # EMA21 > EMA50
        above_200 = row.get("price_vs_ema200", 0) == 1  # 価格 > EMA200
        rsi       = row.get("rsi_14", 50)
        adx       = row.get("adx", 0)
        strong_trend = adx > 25  # ADX>25 = トレンドが強い

        # BUY条件: 上昇トレンド + AI確信度30%以上 + RSI過売りでない
        if uptrend and above_200 and prob >= 0.30 and rsi < 75 and strong_trend:
            return "BUY", prob

        # SELL条件: 下降トレンド or AI確信度低い
        if (not uptrend) and prob <= 0.50:
            return "SELL", prob

        return "HOLD", prob


def run_strategy_b_backtest(df: pd.DataFrame, model, initial_capital=100_000,
                             stop_loss=0.03, take_profit=0.06) -> dict:
    """戦略Bのバックテスト"""
    strategy = StrategyB(model)
    capital  = initial_capital
    position = 0.0
    entry_px = 0.0
    trades   = []
    equity   = []

    for i in range(len(df)):
        ctx = df.iloc[:i+1]
        price = df["close"].iloc[i]

        # 損切り・利確
        if position > 0:
            if price <= entry_px * (1 - stop_loss):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"type": "SELL_SL", "pnl": pnl})
                position = 0.0
            elif price >= entry_px * (1 + take_profit):
                pnl = position * (price - entry_px)
                capital += position * price
                trades.append({"type": "SELL_TP", "pnl": pnl})
                position = 0.0

        action, prob = strategy.get_signal(ctx)

        if action == "BUY" and position == 0 and capital > 0:
            position = capital / price
            entry_px = price
            capital  = 0.0
            trades.append({"type": "BUY", "pnl": 0})
        elif action == "SELL" and position > 0:
            pnl = position * (price - entry_px)
            capital += position * price
            trades.append({"type": "SELL_SIG", "pnl": pnl})
            position = 0.0

        equity.append(capital + position * price)

    if position > 0:
        capital += position * df["close"].iloc[-1]

    eq = pd.Series(equity)
    ret = eq.pct_change().dropna()
    total_return = (capital - initial_capital) / initial_capital
    bh_return    = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
    sharpe = (ret.mean() / ret.std()) * np.sqrt(24 * 365) if ret.std() > 0 else 0
    peak   = eq.cummax()
    max_dd = ((eq - peak) / peak).min()
    sells  = [t for t in trades if t["type"].startswith("SELL")]
    wins   = [t for t in sells if t["pnl"] > 0]
    win_rate = len(wins) / len(sells) if sells else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_pct":     round(bh_return * 100, 2),
        "alpha_pct":        round((total_return - bh_return) * 100, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct":     round(win_rate * 100, 2),
        "total_trades":     len(sells),
        "final_capital":    round(capital, 0),
    }


# ============================================================
# メイン
# ============================================================

def main():
    print(f"[{datetime.now()}] A/B比較バックテスト開始")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    print("データ取得中（180日分）...")
    df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
    df    = add_features(df_1h)
    print(f"データ件数: {len(df)}件")

    split    = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    print(f"学習: {len(train_df)}件 / 検証: {len(test_df)}件")

    # ---- 戦略A ----
    print("\n[戦略A] モデル見直し学習中...")
    model_a = ModelA()
    model_a.train(train_df)

    print("[戦略A] バックテスト実行中...")
    probas_a = []
    for i in range(len(test_df)):
        ctx = pd.concat([train_df, test_df.iloc[:i+1]]) if i > 0 else train_df
        try:
            p = model_a.predict_proba(ctx)
        except Exception:
            p = 0.5
        probas_a.append(p)
        if (i + 1) % 200 == 0:
            print(f"  A: {i+1}/{len(test_df)}")

    test_a = test_df.copy()
    test_a["proba"] = probas_a
    result_a = run_backtest(test_a, buy_threshold=0.30, sell_threshold=0.70,
                            stop_loss=0.03, take_profit=0.06)

    # ---- 戦略B ----
    print("\n[戦略B] トレンドフォロー学習中...")
    model_b = NexusEnsemble()
    model_b.train(train_df)

    print("[戦略B] バックテスト実行中...")
    result_b = run_strategy_b_backtest(test_df, model_b)

    # ---- 結果比較 ----
    print("\n" + "=" * 55)
    print("A/B 戦略比較結果")
    print("=" * 55)
    header = f"{'指標':<18} {'戦略A(モデル改善)':>16} {'戦略B(トレンド)':>14}"
    print(header)
    print("-" * 55)

    metrics = [
        ("総リターン",      "total_return_pct", "%"),
        ("Buy&Hold",        "buy_hold_pct",      "%"),
        ("アルファ",        "alpha_pct",         "%"),
        ("シャープレシオ",  "sharpe",            ""),
        ("最大DD",          "max_drawdown_pct",  "%"),
        ("勝率",            "win_rate_pct",      "%"),
        ("取引回数",        "total_trades",      "回"),
    ]
    for label, key, unit in metrics:
        va = result_a[key]
        vb = result_b[key]
        fmt = "+.2f" if unit == "%" else ".3f" if unit == "" else "d"
        sa = f"{va:{fmt}}{unit}" if unit != "回" else f"{int(va)}{unit}"
        sb = f"{vb:{fmt}}{unit}" if unit != "回" else f"{int(vb)}{unit}"
        print(f"{label:<18} {sa:>16} {sb:>14}")

    print("=" * 55)

    winner = "A（モデル改善）" if result_a["total_return_pct"] > result_b["total_return_pct"] else "B（トレンドフォロー）"
    print(f"\n勝者: 戦略{winner}")
    print(f"  A総リターン: {result_a['total_return_pct']:+.2f}%  アルファ: {result_a['alpha_pct']:+.2f}%")
    print(f"  B総リターン: {result_b['total_return_pct']:+.2f}%  アルファ: {result_b['alpha_pct']:+.2f}%")

    # 保存
    os.makedirs("data", exist_ok=True)
    with open("data/compare_ab.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "strategy_a": {k: v for k, v in result_a.items() if k != "equity"},
            "strategy_b": result_b,
            "winner": winner,
        }, f, ensure_ascii=False, indent=2, default=str)

    print("\ndata/compare_ab.json に保存しました")


if __name__ == "__main__":
    main()
