"""毎日0時に実行：モデル再学習（XGBoostのみ・軽量版）"""
import os
import pickle
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features

API_KEY    = os.environ.get("GMO_API_KEY", "")
API_SECRET = os.environ.get("GMO_API_SECRET", "")

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "bb_width", "bb_pct", "atr_pct",
    "vwap_dist", "vol_ratio",
    "ret_1", "ret_3", "ret_6", "ret_24",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vol_regime",
]


def main():
    print(f"[{datetime.now()}] 日次再学習開始（XGBoost軽量版）")

    client  = GMOClient(API_KEY, API_SECRET)
    fetcher = DataFetcher(client)

    print("データ取得中（60日分）...")
    df_1h = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=60)
    df    = add_features(df_1h)
    print(f"データ件数: {len(df)}件")

    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X = df[FEATURE_COLS].values
    y = (df["close"].shift(-1) > df["close"]).astype(int).values
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    pos, neg = y.sum(), len(y) - y.sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.65,
        min_child_weight=8, gamma=0.3,
        reg_alpha=0.5, reg_lambda=1.5,
        scale_pos_weight=neg/pos if pos > 0 else 1,
        use_label_encoder=False, eval_metric="auc", random_state=42,
    )
    xgb.fit(X_s, y, verbose=False)

    os.makedirs("models", exist_ok=True)
    with open("models/xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/meta.pkl", "wb") as f:
        pickle.dump({"xgb_weight": 1.0, "feature_cols": FEATURE_COLS}, f)

    print(f"[{datetime.now()}] 再学習完了 → models/ に保存")
    print(f"  正例率: {y.mean():.1%}  サンプル数: {n}")


if __name__ == "__main__":
    main()
