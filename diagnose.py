"""データ・モデル診断スクリプト"""
import pandas as pd
import numpy as np
from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

client  = GMOClient()
fetcher = DataFetcher(client)
df_1h   = fetcher.fetch_ohlcv(symbol="BTC", interval="1hour", days=180)
df      = add_features(df_1h)

split = int(len(df) * 0.70)
train = df.iloc[:split]
test  = df.iloc[split:]

print("=== 検証データの市場状態 ===")
print(f"期間: {test['timestamp'].iloc[0]} 〜 {test['timestamp'].iloc[-1]}")
print(f"EMA21>EMA50 (上昇トレンド): {test['ema_cross_21_50'].mean():.1%}")
print(f"Price>EMA200:               {test['price_vs_ema200'].mean():.1%}")
print(f"ADX>20 (トレンド強い):      {(test['adx']>20).mean():.1%}")
print(f"RSI平均:                    {test['rsi_14'].mean():.1f}")
print(f"全BUY条件同時成立:          {((test['ema_cross_21_50']==1) & (test['price_vs_ema200']==1) & (test['adx']>20)).mean():.1%}")

# ラベル分布
future_ret = df["close"].shift(-2) / df["close"] - 1
label_005 = (future_ret >= 0.005).astype(int)
label_001 = (future_ret >= 0.001).astype(int)
label_up  = (future_ret > 0).astype(int)
print(f"\n=== ラベル分布 ===")
print(f"学習 +0.5%以上: {label_005.iloc[:split].mean():.1%}")
print(f"検証 +0.5%以上: {label_005.iloc[split:].mean():.1%}")
print(f"学習 +0.1%以上: {label_001.iloc[:split].mean():.1%}")
print(f"検証 +0.1%以上: {label_001.iloc[split:].mean():.1%}")
print(f"学習 上昇:      {label_up.iloc[:split].mean():.1%}")
print(f"検証 上昇:      {label_up.iloc[split:].mean():.1%}")

# 予測確率分布
COLS = [
    "ema_9", "ema_21", "ema_50", "macd", "rsi_14", "bb_pct", "atr_pct",
    "ret_1", "ret_3", "ema_cross_21_50", "price_vs_ema200", "adx", "vol_regime",
]
X_tr = train[COLS].values
y_tr = label_up.iloc[:split].values[:len(X_tr)]
sc   = StandardScaler()
X_s  = sc.fit_transform(X_tr)
xgb  = XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False,
                     eval_metric="auc", random_state=42)
xgb.fit(X_s, y_tr, verbose=False)
X_te  = sc.transform(test[COLS].values)
probs = xgb.predict_proba(X_te)[:, 1]

print(f"\n=== 予測確率分布（上昇ラベル使用） ===")
print(f"平均:    {probs.mean():.3f}")
print(f"中央値:  {np.median(probs):.3f}")
print(f">0.55:   {(probs>0.55).mean():.1%}")
print(f">0.52:   {(probs>0.52).mean():.1%}")
print(f">0.50:   {(probs>0.50).mean():.1%}")
print(f">0.45:   {(probs>0.45).mean():.1%}")
print(f">0.40:   {(probs>0.40).mean():.1%}")

# BUY条件と確率の組み合わせ
uptrend = (test["ema_cross_21_50"] == 1).values
above200 = (test["price_vs_ema200"] == 1).values
adx_ok = (test["adx"] > 20).values
print(f"\n=== BUY条件 + 確率の組み合わせ ===")
for thr in [0.40, 0.45, 0.50, 0.52, 0.55]:
    n = (uptrend & above200 & adx_ok & (probs > thr)).sum()
    print(f"全条件 + prob>{thr}: {n}回 ({n/len(test):.1%})")
