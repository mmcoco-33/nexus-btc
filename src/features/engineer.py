"""テクニカル指標を特徴量として生成"""
import pandas as pd
import numpy as np
import ta

# ① 3本後に+1%以上上昇をラベルに
LABEL_HORIZON = 3       # 何本後を見るか
LABEL_THRESHOLD = 0.01  # 上昇率の閾値


def add_features(df: pd.DataFrame, df_4h: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # トレンド系
    df["ema_9"]       = ta.trend.ema_indicator(c, window=9)
    df["ema_21"]      = ta.trend.ema_indicator(c, window=21)
    df["ema_50"]      = ta.trend.ema_indicator(c, window=50)
    df["ema_200"]     = ta.trend.ema_indicator(c, window=200).ffill()
    df["macd"]        = ta.trend.macd(c)
    df["macd_signal"] = ta.trend.macd_signal(c)
    df["macd_diff"]   = ta.trend.macd_diff(c)
    df["adx"]         = ta.trend.adx(h, l, c)
    df["adx_pos"]     = ta.trend.adx_pos(h, l, c)
    df["adx_neg"]     = ta.trend.adx_neg(h, l, c)

    # モメンタム系
    df["rsi_14"]  = ta.momentum.rsi(c, window=14)
    df["rsi_7"]   = ta.momentum.rsi(c, window=7)
    df["rsi_21"]  = ta.momentum.rsi(c, window=21)
    df["stoch_k"] = ta.momentum.stoch(h, l, c)
    df["stoch_d"] = ta.momentum.stoch_signal(h, l, c)
    df["cci"]     = ta.trend.cci(h, l, c)
    df["williams_r"] = ta.momentum.williams_r(h, l, c)
    df["roc"]     = ta.momentum.roc(c, window=10)

    # ボラティリティ系
    bb = ta.volatility.BollingerBands(c)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / c
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"]      = ta.volatility.average_true_range(h, l, c)
    df["atr_pct"]  = df["atr"] / c  # ATRを価格で正規化

    # 出来高系
    df["obv"]  = ta.volume.on_balance_volume(c, v)
    df["vwap"] = (c * v).cumsum() / v.cumsum()
    df["vwap_dist"] = (c - df["vwap"]) / df["vwap"]  # VWAPからの乖離率
    df["vol_ma20"]  = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_ma20"]  # 出来高比率

    # 価格変化率
    for n in [1, 2, 3, 6, 12, 24, 48]:
        df[f"ret_{n}"] = c.pct_change(n)

    # EMAクロス・位置
    df["ema_cross_9_21"]  = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema_cross_21_50"] = (df["ema_21"] > df["ema_50"]).astype(int)
    df["price_vs_ema50"]  = (c > df["ema_50"]).astype(int)
    df["price_vs_ema200"] = (c > df["ema_200"]).astype(int)

    # ② 時間帯・曜日（市場の文脈）
    df["hour"]       = df["timestamp"].dt.hour
    df["dow"]        = df["timestamp"].dt.dayofweek  # 0=月曜
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # 時間帯をsin/cosで循環エンコード
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)

    # ボラティリティ局面（高/低）
    df["vol_regime"] = (df["atr_pct"] > df["atr_pct"].rolling(48).mean()).astype(int)

    # ⑤ 4時間足の特徴量をマージ
    if df_4h is not None:
        df_4h = df_4h.copy()
        df_4h["ema_21_4h"]  = ta.trend.ema_indicator(df_4h["close"], window=21)
        df_4h["rsi_14_4h"]  = ta.momentum.rsi(df_4h["close"], window=14)
        df_4h["macd_4h"]    = ta.trend.macd(df_4h["close"])
        df_4h["trend_4h"]   = (df_4h["close"] > df_4h["ema_21_4h"]).astype(int)
        df_4h = df_4h[["timestamp", "ema_21_4h", "rsi_14_4h", "macd_4h", "trend_4h"]].dropna()
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_4h.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )

    # ① ラベル：3本後に+1%以上上昇
    future_ret = c.shift(-LABEL_HORIZON) / c - 1
    df["target"] = (future_ret >= LABEL_THRESHOLD).astype(int)

    return df.dropna(subset=[c for c in df.columns if c not in ["target", "timestamp"]], how="all")
