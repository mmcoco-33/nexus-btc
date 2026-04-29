"""テクニカル指標を特徴量として生成"""
import pandas as pd
import numpy as np
import ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # トレンド系
    df["ema_9"]  = ta.trend.ema_indicator(c, window=9)
    df["ema_21"] = ta.trend.ema_indicator(c, window=21)
    df["ema_50"] = ta.trend.ema_indicator(c, window=50)
    df["macd"]       = ta.trend.macd(c)
    df["macd_signal"] = ta.trend.macd_signal(c)
    df["macd_diff"]   = ta.trend.macd_diff(c)
    df["adx"] = ta.trend.adx(h, l, c)

    # モメンタム系
    df["rsi_14"] = ta.momentum.rsi(c, window=14)
    df["rsi_7"]  = ta.momentum.rsi(c, window=7)
    df["stoch_k"] = ta.momentum.stoch(h, l, c)
    df["stoch_d"] = ta.momentum.stoch_signal(h, l, c)
    df["cci"]     = ta.trend.cci(h, l, c)

    # ボラティリティ系
    bb = ta.volatility.BollingerBands(c)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / c
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"] = ta.volatility.average_true_range(h, l, c)

    # 出来高系
    df["obv"] = ta.volume.on_balance_volume(c, v)
    df["vwap"] = (c * v).cumsum() / v.cumsum()

    # 価格変化率
    for n in [1, 3, 6, 12, 24]:
        df[f"ret_{n}"] = c.pct_change(n)

    # EMAクロス
    df["ema_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)

    # ラベル：次の足が上がるか（学習用）
    df["target"] = (c.shift(-1) > c).astype(int)

    return df.dropna()
