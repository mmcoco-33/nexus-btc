"""売買シグナル生成（V4ベース全要素統合版）"""
import numpy as np
import pandas as pd


class SignalGenerator:
    def __init__(self, model, threshold: float = 0.38):
        self.model     = model
        self.threshold = threshold

    def get_signal(self, df: pd.DataFrame) -> dict:
        """
        戻り値:
          action:     "BUY" | "SELL" | "HOLD"
          confidence: float
          price:      float
          reason:     str
        """
        row   = df.iloc[-1]
        price = row["close"]

        # AI予測
        try:
            prob = self.model.predict_proba(df)
        except Exception:
            prob = 0.5

        # テクニカル条件
        up_slow  = row.get("ema_cross_21_50", 0) == 1   # EMA21 > EMA50
        up_fast  = row.get("ema_cross_9_21",  0) == 1   # EMA9  > EMA21
        above200 = row.get("price_vs_ema200", 0) == 1   # 価格  > EMA200
        adx      = row.get("adx",   0)
        rsi      = row.get("rsi_14", 50)
        bb_pct   = row.get("bb_pct", 0.5)
        macd_d   = row.get("macd_diff", 0)

        # 押し目条件（V4）: RSI低下 or BB下限付近
        dip = (rsi < 45) or (bb_pct < 0.35)

        # トレンド条件（V3緩和）: 強トレンド or 短期上昇
        trend_ok = (up_slow and above200) or (up_fast and above200 and adx > 20)

        # BUY: トレンド + 押し目 + AI確信度 + ADX
        if trend_ok and dip and prob >= self.threshold and adx > 15 and rsi < 70:
            reason = f"trend+dip rsi={rsi:.0f} bb={bb_pct:.2f} prob={prob:.2f}"
            return {"action": "BUY", "confidence": round(prob, 4), "price": price, "reason": reason}

        # SELL: 下降転換 + AI弱気
        if (not up_slow) and prob < 0.42 and macd_d < 0:
            reason = f"trend_break prob={prob:.2f} macd_diff={macd_d:.0f}"
            return {"action": "SELL", "confidence": round(prob, 4), "price": price, "reason": reason}

        return {"action": "HOLD", "confidence": round(prob, 4), "price": price, "reason": "no signal"}
