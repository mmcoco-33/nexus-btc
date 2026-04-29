"""売買シグナル生成"""
import pandas as pd
from src.models.ensemble import NexusEnsemble


class SignalGenerator:
    def __init__(self, model: NexusEnsemble, threshold: float = 0.60):
        self.model = model
        self.threshold = threshold
        self.sell_threshold = 1.0 - threshold  # 売りは逆方向

    def get_signal(self, df: pd.DataFrame) -> dict:
        """
        戻り値:
          action: "BUY" | "SELL" | "HOLD"
          confidence: float
          price: float
        """
        prob = self.model.predict_proba(df)
        price = df["close"].iloc[-1]

        if prob >= self.threshold:
            action = "BUY"
        elif prob <= self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return {"action": action, "confidence": round(prob, 4), "price": price}
