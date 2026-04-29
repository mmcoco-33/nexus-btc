"""リスク管理（トレーリングストップ + ATRベース損切り）"""
import json
import os
from datetime import date


STATE_FILE = "data/risk_state.json"


class RiskManager:
    def __init__(self,
                 atr_mult: float = 2.0,
                 trail_pct: float = 0.03,
                 max_daily_loss_pct: float = 0.05):
        self.atr_mult           = atr_mult    # ATR × N で損切りライン
        self.trail_pct          = trail_pct   # トレーリングストップ幅
        self.max_daily_loss_pct = max_daily_loss_pct
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
        return {
            "daily_loss": 0.0,
            "last_date":  str(date.today()),
            "entry_price": None,
            "entry_atr":   None,
            "peak_price":  None,
        }

    def _save(self):
        os.makedirs("data", exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f)

    def _reset_daily(self):
        today = str(date.today())
        if self.state["last_date"] != today:
            self.state["daily_loss"] = 0.0
            self.state["last_date"]  = today
            self._save()

    # ---- 公開メソッド ----

    def can_trade(self, balance: float) -> bool:
        self._reset_daily()
        return self.state["daily_loss"] < balance * self.max_daily_loss_pct

    def set_entry(self, price: float, atr: float):
        self.state["entry_price"] = price
        self.state["entry_atr"]   = atr
        self.state["peak_price"]  = price
        self._save()

    def update_peak(self, current_price: float):
        """毎時呼び出してトレーリングストップの基準値を更新"""
        if self.state["peak_price"] is not None:
            if current_price > self.state["peak_price"]:
                self.state["peak_price"] = current_price
                self._save()

    def should_exit(self, current_price: float) -> tuple[bool, str]:
        """
        (exit_flag, reason) を返す
        reason: "atr_sl" | "trailing" | ""
        """
        entry = self.state.get("entry_price")
        atr   = self.state.get("entry_atr")
        peak  = self.state.get("peak_price")

        if entry is None:
            return False, ""

        # ATRベース損切り
        if atr:
            atr_sl = entry - atr * self.atr_mult
            if current_price <= atr_sl:
                return True, "atr_sl"

        # トレーリングストップ
        if peak:
            trail_sl = peak * (1 - self.trail_pct)
            if current_price <= trail_sl:
                return True, "trailing"

        return False, ""

    def record_loss(self, loss_jpy: float):
        self.state["daily_loss"] += abs(loss_jpy)
        self._save()

    def clear_position(self):
        self.state["entry_price"] = None
        self.state["entry_atr"]   = None
        self.state["peak_price"]  = None
        self._save()
