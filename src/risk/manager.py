"""リスク管理"""
import json
import os
from datetime import date


STATE_FILE = "data/risk_state.json"


class RiskManager:
    def __init__(self, stop_loss_pct: float = 0.03, take_profit_pct: float = 0.06,
                 max_daily_loss_pct: float = 0.05):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
        return {"daily_loss": 0.0, "last_date": str(date.today()), "entry_price": None}

    def _save_state(self):
        os.makedirs("data", exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f)

    def _reset_daily_if_needed(self):
        today = str(date.today())
        if self.state["last_date"] != today:
            self.state["daily_loss"] = 0.0
            self.state["last_date"] = today
            self._save_state()

    def can_trade(self, balance: float) -> bool:
        """日次損失上限チェック"""
        self._reset_daily_if_needed()
        max_loss = balance * self.max_daily_loss_pct
        return self.state["daily_loss"] < max_loss

    def set_entry(self, price: float):
        self.state["entry_price"] = price
        self._save_state()

    def should_stop_loss(self, current_price: float) -> bool:
        entry = self.state.get("entry_price")
        if entry is None:
            return False
        return current_price <= entry * (1 - self.stop_loss_pct)

    def should_take_profit(self, current_price: float) -> bool:
        entry = self.state.get("entry_price")
        if entry is None:
            return False
        return current_price >= entry * (1 + self.take_profit_pct)

    def record_loss(self, loss_jpy: float):
        self.state["daily_loss"] += loss_jpy
        self._save_state()

    def clear_entry(self):
        self.state["entry_price"] = None
        self._save_state()
