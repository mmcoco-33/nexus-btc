"""NEXUS-BTC メインBot"""
import json
import os
import yaml
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.models.ensemble import NexusEnsemble
from src.strategy.signal import SignalGenerator
from src.risk.manager import RiskManager


def load_config() -> dict:
    with open("config.yml") as f:
        return yaml.safe_load(f)


def save_results(entry: dict):
    """GitHub Pages用にJSONへ結果を追記"""
    os.makedirs("data", exist_ok=True)
    path = "data/results.json"
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results.append(entry)
    results = results[-500:]  # 直近500件のみ保持
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, default=str)


def run():
    cfg = load_config()
    t_cfg = cfg["trading"]
    r_cfg = cfg["risk"]
    m_cfg = cfg["model"]

    api_key    = os.environ.get("GMO_API_KEY", "")
    api_secret = os.environ.get("GMO_API_SECRET", "")

    client  = GMOClient(api_key, api_secret)
    fetcher = DataFetcher(client)
    model   = NexusEnsemble()
    risk    = RiskManager(
        stop_loss_pct=r_cfg["stop_loss_pct"],
        take_profit_pct=r_cfg["take_profit_pct"],
        max_daily_loss_pct=r_cfg["max_daily_loss_pct"],
    )

    print(f"[{datetime.now()}] NEXUS-BTC 起動")

    # データ取得・特徴量生成
    raw_df = fetcher.fetch_ohlcv(
        symbol=t_cfg["symbol"].replace("_JPY", ""),
        interval=t_cfg["interval"],
        days=60,
    )
    df = add_features(raw_df)

    # モデル学習（初回 or 再学習タイミング）
    retrain_flag = "models/xgb.pkl"
    if not os.path.exists(retrain_flag):
        print("モデル学習中...")
        model.train(df)
    else:
        model._load()

    # シグナル生成
    gen = SignalGenerator(model, threshold=m_cfg["signal_threshold"])
    signal = gen.get_signal(df)
    print(f"シグナル: {signal}")

    current_price = signal["price"]
    action = signal["action"]
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "price": current_price,
        "action": action,
        "confidence": signal["confidence"],
        "executed": False,
        "reason": "",
    }

    # ポジション確認
    has_position = False
    position_id = None
    if api_key:
        try:
            pos_resp = client.get_positions(t_cfg["symbol"])
            positions = pos_resp.get("data", {}).get("list", [])
            has_position = len(positions) > 0
            if has_position:
                position_id = positions[0]["positionId"]
        except Exception as e:
            print(f"ポジション取得エラー: {e}")

    # 損切り・利確チェック
    if has_position:
        if risk.should_stop_loss(current_price):
            action = "SELL"
            log_entry["reason"] = "stop_loss"
        elif risk.should_take_profit(current_price):
            action = "SELL"
            log_entry["reason"] = "take_profit"

    # 残高チェック
    balance = 0
    if api_key:
        try:
            margin = client.get_account_margin()
            balance = float(margin["data"]["availableAmount"])
        except Exception as e:
            print(f"残高取得エラー: {e}")

    # 注文執行
    if api_key and balance > 0:
        size = str(round(t_cfg["trade_amount_jpy"] / current_price, 6))

        if action == "BUY" and not has_position and risk.can_trade(balance):
            try:
                client.place_order(t_cfg["symbol"], "BUY", size)
                risk.set_entry(current_price)
                log_entry["executed"] = True
                print(f"BUY注文執行: {size} BTC @ {current_price}")
            except Exception as e:
                print(f"注文エラー: {e}")

        elif action == "SELL" and has_position and position_id:
            try:
                client.close_position(t_cfg["symbol"], position_id, "SELL", size)
                risk.clear_entry()
                log_entry["executed"] = True
                print(f"SELL注文執行: {size} BTC @ {current_price}")
            except Exception as e:
                print(f"決済エラー: {e}")
    else:
        log_entry["reason"] = "dry_run（APIキー未設定）"
        print("ドライラン: 注文は執行されません")

    save_results(log_entry)
    print(f"完了: {log_entry}")


if __name__ == "__main__":
    run()
