"""NEXUS-BTC メインBot（全要素統合版）"""
import json
import os
import yaml
from datetime import datetime

from src.api.gmo_client import GMOClient
from src.data.fetcher import DataFetcher
from src.features.engineer import add_features
from src.strategy.signal import SignalGenerator
from src.risk.manager import RiskManager

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "bb_width", "bb_pct", "atr_pct",
    "vwap_dist", "vol_ratio",
    "ret_1", "ret_3", "ret_6", "ret_24",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "vol_regime",
]


class LightModel:
    """XGBoostのみの軽量モデル（torch不要）"""
    def predict_proba(self, df) -> float:
        import pickle
        try:
            with open("models/xgb.pkl", "rb") as f:
                xgb = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            cols = [c for c in FEATURE_COLS if c in df.columns]
            X_s  = scaler.transform(df[cols].values[-1:])
            return xgb.predict_proba(X_s)[0, 1]
        except Exception:
            return 0.5


def load_config() -> dict:
    with open("config.yml") as f:
        return yaml.safe_load(f)


def save_results(entry: dict):
    os.makedirs("data", exist_ok=True)
    path = "data/results.json"
    results = []
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    results.append(entry)
    results = results[-500:]
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, default=str)


def run():
    cfg   = load_config()
    t_cfg = cfg["trading"]
    r_cfg = cfg["risk"]
    m_cfg = cfg["model"]

    api_key    = os.environ.get("GMO_API_KEY", "")
    api_secret = os.environ.get("GMO_API_SECRET", "")

    client  = GMOClient(api_key, api_secret)
    fetcher = DataFetcher(client)
    model   = LightModel()
    risk    = RiskManager(
        atr_mult=r_cfg.get("atr_mult", 2.0),
        trail_pct=r_cfg.get("trail_pct", 0.03),
        max_daily_loss_pct=r_cfg["max_daily_loss_pct"],
    )
    gen = SignalGenerator(model, threshold=m_cfg["signal_threshold"])

    print(f"[{datetime.now()}] NEXUS-BTC 起動")

    # データ取得・特徴量生成
    symbol = t_cfg["symbol"].replace("_JPY", "")
    try:
        df_1h, df_4h = fetcher.fetch_multi_timeframe(symbol=symbol, days_1h=60, days_4h=120)
        df = add_features(df_1h, df_4h=df_4h)
    except Exception:
        df_1h = fetcher.fetch_ohlcv(symbol=symbol, interval="1hour", days=60)
        df = add_features(df_1h)

    # モデルロード（retrainワークフローが毎日更新）
    if not os.path.exists("models/xgb.pkl"):
        print("モデルファイルなし → ドライラン継続")

    # シグナル生成
    signal = gen.get_signal(df)
    print(f"シグナル: {signal}")

    current_price = signal["price"]
    current_atr   = float(df["atr"].iloc[-1]) if "atr" in df.columns else current_price * 0.01
    action        = signal["action"]

    log_entry = {
        "timestamp":  datetime.now().isoformat(),
        "price":      current_price,
        "action":     action,
        "confidence": signal["confidence"],
        "reason":     signal["reason"],
        "executed":   False,
    }

    # ポジション確認
    has_position = False
    position_id  = None
    if api_key:
        try:
            pos_resp = client.get_positions(t_cfg["symbol"])
            positions = pos_resp.get("data", {}).get("list", [])
            has_position = len(positions) > 0
            if has_position:
                position_id = positions[0]["positionId"]
        except Exception as e:
            print(f"ポジション取得エラー: {e}")

    # トレーリングストップ更新 & 損切り確認
    if has_position:
        risk.update_peak(current_price)
        should_exit, exit_reason = risk.should_exit(current_price)
        if should_exit:
            action = "SELL"
            log_entry["reason"] = exit_reason
            print(f"エグジット判定: {exit_reason} @ {current_price}")

    # 残高取得
    balance = 0
    if api_key:
        try:
            margin  = client.get_account_margin()
            balance = float(margin["data"]["availableAmount"])
        except Exception as e:
            print(f"残高取得エラー: {e}")

    # 注文執行
    if api_key and balance > 0:
        size = str(round(t_cfg["trade_amount_jpy"] / current_price, 6))

        if action == "BUY" and not has_position and risk.can_trade(balance):
            try:
                client.place_order(t_cfg["symbol"], "BUY", size)
                risk.set_entry(current_price, current_atr)
                log_entry["executed"] = True
                print(f"BUY注文: {size} BTC @ ¥{current_price:,.0f}  ATR={current_atr:,.0f}")
            except Exception as e:
                print(f"注文エラー: {e}")

        elif action == "SELL" and has_position and position_id:
            try:
                client.close_position(t_cfg["symbol"], position_id, "SELL", size)
                risk.clear_position()
                log_entry["executed"] = True
                print(f"SELL注文: {size} BTC @ ¥{current_price:,.0f}")
            except Exception as e:
                print(f"決済エラー: {e}")
    else:
        log_entry["reason"] += " | dry_run"
        print("ドライラン: 注文は執行されません")

    save_results(log_entry)
    print(f"完了: {log_entry}")


if __name__ == "__main__":
    run()
