"""XGBoost + LSTM アンサンブルモデル（重み最適化付き）"""
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_signal", "macd_diff", "adx", "adx_pos", "adx_neg",
    "rsi_14", "rsi_7", "rsi_21", "stoch_k", "stoch_d", "cci", "williams_r", "roc",
    "bb_width", "bb_pct", "atr", "atr_pct",
    "obv", "vwap_dist", "vol_ratio",
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48",
    "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema50", "price_vs_ema200",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "vol_regime",
]
# 4時間足特徴量（存在する場合のみ使用）
FEATURE_COLS_4H = ["ema_21_4h", "rsi_14_4h", "macd_4h", "trend_4h"]

MODEL_DIR = "models"
SEQ_LEN = 24


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=0.3, bidirectional=False)
        self.bn   = nn.BatchNorm1d(hidden)
        self.fc   = nn.Linear(hidden, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.sig(self.fc(out))


class NexusEnsemble:
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="auc",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.lstm_model: LSTMModel | None = None
        self.xgb_weight = 0.5  # ④ 最適化される重み
        self.trained = False
        self._feature_cols = FEATURE_COLS[:]

    def _get_cols(self, df: pd.DataFrame) -> list:
        """4時間足特徴量が存在する場合は追加"""
        cols = FEATURE_COLS[:]
        for c in FEATURE_COLS_4H:
            if c in df.columns:
                cols.append(c)
        return cols

    def train(self, df: pd.DataFrame):
        self._feature_cols = self._get_cols(df)
        X = df[self._feature_cols].values
        y = df["target"].values

        X_scaled = self.scaler.fit_transform(X)

        # XGBoost（時系列CV）
        tscv = TimeSeriesSplit(n_splits=5)
        self.xgb.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            verbose=False,
        )

        # LSTM
        self.lstm_model = LSTMModel(input_size=len(self._feature_cols))
        self._train_lstm(X_scaled, y)

        # ④ アンサンブル重みを最適化
        self._optimize_weights(X_scaled, y)

        self.trained = True
        self._save()
        print(f"学習完了 XGB重み={self.xgb_weight:.2f} LSTM重み={1-self.xgb_weight:.2f}")

    def _train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        seqs, labels = [], []
        for i in range(SEQ_LEN, len(X)):
            seqs.append(X[i - SEQ_LEN:i])
            labels.append(y[i])

        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        y_t = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

        opt     = torch.optim.AdamW(self.lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.BCELoss()

        self.lstm_model.train()
        for ep in range(epochs):
            opt.zero_grad()
            pred = self.lstm_model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f"  LSTM epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    def _optimize_weights(self, X_scaled: np.ndarray, y: np.ndarray):
        """④ XGBとLSTMの最適な重みをAUCで探索"""
        xgb_probs = self.xgb.predict_proba(X_scaled)[:, 1]

        # LSTM予測
        seqs = []
        for i in range(SEQ_LEN, len(X_scaled)):
            seqs.append(X_scaled[i - SEQ_LEN:i])
        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_probs = self.lstm_model(X_t).squeeze().numpy()

        xgb_aligned = xgb_probs[SEQ_LEN:]
        y_aligned   = y[SEQ_LEN:]

        def neg_auc(w):
            ensemble = w * xgb_aligned + (1 - w) * lstm_probs
            return -roc_auc_score(y_aligned, ensemble)

        result = minimize_scalar(neg_auc, bounds=(0.1, 0.9), method="bounded")
        self.xgb_weight = result.x
        print(f"  最適重み: XGB={self.xgb_weight:.2f} AUC={-result.fun:.4f}")

    def predict_proba(self, df: pd.DataFrame) -> float:
        if not self.trained:
            self._load()

        cols = [c for c in self._feature_cols if c in df.columns]
        X = df[cols].values
        X_scaled = self.scaler.transform(X)

        xgb_prob = self.xgb.predict_proba(X_scaled[-1:])[:, 1][0]

        if len(X_scaled) >= SEQ_LEN:
            seq = torch.tensor(X_scaled[-SEQ_LEN:][np.newaxis], dtype=torch.float32)
            self.lstm_model.eval()
            with torch.no_grad():
                lstm_prob = self.lstm_model(seq).item()
        else:
            lstm_prob = xgb_prob

        return self.xgb_weight * xgb_prob + (1 - self.xgb_weight) * lstm_prob

    def _save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(f"{MODEL_DIR}/xgb.pkl", "wb") as f:
            pickle.dump(self.xgb, f)
        with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{MODEL_DIR}/meta.pkl", "wb") as f:
            pickle.dump({"xgb_weight": self.xgb_weight, "feature_cols": self._feature_cols}, f)
        torch.save(self.lstm_model.state_dict(), f"{MODEL_DIR}/lstm.pt")

    def _load(self):
        with open(f"{MODEL_DIR}/xgb.pkl", "rb") as f:
            self.xgb = pickle.load(f)
        with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{MODEL_DIR}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.xgb_weight    = meta["xgb_weight"]
            self._feature_cols = meta["feature_cols"]
        self.lstm_model = LSTMModel(input_size=len(self._feature_cols))
        self.lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm.pt", weights_only=True))
        self.trained = True
