"""XGBoost + LSTM アンサンブルモデル"""
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50", "macd", "macd_signal", "macd_diff", "adx",
    "rsi_14", "rsi_7", "stoch_k", "stoch_d", "cci",
    "bb_width", "bb_pct", "atr", "obv", "vwap",
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ema_cross",
]
MODEL_DIR = "models"
SEQ_LEN = 24  # LSTMの入力シーケンス長


# --- LSTM ---

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


# --- アンサンブル ---

class NexusEnsemble:
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
        )
        self.scaler = StandardScaler()
        self.lstm_model: LSTMModel | None = None
        self.trained = False

    def train(self, df: pd.DataFrame):
        X = df[FEATURE_COLS].values
        y = df["target"].values

        X_scaled = self.scaler.fit_transform(X)

        # XGBoost（時系列CV）
        tscv = TimeSeriesSplit(n_splits=5)
        self.xgb.fit(X_scaled, y)

        # LSTM
        self.lstm_model = LSTMModel(input_size=len(FEATURE_COLS))
        self._train_lstm(X_scaled, y)

        self.trained = True
        self._save()

    def _train_lstm(self, X: np.ndarray, y: np.ndarray, epochs: int = 30):
        seqs, labels = [], []
        for i in range(SEQ_LEN, len(X)):
            seqs.append(X[i - SEQ_LEN:i])
            labels.append(y[i])

        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        y_t = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

        opt = torch.optim.Adam(self.lstm_model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        self.lstm_model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.lstm_model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

    def predict_proba(self, df: pd.DataFrame) -> float:
        """買いシグナルの確率を返す（0〜1）"""
        if not self.trained:
            self._load()

        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)

        # XGBoost予測
        xgb_prob = self.xgb.predict_proba(X_scaled[-1:])[:, 1][0]

        # LSTM予測
        if len(X_scaled) >= SEQ_LEN:
            seq = torch.tensor(X_scaled[-SEQ_LEN:][np.newaxis], dtype=torch.float32)
            self.lstm_model.eval()
            with torch.no_grad():
                lstm_prob = self.lstm_model(seq).item()
        else:
            lstm_prob = xgb_prob

        # アンサンブル（重み付き平均）
        return 0.5 * xgb_prob + 0.5 * lstm_prob

    def _save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(f"{MODEL_DIR}/xgb.pkl", "wb") as f:
            pickle.dump(self.xgb, f)
        with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        torch.save(self.lstm_model.state_dict(), f"{MODEL_DIR}/lstm.pt")

    def _load(self):
        with open(f"{MODEL_DIR}/xgb.pkl", "rb") as f:
            self.xgb = pickle.load(f)
        with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        n_features = len(FEATURE_COLS)
        self.lstm_model = LSTMModel(input_size=n_features)
        self.lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm.pt"))
        self.trained = True
