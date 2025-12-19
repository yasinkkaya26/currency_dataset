import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, augment=False, augment_prob=0.3):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.augment = augment
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment and torch.rand(1).item() < self.augment_prob:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        aug_type = torch.randint(0, 3, (1,)).item()

        if aug_type == 0:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        elif aug_type == 1:
            scale = torch.FloatTensor(1).uniform_(0.95, 1.05)
            x = x * scale
        else:
            mask = torch.rand(x.size(0)) > 0.1
            x = x * mask.unsqueeze(-1)

        return x


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_fix = ['close_USD', 'open_USD', 'high_USD', 'low_USD', 'volume']
    for col in cols_to_fix:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    close = df["close_USD"]

    df["return_1d"] = close.pct_change(fill_method=None)
    df["log_return_1d"] = np.log(close).diff()

    df["ema_7"] = close.ewm(span=7, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["sma_30"] = close.rolling(window=30).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    period = 14
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    df["volatility_7"] = close.rolling(window=7).std()
    df["volatility_30"] = close.rolling(window=30).std()

    df["momentum_5"] = close.diff(5)
    df["momentum_10"] = close.diff(10)

    df = df.dropna().reset_index(drop=True)
    return df


def create_sequences(df: pd.DataFrame, feature_cols, target_col: str, window: int):
    values = df[feature_cols].values.astype(np.float32)
    t_index = feature_cols.index(target_col)

    X_list, y_list = [], []
    for i in range(len(df) - window):
        x = values[i:i + window]
        y = values[i + window, t_index]
        X_list.append(x)
        y_list.append(y)

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


def load_time_series_dataloaders(
    csv_path: str,
    target_col: str = "close_USD",
    window_size: int = 90,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    df = pd.read_csv(csv_path, thousands=',')

    if df.columns[0].lower() in ["date", "time", "timestamp", "datetime"] or df.columns[0].startswith("Unnamed"):
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    keep_cols = ["date", "open_USD", "high_USD", "low_USD", "close_USD", "volume"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df = _add_technical_indicators(df)

    feature_cols = [
        "open_USD", "high_USD", "low_USD", "close_USD", "volume",
        "return_1d", "log_return_1d",
        "ema_7", "ema_21", "sma_30",
        "rsi_14",
        "volatility_7", "volatility_30",
        "momentum_5", "momentum_10"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    assert target_col in feature_cols, f"{target_col} not in feature columns"

    X_all, y_all = create_sequences(df, feature_cols, target_col, window_size)
    num_samples, seq_len, num_features = X_all.shape

    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_val = X_all[train_size:train_size + val_size]
    y_val = y_all[train_size:train_size + val_size]
    X_test = X_all[train_size + val_size:]
    y_test = y_all[train_size + val_size:]

    feature_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    feature_scaler.fit(X_train_flat)

    def scale_X(X):
        X_flat = X.reshape(-1, num_features)
        X_scaled = feature_scaler.transform(X_flat)
        return X_scaled.reshape(-1, seq_len, num_features)

    X_train = scale_X(X_train)
    X_val = scale_X(X_val)
    X_test = scale_X(X_test)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    train_ds = TimeSeriesDataset(X_train, y_train, augment=True, augment_prob=0.3)
    val_ds = TimeSeriesDataset(X_val, y_val, augment=False)
    test_ds = TimeSeriesDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": feature_scaler,
        "target_scaler": target_scaler,
        "seq_len": seq_len,
        "num_features": num_features,
        "feature_cols": feature_cols,
    }