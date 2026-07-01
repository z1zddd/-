from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def add_local_dll_dirs() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        Path(r"C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"),
        Path(r"C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\jxrlib\jxrlib\bin"),
        Path(r"C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\libheif\libheif\bin"),
        Path(r"C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python"),
    ]
    for path in candidates:
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass


add_local_dll_dirs()

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
TARGET = "dce_corn_close"
DATE_COL = "month"
HORIZON = 1
LEAKAGE_COLS = {
    "dce_corn_close_next_month",
    "dce_corn_close_next_month_ret",
    "dce_corn_close_next_month_direction",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_monthly_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing {DATE_COL} column.")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}.")

    df[DATE_COL] = pd.PeriodIndex(df[DATE_COL].astype(str), freq="M").to_timestamp()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame, max_missing_ratio: float) -> list[str]:
    excluded = {
        DATE_COL,
        "first_trade_date",
        "last_trade_date",
        *LEAKAGE_COLS,
    }
    candidates = []
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        missing_ratio = float(df[col].isna().mean())
        if missing_ratio <= max_missing_ratio:
            candidates.append(col)
    if TARGET not in candidates:
        raise ValueError(f"{TARGET} must be present in feature columns.")
    return candidates


def make_windows(
    df: pd.DataFrame,
    features: list[str],
    seq_len: int,
    target_col: str = TARGET,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = df[features].to_numpy(dtype=float)
    target = df[target_col].to_numpy(dtype=float)
    months = df[DATE_COL].to_numpy()

    xs: list[np.ndarray] = []
    ys: list[float] = []
    rows: list[dict[str, Any]] = []

    for end_idx in range(seq_len - 1, len(df) - 1):
        target_idx = end_idx + 1
        xs.append(values[end_idx - seq_len + 1 : end_idx + 1])
        ys.append(float(target[target_idx]))
        rows.append(
            {
                "input_end_month": months[end_idx],
                "target_month": months[target_idx],
                "today_close": float(target[end_idx]),
                "actual_close": float(target[target_idx]),
                "actual_change": float(target[target_idx] - target[end_idx]),
                "actual_return": float(target[target_idx] / target[end_idx] - 1.0),
            }
        )

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(rows)


class Standardizer:
    def __init__(self) -> None:
        self.x_median: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_median = np.nanmedian(x, axis=(0, 1)).astype(np.float32)
        self.x_median = np.where(np.isfinite(self.x_median), self.x_median, 0.0)
        filled = self.fill(x)
        self.x_mean = filled.mean(axis=(0, 1)).astype(np.float32)
        self.x_std = filled.std(axis=(0, 1)).astype(np.float32)
        self.x_std = np.where(self.x_std < 1e-6, 1.0, self.x_std)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y))
        if self.y_std < 1e-6:
            self.y_std = 1.0

    def fill(self, x: np.ndarray) -> np.ndarray:
        assert self.x_median is not None
        return np.where(np.isfinite(x), x, self.x_median.reshape(1, 1, -1)).astype(np.float32)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None and self.x_std is not None
        return ((self.fill(x) - self.x_mean.reshape(1, 1, -1)) / self.x_std.reshape(1, 1, -1)).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_std + self.y_mean


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)).view(-1, 1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, x: np.ndarray, scaler: Standardizer, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return scaler.inverse_y(np.concatenate(preds))


def metrics(y_true: np.ndarray, y_pred: np.ndarray, today_close: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    today_close = np.asarray(today_close, dtype=float)

    actual_dir = (y_true > today_close).astype(int)
    pred_dir = (y_pred > today_close).astype(int)
    pred_score = y_pred - today_close
    nonzero = y_true != 0

    out = {
        "samples": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))),
        "direction_accuracy": float(accuracy_score(actual_dir, pred_dir)),
        "balanced_accuracy": float(balanced_accuracy_score(actual_dir, pred_dir)),
        "precision_up": float(precision_score(actual_dir, pred_dir, zero_division=0)),
        "recall_up": float(recall_score(actual_dir, pred_dir, zero_division=0)),
        "f1_up": float(f1_score(actual_dir, pred_dir, zero_division=0)),
        "actual_up_count": float((actual_dir == 1).sum()),
        "predicted_up_count": float((pred_dir == 1).sum()),
    }
    try:
        out["auc_from_predicted_change"] = float(roc_auc_score(actual_dir, pred_score))
    except ValueError:
        out["auc_from_predicted_change"] = float("nan")
    return out


def split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_end = max(train_end, 8)
    val_end = max(val_end, train_end + 4)
    val_end = min(val_end, n - 4)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_monthly_data(Path(args.data_path))
    features = feature_columns(df, args.max_missing_ratio)
    x_all, y_all, meta = make_windows(df, features, args.seq_len)

    train_slice, val_slice, test_slice = split_indices(len(x_all), args.train_ratio, args.val_ratio)
    x_train_raw, y_train_raw = x_all[train_slice], y_all[train_slice]
    x_val_raw, y_val_raw = x_all[val_slice], y_all[val_slice]
    x_test_raw, y_test_raw = x_all[test_slice], y_all[test_slice]

    scaler = Standardizer()
    scaler.fit(x_train_raw, y_train_raw)
    x_train = scaler.transform_x(x_train_raw)
    y_train = scaler.transform_y(y_train_raw)
    x_val = scaler.transform_x(x_val_raw)
    y_val = scaler.transform_y(y_val_raw)
    x_test = scaler.transform_x(x_test_raw)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model = LSTMRegressor(
        input_size=x_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_mae = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        val_pred = predict(model, x_val, scaler, args.batch_size, device)
        val_mae = float(mean_absolute_error(y_val_raw, val_pred))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(batch_losses)), "val_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred = predict(model, scaler.transform_x(x_train_raw), scaler, args.batch_size, device)
    val_pred = predict(model, x_val, scaler, args.batch_size, device)
    test_pred = predict(model, x_test, scaler, args.batch_size, device)

    meta_train = meta.iloc[train_slice].reset_index(drop=True)
    meta_val = meta.iloc[val_slice].reset_index(drop=True)
    meta_test = meta.iloc[test_slice].reset_index(drop=True)

    train_metrics = metrics(y_train_raw, train_pred, meta_train["today_close"].to_numpy(float))
    val_metrics = metrics(y_val_raw, val_pred, meta_val["today_close"].to_numpy(float))
    test_metrics = metrics(y_test_raw, test_pred, meta_test["today_close"].to_numpy(float))

    def make_pred_df(part: str, part_meta: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        out = part_meta.copy()
        out.insert(0, "split", part)
        out["input_end_month"] = pd.to_datetime(out["input_end_month"]).dt.strftime("%Y-%m")
        out["target_month"] = pd.to_datetime(out["target_month"]).dt.strftime("%Y-%m")
        out["pred_close"] = preds
        out["pred_change"] = out["pred_close"] - out["today_close"]
        out["actual_direction"] = (out["actual_close"] > out["today_close"]).astype(int)
        out["pred_direction"] = (out["pred_close"] > out["today_close"]).astype(int)
        return out

    predictions = pd.concat(
        [
            make_pred_df("train", meta_train, train_pred),
            make_pred_df("val", meta_val, val_pred),
            make_pred_df("test", meta_test, test_pred),
        ],
        ignore_index=True,
    )

    summary = {
        "data_path": str(Path(args.data_path).resolve()),
        "target": TARGET,
        "task": "past seq_len months -> next month dce_corn_close regression",
        "seq_len": args.seq_len,
        "horizon": HORIZON,
        "features": len(features),
        "windows": int(len(x_all)),
        "train_windows": int(len(x_train_raw)),
        "val_windows": int(len(x_val_raw)),
        "test_windows": int(len(x_test_raw)),
        "first_target_month": pd.to_datetime(meta["target_month"].iloc[0]).strftime("%Y-%m"),
        "last_target_month": pd.to_datetime(meta["target_month"].iloc[-1]).strftime("%Y-%m"),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "params": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    predictions.to_csv(out_dir / "monthly_lstm_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(out_dir / "monthly_lstm_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly LSTM for next-month dce_corn_close regression.")
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "monthly_corn_data" / "玉米价格月度_混合特征版.csv"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "monthly_lstm_outputs"))
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-missing-ratio", type=float, default=0.30)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
