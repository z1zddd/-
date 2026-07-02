from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MONTH_COL = "month"
TARGET_COL = "spike"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def normalize_month(value: object) -> str:
    text = str(value)
    parsed = pd.NaT
    for fmt in ("%Y-%m", "%b-%y"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            break
    if pd.isna(parsed):
        parsed = pd.to_datetime(text[:7] + "-01", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot parse month value: {value!r}")
    return parsed.strftime("%Y-%m")


def load_inputs(csv_path: Path, feature_rank_path: Path, top_n: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    ranks = pd.read_csv(feature_rank_path, encoding="utf-8-sig").head(top_n)
    df = df.copy()
    df[MONTH_COL] = df[MONTH_COL].map(normalize_month)
    df = df.sort_values(MONTH_COL).reset_index(drop=True)
    for col in ranks["feature"]:
        if col not in df.columns:
            raise SystemExit(f"Feature from ranking not found in data: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    news_cols = [col for col in ranks["feature"].tolist() if col.startswith("pca_")]
    structured_cols = [col for col in ranks["feature"].tolist() if not col.startswith("pca_")]
    return df[[MONTH_COL, TARGET_COL, *structured_cols, *news_cols]], structured_cols, news_cols


def train_input_rows(sample_end_indices: np.ndarray, lookback: int) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                row
                for end_idx in sample_end_indices
                for row in range(end_idx - lookback + 1, end_idx + 1)
            }
        ),
        dtype=np.int64,
    )


def fit_transform(
    df: pd.DataFrame,
    structured_cols: list[str],
    news_cols: list[str],
    train_sample_end_indices: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = train_input_rows(train_sample_end_indices, lookback)
    structured_raw = df[structured_cols].to_numpy(dtype=np.float32)
    news_raw = df[news_cols].to_numpy(dtype=np.float32)

    structured_imputer = SimpleImputer(strategy="median")
    news_imputer = SimpleImputer(strategy="median")
    structured_scaler = StandardScaler()
    news_scaler = StandardScaler()

    structured_train = structured_imputer.fit_transform(structured_raw[rows])
    news_train = news_imputer.fit_transform(news_raw[rows])
    structured_scaler.fit(structured_train)
    news_scaler.fit(news_train)

    structured_x = structured_scaler.transform(structured_imputer.transform(structured_raw)).astype(np.float32)
    news_x = news_scaler.transform(news_imputer.transform(news_raw)).astype(np.float32)
    return structured_x, news_x


def build_sequences(
    df: pd.DataFrame,
    structured_x: np.ndarray,
    news_x: np.ndarray,
    sample_end_indices: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    labels = df[TARGET_COL].to_numpy(dtype=np.float32)
    structured_seq, news_seq, y, meta = [], [], [], []
    for end_idx in sample_end_indices:
        start_idx = end_idx - lookback + 1
        target_idx = end_idx + horizon
        target = labels[target_idx]
        if np.isnan(target):
            continue
        structured_seq.append(structured_x[start_idx : end_idx + 1])
        news_seq.append(news_x[start_idx : end_idx + 1])
        y.append(target)
        meta.append(
            {
                "input_start_month": df.loc[start_idx, MONTH_COL],
                "input_end_month": df.loc[end_idx, MONTH_COL],
                "target_month": df.loc[target_idx, MONTH_COL],
                "target": int(target),
            }
        )
    return (
        np.asarray(structured_seq, dtype=np.float32),
        np.asarray(news_seq, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(meta),
    )


class DualStreamLSTM(nn.Module):
    def __init__(self, structured_dim: int, news_dim: int, hidden_dim: int, attn_dim: int, dense_dim: int, dropout: float) -> None:
        super().__init__()
        self.structured_lstm = nn.LSTM(structured_dim, hidden_dim, batch_first=True)
        self.news_lstm = nn.LSTM(news_dim, hidden_dim, batch_first=True)
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, attn_dim)
        self.fc1 = nn.Linear(hidden_dim + attn_dim, dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dense_dim, 1)

    def forward(self, structured_seq: torch.Tensor, news_seq: torch.Tensor) -> torch.Tensor:
        _, (structured_h, _) = self.structured_lstm(structured_seq)
        h_structured = structured_h[-1]

        news_h, _ = self.news_lstm(news_seq)
        q = self.q_proj(news_h)
        k = self.k_proj(news_h)
        v = self.v_proj(news_h)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
        h_news = torch.matmul(attn, v).mean(dim=1)

        fused = torch.cat([h_structured, h_news], dim=1)
        z = self.dropout(torch.relu(self.fc1(fused)))
        return self.out(z).squeeze(-1)


def make_loader(structured_seq: np.ndarray, news_seq: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(structured_seq, dtype=torch.float32),
        torch.tensor(news_seq, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(
    structured_train: np.ndarray,
    news_train: np.ndarray,
    y_train: np.ndarray,
    structured_val: np.ndarray,
    news_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[DualStreamLSTM, dict]:
    device = torch.device(args.device)
    model = DualStreamLSTM(
        structured_dim=structured_train.shape[-1],
        news_dim=news_train.shape[-1],
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
    ).to(device)
    dense_params = list(model.fc1.parameters()) + list(model.out.parameters())
    dense_ids = {id(param) for param in dense_params}
    other_params = [param for param in model.parameters() if id(param) not in dense_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )
    criterion = nn.BCEWithLogitsLoss()
    train_loader = make_loader(structured_train, news_train, y_train, args.batch_size, True)
    val_loader = make_loader(structured_val, news_val, y_val, args.batch_size, False)
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    patience_left = args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for structured_batch, news_batch, y_batch in train_loader:
            structured_batch = structured_batch.to(device)
            news_batch = news_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(structured_batch, news_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for structured_batch, news_batch, y_batch in val_loader:
                structured_batch = structured_batch.to(device)
                news_batch = news_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(float(criterion(model(structured_batch, news_batch), y_batch).detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "epochs_ran": len(history)}


def predict_prob(model: nn.Module, structured_seq: np.ndarray, news_seq: np.ndarray, device: str) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(structured_seq[None, :, :], dtype=torch.float32).to(device),
            torch.tensor(news_seq[None, :, :], dtype=torch.float32).to(device),
        )
        return float(torch.sigmoid(logits).detach().cpu().item())


def predict_probs(model: nn.Module, structured_seq: np.ndarray, news_seq: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    ds = TensorDataset(
        torch.tensor(structured_seq, dtype=torch.float32),
        torch.tensor(news_seq, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    probs = []
    with torch.no_grad():
        for structured_batch, news_batch in loader:
            logits = model(structured_batch.to(device), news_batch.to(device))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs).astype(float)


def metrics_from_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    threshold: float | None = None,
    threshold_rule: str | None = None,
) -> dict:
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out = {
        "threshold": None if threshold is None else float(threshold),
        "threshold_rule": threshold_rule,
        "n": int(len(y_true)),
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y_true.astype(int), return_counts=True))},
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else None,
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "precision_positive": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "auc": None,
        "average_precision": None,
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    return out


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return metrics_from_predictions(y_true, pred, prob, threshold=threshold)


def best_threshold_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    rows = [metrics(y_true, prob, threshold) for threshold in np.linspace(0.05, 0.95, 91)]
    return max(rows, key=lambda row: (row["f1_weighted"], row["balanced_accuracy"] or -1, row["f1_positive"]))


def rolling_backtest(df: pd.DataFrame, structured_cols: list[str], news_cols: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    labels = df[TARGET_COL].to_numpy(dtype=np.float32)
    max_end = len(df) - args.horizon
    all_end_indices = np.asarray(
        [idx for idx in range(args.lookback - 1, max_end) if not np.isnan(labels[idx + args.horizon])],
        dtype=np.int64,
    )
    test_start = max(args.min_train_samples, int(math.ceil(len(all_end_indices) * args.initial_train_fraction)))
    test_positions = list(range(test_start, len(all_end_indices), args.step))
    if args.max_tests > 0:
        test_positions = test_positions[-args.max_tests :]
    rows = []
    histories = []
    for run_no, test_pos in enumerate(test_positions, start=1):
        test_end_idx = int(all_end_indices[test_pos])
        train_val_end_indices = all_end_indices[:test_pos]
        val_size = max(1, int(math.ceil(len(train_val_end_indices) * args.val_fraction)))
        train_end_indices = train_val_end_indices[:-val_size]
        val_end_indices = train_val_end_indices[-val_size:]
        if len(train_end_indices) < 2:
            continue

        set_seed(args.seed + run_no)
        structured_x, news_x = fit_transform(df, structured_cols, news_cols, train_end_indices, args.lookback)
        structured_train, news_train, y_train, train_meta = build_sequences(df, structured_x, news_x, train_end_indices, args.lookback, args.horizon)
        structured_val, news_val, y_val, val_meta = build_sequences(df, structured_x, news_x, val_end_indices, args.lookback, args.horizon)
        structured_test, news_test, y_test, test_meta = build_sequences(df, structured_x, news_x, np.asarray([test_end_idx], dtype=np.int64), args.lookback, args.horizon)
        model, train_info = train_model(structured_train, news_train, y_train, structured_val, news_val, y_val, args)
        val_prob = predict_probs(model, structured_val, news_val, args.device, args.batch_size)
        val_threshold_info = best_threshold_metrics(y_val.astype(int), val_prob)
        val_threshold = float(val_threshold_info["threshold"])
        probability = predict_prob(model, structured_test[0], news_test[0], args.device)
        rows.append(
            {
                "run_no": run_no,
                "input_start_month": test_meta.iloc[0]["input_start_month"],
                "input_end_month": test_meta.iloc[0]["input_end_month"],
                "target_month": test_meta.iloc[0]["target_month"],
                "target": int(y_test[0]),
                "probability": probability,
                "pred_0p5": int(probability >= 0.5),
                "val_threshold": val_threshold,
                "pred_val_threshold": int(probability >= val_threshold),
                "val_threshold_accuracy": val_threshold_info["accuracy"],
                "val_threshold_balanced_accuracy": val_threshold_info["balanced_accuracy"],
                "val_threshold_f1_weighted": val_threshold_info["f1_weighted"],
                "val_threshold_f1_positive": val_threshold_info["f1_positive"],
                "train_start_target_month": train_meta.iloc[0]["target_month"],
                "train_end_target_month": train_meta.iloc[-1]["target_month"],
                "val_start_target_month": val_meta.iloc[0]["target_month"],
                "val_end_target_month": val_meta.iloc[-1]["target_month"],
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "best_epoch": train_info["best_epoch"],
                "epochs_ran": train_info["epochs_ran"],
                "best_val_loss": train_info["best_val_loss"],
            }
        )
        histories.append({"run_no": run_no, **train_info})
        print(
            f"test {run_no}/{len(test_positions)} target={test_meta.iloc[0]['target_month']} "
            f"y={int(y_test[0])} p={probability:.4f} val_thr={val_threshold:.2f}",
            flush=True,
        )
    pred_df = pd.DataFrame(rows)
    y_true = pred_df["target"].to_numpy(dtype=int)
    prob = pred_df["probability"].to_numpy(dtype=float)
    pred_val_threshold = pred_df["pred_val_threshold"].to_numpy(dtype=int)
    summary = {
        "threshold_0p5": metrics(y_true, prob, 0.5),
        "validation_selected_threshold": metrics_from_predictions(
            y_true,
            pred_val_threshold,
            prob,
            threshold_rule="For each rolling test month, choose the threshold on that run's past validation set only.",
        ),
        "validation_threshold_summary": {
            "mean": float(pred_df["val_threshold"].mean()),
            "median": float(pred_df["val_threshold"].median()),
            "min": float(pred_df["val_threshold"].min()),
            "max": float(pred_df["val_threshold"].max()),
        },
        "best_global_threshold_leaky_reference": best_threshold_metrics(y_true, prob),
        "history_summary": {
            "mean_best_epoch": float(np.mean([item["best_epoch"] for item in histories])),
            "mean_epochs_ran": float(np.mean([item["epochs_ran"] for item in histories])),
        },
    }
    return pred_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling backtest dual-stream LSTM using top RF features.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--feature-rank", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=79)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--initial-train-fraction", type=float, default=0.45)
    parser.add_argument("--min-train-samples", type=int, default=36)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-tests", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attn-dim", type=int, default=32)
    parser.add_argument("--dense-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.horizon != 1:
        raise SystemExit("This experiment keeps horizon=1.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, structured_cols, news_cols = load_inputs(args.csv, args.feature_rank, args.top_n)
    pred_df, summary_metrics = rolling_backtest(df, structured_cols, news_cols, args)

    selected_features = pd.DataFrame(
        {
            "branch": ["structured"] * len(structured_cols) + ["news"] * len(news_cols),
            "feature": structured_cols + news_cols,
        }
    )
    selected_features.to_csv(args.out_dir / "top79_selected_features_for_dual_lstm.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(args.out_dir / "top79_dual_lstm_rolling_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Dual-stream LSTM using top RF importance features",
        "csv": str(args.csv),
        "feature_rank": str(args.feature_rank),
        "top_n": args.top_n,
        "structured_feature_count": len(structured_cols),
        "news_feature_count": len(news_cols),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "protocol": {
            "rolling_backtest": True,
            "initial_train_fraction": args.initial_train_fraction,
            "min_train_samples": args.min_train_samples,
            "val_fraction": args.val_fraction,
            "step": args.step,
            "max_tests": args.max_tests,
        },
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "attn_dim": args.attn_dim,
            "dense_dim": args.dense_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "metrics": summary_metrics,
    }
    (args.out_dir / "top79_dual_lstm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
