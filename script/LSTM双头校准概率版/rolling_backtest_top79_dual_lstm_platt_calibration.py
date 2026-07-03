from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_SCRIPT = Path(__file__).with_name("rolling_backtest选阈值版（保留前79个特征）.py")


def load_base_module(base_script: Path):
    spec = importlib.util.spec_from_file_location("top79_dual_lstm_base", base_script)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import base script: {base_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def predict_logits(
    model: nn.Module,
    structured_seq: np.ndarray,
    news_seq: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(
        torch.tensor(structured_seq, dtype=torch.float32),
        torch.tensor(news_seq, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    logits = []
    with torch.no_grad():
        for structured_batch, news_batch in loader:
            out = model(structured_batch.to(device), news_batch.to(device))
            logits.append(out.detach().cpu().numpy())
    return np.concatenate(logits).astype(float)


def fit_positive_platt(
    logits: np.ndarray,
    y: np.ndarray,
    l2: float,
    max_iter: int,
) -> dict[str, float | str]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(np.unique(y.astype(int))) < 2:
        return {"method": "raw_fallback_single_class_validation", "a": 1.0, "b": 0.0}

    z = torch.tensor(logits, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    log_a = torch.zeros((), dtype=torch.float32, requires_grad=True)
    prior = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    bias_init = math.log(prior / (1.0 - prior))
    b = torch.tensor(bias_init, dtype=torch.float32, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_a, b], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        a = torch.exp(log_a)
        calibrated_logits = a * z + b
        loss = criterion(calibrated_logits, target) + l2 * (log_a.square() + b.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"method": "positive_platt", "a": float(torch.exp(log_a).detach().cpu()), "b": float(b.detach().cpu())}


def apply_calibrator(logits: np.ndarray, calibrator: dict[str, float | str]) -> np.ndarray:
    a = float(calibrator["a"])
    b = float(calibrator["b"])
    return sigmoid_np(a * np.asarray(logits, dtype=float) + b)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def add_probability_diagnostics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_brier": float(brier_score_loss(y_true, prob)),
        f"{prefix}_ece_10bins": expected_calibration_error(y_true, prob, n_bins=10),
        f"{prefix}_prob_mean": float(np.mean(prob)),
        f"{prefix}_prob_median": float(np.median(prob)),
    }


def rolling_backtest(
    base,
    df: pd.DataFrame,
    structured_cols: list[str],
    news_cols: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = df[base.TARGET_COL].to_numpy(dtype=np.float32)
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

        base.set_seed(args.seed + run_no)
        structured_x, news_x = base.fit_transform(df, structured_cols, news_cols, train_end_indices, args.lookback)
        structured_train, news_train, y_train, train_meta = base.build_sequences(
            df, structured_x, news_x, train_end_indices, args.lookback, args.horizon
        )
        structured_val, news_val, y_val, val_meta = base.build_sequences(
            df, structured_x, news_x, val_end_indices, args.lookback, args.horizon
        )
        structured_test, news_test, y_test, test_meta = base.build_sequences(
            df, structured_x, news_x, np.asarray([test_end_idx], dtype=np.int64), args.lookback, args.horizon
        )

        model, train_info = base.train_model(
            structured_train,
            news_train,
            y_train,
            structured_val,
            news_val,
            y_val,
            args,
        )
        val_logits = predict_logits(model, structured_val, news_val, args.device, args.batch_size)
        val_raw_prob = sigmoid_np(val_logits)
        raw_threshold_info = base.best_threshold_metrics(y_val.astype(int), val_raw_prob)
        raw_val_threshold = float(raw_threshold_info["threshold"])

        calibrator = fit_positive_platt(val_logits, y_val, l2=args.calibrator_l2, max_iter=args.calibrator_max_iter)
        val_calibrated_prob = apply_calibrator(val_logits, calibrator)
        val_calibrated_threshold_info = base.best_threshold_metrics(y_val.astype(int), val_calibrated_prob)
        val_calibrated_threshold = float(val_calibrated_threshold_info["threshold"])

        test_logit = float(predict_logits(model, structured_test, news_test, args.device, args.batch_size)[0])
        raw_probability = float(sigmoid_np(np.asarray([test_logit]))[0])
        calibrated_probability = float(apply_calibrator(np.asarray([test_logit]), calibrator)[0])

        rows.append(
            {
                "run_no": run_no,
                "input_start_month": test_meta.iloc[0]["input_start_month"],
                "input_end_month": test_meta.iloc[0]["input_end_month"],
                "target_month": test_meta.iloc[0]["target_month"],
                "target": int(y_test[0]),
                "logit": test_logit,
                "raw_probability": raw_probability,
                "calibrated_probability": calibrated_probability,
                "raw_pred_0p5": int(raw_probability >= 0.5),
                "calibrated_pred_0p5": int(calibrated_probability >= 0.5),
                "calibrated_pred_0p6": int(calibrated_probability >= 0.6),
                "calibrated_pred_0p7": int(calibrated_probability >= 0.7),
                "raw_val_threshold": raw_val_threshold,
                "raw_pred_val_threshold": int(raw_probability >= raw_val_threshold),
                "calibrated_val_threshold": val_calibrated_threshold,
                "calibrated_pred_val_threshold": int(calibrated_probability >= val_calibrated_threshold),
                "calibrator_method": calibrator["method"],
                "calibrator_a": float(calibrator["a"]),
                "calibrator_b": float(calibrator["b"]),
                "raw_val_threshold_accuracy": raw_threshold_info["accuracy"],
                "raw_val_threshold_balanced_accuracy": raw_threshold_info["balanced_accuracy"],
                "raw_val_threshold_f1_weighted": raw_threshold_info["f1_weighted"],
                "raw_val_threshold_f1_positive": raw_threshold_info["f1_positive"],
                "calibrated_val_brier": float(brier_score_loss(y_val.astype(int), val_calibrated_prob)),
                "raw_val_brier": float(brier_score_loss(y_val.astype(int), val_raw_prob)),
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
            f"y={int(y_test[0])} raw={raw_probability:.4f} cal={calibrated_probability:.4f} "
            f"method={calibrator['method']} a={float(calibrator['a']):.3f} b={float(calibrator['b']):.3f}",
            flush=True,
        )

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["target"].to_numpy(dtype=int)
    raw_prob = pred_df["raw_probability"].to_numpy(dtype=float)
    calibrated_prob = pred_df["calibrated_probability"].to_numpy(dtype=float)

    summary = {
        "raw_threshold_0p5": base.metrics(y_true, raw_prob, 0.5),
        "raw_validation_selected_threshold_reference": base.metrics_from_predictions(
            y_true,
            pred_df["raw_pred_val_threshold"].to_numpy(dtype=int),
            raw_prob,
            threshold_rule="Reference only: each rolling test month uses that run's past validation set to choose a raw-probability threshold.",
        ),
        "calibrated_threshold_0p5": base.metrics(y_true, calibrated_prob, 0.5),
        "calibrated_threshold_0p6": base.metrics(y_true, calibrated_prob, 0.6),
        "calibrated_threshold_0p7": base.metrics(y_true, calibrated_prob, 0.7),
        "calibrated_validation_selected_threshold_reference": base.metrics_from_predictions(
            y_true,
            pred_df["calibrated_pred_val_threshold"].to_numpy(dtype=int),
            calibrated_prob,
            threshold_rule="Reference only: threshold is chosen on calibrated validation probabilities, not used as the deployable main result.",
        ),
        "probability_diagnostics": {
            **add_probability_diagnostics(y_true, raw_prob, "raw"),
            **add_probability_diagnostics(y_true, calibrated_prob, "calibrated"),
        },
        "raw_validation_threshold_summary": {
            "mean": float(pred_df["raw_val_threshold"].mean()),
            "median": float(pred_df["raw_val_threshold"].median()),
            "min": float(pred_df["raw_val_threshold"].min()),
            "max": float(pred_df["raw_val_threshold"].max()),
        },
        "calibrated_validation_threshold_summary_reference": {
            "mean": float(pred_df["calibrated_val_threshold"].mean()),
            "median": float(pred_df["calibrated_val_threshold"].median()),
            "min": float(pred_df["calibrated_val_threshold"].min()),
            "max": float(pred_df["calibrated_val_threshold"].max()),
        },
        "calibrator_summary": {
            "positive_platt_count": int((pred_df["calibrator_method"] == "positive_platt").sum()),
            "fallback_count": int((pred_df["calibrator_method"] != "positive_platt").sum()),
            "a_mean": float(pred_df["calibrator_a"].mean()),
            "a_median": float(pred_df["calibrator_a"].median()),
            "b_mean": float(pred_df["calibrator_b"].mean()),
            "b_median": float(pred_df["calibrator_b"].median()),
        },
        "history_summary": {
            "mean_best_epoch": float(np.mean([item["best_epoch"] for item in histories])),
            "mean_epochs_ran": float(np.mean([item["epochs_ran"] for item in histories])),
        },
    }
    return pred_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Rolling backtest dual-stream LSTM with validation-only Platt calibration.")
    parser.add_argument("--base-script", type=Path, default=BASE_SCRIPT)
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
    parser.add_argument("--calibrator-l2", type=float, default=1e-3)
    parser.add_argument("--calibrator-max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.horizon != 1:
        raise SystemExit("This experiment keeps horizon=1.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(args.base_script)
    df, structured_cols, news_cols = base.load_inputs(args.csv, args.feature_rank, args.top_n)
    pred_df, summary_metrics = rolling_backtest(base, df, structured_cols, news_cols, args)

    selected_features = pd.DataFrame(
        {
            "branch": ["structured"] * len(structured_cols) + ["news"] * len(news_cols),
            "feature": structured_cols + news_cols,
        }
    )
    selected_features.to_csv(args.out_dir / "top79_selected_features_for_dual_lstm_platt.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(args.out_dir / "top79_dual_lstm_platt_rolling_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Dual-stream LSTM with validation-only positive Platt calibration",
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
            "calibration": "For each rolling run, fit p=sigmoid(a*logit+b) on that run's past validation set only; deployable main metrics use fixed thresholds.",
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
            "calibrator_l2": args.calibrator_l2,
            "calibrator_max_iter": args.calibrator_max_iter,
            "seed": args.seed,
        },
        "metrics": summary_metrics,
    }
    (args.out_dir / "top79_dual_lstm_platt_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
