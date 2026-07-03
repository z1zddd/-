from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


BASE_SCRIPT = Path(__file__).with_name("rolling_backtest选阈值版（保留前79个特征）.py")
HELPER_SCRIPT = Path(__file__).with_name("rolling_backtest_top79_dual_lstm_platt_calibration.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_expanding_calibration_split(
    train_val_end_indices: np.ndarray,
    labels: np.ndarray,
    horizon: int,
    min_train_samples: int,
    min_pos: int,
    min_neg: int,
    fallback_val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    for start in range(len(train_val_end_indices) - 1, min_train_samples - 1, -1):
        train_end_indices = train_val_end_indices[:start]
        calibration_end_indices = train_val_end_indices[start:]
        y_cal = labels[calibration_end_indices + horizon].astype(int)
        pos = int((y_cal == 1).sum())
        neg = int((y_cal == 0).sum())
        if pos >= min_pos and neg >= min_neg:
            return train_end_indices, calibration_end_indices, {
                "split_method": "expanding_balanced_calibration",
                "calibration_samples": int(len(calibration_end_indices)),
                "calibration_pos": pos,
                "calibration_neg": neg,
                "train_samples_before_sequence_filter": int(len(train_end_indices)),
            }

    val_size = max(1, int(math.ceil(len(train_val_end_indices) * fallback_val_fraction)))
    train_end_indices = train_val_end_indices[:-val_size]
    calibration_end_indices = train_val_end_indices[-val_size:]
    y_cal = labels[calibration_end_indices + horizon].astype(int)
    return train_end_indices, calibration_end_indices, {
        "split_method": "fallback_recent_validation",
        "calibration_samples": int(len(calibration_end_indices)),
        "calibration_pos": int((y_cal == 1).sum()),
        "calibration_neg": int((y_cal == 0).sum()),
        "train_samples_before_sequence_filter": int(len(train_end_indices)),
    }


def rolling_backtest(
    base,
    helper,
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
        train_end_indices, cal_end_indices, split_info = select_expanding_calibration_split(
            train_val_end_indices=train_val_end_indices,
            labels=labels,
            horizon=args.horizon,
            min_train_samples=args.calibration_min_train_samples,
            min_pos=args.calibration_min_pos,
            min_neg=args.calibration_min_neg,
            fallback_val_fraction=args.val_fraction,
        )
        if len(train_end_indices) < 2:
            continue

        base.set_seed(args.seed + run_no)
        structured_x, news_x = base.fit_transform(df, structured_cols, news_cols, train_end_indices, args.lookback)
        structured_train, news_train, y_train, train_meta = base.build_sequences(
            df, structured_x, news_x, train_end_indices, args.lookback, args.horizon
        )
        structured_cal, news_cal, y_cal, cal_meta = base.build_sequences(
            df, structured_x, news_x, cal_end_indices, args.lookback, args.horizon
        )
        structured_test, news_test, y_test, test_meta = base.build_sequences(
            df, structured_x, news_x, np.asarray([test_end_idx], dtype=np.int64), args.lookback, args.horizon
        )

        model, train_info = base.train_model(
            structured_train,
            news_train,
            y_train,
            structured_cal,
            news_cal,
            y_cal,
            args,
        )
        cal_logits = helper.predict_logits(model, structured_cal, news_cal, args.device, args.batch_size)
        cal_raw_prob = helper.sigmoid_np(cal_logits)
        raw_threshold_info = base.best_threshold_metrics(y_cal.astype(int), cal_raw_prob)
        raw_cal_threshold = float(raw_threshold_info["threshold"])

        calibrator = helper.fit_positive_platt(cal_logits, y_cal, l2=args.calibrator_l2, max_iter=args.calibrator_max_iter)
        cal_calibrated_prob = helper.apply_calibrator(cal_logits, calibrator)
        calibrated_threshold_info = base.best_threshold_metrics(y_cal.astype(int), cal_calibrated_prob)
        calibrated_cal_threshold = float(calibrated_threshold_info["threshold"])

        test_logit = float(helper.predict_logits(model, structured_test, news_test, args.device, args.batch_size)[0])
        raw_probability = float(helper.sigmoid_np(np.asarray([test_logit]))[0])
        calibrated_probability = float(helper.apply_calibrator(np.asarray([test_logit]), calibrator)[0])

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
                "raw_calibration_threshold": raw_cal_threshold,
                "raw_pred_calibration_threshold": int(raw_probability >= raw_cal_threshold),
                "calibrated_calibration_threshold": calibrated_cal_threshold,
                "calibrated_pred_calibration_threshold": int(calibrated_probability >= calibrated_cal_threshold),
                "calibrator_method": calibrator["method"],
                "calibrator_a": float(calibrator["a"]),
                "calibrator_b": float(calibrator["b"]),
                "split_method": split_info["split_method"],
                "calibration_samples": split_info["calibration_samples"],
                "calibration_pos": split_info["calibration_pos"],
                "calibration_neg": split_info["calibration_neg"],
                "train_samples_before_sequence_filter": split_info["train_samples_before_sequence_filter"],
                "raw_calibration_threshold_accuracy": raw_threshold_info["accuracy"],
                "raw_calibration_threshold_balanced_accuracy": raw_threshold_info["balanced_accuracy"],
                "raw_calibration_threshold_f1_weighted": raw_threshold_info["f1_weighted"],
                "raw_calibration_threshold_f1_positive": raw_threshold_info["f1_positive"],
                "calibrated_cal_brier": float(brier_score_loss(y_cal.astype(int), cal_calibrated_prob)),
                "raw_cal_brier": float(brier_score_loss(y_cal.astype(int), cal_raw_prob)),
                "train_start_target_month": train_meta.iloc[0]["target_month"],
                "train_end_target_month": train_meta.iloc[-1]["target_month"],
                "calibration_start_target_month": cal_meta.iloc[0]["target_month"],
                "calibration_end_target_month": cal_meta.iloc[-1]["target_month"],
                "train_samples": int(len(y_train)),
                "calibration_sequence_samples": int(len(y_cal)),
                "best_epoch": train_info["best_epoch"],
                "epochs_ran": train_info["epochs_ran"],
                "best_val_loss": train_info["best_val_loss"],
            }
        )
        histories.append({"run_no": run_no, **train_info})
        print(
            f"test {run_no}/{len(test_positions)} target={test_meta.iloc[0]['target_month']} "
            f"y={int(y_test[0])} raw={raw_probability:.4f} cal={calibrated_probability:.4f} "
            f"cal_n={split_info['calibration_samples']} pos={split_info['calibration_pos']} "
            f"neg={split_info['calibration_neg']} method={calibrator['method']}",
            flush=True,
        )

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["target"].to_numpy(dtype=int)
    raw_prob = pred_df["raw_probability"].to_numpy(dtype=float)
    calibrated_prob = pred_df["calibrated_probability"].to_numpy(dtype=float)

    summary = {
        "raw_threshold_0p5": base.metrics(y_true, raw_prob, 0.5),
        "raw_calibration_set_selected_threshold_reference": base.metrics_from_predictions(
            y_true,
            pred_df["raw_pred_calibration_threshold"].to_numpy(dtype=int),
            raw_prob,
            threshold_rule="Reference only: threshold is chosen on the same past expanding calibration set.",
        ),
        "calibrated_threshold_0p5": base.metrics(y_true, calibrated_prob, 0.5),
        "calibrated_threshold_0p6": base.metrics(y_true, calibrated_prob, 0.6),
        "calibrated_threshold_0p7": base.metrics(y_true, calibrated_prob, 0.7),
        "calibrated_calibration_set_selected_threshold_reference": base.metrics_from_predictions(
            y_true,
            pred_df["calibrated_pred_calibration_threshold"].to_numpy(dtype=int),
            calibrated_prob,
            threshold_rule="Reference only: threshold is chosen on calibrated probabilities in the past expanding calibration set.",
        ),
        "probability_diagnostics": {
            **helper.add_probability_diagnostics(y_true, raw_prob, "raw"),
            **helper.add_probability_diagnostics(y_true, calibrated_prob, "calibrated"),
        },
        "calibration_split_summary": {
            "expanding_balanced_count": int((pred_df["split_method"] == "expanding_balanced_calibration").sum()),
            "fallback_recent_validation_count": int((pred_df["split_method"] != "expanding_balanced_calibration").sum()),
            "calibration_samples_mean": float(pred_df["calibration_samples"].mean()),
            "calibration_samples_median": float(pred_df["calibration_samples"].median()),
            "calibration_samples_min": int(pred_df["calibration_samples"].min()),
            "calibration_samples_max": int(pred_df["calibration_samples"].max()),
            "calibration_pos_min": int(pred_df["calibration_pos"].min()),
            "calibration_neg_min": int(pred_df["calibration_neg"].min()),
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
    parser = argparse.ArgumentParser(description="Rolling backtest dual-stream LSTM with expanding balanced Platt calibration.")
    parser.add_argument("--base-script", type=Path, default=BASE_SCRIPT)
    parser.add_argument("--helper-script", type=Path, default=HELPER_SCRIPT)
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
    parser.add_argument("--calibration-min-train-samples", type=int, default=24)
    parser.add_argument("--calibration-min-pos", type=int, default=5)
    parser.add_argument("--calibration-min-neg", type=int, default=5)
    parser.add_argument("--calibrator-l2", type=float, default=1e-3)
    parser.add_argument("--calibrator-max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.horizon != 1:
        raise SystemExit("This experiment keeps horizon=1.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = load_module("top79_dual_lstm_base", args.base_script)
    helper = load_module("top79_dual_lstm_platt_helper", args.helper_script)
    df, structured_cols, news_cols = base.load_inputs(args.csv, args.feature_rank, args.top_n)
    pred_df, summary_metrics = rolling_backtest(base, helper, df, structured_cols, news_cols, args)

    selected_features = pd.DataFrame(
        {
            "branch": ["structured"] * len(structured_cols) + ["news"] * len(news_cols),
            "feature": structured_cols + news_cols,
        }
    )
    selected_features.to_csv(args.out_dir / "top79_selected_features_for_dual_lstm_expanding_platt.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(args.out_dir / "top79_dual_lstm_expanding_platt_rolling_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Dual-stream LSTM with expanding balanced validation-only positive Platt calibration",
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
            "min_train_samples_for_test_start": args.min_train_samples,
            "calibration_min_train_samples": args.calibration_min_train_samples,
            "calibration_min_pos": args.calibration_min_pos,
            "calibration_min_neg": args.calibration_min_neg,
            "step": args.step,
            "max_tests": args.max_tests,
            "calibration": "For each rolling run, choose the most recent past calibration block that has enough positive and negative labels while keeping the earlier training block strictly before it.",
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
    (args.out_dir / "top79_dual_lstm_expanding_platt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
