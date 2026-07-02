from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "玉米价格原始数据.csv"
OUTPUT_DIR = ROOT / "monthly_corn_data"

MONTH_END_OUTPUT = OUTPUT_DIR / "玉米价格月度_月末版.csv"
MIXED_OUTPUT = OUTPUT_DIR / "玉米价格月度_混合特征版.csv"

TARGET_COL = "dce_corn_close"
LABEL_THRESHOLD = 0.0

MARKET_PREFIXES = ("dce_corn", "dce_corn_starch", "cbot_corn", "cbot_wheat")
MARKET_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "settle": "last",
    "volume": "sum",
    "open_interest": "last",
}


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def market_field(col: str) -> tuple[str, str] | None:
    for prefix in MARKET_PREFIXES:
        head = prefix + "_"
        if col.startswith(head):
            field = col[len(head) :]
            if field in MARKET_AGG_RULES:
                return prefix, field
    return None


def compound_return(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() - 1.0)


def month_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df.groupby("month", sort=True)["date"]
        .agg(first_trade_date="min", last_trade_date="max", n_obs="size")
        .reset_index()
    )
    meta["first_trade_date"] = meta["first_trade_date"].dt.strftime("%Y-%m-%d")
    meta["last_trade_date"] = meta["last_trade_date"].dt.strftime("%Y-%m-%d")
    return meta


def add_group_agg(
    out: pd.DataFrame,
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    col: str,
    out_col: str,
    func: str | Callable[[pd.Series], float],
) -> None:
    if isinstance(func, str):
        series = grouped[col].agg(func)
    else:
        series = grouped[col].apply(func)
    out[out_col] = series.reindex(out["month"]).to_numpy()


def add_next_month_labels(out: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    out = out.copy()
    if target_col not in out.columns:
        return out

    next_col = f"{target_col}_next_month"
    ret_col = f"{target_col}_next_month_ret"
    dir_col = f"{target_col}_next_month_direction"

    out[next_col] = out[target_col].shift(-1)
    out[ret_col] = out[next_col] / out[target_col] - 1.0

    direction = (out[ret_col] > LABEL_THRESHOLD).astype("Int64")
    direction[out[ret_col].isna()] = pd.NA
    out[dir_col] = direction
    return out


def add_close_rolling_features(out: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    out = out.copy()
    if target_col not in out.columns:
        return out

    close = out[target_col]
    ret_1m = close.pct_change()
    out[f"{target_col}_ret_1m"] = ret_1m
    for window in (3, 6, 12):
        out[f"{target_col}_ma{window}"] = close.rolling(window, min_periods=window).mean()
        out[f"{target_col}_ret_{window}m"] = close.pct_change(window)
        out[f"{target_col}_vol_{window}m"] = ret_1m.rolling(window, min_periods=window).std()
    return out


def build_month_end(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("month", sort=True)
    out = month_meta(df)

    for col in df.columns:
        if col in {"date", "month"}:
            continue

        field_info = market_field(col)
        if field_info is not None:
            _, field = field_info
            add_group_agg(out, grouped, col, col, MARKET_AGG_RULES[field])
        elif col.endswith("_ret_1d"):
            add_group_agg(out, grouped, col, col.replace("_ret_1d", "_ret_1m_compound"), compound_return)
        elif col.endswith("precip_mm"):
            add_group_agg(out, grouped, col, col, "sum")
        elif col.endswith("t2m_c"):
            add_group_agg(out, grouped, col, col, "mean")
        elif col == "is_harvest_season_ne_cn":
            add_group_agg(out, grouped, col, col, "max")
        elif col == "reserve_corn_release_volume_ton":
            add_group_agg(out, grouped, col, col, "sum")
        elif col == "corn_import_volume_ton_ffill":
            add_group_agg(out, grouped, col, col, "last")
        else:
            add_group_agg(out, grouped, col, col, "last")

    out = add_close_rolling_features(out, TARGET_COL)
    out = add_next_month_labels(out, TARGET_COL)
    return out


def build_mixed_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("month", sort=True)
    out = month_meta(df)
    processed: set[str] = set()

    for prefix in MARKET_PREFIXES:
        close_last_col = f"{prefix}_close_last"
        high_max_col = f"{prefix}_high_max"
        low_min_col = f"{prefix}_low_min"

        for field in ("open", "high", "low", "close", "settle", "volume", "open_interest"):
            col = f"{prefix}_{field}"
            if col not in df.columns:
                continue
            processed.add(col)

            if field == "open":
                add_group_agg(out, grouped, col, f"{col}_first", "first")
            elif field == "high":
                add_group_agg(out, grouped, col, high_max_col, "max")
            elif field == "low":
                add_group_agg(out, grouped, col, low_min_col, "min")
            elif field == "close":
                add_group_agg(out, grouped, col, close_last_col, "last")
                add_group_agg(out, grouped, col, f"{col}_mean", "mean")
                add_group_agg(out, grouped, col, f"{col}_std", "std")
            elif field == "settle":
                add_group_agg(out, grouped, col, f"{col}_last", "last")
            elif field == "volume":
                add_group_agg(out, grouped, col, f"{col}_sum", "sum")
            elif field == "open_interest":
                add_group_agg(out, grouped, col, f"{col}_last", "last")

        if close_last_col in out.columns:
            out[f"{prefix}_ret_1m"] = out[close_last_col].pct_change()
        if high_max_col in out.columns and low_min_col in out.columns and close_last_col in out.columns:
            out[f"{prefix}_month_range_pct"] = (out[high_max_col] - out[low_min_col]) / out[close_last_col]

    if "dce_corn_close_last" in out.columns:
        out[TARGET_COL] = out["dce_corn_close_last"]

    for col in df.columns:
        if col in {"date", "month"} or col in processed:
            continue

        if col.endswith("_ret_1d"):
            add_group_agg(out, grouped, col, col.replace("_ret_1d", "_ret_1m_compound"), compound_return)
        elif col.endswith("precip_mm"):
            add_group_agg(out, grouped, col, f"{col}_sum", "sum")
            add_group_agg(out, grouped, col, f"{col}_mean", "mean")
        elif col.endswith("t2m_c"):
            add_group_agg(out, grouped, col, f"{col}_mean", "mean")
            add_group_agg(out, grouped, col, f"{col}_last", "last")
        elif col == "is_harvest_season_ne_cn":
            add_group_agg(out, grouped, col, f"{col}_flag", "max")
            add_group_agg(out, grouped, col, f"{col}_share", "mean")
        elif col == "reserve_corn_release_volume_ton":
            add_group_agg(out, grouped, col, f"{col}_sum", "sum")
        elif col == "corn_import_volume_ton_ffill":
            add_group_agg(out, grouped, col, f"{col}_last", "last")
        else:
            add_group_agg(out, grouped, col, f"{col}_mean", "mean")
            add_group_agg(out, grouped, col, f"{col}_last", "last")

    out = add_close_rolling_features(out, TARGET_COL)
    out = add_next_month_labels(out, TARGET_COL)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_source(INPUT_PATH)

    month_end = build_month_end(df)
    mixed = build_mixed_features(df)

    month_end.to_csv(MONTH_END_OUTPUT, index=False, encoding="utf-8-sig")
    mixed.to_csv(MIXED_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"source_rows={len(df)} source_cols={len(df.columns) - 1}")
    print(f"monthly_rows={len(month_end)}")
    print(f"month_end_cols={len(month_end.columns)}")
    print(f"mixed_cols={len(mixed.columns)}")
    print(f"month_end_output={MONTH_END_OUTPUT}")
    print(f"mixed_output={MIXED_OUTPUT}")


if __name__ == "__main__":
    main()
