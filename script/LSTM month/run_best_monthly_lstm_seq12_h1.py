from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from run_monthly_lstm_dce_corn import run


ROOT = Path(__file__).resolve().parent


def main() -> None:
    args = Namespace(
        data_path=str(ROOT / "monthly_corn_data" / "玉米价格月度_混合特征版.csv"),
        output_dir=str(ROOT / "monthly_lstm_best_seq12_h1"),
        seq_len=12,
        hidden_size=32,
        num_layers=2,
        dropout=0.20,
        batch_size=16,
        epochs=300,
        patience=40,
        lr=0.0005,
        weight_decay=0.0001,
        grad_clip=1.0,
        train_ratio=0.70,
        val_ratio=0.15,
        max_missing_ratio=0.30,
        device="cpu",
        seed=42,
    )
    run(args)


if __name__ == "__main__":
    main()
