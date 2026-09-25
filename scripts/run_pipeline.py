from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, predict, select a validation threshold, and evaluate one "
            "residue-level experiment"
        )
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--architecture",
        choices=("linear", "mlp", "tcn", "bigru", "conv_bigru"),
        default="mlp",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--tcn-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--loss-weighting",
        choices=("residue", "protein"),
        default="residue",
        help="Residue-weighted BCE or equal-weighted mean loss across proteins",
    )
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--embedding-normalization",
        choices=("none", "standardize"),
        default="none",
        help="Optional train-split feature standardization",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("mcc", "f1"),
        default="mcc",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> dict[str, object]:
    print(f"== {name} ==")
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "command": command,
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; use a new named directory or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--train-manifest",
        args.train_manifest,
        "--val-manifest",
        args.val_manifest,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--architecture",
        args.architecture,
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--tcn-kernel-size",
        str(args.tcn_kernel_size),
        "--tcn-layers",
        str(args.tcn_layers),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--embedding-normalization",
        args.embedding_normalization,
        "--loss-weighting",
        args.loss_weighting,
    ]
    if args.pos_weight is not None:
        train_command.extend(["--pos-weight", str(args.pos_weight)])

    steps = [
        run_step("train", train_command),
    ]

    checkpoint = output_dir / "best.pt"
    val_prediction = output_dir / "val.prediction.tsv"
    test_prediction = output_dir / "test.prediction.tsv"
    val_metrics_path = output_dir / "val.metrics.json"
    test_metrics_path = output_dir / "test.metrics.json"
    val_protein_path = output_dir / "val.protein.tsv"
    test_protein_path = output_dir / "test.protein.tsv"

    steps.append(
        run_step(
            "predict-validation",
            [
                sys.executable,
                "-m",
                "scripts.predict_manifest",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                args.val_manifest,
                "--output",
                str(val_prediction),
                "--device",
                args.device,
            ],
        )
    )
    steps.append(
        run_step(
            "evaluate-validation",
            [
                sys.executable,
                "-m",
                "scripts.evaluate",
                "--prediction",
                str(val_prediction),
                "--manifest",
                args.val_manifest,
                "--select-threshold",
                args.threshold_objective,
                "--output-json",
                str(val_metrics_path),
                "--output-protein-tsv",
                str(val_protein_path),
            ],
        )
    )

    validation_metrics = json.loads(val_metrics_path.read_text(encoding="utf-8"))
    threshold = float(validation_metrics["threshold"])

    steps.append(
        run_step(
            "predict-test",
            [
                sys.executable,
                "-m",
                "scripts.predict_manifest",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                args.test_manifest,
                "--output",
                str(test_prediction),
                "--device",
                args.device,
            ],
        )
    )
    steps.append(
        run_step(
            "evaluate-test",
            [
                sys.executable,
                "-m",
                "scripts.evaluate",
                "--prediction",
                str(test_prediction),
                "--manifest",
                args.test_manifest,
                "--threshold",
                str(threshold),
                "--output-json",
                str(test_metrics_path),
                "--output-protein-tsv",
                str(test_protein_path),
            ],
        )
    )

    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    summary = {
        "arguments": vars(args),
        "checkpoint": str(checkpoint.resolve()),
        "validation_threshold_objective": args.threshold_objective,
        "validation_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "steps": steps,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
