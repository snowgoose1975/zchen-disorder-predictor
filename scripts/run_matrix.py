from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


EMBEDDING_FAMILIES = ("esm2-layer30", "prott5-layer24", "esmc-layer30")
SUPPORTED_EMBEDDING_FAMILIES = EMBEDDING_FAMILIES + ("fusion-esm2-prott5-esmc",)
ARCHITECTURES = ("linear", "mlp", "tcn", "bigru", "conv_bigru")


def parse_names(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(
            f"unsupported {name}: {unknown}; choose from {', '.join(allowed)}"
        )
    return names


def parse_seed_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("seeds must not be empty")
    try:
        seeds = [int(item) for item in values]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return list(dict.fromkeys(seeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or print a reproducible embedding/head experiment matrix"
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--embedding-families",
        default=",".join(EMBEDDING_FAMILIES),
        help="Comma-separated manifest directory names",
    )
    parser.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURES),
        help="Comma-separated prediction heads",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds; overrides --seed and creates one run per seed",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--embedding-normalization",
        choices=("none", "standardize"),
        default="none",
        help="Feature preprocessing applied using train-split statistics",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("residue", "protein"),
        default="residue",
        help="Training loss weighting passed to each pipeline run",
    )
    parser.add_argument("--threshold-objective", choices=("mcc", "f1"), default="mcc")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting training or creating output directories",
    )
    return parser.parse_args()


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"unsafe matrix name: {name!r}")


def build_command(
    args: argparse.Namespace,
    manifest_dir: Path,
    output_dir: Path,
    architecture: str,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.run_pipeline",
        "--train-manifest",
        str(manifest_dir / "train.tsv"),
        "--val-manifest",
        str(manifest_dir / "val.tsv"),
        "--test-manifest",
        str(manifest_dir / "test.tsv"),
        "--output-dir",
        str(output_dir),
        "--architecture",
        architecture,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(seed),
        "--num-workers",
        str(args.num_workers),
        "--threshold-objective",
        args.threshold_objective,
        "--device",
        args.device,
        "--embedding-normalization",
        args.embedding_normalization,
        "--loss-weighting",
        args.loss_weighting,
    ]
    if args.pos_weight is not None:
        command.extend(["--pos-weight", str(args.pos_weight)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def write_matrix_results(path: Path, runs: list[dict[str, object]]) -> None:
    fields = [
        "embedding_family",
        "architecture",
        "seed",
        "status",
        "test_average_precision",
        "test_auroc",
        "test_mcc",
        "test_f1",
        "test_brier_score",
        "test_expected_calibration_error",
        "test_macro_average_precision",
        "test_macro_auroc",
        "elapsed_seconds",
        "output_dir",
    ]
    rows: list[dict[str, object]] = []
    for record in runs:
        test_metrics: dict[str, object] = {}
        protein_level: dict[str, object] = {}
        summary_path = Path(str(record["output_dir"])) / "run_summary.json"
        if record.get("status") == "passed" and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            test_metrics = summary.get("test_metrics", {})
            protein_level = test_metrics.get("protein_level", {})
        rows.append(
            {
                "embedding_family": record.get("embedding_family", ""),
                "architecture": record.get("architecture", ""),
                "seed": record.get("seed", ""),
                "status": record.get("status", ""),
                "test_average_precision": test_metrics.get("average_precision", ""),
                "test_auroc": test_metrics.get("auroc", ""),
                "test_mcc": test_metrics.get("mcc", ""),
                "test_f1": test_metrics.get("f1", ""),
                "test_brier_score": test_metrics.get("brier_score", ""),
                "test_expected_calibration_error": test_metrics.get(
                    "expected_calibration_error", ""
                ),
                "test_macro_average_precision": protein_level.get(
                    "macro_average_precision", ""
                ),
                "test_macro_auroc": protein_level.get("macro_auroc", ""),
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "output_dir": record.get("output_dir", ""),
            }
        )

    def ranking(row: dict[str, object]) -> float:
        value = row["test_average_precision"]
        try:
            return float(value)
        except (TypeError, ValueError):
            return -float("inf")

    rows.sort(key=ranking, reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    families = parse_names(
        args.embedding_families,
        SUPPORTED_EMBEDDING_FAMILIES,
        "embedding-families",
    )
    architectures = parse_names(
        args.architectures,
        ARCHITECTURES,
        "architectures",
    )
    seeds = parse_seed_list(
        args.seeds if args.seeds is not None else str(args.seed)
    )
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.hidden_dim < 1 or args.learning_rate <= 0:
        raise ValueError("hidden-dim and learning-rate must be positive")

    manifest_root = Path(args.manifest_root).resolve()
    output_root = Path(args.output_root).resolve()
    runs: list[dict[str, object]] = []

    for seed in seeds:
        for family in families:
            validate_name(family)
            manifest_dir = manifest_root / family
            for architecture in architectures:
                validate_name(architecture)
                normalization_suffix = (
                    ""
                    if args.embedding_normalization == "none"
                    else f"-norm-{args.embedding_normalization}"
                )
                loss_suffix = (
                    ""
                    if args.loss_weighting == "residue"
                    else f"-loss-{args.loss_weighting}"
                )
                output_dir = output_root / (
                    f"own-{architecture}-{family}{normalization_suffix}"
                    f"{loss_suffix}-seed{seed}"
                )
                command = build_command(
                    args, manifest_dir, output_dir, architecture, seed
                )
                record: dict[str, object] = {
                    "embedding_family": family,
                    "architecture": architecture,
                    "seed": seed,
                    "manifest_dir": str(manifest_dir),
                    "output_dir": str(output_dir),
                    "command": command,
                }
                print("$ " + " ".join(command), flush=True)
                if args.dry_run:
                    record["status"] = "dry-run"
                    runs.append(record)
                    continue

                if not manifest_dir.is_dir():
                    raise FileNotFoundError(manifest_dir)
                for filename in ("train.tsv", "val.tsv", "test.tsv"):
                    if not (manifest_dir / filename).is_file():
                        raise FileNotFoundError(manifest_dir / filename)
                if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
                    raise FileExistsError(
                        f"{output_dir} is not empty; use --overwrite or a new seed"
                    )

                started = time.perf_counter()
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as exc:
                    record["status"] = "failed"
                    record["returncode"] = exc.returncode
                    runs.append(record)
                    if not args.continue_on_error:
                        raise
                else:
                    record["status"] = "passed"
                    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    runs.append(record)

    summary = {
        "manifest_root": str(manifest_root),
        "output_root": str(output_root),
        "arguments": vars(args),
        "run_count": len(runs),
        "runs": runs,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "matrix_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_matrix_results(output_root / "matrix_results.tsv", runs)


if __name__ == "__main__":
    main()
