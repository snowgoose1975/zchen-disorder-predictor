from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from my_predictor.io import load_vector, read_manifest
from my_predictor.metrics import classification_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate residue-level predictions against a manifest"
    )
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-protein-tsv")
    parser.add_argument(
        "--stratum-file",
        help="Optional TSV with columns id and stratum for grouped evaluation",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--select-threshold", choices=("mcc", "f1"))
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Optional protein-level bootstrap replicate count; 0 disables it",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260925,
        help="Seed for the optional protein-level bootstrap",
    )
    return parser.parse_args()


def read_predictions(path: str | Path) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "position", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns id, position, score")
        for row in reader:
            identifier = row["id"]
            position = int(row["position"])
            if identifier not in result:
                result[identifier] = {}
            if position in result[identifier]:
                raise ValueError(f"duplicate prediction: {identifier}:{position}")
            result[identifier][position] = float(row["score"])
    return result


def read_strata(
    path: str | Path,
) -> tuple[str, dict[str, str] | dict[str, dict[int, str]]]:
    """Read protein-level or residue-level strata."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=chr(9))
        fields = set(reader.fieldnames or ())
        if {"id", "position", "stratum"}.issubset(fields):
            result: dict[str, dict[int, str]] = defaultdict(dict)
            for row in reader:
                identifier = row["id"]
                stratum = row["stratum"].strip()
                if not identifier or not stratum:
                    raise ValueError(f"{path}: id and stratum must not be empty")
                try:
                    position = int(row["position"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: position must be an integer") from exc
                if position < 1:
                    raise ValueError(f"{path}: position must be >= 1")
                if position in result[identifier]:
                    raise ValueError(
                        f"{path}: duplicate stratum for {identifier}:{position}"
                    )
                result[identifier][position] = stratum
            return "residue", dict(result)

        required = {"id", "stratum"}
        if not required.issubset(fields):
            raise ValueError(
                f"{path}: expected columns id and stratum, or id, position, and stratum"
            )
        result_protein: dict[str, str] = {}
        for row in reader:
            identifier = row["id"]
            stratum = row["stratum"].strip()
            if not identifier or not stratum:
                raise ValueError(f"{path}: id and stratum must not be empty")
            if identifier in result_protein:
                raise ValueError(f"{path}: duplicate stratum for {identifier}")
            result_protein[identifier] = stratum
        return "protein", result_protein


def collect_arrays(
    manifest_path: str | Path,
    predictions: dict[str, dict[int, float]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    list[tuple[str, int, int]],
]:
    scores: list[float] = []
    labels: list[int] = []
    valid: list[bool] = []
    spans: list[tuple[str, int, int]] = []
    total_residues = 0
    manifest_ids: set[str] = set()

    for row in read_manifest(manifest_path):
        identifier = row["id"]
        if identifier in manifest_ids:
            raise ValueError(f"duplicate manifest id: {identifier}")
        manifest_ids.add(identifier)
        labels_array = load_vector(row["labels"]).astype(np.int64)
        valid_array = np.isin(labels_array, (0, 1))
        mask_path = row.get("mask", "")
        if mask_path:
            mask_array = load_vector(mask_path).astype(bool)
            if labels_array.shape != mask_array.shape:
                raise ValueError(f"{identifier}: labels and mask lengths differ")
            valid_array &= mask_array
        positions = predictions.get(identifier)
        if positions is None:
            raise ValueError(f"missing predictions for {identifier}")
        expected = set(range(1, len(labels_array) + 1))
        if set(positions) != expected:
            raise ValueError(
                f"{identifier}: prediction positions do not exactly match "
                f"1..{len(labels_array)}"
            )
        start = len(scores)
        scores.extend(positions[position] for position in range(1, len(labels_array) + 1))
        labels.extend(labels_array.tolist())
        valid.extend(valid_array.tolist())
        end = len(scores)
        spans.append((identifier, start, end))
        total_residues += len(labels_array)

    extra_ids = set(predictions) - manifest_ids
    if extra_ids:
        raise ValueError(
            "prediction contains IDs absent from manifest: "
            + ", ".join(sorted(extra_ids)[:5])
        )

    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(valid, dtype=bool),
        total_residues,
        spans,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def protein_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    spans: list[tuple[str, int, int]],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, start, end in spans:
        row = classification_metrics(
            scores[start:end],
            labels[start:end],
            valid[start:end],
            threshold=threshold,
            total_residues=end - start,
        )
        rows.append(
            _json_safe(
                {
                    "id": identifier,
                    "length": end - start,
                    **row,
                }
            )
        )
    return rows


def summarize_protein_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["valid_residues"] > 0]

    def summarize(name: str) -> tuple[float | None, float | None, int]:
        values = [
            float(row[name])
            for row in eligible
            if row.get(name) is not None
        ]
        if not values:
            return None, None, 0
        return float(np.mean(values)), float(np.median(values)), len(values)

    mean_ap, median_ap, ap_count = summarize("average_precision")
    mean_auroc, median_auroc, auroc_count = summarize("auroc")
    return {
        "protein_count": len(rows),
        "proteins_with_valid_residues": len(eligible),
        "ap_defined_proteins": ap_count,
        "macro_average_precision": mean_ap,
        "median_protein_average_precision": median_ap,
        "auroc_defined_proteins": auroc_count,
        "macro_auroc": mean_auroc,
        "median_protein_auroc": median_auroc,
    }


def protein_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    spans: list[tuple[str, int, int]],
    threshold: float,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not spans:
        raise ValueError("cannot bootstrap an empty protein list")

    rng = np.random.default_rng(seed)
    metric_names = (
        "average_precision",
        "auroc",
        "brier_score",
        "mcc",
        "f1",
        "expected_calibration_error",
    )
    sampled_values = {name: [] for name in metric_names}
    protein_count = len(spans)

    for _ in range(replicates):
        selected = rng.integers(0, protein_count, size=protein_count)
        sampled_scores = np.concatenate(
            [scores[spans[int(index)][1] : spans[int(index)][2]] for index in selected]
        )
        sampled_labels = np.concatenate(
            [labels[spans[int(index)][1] : spans[int(index)][2]] for index in selected]
        )
        sampled_valid = np.concatenate(
            [valid[spans[int(index)][1] : spans[int(index)][2]] for index in selected]
        )
        values = classification_metrics(
            sampled_scores,
            sampled_labels,
            sampled_valid,
            threshold=threshold,
            total_residues=len(sampled_scores),
        )
        for name in metric_names:
            value = values[name]
            if value is not None and np.isfinite(value):
                sampled_values[name].append(float(value))

    intervals: dict[str, Any] = {}
    for name, values in sampled_values.items():
        if not values:
            intervals[name] = {
                "mean": None,
                "lower": None,
                "upper": None,
                "defined_replicates": 0,
            }
            continue
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "mean": float(np.mean(array)),
            "lower": float(np.percentile(array, 2.5)),
            "upper": float(np.percentile(array, 97.5)),
            "defined_replicates": int(array.size),
        }

    return {
        "available": True,
        "unit": "protein",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "metrics": intervals,
    }


def write_protein_metrics(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "length",
        "valid_residues",
        "positive_residues",
        "coverage",
        "positive_fraction",
        "average_precision",
        "auroc",
        "brier_score",
        "expected_calibration_error",
        "mcc",
        "f1",
        "threshold",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def grouped_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    spans: list[tuple[str, int, int]],
    strata: dict[str, str] | dict[str, dict[int, str]],
    mode: str,
    threshold: float,
) -> dict[str, Any]:
    manifest_ids = {identifier for identifier, _, _ in spans}
    missing = manifest_ids - set(strata)
    extra = set(strata) - manifest_ids
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:5]}")
        if extra:
            details.append(f"extra={sorted(extra)[:5]}")
        raise ValueError("stratum file IDs do not match manifest: " + ", ".join(details))

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    grouped_proteins: dict[str, set[str]] = defaultdict(set)
    if mode == "protein":
        protein_strata = strata
        for identifier, start, end in spans:
            stratum = protein_strata[identifier]  # type: ignore[index]
            grouped_indices[stratum].extend(range(start, end))
            grouped_proteins[stratum].add(identifier)
    elif mode == "residue":
        residue_strata = strata
        for identifier, start, end in spans:
            mapping = residue_strata[identifier]  # type: ignore[index]
            expected = set(range(1, end - start + 1))
            if set(mapping) != expected:
                missing_positions = sorted(expected - set(mapping))[:5]
                extra_positions = sorted(set(mapping) - expected)[:5]
                raise ValueError(
                    f"{identifier}: residue strata must cover positions 1..{end - start}; "
                    f"missing={missing_positions}, extra={extra_positions}"
                )
            for offset in range(end - start):
                grouped_indices[mapping[offset + 1]].append(start + offset)
            for stratum in set(mapping.values()):
                grouped_proteins[stratum].add(identifier)
    else:
        raise ValueError(f"unknown stratum mode: {mode!r}")

    result: dict[str, Any] = {}
    for stratum, indices in sorted(grouped_indices.items()):
        index_array = np.asarray(indices, dtype=np.int64)
        group_scores = scores[index_array]
        group_labels = labels[index_array]
        group_valid = valid[index_array]
        result[stratum] = _json_safe(
            {
                "unit": mode,
                "protein_count": len(grouped_proteins[stratum]),
                **classification_metrics(
                    group_scores,
                    group_labels,
                    group_valid,
                    threshold=threshold,
                    total_residues=len(group_scores),
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    predictions = read_predictions(args.prediction)
    scores, labels, valid, total_residues, spans = collect_arrays(
        args.manifest,
        predictions,
    )
    threshold = args.threshold
    if args.select_threshold:
        threshold = select_threshold(
            scores,
            labels,
            valid,
            objective=args.select_threshold,
        )

    metrics = _json_safe(
        classification_metrics(
            scores,
            labels,
            valid,
            threshold=threshold,
            total_residues=total_residues,
        )
    )
    rows = protein_metrics(scores, labels, valid, spans, threshold)
    metrics["protein_level"] = summarize_protein_metrics(rows)
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative")
    if args.bootstrap_replicates:
        metrics["protein_bootstrap"] = protein_bootstrap(
            scores,
            labels,
            valid,
            spans,
            threshold,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
    else:
        metrics["protein_bootstrap"] = {
            "available": False,
            "reason": "Bootstrap disabled; pass --bootstrap-replicates to enable it.",
        }
    if args.stratum_file:
        stratum_mode, strata = read_strata(args.stratum_file)
        metrics["stratification"] = {
            "available": True,
            "source": str(Path(args.stratum_file).resolve()),
            "unit": stratum_mode,
            "groups": grouped_metrics(
                scores,
                labels,
                valid,
                spans,
                strata,
                stratum_mode,
                threshold,
            ),
        }
    else:
        metrics["stratification"] = {
            "available": False,
            "reason": (
                "No --stratum-file was supplied. The current handoff does not "
                "contain a topology stratum column."
            ),
        }
    metrics["prediction_file"] = str(Path(args.prediction).resolve())
    metrics["manifest_file"] = str(Path(args.manifest).resolve())

    output = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    if args.output_protein_tsv:
        write_protein_metrics(args.output_protein_tsv, rows)


if __name__ == "__main__":
    main()
