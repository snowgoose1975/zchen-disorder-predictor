from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test manifests from the TM disorder handoff"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-type", default="esm2")
    parser.add_argument("--embedding-key", default="layer_30")
    parser.add_argument(
        "--label-channel",
        choices=("merged", "curated", "nmr_chemical_shift", "structure_derived"),
        default="merged",
        help="Use merged labels or one selected source channel",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_vector(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = [2 if character == "-" else int(character) for character in stripped]
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a JSON/list vector")
    return value


def select_labels(
    record: dict[str, Any],
    channel: str,
    *,
    name: str,
) -> list[int]:
    if channel == "merged":
        raw = record.get("labels")
    else:
        channels = record.get("selected_channel_labels", {})
        if channel not in channels:
            raise ValueError(f"{name}: missing selected label channel {channel!r}")
        raw = channels[channel]
    labels = [int(value) for value in parse_vector(raw, name=name)]
    if any(label not in (0, 1, 2) for label in labels):
        raise ValueError(f"{name}: labels must contain only 0, 1, or 2")
    return labels


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("record_key")
            if not key:
                raise ValueError(f"{path}:{line_number}: missing record_key")
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate record_key {key}")
            records[key] = record
    return records


def build_manifests(args: argparse.Namespace) -> None:
    root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not requested_splits:
        raise ValueError("splits must not be empty")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --overwrite to regenerate manifests"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    catalogue_rows = read_tsv(root / "catalogue" / "records.tsv")
    records = read_records(root / "development" / "records.jsonl.gz")
    asset_rows = read_tsv(root / "catalogue" / "assets.tsv")

    assets: dict[str, dict[str, str]] = {}
    for asset in asset_rows:
        if asset.get("asset_type") != args.asset_type:
            continue
        sequence_key = asset.get("sequence_key", "")
        if not sequence_key:
            continue
        if sequence_key in assets:
            raise ValueError(
                f"multiple {args.asset_type} assets for sequence_key={sequence_key}"
            )
        assets[sequence_key] = asset

    fieldnames = [
        "id",
        "native_id",
        "sequence_key",
        "split",
        "embedding",
        "embedding_key",
        "labels",
        "mask",
        "length",
    ]
    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    try:
        for split in requested_splits:
            manifest_path = output_dir / f"{split}.tsv"
            handle = manifest_path.open("w", encoding="utf-8", newline="")
            handles[split] = handle
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writers[split] = writer

        selected = 0
        for catalogue in catalogue_rows:
            split = catalogue.get("split", "")
            if split not in requested_splits:
                continue
            if args.limit is not None and selected >= args.limit:
                break
            record_key = catalogue.get("record_key", "")
            record = records.get(record_key)
            if record is None:
                raise ValueError(f"missing JSONL record for {record_key}")
            sequence = record.get("sequence", "")
            length = int(catalogue.get("length", len(sequence)))
            if len(sequence) != length:
                raise ValueError(
                    f"{record_key}: catalogue length {length} != sequence length {len(sequence)}"
                )
            labels = select_labels(
                record,
                args.label_channel,
                name=f"{record_key}.{args.label_channel}",
            )
            if args.label_channel == "merged":
                mask = parse_vector(
                    record.get("label_mask"),
                    name=f"{record_key}.label_mask",
                )
                mask = [bool(item) for item in mask]
            else:
                mask = [label != 2 for label in labels]
            if len(labels) != length or len(mask) != length:
                raise ValueError(f"{record_key}: labels/mask length does not match {length}")
            sequence_key = catalogue.get("sequence_key", "")
            asset = assets.get(sequence_key)
            if asset is None:
                raise ValueError(
                    f"{record_key}: missing {args.asset_type} asset for {sequence_key}"
                )
            embedding_path = Path(asset["path"])
            if not embedding_path.is_absolute():
                embedding_path = root / embedding_path
            if not embedding_path.exists():
                raise FileNotFoundError(embedding_path)

            label_dir = output_dir / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = safe_name(record_key)
            label_file = label_dir / f"{stem}.labels.json"
            mask_file = label_dir / f"{stem}.mask.json"
            label_file.write_text(json.dumps(labels), encoding="utf-8")
            mask_file.write_text(json.dumps(mask), encoding="utf-8")
            writers[split].writerow(
                {
                    "id": record_key,
                    "native_id": catalogue.get("native_id", record.get("accession", "")),
                    "sequence_key": sequence_key,
                    "split": split,
                    "embedding": str(embedding_path),
                    "embedding_key": args.embedding_key,
                    "labels": str(label_file.relative_to(output_dir)),
                    "mask": str(mask_file.relative_to(output_dir)),
                    "length": length,
                }
            )
            counts[split] += 1
            label_counts["known"] += sum(mask)
            label_counts["unknown"] += len(mask) - sum(mask)
            label_counts["disorder"] += sum(label == 1 and is_valid for label, is_valid in zip(labels, mask))
            label_counts["order"] += sum(label == 0 and is_valid for label, is_valid in zip(labels, mask))
            policy_counts[str(record.get("merged_policy_id", "unknown"))] += 1
            selected += 1
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "data_root": str(root),
        "asset_type": args.asset_type,
        "embedding_key": args.embedding_key,
        "label_channel": args.label_channel,
        "unknown_label_value": 2,
        "mask_policy": (
            "record.label_mask"
            if args.label_channel == "merged"
            else "selected_channel_labels:unknown_only"
        ),
        "merged_policy_ids": dict(policy_counts),
        "topology_stratification": {
            "available": False,
            "reason": "No topology stratum column in the current handoff",
        },
        "splits": requested_splits,
        "sample_counts": dict(counts),
        "label_counts": dict(label_counts),
        "record_source": str(root / "development" / "records.jsonl.gz"),
        "catalogue_source": str(root / "catalogue" / "records.tsv"),
        "asset_source": str(root / "catalogue" / "assets.tsv"),
    }
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    build_manifests(parse_args())
