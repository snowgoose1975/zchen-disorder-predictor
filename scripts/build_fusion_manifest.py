from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_manifest import parse_vector, read_records, read_tsv, safe_name, select_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifests that concatenate multiple residue embedding families"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-types", default="esm2,prott5,esmc")
    parser.add_argument("--embedding-keys", default="layer_30,layer_24,layer_30")
    parser.add_argument(
        "--label-channel",
        choices=("merged", "curated", "nmr_chemical_shift", "structure_derived"),
        default="merged",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_names(value: str, name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in values):
        raise ValueError(f"{name} contains an unsafe name")
    return values


def build_manifests(args: argparse.Namespace) -> None:
    asset_types = parse_names(args.asset_types, "asset-types")
    embedding_keys = [item.strip() for item in args.embedding_keys.split(",") if item.strip()]
    if len(asset_types) != len(embedding_keys):
        raise ValueError("--asset-types and --embedding-keys must have equal lengths")
    if len(asset_types) < 2:
        raise ValueError("fusion requires at least two embedding families")

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
    assets: dict[tuple[str, str], dict[str, str]] = {}
    for asset in read_tsv(root / "catalogue" / "assets.tsv"):
        asset_type = asset.get("asset_type", "")
        sequence_key = asset.get("sequence_key", "")
        if asset_type in asset_types and sequence_key:
            key = (sequence_key, asset_type)
            if key in assets:
                raise ValueError(f"multiple {asset_type} assets for sequence_key={sequence_key}")
            assets[key] = asset

    fieldnames = [
        "id", "native_id", "sequence_key", "split",
        "embedding_sources", "labels", "mask", "length",
    ]
    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    try:
        for split in requested_splits:
            handle = (output_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="")
            handles[split] = handle
            writers[split] = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writers[split].writeheader()

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
            sequence = str(record.get("sequence", ""))
            length = int(catalogue.get("length") or len(sequence))
            if len(sequence) != length:
                raise ValueError(f"{record_key}: catalogue length does not match sequence")
            labels = select_labels(record, args.label_channel, name=f"{record_key}.{args.label_channel}")
            if args.label_channel == "merged":
                mask = [bool(item) for item in parse_vector(record.get("label_mask"), name=f"{record_key}.label_mask")]
            else:
                mask = [label != 2 for label in labels]
            if len(labels) != length or len(mask) != length:
                raise ValueError(f"{record_key}: labels/mask length does not match {length}")

            sequence_key = catalogue.get("sequence_key", "")
            sources: list[dict[str, str]] = []
            for asset_type, embedding_key in zip(asset_types, embedding_keys):
                asset = assets.get((sequence_key, asset_type))
                if asset is None:
                    raise ValueError(f"{record_key}: missing {asset_type} asset for {sequence_key}")
                asset_path = Path(asset["path"])
                if not asset_path.is_absolute():
                    asset_path = root / asset_path
                asset_path = asset_path.resolve()
                if not asset_path.exists():
                    raise FileNotFoundError(asset_path)
                sources.append({
                    "family": asset_type,
                    "key": embedding_key,
                    "path": str(asset_path),
                })

            label_dir = output_dir / "labels" / split
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = safe_name(record_key)
            label_file = label_dir / f"{stem}.labels.json"
            mask_file = label_dir / f"{stem}.mask.json"
            label_file.write_text(json.dumps(labels), encoding="utf-8")
            mask_file.write_text(json.dumps(mask), encoding="utf-8")
            writers[split].writerow({
                "id": record_key,
                "native_id": catalogue.get("native_id", ""),
                "sequence_key": sequence_key,
                "split": split,
                "embedding_sources": json.dumps(sources, separators=(",", ":"), sort_keys=True),
                "labels": str(label_file.relative_to(output_dir)),
                "mask": str(mask_file.relative_to(output_dir)),
                "length": length,
            })
            known = sum(mask)
            counts[split] += 1
            label_counts["known"] += known
            label_counts["unknown"] += len(mask) - known
            label_counts["disorder"] += sum(label == 1 and valid for label, valid in zip(labels, mask))
            label_counts["order"] += sum(label == 0 and valid for label, valid in zip(labels, mask))
            policy_counts[str(record.get("merged_policy_id", "unknown"))] += 1
            selected += 1
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "data_root": str(root),
        "mode": "concatenate_residue_embeddings",
        "asset_types": asset_types,
        "embedding_keys": embedding_keys,
        "unknown_label_value": 2,
        "label_channel": args.label_channel,
        "splits": requested_splits,
        "sample_counts": dict(counts),
        "label_counts": dict(label_counts),
        "merged_policy_ids": dict(policy_counts),
        "topology_stratification": {
            "available": False,
            "reason": "No per-residue topology stratum is included in the current handoff",
        },
        "record_source": str(root / "development" / "records.jsonl.gz"),
        "catalogue_source": str(root / "catalogue" / "records.tsv"),
        "asset_source": str(root / "catalogue" / "assets.tsv"),
    }
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    build_manifests(parse_args())
