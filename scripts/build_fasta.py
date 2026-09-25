from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export split FASTA files from a TM disorder handoff"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def wrap(sequence: str, width: int = 80) -> list[str]:
    return [sequence[start : start + width] for start in range(0, len(sequence), width)]


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    data_root = Path(args.data_root).resolve()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    catalogue = read_tsv(data_root / "catalogue" / "records.tsv")
    records = read_records(data_root / "development" / "records.jsonl.gz")
    selected: list[dict[str, str]] = []
    for row in catalogue:
        if row.get("split") != args.split:
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError(f"no records found for split {args.split!r}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            record_key = row.get("record_key", "")
            record = records.get(record_key)
            if record is None:
                raise ValueError(f"missing JSONL record for {record_key}")
            sequence = str(record.get("sequence", "")).strip().replace(" ", "").upper()
            expected_length = int(row.get("length") or len(sequence))
            if len(sequence) != expected_length:
                raise ValueError(
                    f"{record_key}: catalogue length {expected_length} "
                    f"!= sequence length {len(sequence)}"
                )
            native_id = row.get("native_id") or record.get("accession", "")
            sequence_key = row.get("sequence_key", "")
            handle.write(
                f">{record_key} native_id={native_id} sequence_key={sequence_key}\n"
            )
            handle.write("\n".join(wrap(sequence)) + "\n")

    metadata = {
        "data_root": str(data_root),
        "output": str(output.resolve()),
        "split": args.split,
        "record_count": len(selected),
        "ids": [row["record_key"] for row in selected],
        "lengths": {row["record_key"]: int(row["length"]) for row in selected},
        "record_source": str((data_root / "development" / "records.jsonl.gz").resolve()),
        "catalogue_source": str((data_root / "catalogue" / "records.tsv").resolve()),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
