from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from my_predictor.io import FastaRecord, read_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a baseline raw output to id/position/score TSV"
    )
    parser.add_argument(
        "--tool",
        choices=("iupred2a", "aiupred", "metapredict", "fldpnn", "tabular"),
        required=True,
    )
    parser.add_argument("--input", required=True, help="Raw predictor output")
    parser.add_argument("--output", required=True, help="Normalized TSV output")
    parser.add_argument("--fasta", help="Optional FASTA used to validate IDs and lengths")
    parser.add_argument(
        "--id",
        help="Fallback protein ID for single-protein outputs without an ID column",
    )
    parser.add_argument(
        "--score-direction",
        choices=("higher-is-disorder", "lower-is-disorder"),
        default="higher-is-disorder",
        help="Whether larger raw values indicate more disorder",
    )
    parser.add_argument("--metadata", help="Optional metadata JSON path")
    return parser.parse_args()


def numeric(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"not a numeric score: {value!r}") from exc


def fallback_id(records: list[FastaRecord], explicit_id: str | None) -> str | None:
    if explicit_id:
        return explicit_id
    if len(records) == 1:
        return records[0].id
    return None


def parse_positioned_lines(
    path: Path,
    identifier: str | None,
) -> list[tuple[str, int, float]]:
    if identifier is None:
        raise ValueError(
            f"{path}: positional output has no protein ID; pass --id or a single-record --fasta"
        )
    rows: list[tuple[str, int, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        try:
            position = int(fields[0])
            score = numeric(fields[2])
        except ValueError:
            continue
        rows.append((identifier, position, score))
    if not rows:
        raise ValueError(f"{path}: no positional scores were found")
    return rows


def parse_metapredict(path: Path) -> list[tuple[str, int, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"protein_id", "position", "disorder_score"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path}: expected columns {sorted(required)}; got {reader.fieldnames}"
            )
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    (
                        row["protein_id"],
                        int(row["position"]),
                        numeric(row["disorder_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed metapredict row") from exc
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def parse_tabular(path: Path, identifier: str | None) -> list[tuple[str, int, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "," if "," in sample.splitlines()[0] else "	"
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        names = {name.strip().lower(): name for name in reader.fieldnames}
        id_name = next(
            (names[name] for name in ("id", "protein_id", "sequence_id") if name in names),
            None,
        )
        position_name = next(
            (names[name] for name in ("position", "pos", "residue_position") if name in names),
            None,
        )
        score_name = next(
            (
                names[name]
                for name in ("score", "disorder_score", "propensity", "prediction")
                if name in names
            ),
            None,
        )
        if position_name is None or score_name is None:
            raise ValueError(
                f"{path}: need position and score-like columns; got {reader.fieldnames}"
            )
        rows: list[tuple[str, int, float]] = []
        for line_number, row in enumerate(reader, start=2):
            current_id = row.get(id_name, "") if id_name else identifier
            if not current_id:
                raise ValueError(f"{path}:{line_number}: missing protein ID")
            rows.append(
                (current_id, int(row[position_name]), numeric(row[score_name]))
            )
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def parse_fldpnn(path: Path, identifier_override: str | None) -> list[tuple[str, int, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[tuple[str, int, float]] = []
    index = 0
    while index < len(lines):
        header = lines[index]
        if not header.startswith(">") or index + 3 >= len(lines):
            index += 1
            continue
        identifier = identifier_override or header[1:].split()[0]
        try:
            scores = [numeric(value) for value in lines[index + 3].split(",")]
        except ValueError:
            index += 1
            continue
        rows.extend((identifier, position, score) for position, score in enumerate(scores, start=1))
        index += 4
    if not rows:
        raise ValueError(f"{path}: no flDPnn propensity block was found")
    return rows


def validate_rows(
    rows: list[tuple[str, int, float]],
    records: list[FastaRecord],
) -> None:
    if not rows:
        raise ValueError("no scores to validate")
    expected = {record.id: len(record.sequence) for record in records}
    by_id: dict[str, list[tuple[int, float]]] = {}
    for identifier, position, score in rows:
        if not identifier:
            raise ValueError("empty protein ID")
        if position < 1:
            raise ValueError(f"{identifier}: residue positions must start at 1")
        if not score == score or score in (float("inf"), float("-inf")):
            raise ValueError(f"{identifier}:{position}: score is not finite")
        by_id.setdefault(identifier, []).append((position, score))

    if records:
        unknown = sorted(set(by_id) - set(expected))
        if unknown:
            raise ValueError(f"output IDs absent from FASTA: {unknown[:5]}")
        for identifier, length in expected.items():
            if identifier not in by_id:
                raise ValueError(f"missing output for FASTA record {identifier}")
            positions = [position for position, _ in by_id[identifier]]
            if positions != list(range(1, length + 1)):
                raise ValueError(
                    f"{identifier}: positions do not exactly match 1..{length}"
                )
    else:
        for identifier, values in by_id.items():
            positions = [position for position, _ in values]
            if len(set(positions)) != len(positions):
                raise ValueError(f"{identifier}: duplicate residue positions")


def transform_scores(
    rows: Iterable[tuple[str, int, float]],
    direction: str,
) -> list[tuple[str, int, float]]:
    if direction == "higher-is-disorder":
        return list(rows)
    return [(identifier, position, -score) for identifier, position, score in rows]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = read_fasta(args.fasta) if args.fasta else []
    single_id = fallback_id(records, args.id)

    if args.tool in {"iupred2a", "aiupred"}:
        rows = parse_positioned_lines(input_path, single_id)
    elif args.tool == "metapredict":
        rows = parse_metapredict(input_path)
    elif args.tool == "fldpnn":
        rows = parse_fldpnn(input_path, single_id)
    else:
        rows = parse_tabular(input_path, single_id)

    rows = transform_scores(rows, args.score_direction)
    validate_rows(rows, records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "position", "score"])
        writer.writerows(rows)

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )
    metadata = {
        "tool": args.tool,
        "raw_input": str(input_path.resolve()),
        "normalized_output": str(output_path.resolve()),
        "fasta": str(Path(args.fasta).resolve()) if args.fasta else None,
        "score_direction": args.score_direction,
        "score_transform": "identity"
        if args.score_direction == "higher-is-disorder"
        else "negate",
        "record_count": len({identifier for identifier, _, _ in rows}),
        "row_count": len(rows),
        "records": [
            {
                "id": identifier,
                "positions": len(values),
                "first_position": min(position for position, _ in values),
                "last_position": max(position for position, _ in values),
            }
            for identifier, values in sorted(
                (
                    (identifier, [(position, score) for current_id, position, score in rows if current_id == identifier])
                    for identifier in {current_id for current_id, _, _ in rows}
                ),
                key=lambda item: item[0],
            )
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
