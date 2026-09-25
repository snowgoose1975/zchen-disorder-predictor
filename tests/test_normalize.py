import csv
from pathlib import Path

from scripts.normalize_baseline import main


def run_cli(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["normalize_baseline.py", *arguments],
    )
    main()


def write_fasta(path: Path) -> None:
    path.write_text(">P1 example\nMSE\n", encoding="utf-8")


def test_iupred_and_aiupred_positioned_output(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    write_fasta(fasta)
    raw = tmp_path / "raw.txt"
    raw.write_text("# header\n1 M 0.2\n2 S 0.8\n3 E 0.4\n", encoding="utf-8")
    output = tmp_path / "normalized.tsv"

    run_cli(
        monkeypatch,
        "--tool",
        "iupred2a",
        "--input",
        str(raw),
        "--fasta",
        str(fasta),
        "--output",
        str(output),
    )
    assert output.read_text(encoding="utf-8").splitlines() == [
        "id\tposition\tscore",
        "P1\t1\t0.2",
        "P1\t2\t0.8",
        "P1\t3\t0.4",
    ]


def test_metapredict_csv(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    write_fasta(fasta)
    raw = tmp_path / "raw.csv"
    with raw.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "position", "residue", "disorder_score"])
        writer.writerows([["P1", 1, "M", 0.1], ["P1", 2, "S", 0.2], ["P1", 3, "E", 0.3]])
    output = tmp_path / "normalized.tsv"

    run_cli(
        monkeypatch,
        "--tool",
        "metapredict",
        "--input",
        str(raw),
        "--fasta",
        str(fasta),
        "--output",
        str(output),
    )
    assert output.read_text(encoding="utf-8").count("\n") == 4


def test_fldpnn_propensity_block(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "input.fasta"
    write_fasta(fasta)
    raw = tmp_path / "raw.csv"
    raw.write_text(
        ">P1\nM,S,E\n1,0,1\n0.1,0.7,0.2\n"
        "1,1,1\n0.2,0.2,0.2\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.tsv"

    run_cli(
        monkeypatch,
        "--tool",
        "fldpnn",
        "--input",
        str(raw),
        "--fasta",
        str(fasta),
        "--output",
        str(output),
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == "P1\t3\t0.2"


def test_lower_score_direction_negates_scores(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw.tsv"
    raw.write_text("id\tposition\tscore\nP1\t1\t2.0\n", encoding="utf-8")
    output = tmp_path / "normalized.tsv"

    run_cli(
        monkeypatch,
        "--tool",
        "tabular",
        "--input",
        str(raw),
        "--id",
        "P1",
        "--score-direction",
        "lower-is-disorder",
        "--output",
        str(output),
    )
    assert output.read_text(encoding="utf-8").splitlines()[-1] == "P1\t1\t-2.0"
