from pathlib import Path

import numpy as np

from scripts.evaluate import grouped_metrics, read_strata


def test_read_protein_and_residue_strata(tmp_path: Path) -> None:
    protein_path = tmp_path / "protein.tsv"
    protein_path.write_text("id\tstratum\np1\tglobular\np2\tdisordered\n", encoding="utf-8")
    mode, values = read_strata(protein_path)
    assert mode == "protein"
    assert values == {"p1": "globular", "p2": "disordered"}

    residue_path = tmp_path / "residue.tsv"
    residue_path.write_text(
        "id\tposition\tstratum\n"
        "p1\t1\ttm\n"
        "p1\t2\ttm\n"
        "p1\t3\tsoluble\n",
        encoding="utf-8",
    )
    mode, values = read_strata(residue_path)
    assert mode == "residue"
    assert values == {"p1": {1: "tm", 2: "tm", 3: "soluble"}}


def test_residue_strata_grouping() -> None:
    scores = np.asarray([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    labels = np.asarray([1, 1, 0, 0, 1, 0])
    valid = np.ones(6, dtype=bool)
    spans = [("p1", 0, 4), ("p2", 4, 6)]
    strata = {
        "p1": {1: "tm", 2: "tm", 3: "soluble", 4: "soluble"},
        "p2": {1: "soluble", 2: "soluble"},
    }

    result = grouped_metrics(scores, labels, valid, spans, strata, "residue", 0.5)
    assert result["tm"]["unit"] == "residue"
    assert result["tm"]["protein_count"] == 1
    assert result["tm"]["valid_residues"] == 2
    assert result["soluble"]["protein_count"] == 2
    assert result["soluble"]["valid_residues"] == 4


def test_residue_strata_require_complete_positions() -> None:
    scores = np.asarray([0.9, 0.8])
    labels = np.asarray([1, 0])
    valid = np.ones(2, dtype=bool)
    try:
        grouped_metrics(
            scores,
            labels,
            valid,
            [("p1", 0, 2)],
            {"p1": {1: "tm"}},
            "residue",
            0.5,
        )
    except ValueError as exc:
        assert "must cover positions" in str(exc)
    else:
        raise AssertionError("expected incomplete residue strata to fail")
