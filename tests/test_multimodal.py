from pathlib import Path
import csv
import json

import numpy as np

from my_predictor.data import ManifestDataset, load_manifest_embedding
from my_predictor.io import load_embedding_sources, read_manifest


def test_manifest_concatenates_embedding_sources(tmp_path: Path) -> None:
    first = np.arange(6, dtype=np.float32).reshape(3, 2)
    second = np.arange(9, dtype=np.float32).reshape(3, 3) + 100
    first_path = tmp_path / "esm2.npy"
    second_path = tmp_path / "prott5.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps([0, 1, 2]), encoding="utf-8")
    manifest_path = tmp_path / "train.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "embedding_sources", "labels"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerow({
            "id": "protein-1",
            "embedding_sources": json.dumps([
                {"family": "esm2", "path": first_path.name},
                {"family": "prott5", "path": second_path.name},
            ]),
            "labels": labels_path.name,
        })

    rows = read_manifest(manifest_path)
    embedding = load_manifest_embedding(rows[0])
    assert embedding.shape == (3, 5)
    np.testing.assert_array_equal(embedding[:, :2], first)
    np.testing.assert_array_equal(embedding[:, 2:], second)
    sample = ManifestDataset(manifest_path)[0]
    assert sample["embedding"].shape == (3, 5)
    np.testing.assert_array_equal(sample["labels"], [0.0, 1.0, 0.0])


def test_embedding_sources_reject_mismatched_lengths(tmp_path: Path) -> None:
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros((2, 2), dtype=np.float32))
    np.save(second, np.zeros((3, 2), dtype=np.float32))
    try:
        load_embedding_sources([{"path": str(first)}, {"path": str(second)}])
    except ValueError as exc:
        assert "different residue lengths" in str(exc)
    else:
        raise AssertionError("expected a length mismatch error")
