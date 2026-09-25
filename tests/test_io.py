from pathlib import Path

import numpy as np

from my_predictor.io import load_embedding, read_fasta


def test_read_fasta_preserves_first_header_token(tmp_path: Path) -> None:
    path = tmp_path / "sample.fasta"
    path.write_text(
        ">P12345 component=one\nACD\nEF\n>Q9XYZ\nMNP\n",
        encoding="utf-8",
    )
    records = read_fasta(path)
    assert [record.id for record in records] == ["P12345", "Q9XYZ"]
    assert records[0].sequence == "ACDEF"


def test_load_embedding_npz_prefers_embedding_key(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    expected = np.ones((3, 4), dtype=np.float16)
    np.savez(path, embedding=expected, other=np.zeros((2, 2)))
    actual = load_embedding(path)
    assert actual.shape == (3, 4)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)
