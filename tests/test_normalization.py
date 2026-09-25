from pathlib import Path

import numpy as np

from my_predictor.normalization import (
    apply_normalization,
    compute_standardization,
    make_metadata,
)


def test_train_statistics_and_application(tmp_path: Path) -> None:
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.asarray([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32))
    np.save(second, np.asarray([[5.0, 18.0]], dtype=np.float32))

    mean, scale = compute_standardization([first, second])
    np.testing.assert_allclose(mean, [3.0, 14.0])
    np.testing.assert_allclose(scale, [np.sqrt(8.0 / 3.0), np.sqrt(32.0 / 3.0)])

    metadata = make_metadata("standardize", mean=mean, scale=scale)
    transformed = apply_normalization(np.load(first), metadata)
    np.testing.assert_allclose(
        transformed.mean(axis=0),
        [(-1.0) / np.sqrt(8.0 / 3.0), (-2.0) / np.sqrt(32.0 / 3.0)],
    )


def test_none_normalization_is_identity() -> None:
    values = np.ones((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(values, apply_normalization(values, make_metadata("none")))
