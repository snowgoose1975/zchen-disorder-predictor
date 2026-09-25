from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_embedding


def compute_standardization(
    embeddings: Iterable[str | Path | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute population mean/std over all training residues."""
    total: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    count = 0

    for source in embeddings:
        if isinstance(source, (str, Path)):
            values = load_embedding(source).astype(np.float64, copy=False)
        else:
            values = np.asarray(source)
            if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
                raise ValueError("embedding arrays must be numeric [length, dimension]")
            values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("embedding contains NaN or infinite values")
        if total is None:
            dimension = values.shape[1]
            total = np.zeros(dimension, dtype=np.float64)
            total_squared = np.zeros(dimension, dtype=np.float64)
        if values.shape[1] != total.shape[0]:
            raise ValueError(
                f"embedding dimension {values.shape[1]} does not match "
                f"the first dimension {total.shape[0]} while computing statistics"
            )
        total += values.sum(axis=0)
        total_squared += np.square(values).sum(axis=0)
        count += values.shape[0]

    if total is None or total_squared is None or count == 0:
        raise ValueError("cannot compute embedding statistics from an empty collection")

    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def make_metadata(
    method: str,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> dict[str, Any]:
    if method == "none":
        return {"method": "none"}
    if method != "standardize":
        raise ValueError(f"unknown embedding normalization method: {method!r}")
    if mean is None or scale is None:
        raise ValueError("standardize metadata needs mean and scale")
    if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
        raise ValueError("mean and scale must be one-dimensional arrays of equal length")
    return {
        "method": "standardize",
        "mean": mean.astype(np.float32).tolist(),
        "scale": scale.astype(np.float32).tolist(),
    }


def apply_normalization(
    embedding: np.ndarray,
    metadata: dict[str, Any] | None,
) -> np.ndarray:
    if not metadata or metadata.get("method", "none") == "none":
        return embedding
    if metadata.get("method") != "standardize":
        raise ValueError(f"unknown checkpoint normalization method: {metadata.get('method')!r}")
    mean = np.asarray(metadata.get("mean"), dtype=np.float32)
    scale = np.asarray(metadata.get("scale"), dtype=np.float32)
    if mean.ndim != 1 or scale.ndim != 1 or mean.shape != scale.shape:
        raise ValueError("checkpoint normalization mean/scale are malformed")
    if embedding.ndim != 2 or embedding.shape[1] != mean.shape[0]:
        raise ValueError(
            f"embedding dimension {embedding.shape[1]} does not match "
            f"normalization dimension {mean.shape[0]}"
        )
    normalized = (embedding.astype(np.float32, copy=False) - mean[None, :]) / scale[None, :]
    if not np.isfinite(normalized).all():
        raise ValueError("normalized embedding contains NaN or infinite values")
    return normalized
