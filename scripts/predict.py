from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from my_predictor.embedding import ESM2Encoder
from my_predictor.io import (
    FastaRecord,
    load_embedding,
    read_embedding_index,
    read_fasta,
)
from my_predictor.model import build_model
from my_predictor.normalization import apply_normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict residue-level disorder scores")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embedding", help="One .npy/.npz residue embedding")
    source.add_argument("--fasta", help="FASTA file with one or more sequences")
    parser.add_argument("--id", help="ID used with a standalone --embedding")
    parser.add_argument(
        "--embedding-dir",
        help="Directory containing <FASTA_ID>.npy or <FASTA_ID>.npz files",
    )
    parser.add_argument(
        "--embedding-index",
        help="TSV mapping sequence IDs to embedding files",
    )
    parser.add_argument("--embedding-key", help="Array key inside an NPZ embedding archive")
    parser.add_argument(
        "--esm2-model",
        help="ESM2 model name or local directory for FASTA input",
    )
    parser.add_argument(
        "--esm2-layer",
        type=int,
        default=None,
        help="Hidden-state layer for ESM2; default uses the final hidden state",
    )
    parser.add_argument(
        "--esm2-window-size",
        type=int,
        default=None,
        help="Residues per ESM2 window; use 1022 for the current handoff",
    )
    parser.add_argument(
        "--esm2-overlap",
        type=int,
        default=128,
        help="Overlap between long-sequence ESM2 windows",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_embedding(
    record: FastaRecord,
    args: argparse.Namespace,
    index: dict[str, Path] | None,
    encoder: ESM2Encoder | None,
) -> np.ndarray:
    sources = sum(
        value is not None
        for value in (args.embedding_dir, args.embedding_index, args.esm2_model)
    )
    if sources != 1:
        raise ValueError(
            "with --fasta, choose exactly one of --embedding-dir, "
            "--embedding-index, or --esm2-model"
        )

    if args.embedding_dir:
        directory = Path(args.embedding_dir)
        candidates = [directory / f"{record.id}.npz", directory / f"{record.id}.npy"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"no embedding for {record.id} in {directory}; tried {candidates}"
            )
        return load_embedding(path, args.embedding_key)

    if args.embedding_index:
        assert index is not None
        if record.id not in index:
            raise KeyError(f"{record.id} is missing from embedding index")
        return load_embedding(index[record.id], args.embedding_key)

    if encoder is None:
        raise RuntimeError("the ESM2 encoder was not initialized")
    return encoder.embed(record.sequence)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    architecture = checkpoint.get("architecture", "mlp")
    model = build_model(
        architecture,
        input_dim,
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
        int(checkpoint.get("tcn_kernel_size", 5)),
        int(checkpoint.get("tcn_layers", 3)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, input_dim, checkpoint.get("embedding_normalization")


def check_embedding_shape(
    identifier: str,
    embedding: np.ndarray,
    expected_dim: int,
    expected_length: int | None = None,
) -> None:
    if embedding.shape[1] != expected_dim:
        raise ValueError(
            f"{identifier}: embedding dimension {embedding.shape[1]} does not match "
            f"checkpoint input_dim={expected_dim}; use the same encoder family/layer "
            "as the training data"
        )
    if expected_length is not None and embedding.shape[0] != expected_length:
        raise ValueError(
            f"{identifier}: embedding length {embedding.shape[0]} "
            f"!= FASTA length {expected_length}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, expected_dim, normalization = load_checkpoint(args.checkpoint, device)

    jobs: list[tuple[str, np.ndarray]] = []
    if args.embedding:
        identifier = args.id or Path(args.embedding).stem
        embedding = load_embedding(args.embedding, args.embedding_key)
        check_embedding_shape(identifier, embedding, expected_dim)
        embedding = apply_normalization(embedding, normalization)
        jobs.append((identifier, embedding))
    else:
        records = read_fasta(args.fasta)
        index = read_embedding_index(args.embedding_index) if args.embedding_index else None
        encoder = (
            ESM2Encoder(
                args.esm2_model,
                device=str(device),
                layer=args.esm2_layer,
                window_size=args.esm2_window_size,
                overlap=args.esm2_overlap,
            )
            if args.esm2_model
            else None
        )
        for record in records:
            embedding = resolve_embedding(record, args, index, encoder)
            check_embedding_shape(
                record.id,
                embedding,
                expected_dim,
                expected_length=len(record.sequence),
            )
            embedding = apply_normalization(embedding, normalization)
            jobs.append((record.id, embedding))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "position", "score"])
        for identifier, embedding in jobs:
            tensor = torch.from_numpy(embedding).unsqueeze(0).to(device)
            with torch.inference_mode():
                scores = torch.sigmoid(model(tensor))[0].cpu().numpy()
            for position, score in enumerate(scores, start=1):
                writer.writerow([identifier, position, f"{float(score):.8f}"])


if __name__ == "__main__":
    main()
