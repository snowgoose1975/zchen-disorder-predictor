from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from my_predictor.data import load_manifest_embedding
from my_predictor.io import read_manifest
from my_predictor.model import build_model
from my_predictor.normalization import apply_normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict every record listed in an embedding manifest"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint.get("architecture", "mlp"),
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
        int(checkpoint.get("tcn_kernel_size", 5)),
        int(checkpoint.get("tcn_layers", 3)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("embedding_normalization")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, normalization = load_checkpoint(args.checkpoint, device)
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "position", "score"])
        for row in rows:
            embedding = load_manifest_embedding(row)
            embedding = apply_normalization(embedding, normalization)
            tensor = torch.from_numpy(embedding).unsqueeze(0).to(device)
            with torch.inference_mode():
                scores = torch.sigmoid(model(tensor))[0].cpu().numpy()
            for position, score in enumerate(scores, start=1):
                writer.writerow([row["id"], position, f"{float(score):.8f}"])


if __name__ == "__main__":
    main()
