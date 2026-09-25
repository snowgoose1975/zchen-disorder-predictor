from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from my_predictor.data import ManifestDataset, load_manifest_embedding, pad_batch
from my_predictor.metrics import average_precision
from my_predictor.model import build_model
from my_predictor.normalization import compute_standardization, make_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residue-level disorder head")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--architecture", choices=("linear", "mlp", "tcn", "bigru", "conv_bigru"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--tcn-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--loss-weighting",
        choices=("residue", "protein"),
        default="residue",
        help="Residue-weighted BCE or equal-weighted mean loss across proteins",
    )
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--embedding-normalization",
        choices=("none", "standardize"),
        default="none",
        help="Optional train-split feature standardization saved in the checkpoint",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reduce_masked_loss(
    losses: torch.Tensor,
    valid: torch.Tensor,
    weighting: str,
) -> tuple[torch.Tensor, int]:
    """Return the batch loss and its reporting unit count."""
    if weighting == "residue":
        masked = losses[valid]
        if masked.numel() == 0:
            raise ValueError("a batch contains no known residue labels")
        return masked.mean(), int(masked.numel())
    if weighting == "protein":
        per_protein: list[torch.Tensor] = []
        for row_index in range(losses.shape[0]):
            protein_losses = losses[row_index][valid[row_index]]
            if protein_losses.numel():
                per_protein.append(protein_losses.mean())
        if not per_protein:
            raise ValueError("a batch contains no known residue labels")
        return torch.stack(per_protein).mean(), len(per_protein)
    raise ValueError(f"unknown loss weighting: {weighting!r}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    loss_weighting: str,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_loss_units = 0
    total_correct = 0
    total_valid = 0
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []

    for embeddings, labels, valid, _ids, lengths in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        valid = valid.to(device)
        with torch.set_grad_enabled(training):
            logits = model(embeddings, lengths)
            losses = criterion(logits, labels)
            loss, loss_units = reduce_masked_loss(losses, valid, loss_weighting)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits >= 0
        total_correct += (predictions[valid] == labels[valid].bool()).sum().item()
        total_valid += valid.sum().item()
        total_loss += loss.item() * loss_units
        total_loss_units += loss_units
        score_chunks.append(logits.detach().cpu().numpy())
        label_chunks.append(labels.detach().cpu().numpy())
        valid_chunks.append(valid.detach().cpu().numpy())

    if total_valid == 0 or total_loss_units == 0:
        raise ValueError("no known residue labels in this split")
    scores = np.concatenate([chunk.reshape(-1) for chunk in score_chunks])
    gold = np.concatenate([chunk.reshape(-1) for chunk in label_chunks])
    masks = np.concatenate([chunk.reshape(-1) for chunk in valid_chunks])
    ap = average_precision(scores, gold, masks)
    return total_loss / total_loss_units, total_correct / total_valid, ap

def should_save_checkpoint(
    checkpoint_path: Path,
    validation_ap: float,
    best_validation_ap: float,
) -> bool:
    """Always retain a first-epoch fallback when validation AP is undefined."""
    return (
        not checkpoint_path.exists()
        or (
            np.isfinite(validation_ap)
            and validation_ap > best_validation_ap
        )
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)

    train_probe = ManifestDataset(args.train_manifest)
    if args.embedding_normalization == "standardize":
        mean, scale = compute_standardization(
            load_manifest_embedding(row) for row in train_probe.rows
        )
        feature_normalization = make_metadata(
            "standardize",
            mean=mean,
            scale=scale,
        )
    else:
        feature_normalization = make_metadata("none")

    train_data = ManifestDataset(args.train_manifest, feature_normalization)
    val_data = ManifestDataset(args.val_manifest, feature_normalization)
    first_embedding = train_data[0]["embedding"]
    input_dim = int(first_embedding.shape[1])

    device = choose_device(args.device)
    model = build_model(
        args.architecture,
        input_dim,
        args.hidden_dim,
        args.dropout,
        args.tcn_kernel_size,
        args.tcn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    pos_weight = None
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=pos_weight,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_batch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "val_manifest": str(Path(args.val_manifest).resolve()),
        "embedding_key": train_data.rows[0].get("embedding_key") or None,
        "embedding_sources": train_data.rows[0].get("embedding_sources") or None,
        "input_dim": input_dim,
        "device_used": str(device),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "embedding_normalization": feature_normalization["method"],
        "loss_weighting": args.loss_weighting,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    best_val_ap = -float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_ap = run_epoch(
            model, train_loader, device, criterion, optimizer, args.loss_weighting
        )
        with torch.no_grad():
            val_loss, val_accuracy, val_ap = run_epoch(
                model, val_loader, device, criterion, optimizer=None,
                loss_weighting=args.loss_weighting,
            )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} "
            f"train_ap={train_ap:.6f} val_loss={val_loss:.6f} "
            f"val_acc={val_accuracy:.4f} val_ap={val_ap:.6f}"
        )
        checkpoint_path = output_dir / "best.pt"
        if should_save_checkpoint(checkpoint_path, val_ap, best_val_ap):
            if np.isfinite(val_ap):
                best_val_ap = val_ap
            torch.save(
                {
                    "model": args.architecture,
                    "architecture": args.architecture,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "tcn_kernel_size": args.tcn_kernel_size,
                    "tcn_layers": args.tcn_layers,
                    "embedding_normalization": feature_normalization,
                    "loss_weighting": args.loss_weighting,
                    "state_dict": model.state_dict(),
                    "config": config,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
