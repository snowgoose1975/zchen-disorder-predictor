from __future__ import annotations

from typing import Any

import numpy as np


def window_starts(length: int, window_size: int, overlap: int) -> list[int]:
    """Return deterministic overlapping window starts covering a sequence."""
    if length < 1:
        raise ValueError("length must be positive")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    if length <= window_size:
        return [0]

    stride = window_size - overlap
    last_start = length - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def esm2_embed(
    sequence: str,
    model_name: str,
    *,
    device: str = "auto",
    layer: int | None = None,
    window_size: int | None = None,
    overlap: int = 128,
) -> np.ndarray:
    """Generate one residue embedding with a local/Hugging Face ESM2 model.

    This is an optional adapter. The fixed-embedding training path does not
    import Transformers, so it remains usable without the optional dependency.
    """
    encoder = ESM2Encoder(
        model_name,
        device=device,
        layer=layer,
        window_size=window_size,
        overlap=overlap,
    )
    return encoder.embed(sequence)


class ESM2Encoder:
    """Reusable ESM2 encoder for the FASTA input path.

    The prediction head is trained on a fixed embedding family and dimension.
    This adapter must therefore use the same ESM2 checkpoint and layer as the
    embeddings used during training. Long sequences use overlapping windows
    and average the embeddings in overlap regions.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        layer: int | None = None,
        window_size: int | None = None,
        overlap: int = 128,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ESM2 generation requires requirements-esm2.txt"
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected_device == "auto":
            selected_device = "cpu"

        self.model_name = model_name
        self.layer = layer
        self.device = torch.device(selected_device)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=layer is not None,
        ).to(self.device)
        self.model.eval()

        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions is None:
            raise ValueError("ESM2 model does not declare max_position_embeddings")
        self.max_positions = int(max_positions)
        max_residue_length = self.max_positions - 2
        if max_residue_length < 1:
            raise ValueError("ESM2 model context is too small for residue encoding")
        self.window_size = (
            max_residue_length if window_size is None else int(window_size)
        )
        if self.window_size > max_residue_length:
            raise ValueError(
                f"window_size={self.window_size} exceeds model residue context "
                f"{max_residue_length}"
            )
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= self.window_size:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")
        self.overlap = int(overlap)

    def _embed_window(self, sequence: str) -> np.ndarray:
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        if tokens["input_ids"].shape[1] > self.max_positions:
            raise ValueError(
                f"window length {len(sequence)} exceeds this ESM2 model context "
                f"({self.max_positions - 2} residues)"
            )

        with self._torch.inference_mode():
            outputs: Any = self.model(**tokens)

        if self.layer is None:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[self.layer]

        residue = hidden[0, 1 : 1 + len(sequence), :]
        if residue.shape[0] != len(sequence):
            raise ValueError(
                "tokenizer/model output is not one embedding per input residue; "
                "check the tokenizer and sequence alphabet"
            )
        return residue.detach().float().cpu().numpy()

    def embed(self, sequence: str) -> np.ndarray:
        """Return a float32 residue matrix with shape [length, dimension]."""
        if not sequence:
            raise ValueError("cannot embed an empty sequence")
        if any(character.isspace() for character in sequence):
            raise ValueError("sequence must not contain whitespace")

        starts = window_starts(len(sequence), self.window_size, self.overlap)
        accumulator: np.ndarray | None = None
        counts = np.zeros(len(sequence), dtype=np.float32)
        for start in starts:
            end = min(start + self.window_size, len(sequence))
            values = self._embed_window(sequence[start:end])
            if accumulator is None:
                accumulator = np.zeros(
                    (len(sequence), values.shape[1]),
                    dtype=np.float32,
                )
            accumulator[start:end] += values
            counts[start:end] += 1.0

        assert accumulator is not None
        return accumulator / counts[:, None]
