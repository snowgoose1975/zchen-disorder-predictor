from __future__ import annotations

import torch
from torch import nn


class ResidueMLP(nn.Module):
    """A small independent prediction head applied to every residue."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        _lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, length, dimension = embeddings.shape
        if dimension != self.input_dim:
            raise ValueError(
                f"expected embedding dimension {self.input_dim}, got {dimension}"
            )
        logits = self.network(embeddings.reshape(batch_size * length, dimension))
        return logits.reshape(batch_size, length)


class ResidueLinear(nn.Module):
    """A linear residue classifier used as the simplest frozen-embedding baseline."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 0
        self.dropout = 0.0
        self.network = nn.Linear(input_dim, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        _lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[2] != self.input_dim:
            raise ValueError(
                f"expected [batch, length, {self.input_dim}], got {tuple(embeddings.shape)}"
            )
        return self.network(embeddings).squeeze(-1)


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = values
        values = self.activation(self.conv1(values))
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)
        values = self.dropout(self.conv2(values))
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)
        return self.activation(values + residual)


class ResidueTCN(nn.Module):
    """A compact residual temporal convolutional head for local residue context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        kernel_size: int = 5,
        layers: int = 3,
    ):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if layers < 1:
            raise ValueError("layers must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.layers = layers
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            _ResidualConvBlock(
                hidden_dim,
                kernel_size,
                dilation=2 ** layer_index,
                dropout=dropout,
            )
            for layer_index in range(layers)
        )
        self.output = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[2] != self.input_dim:
            raise ValueError(
                f"expected [batch, length, {self.input_dim}], got {tuple(embeddings.shape)}"
            )

        valid_mask = None
        if lengths is not None:
            lengths = lengths.reshape(-1).to(dtype=torch.long, device="cpu")
            if lengths.shape[0] != embeddings.shape[0]:
                raise ValueError("lengths must contain one value per batch item")
            if torch.any(lengths < 1) or torch.any(lengths > embeddings.shape[1]):
                raise ValueError("lengths must be within the padded sequence shape")
            positions = torch.arange(
                embeddings.shape[1], device=embeddings.device
            ).unsqueeze(0)
            valid_mask = (
                positions < lengths.to(device=embeddings.device).unsqueeze(1)
            ).unsqueeze(1)
            embeddings = embeddings.masked_fill(~valid_mask.transpose(1, 2), 0.0)

        values = self.input_projection(embeddings.transpose(1, 2))
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)
        for block in self.blocks:
            values = block(values, valid_mask)
            if valid_mask is not None:
                values = values.masked_fill(~valid_mask, 0.0)
        return self.output(values).squeeze(1)


class ResidueBiGRU(nn.Module):
    """A bidirectional recurrent head for broader residue context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_dim * 2, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[2] != self.input_dim:
            raise ValueError(
                f"expected [batch, length, {self.input_dim}], got {tuple(embeddings.shape)}"
            )
        if lengths is None:
            values, _ = self.gru(embeddings)
        else:
            lengths = lengths.reshape(-1).to(dtype=torch.long, device="cpu")
            if lengths.shape[0] != embeddings.shape[0]:
                raise ValueError("lengths must contain one value per batch item")
            if torch.any(lengths < 1) or torch.any(lengths > embeddings.shape[1]):
                raise ValueError("lengths must be within the padded sequence shape")
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_values, _ = self.gru(packed)
            values, _ = nn.utils.rnn.pad_packed_sequence(
                packed_values,
                batch_first=True,
                total_length=embeddings.shape[1],
            )
        return self.output(self.dropout_layer(values)).squeeze(-1)


class ResidueConvBiGRU(nn.Module):
    """A linear-memory local-context plus bidirectional recurrent head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.local_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_dim * 2, 1)

    @staticmethod
    def _valid_mask(
        lengths: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lengths.shape[0] != batch_size:
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(lengths < 1) or torch.any(lengths > sequence_length):
            raise ValueError("lengths must be within the padded sequence shape")
        positions = torch.arange(sequence_length, device=device).unsqueeze(0)
        return positions < lengths.to(device=device).unsqueeze(1)

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[2] != self.input_dim:
            raise ValueError(
                f"expected [batch, length, {self.input_dim}], got {tuple(embeddings.shape)}"
            )

        valid_mask = None
        if lengths is not None:
            cpu_lengths = lengths.reshape(-1).to(dtype=torch.long, device="cpu")
            valid_mask = self._valid_mask(
                cpu_lengths,
                embeddings.shape[0],
                embeddings.shape[1],
                embeddings.device,
            ).unsqueeze(-1)
            embeddings = embeddings.masked_fill(~valid_mask, 0.0)

        values = self.input_projection(embeddings)
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)
        values = self.normalization(values)
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)

        local = self.local_conv(values.transpose(1, 2)).transpose(1, 2)
        values = values + self.dropout_layer(self.activation(local))
        if valid_mask is not None:
            values = values.masked_fill(~valid_mask, 0.0)

        if lengths is None:
            contextual, _ = self.gru(values)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                values,
                cpu_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_contextual, _ = self.gru(packed)
            contextual, _ = nn.utils.rnn.pad_packed_sequence(
                packed_contextual,
                batch_first=True,
                total_length=embeddings.shape[1],
            )
        return self.output(self.dropout_layer(contextual)).squeeze(-1)


def build_model(
    architecture: str,
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    kernel_size: int = 5,
    layers: int = 3,
) -> nn.Module:
    if architecture == "linear":
        return ResidueLinear(input_dim)
    if architecture == "mlp":
        return ResidueMLP(input_dim, hidden_dim, dropout)
    if architecture == "tcn":
        return ResidueTCN(input_dim, hidden_dim, dropout, kernel_size, layers)
    if architecture == "bigru":
        return ResidueBiGRU(input_dim, hidden_dim, dropout)
    if architecture == "conv_bigru":
        return ResidueConvBiGRU(input_dim, hidden_dim, dropout, kernel_size)
    raise ValueError(f"unknown architecture: {architecture!r}")
