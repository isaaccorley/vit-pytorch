import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)