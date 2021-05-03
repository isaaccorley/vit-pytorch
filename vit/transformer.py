import torch
import torch.nn as nn



class Attention(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.model = nn.Sequential()


class TransformerEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()


class TransformerEncoderSimple(nn.Module):

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
